"""Orchestrate all runtime stages for shelf-image processing.

The pipeline owns stage ordering and shared runtime components. For each image
it detects products and price tags, recognizes product crops, extracts prices,
classifies promotional tags, aligns tags to products, and persists tabular and
annotated outputs. Stage-specific algorithms remain in their domain modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .config import UnifiedPipelineConfig
from .diagnostics import environment_report
from .image_processing.image_io import (
    crop_image,
    crop_price_tag,
    list_image_paths,
    load_image_bgr,
    resolve_path,
    safe_stem,
    save_crop,
)
from .image_processing.object_detection import YoloDetector, custom_nms
from .models import Detection, OcrItem, PipelineResult, PriceCandidate
from .price_extraction.price_parsing import (
    flatten_ocr_result,
    price_candidates_from_ocr,
    select_price_candidates,
)
from .price_extraction.text_extraction import PaddlePriceExtractor
from .result_output.annotations import build_annotated_image
from .result_output.schemas import (
    PRICE_TAG_COLUMNS,
    PRODUCT_COLUMNS,
    join_price_values,
    join_prices,
    price_count,
)
from .result_output.writers import (
    write_combined_outputs,
    write_image_outputs,
)
from .shelf_analysis.product_alignment import align_price_tags_to_products
from .shelf_analysis.promotion_classification import (
    SALE_CLASSIFICATION_COLUMNS,
    SaleClassifier,
    SaleClassifierConfig,
    SaleTag,
)


def _disabled_ocr_result() -> dict:
    """Build an empty price-extraction result with the normal schema.

    Input:
        None.
    Output:
        A result dictionary marking text extraction as disabled.
    """
    return {
        "primary_price": None,
        "primary_value": None,
        "primary_price_confidence": None,
        "secondary_price": None,
        "secondary_value": None,
        "secondary_price_confidence": None,
        "tertiary_price": None,
        "tertiary_value": None,
        "tertiary_price_confidence": None,
        "price_count": 0,
        "prices": None,
        "price_values": None,
        "ocr_error": "ocr_disabled",
        "ocr_text": "",
    }


class ShelfPipeline:
    """Coordinate all runtime stages for one or more shelf images."""

    def __init__(
        self,
        config: Optional[UnifiedPipelineConfig] = None,
    ):
        """Initialize reusable adapters for all configured pipeline stages.

        Input:
            config: Optional runtime settings. Project defaults are used when
                omitted.
        Output:
            None. The pipeline instance stores configured stage adapters.
        """
        self.config = config or UnifiedPipelineConfig()

        # Detector and price-extraction adapters load heavyweight models lazily.
        self.product_detector = YoloDetector(
            self.config.shelf_detector_path,
            self.config.device,
        )
        self.price_tag_detector = YoloDetector(
            self.config.price_tag_detector_path,
            self.config.device,
        )
        self.price_extractor = PaddlePriceExtractor(
            ocr_version=self.config.ocr_version,
            language=self.config.ocr_language,
            min_confidence=self.config.ocr_min_confidence,
            max_prices=self.config.ocr_max_prices,
            upscale=self.config.ocr_upscale,
        )
        self.sale_classifier = SaleClassifier(
            SaleClassifierConfig(
                vertical_aspect_ratio=(
                    self.config.sale_vertical_aspect_ratio
                ),
                large_area_ratio=self.config.sale_large_area_ratio,
                red_pixel_ratio=self.config.sale_red_pixel_ratio,
                yellow_pixel_ratio=self.config.sale_yellow_pixel_ratio,
                shelf_min_y_threshold_px=(
                    self.config.sale_shelf_min_y_threshold_px
                ),
                shelf_image_y_ratio=self.config.sale_shelf_image_y_ratio,
                baseline_min_confidence=(
                    self.config.sale_baseline_min_confidence
                ),
                baseline_min_tags=self.config.sale_baseline_min_tags,
                regular_tag_color=self.config.sale_regular_tag_color,
            )
        )
        self._product_recognizer = None

    def _load_product_recognizer(self):
        """Load and cache the product-recognition runtime.

        Input:
            None. Artifact paths and inference settings come from ``config``.
        Output:
            The recognizer dictionary containing model, index, and thresholds.
        """
        if self._product_recognizer is not None:
            return self._product_recognizer
        from .product_recognition.inference import load_recognizer

        self._product_recognizer = load_recognizer(
            memory_bank_path=self.config.product_memory_bank_path,
            threshold_path=self.config.product_threshold_path,
            model_name=self.config.product_model_name,
            device=self.config.device,
            top_k=self.config.product_top_k,
        )
        if self.config.use_global_product_threshold:
            self._product_recognizer["class_score_thresholds"] = {}
        return self._product_recognizer

    def process(self, input_path: Path | str) -> List[PipelineResult]:
        """Process one image or every supported image in a directory.

        Input:
            input_path: Source image path or directory containing images.
        Output:
            One ``PipelineResult`` per processed image, in resolved path order.
        """
        results = [
            self.process_image(image_path)
            for image_path in list_image_paths(input_path)
        ]
        if results:
            write_combined_outputs(self.config.output_dir, results)
        return results

    def _recognize_products(
        self,
        image,
        image_path: Path,
        image_output_dir: Path,
        detections: Sequence[Detection],
    ) -> list[dict]:
        """Crop, persist, and recognize detected shelf products.

        Input:
            image: Source image in BGR array format.
            image_path: Resolved source-image path used in audit fields.
            image_output_dir: Per-image directory for optional product crops.
            detections: Product detector results to recognize.
        Output:
            Product result rows ready for DataFrame construction.
        """
        if not detections:
            return []

        # Recognition is imported only when product detections need inference.
        from .product_recognition.inference import predict_image

        recognizer = self._load_product_recognizer()
        product_crop_dir = image_output_dir / "product_crops"
        rows = []
        for detection in detections:
            # Persist each crop because the recognizer accepts an image path.
            crop = crop_image(image, detection.bbox)
            crop_file = (
                product_crop_dir
                / (
                    f"product_{detection.index:04d}_"
                    f"conf_{detection.confidence:.2f}.jpg"
                )
            )
            if self.config.save_crops:
                save_crop(crop, crop_file)
            prediction = (
                predict_image(recognizer, str(crop_file))
                if self.config.save_crops
                else {}
            )

            # Keep detector geometry and recognition evidence in one audit row.
            rows.append(
                {
                    "source_image": image_path.name,
                    "product_index": detection.index,
                    "detector_confidence": detection.confidence,
                    "bbox_xmin": detection.bbox[0],
                    "bbox_ymin": detection.bbox[1],
                    "bbox_xmax": detection.bbox[2],
                    "bbox_ymax": detection.bbox[3],
                    "crop_file": (
                        str(crop_file.relative_to(image_output_dir))
                        if self.config.save_crops
                        else None
                    ),
                    "final_pred": prediction.get("final_pred"),
                    "is_rejected": prediction.get("is_rejected"),
                    "best_label": prediction.get("best_label"),
                    "best_score": prediction.get("best_score"),
                    "effective_score_threshold": prediction.get(
                        "effective_score_threshold"
                    ),
                    "threshold_offset": prediction.get("threshold_offset"),
                    "second_label": prediction.get("second_label"),
                    "second_score": prediction.get("second_score"),
                    "margin": prediction.get("margin"),
                    "reject_reasons": prediction.get("reject_reasons"),
                }
            )
        return rows

    def _extract_price_tags(
        self,
        image,
        image_path: Path,
        image_output_dir: Path,
        detections: Sequence[Detection],
    ) -> list[dict]:
        """Extract prices and audit fields for detected price tags.

        Input:
            image: Source image in BGR array format.
            image_path: Resolved source-image path used in audit fields.
            image_output_dir: Per-image directory for optional tag crops.
            detections: Price-tag detector results to process.
        Output:
            Price-tag rows ready for classification, alignment, and persistence.
        """
        tag_crop_dir = image_output_dir / "price_tag_crops"
        rows = []
        for detection in detections:
            # Extract normalized prices while preserving a disabled-stage schema.
            crop = crop_price_tag(image, detection.bbox)
            sale_fields = {
                column: None
                for column in SALE_CLASSIFICATION_COLUMNS
            }
            ocr_result = (
                self.price_extractor.extract(crop)
                if self.config.run_ocr
                else _disabled_ocr_result()
            )
            extracted_prices = [
                ocr_result.get("primary_price"),
                ocr_result.get("secondary_price"),
                ocr_result.get("tertiary_price"),
            ]
            extracted_values = [
                ocr_result.get("primary_value"),
                ocr_result.get("secondary_value"),
                ocr_result.get("tertiary_value"),
            ]
            prices = join_prices(extracted_prices)
            price_values = join_price_values(extracted_values)

            # Use extracted prices in crop names to simplify manual inspection.
            price_slug = (
                prices.replace(",", "_").replace(" / ", "__")
                if prices
                else "not_found"
            )
            crop_file = (
                tag_crop_dir
                / f"tag_{detection.index:04d}_price_{price_slug}.jpg"
            )
            if self.config.save_crops:
                save_crop(crop, crop_file)

            # Promotion fields are populated after all tags establish a baseline.
            rows.append(
                {
                    "source_image": image_path.name,
                    "tag_index": detection.index,
                    "detector_confidence": detection.confidence,
                    "bbox_xmin": detection.bbox[0],
                    "bbox_ymin": detection.bbox[1],
                    "bbox_xmax": detection.bbox[2],
                    "bbox_ymax": detection.bbox[3],
                    **sale_fields,
                    "primary_price": ocr_result.get("primary_price"),
                    "primary_value": ocr_result.get("primary_value"),
                    "primary_price_confidence": ocr_result.get(
                        "primary_price_confidence"
                    ),
                    "secondary_price": ocr_result.get("secondary_price"),
                    "secondary_value": ocr_result.get("secondary_value"),
                    "secondary_price_confidence": ocr_result.get(
                        "secondary_price_confidence"
                    ),
                    "tertiary_price": ocr_result.get("tertiary_price"),
                    "tertiary_value": ocr_result.get("tertiary_value"),
                    "tertiary_price_confidence": ocr_result.get(
                        "tertiary_price_confidence"
                    ),
                    "price_count": price_count(extracted_prices),
                    "prices": prices,
                    "price_values": price_values,
                    "crop_file": (
                        str(crop_file.relative_to(image_output_dir))
                        if self.config.save_crops
                        else None
                    ),
                    "ocr_error": ocr_result.get("ocr_error"),
                    "ocr_text": ocr_result.get("ocr_text"),
                }
            )
        return rows

    def _classify_sales(self, image, tag_rows: list[dict]) -> None:
        """Add promotion-classification evidence to price-tag rows in place.

        Input:
            image: Source image in BGR array format.
            tag_rows: Mutable price-tag rows containing boxes and extracted text.
        Output:
            None. Each input row is updated with classification fields.
        """
        if not self.config.run_sale_classification:
            return
        classifications = self.sale_classifier.classify(
            image,
            [
                SaleTag(
                    tag_index=int(row["tag_index"]),
                    bbox=(
                        int(row["bbox_xmin"]),
                        int(row["bbox_ymin"]),
                        int(row["bbox_xmax"]),
                        int(row["bbox_ymax"]),
                    ),
                    ocr_text=row.get("ocr_text"),
                )
                for row in tag_rows
            ],
        )
        for row in tag_rows:
            result = classifications.get(int(row["tag_index"]))
            if result is not None:
                row.update(result.to_row())

    def process_image(self, image_path: Path | str) -> PipelineResult:
        """Run every configured pipeline stage for one source image.

        Input:
            image_path: Path to a supported shelf image.
        Output:
            Paths, annotated image, and result tables for the processed image.
        """
        # Resolve the source and create its isolated output directory.
        image_path = resolve_path(image_path)
        image = load_image_bgr(image_path)
        image_output_dir = self.config.output_dir / safe_stem(image_path)
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Detect both object types and optionally de-duplicate product boxes.
        product_detections = self.product_detector.predict(
            image,
            conf=self.config.product_detector_conf,
        )
        if self.config.apply_product_nms:
            product_detections = custom_nms(
                product_detections,
                self.config.product_detector_iou,
            )
        price_tag_detections = self.price_tag_detector.predict(
            image,
            conf=self.config.price_tag_detector_conf,
        )

        # Convert detector boxes into product and price-tag audit records.
        product_rows = self._recognize_products(
            image,
            image_path,
            image_output_dir,
            product_detections,
        )
        tag_rows = self._extract_price_tags(
            image,
            image_path,
            image_output_dir,
            price_tag_detections,
        )
        self._classify_sales(image, tag_rows)

        # Align tags to products and enforce stable persisted column order.
        products_df = pd.DataFrame(product_rows)
        price_tags_df = pd.DataFrame(
            tag_rows,
            columns=PRICE_TAG_COLUMNS,
        )
        products_df, price_tags_df = align_price_tags_to_products(
            products_df,
            price_tags_df,
        )
        products_df = products_df.reindex(columns=PRODUCT_COLUMNS)
        price_tags_df = price_tags_df.reindex(columns=PRICE_TAG_COLUMNS)

        # Render and persist the final per-image outputs.
        annotated = build_annotated_image(
            image,
            products_df,
            price_tags_df,
            draw_unknown_products=self.config.draw_unknown_products,
        )
        annotated_path = write_image_outputs(
            image_output_dir,
            annotated,
            products_df,
            price_tags_df,
        )
        return PipelineResult(
            image_path=image_path,
            image_output_dir=image_output_dir,
            annotated_image_path=annotated_path,
            products_df=products_df,
            price_tags_df=price_tags_df,
        )


__all__ = [
    "Detection",
    "OcrItem",
    "PaddlePriceExtractor",
    "PipelineResult",
    "PriceCandidate",
    "ShelfPipeline",
    "UnifiedPipelineConfig",
    "environment_report",
    "flatten_ocr_result",
    "price_candidates_from_ocr",
    "select_price_candidates",
]
