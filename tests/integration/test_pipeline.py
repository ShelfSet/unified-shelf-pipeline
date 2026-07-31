"""Integration tests for orchestration and output persistence."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from unified_shelf_pipeline.models import Detection
from unified_shelf_pipeline.pipeline import ShelfPipeline
from unified_shelf_pipeline.config import UnifiedPipelineConfig
from unified_shelf_pipeline.shelf_analysis.promotion_classification import (
    SALE_CLASSIFICATION_COLUMNS,
)


YELLOW = (0, 255, 255)


def paint(image, bbox, color):
    """Fill one rectangular image region with a test color.

    Input:
        image: Mutable test image array.
        bbox: Integer ``x1, y1, x2, y2`` coordinates.
        color: BGR color assigned to the region.
    Output:
        None. The test image is modified in place.
    """
    x1, y1, x2, y2 = bbox
    image[y1:y2, x1:x2] = color


class FakeDetector:
    """Return deterministic detections without loading YOLO."""

    def __init__(self, detections):
        """Store detector results used by the integration test.

        Input:
            detections: Detection objects returned by every prediction.
        Output:
            None.
        """
        self.detections = detections

    def predict(self, _image, conf):
        """Return the configured detector results.

        Input:
            _image: Ignored source image.
            conf: Ignored detector confidence threshold.
        Output:
            A copy of the configured detection list.
        """
        return list(self.detections)


class SequencedPriceExtractor:
    """Return deterministic OCR text in crop-processing order."""

    def __init__(self, ocr_texts):
        """Store the ordered OCR text fixtures.

        Input:
            ocr_texts: Text returned for successive tag crops.
        Output:
            None.
        """
        self.ocr_texts = list(ocr_texts)
        self.call_count = 0

    def extract(self, _crop):
        """Build a normal extraction result using the next text fixture.

        Input:
            _crop: Ignored price-tag crop.
        Output:
            Empty price fields plus the next configured OCR text.
        """
        text = self.ocr_texts[self.call_count]
        self.call_count += 1
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
            "ocr_error": None,
            "ocr_text": text,
        }


class PipelineIntegrationTests(unittest.TestCase):
    """Verify orchestration across extraction, classification, and persistence."""

    def test_pipeline_classifies_after_ocr_and_writes_evidence_fields(self):
        """Verify OCR evidence drives classification and persisted outputs.

        Input:
            None. The test creates a temporary shelf image and fake adapters.
        Output:
            None. Assertions validate evidence fields and written files.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange a shelf image with four same-row yellow price tags.
            temp_path = Path(temp_dir)
            image_path = temp_path / "shelf.jpg"
            image = np.full((180, 500, 3), 255, dtype=np.uint8)
            boxes = [
                (20, 80, 80, 110),
                (100, 80, 160, 110),
                (180, 80, 240, 110),
                (260, 80, 320, 110),
            ]
            for bbox in boxes:
                paint(image, bbox, YELLOW)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            # Replace heavyweight runtime adapters with deterministic fakes.
            pipeline = ShelfPipeline(
                UnifiedPipelineConfig(
                    output_dir=temp_path / "outputs",
                    run_ocr=True,
                    save_crops=False,
                )
            )
            pipeline.product_detector = FakeDetector([])
            pipeline.price_tag_detector = FakeDetector(
                [
                    Detection(index, bbox, 0.95 - index * 0.01)
                    for index, bbox in enumerate(boxes)
                ]
            )
            pipeline.price_extractor = SequencedPriceExtractor(
                ["OFERTA | 4,19", "6,29", "6,29", "6,29"]
            )

            # Run the real orchestration, classification, annotation, and writers.
            result = pipeline.process_image(image_path)

            # Validate promotion evidence and persisted per-image outputs.
            self.assertTrue(
                set(SALE_CLASSIFICATION_COLUMNS).issubset(
                    result.price_tags_df.columns
                )
            )
            offer_row = result.price_tags_df.loc[
                result.price_tags_df["tag_index"] == 0
            ].iloc[0]
            self.assertEqual(offer_row["sale_classification"], "promo")
            self.assertEqual(offer_row["sale_reason"], "offer_keyword")
            self.assertEqual(offer_row["sale_keyword_match"], "OFERTA")
            self.assertEqual(offer_row["sale_baseline_color"], "yellow")
            self.assertTrue(result.annotated_image_path.exists())
            self.assertTrue(
                (result.image_output_dir / "price_tags.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
