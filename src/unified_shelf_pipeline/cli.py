"""Expose the supported command-line interface for the shelf pipeline.

The module converts user-supplied command-line options into one
``UnifiedPipelineConfig`` and then runs either the environment check or the
image-processing pipeline. Artifact options accept direct file paths so the
runtime has one unambiguous source for each model or inference artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import UnifiedPipelineConfig
from .diagnostics import environment_report
from .pipeline import ShelfPipeline


def parse_args() -> argparse.Namespace:
    """Parse the command-line options supported by the runtime.

    Input:
        None. Values are read from the process command line.
    Output:
        An ``argparse.Namespace`` containing validated option values.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect and classify shelf products, detect price tags, classify "
            "promotions, OCR prices, and write annotated outputs."
        )
    )
    # General input, output, artifact, and runtime options.
    parser.add_argument("--input", default=None, help="Image file or folder of shelf images.")
    parser.add_argument("--output-dir", default=None, help="Output folder for annotated images and CSVs.")
    parser.add_argument(
        "--shelf-detector",
        type=Path,
        default=None,
        help="Path to the shelf/product detector model file.",
    )
    parser.add_argument(
        "--price-tag-detector",
        type=Path,
        default=None,
        help="Path to the price-tag detector model file.",
    )
    parser.add_argument(
        "--product-memory-bank",
        type=Path,
        default=None,
        help=(
            "Optional path to a product memory-bank .pkl file. By default, "
            "the only .pkl file in artifacts/product_memory_bank is used."
        ),
    )
    parser.add_argument(
        "--product-threshold",
        type=Path,
        default=None,
        help="Path to the product thresholds.csv file.",
    )
    parser.add_argument("--product-model-name", default=None, help="SentenceTransformer model id or local path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--product-conf", type=float, default=None, help="Shelf product detector confidence.")
    parser.add_argument("--price-tag-conf", type=float, default=None, help="Price tag detector confidence.")
    parser.add_argument("--no-product-nms", action="store_true", help="Disable product overlap filtering.")

    # Price text-extraction options.
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR and only detect price-tag boxes.")
    parser.add_argument("--ocr-version", default=None, help="PaddleOCR model family (default: PP-OCRv6).")
    parser.add_argument("--ocr-language", default=None, help="PaddleOCR language code (default: pt).")
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=None,
        help="Minimum OCR confidence for a price candidate (default: 0.40).",
    )
    parser.add_argument(
        "--ocr-max-prices",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Maximum prices extracted per tag (default: 3).",
    )
    parser.add_argument(
        "--ocr-upscale",
        type=float,
        default=None,
        help="OCR crop upscaling factor from 1 to 4 (default: 3).",
    )

    # Promotion-classification options.
    parser.add_argument(
        "--no-sale-classification",
        action="store_true",
        help="Skip heuristic promo/regular classification of detected price tags.",
    )
    parser.add_argument(
        "--sale-vertical-aspect-ratio",
        type=float,
        default=None,
        help="Aspect ratio above which a price tag is promotional (default: 1.20).",
    )
    parser.add_argument(
        "--sale-large-area-ratio",
        type=float,
        default=None,
        help="Area-to-shelf-median ratio above which a price tag is promotional (default: 1.80).",
    )
    parser.add_argument(
        "--sale-red-pixel-ratio",
        type=float,
        default=None,
        help="Minimum red pixel share used to classify a tag as red (default: 0.10).",
    )
    parser.add_argument(
        "--sale-yellow-pixel-ratio",
        type=float,
        default=None,
        help="Minimum yellow pixel share used to classify a tag as yellow (default: 0.25).",
    )
    parser.add_argument(
        "--sale-shelf-min-y-threshold-px",
        type=int,
        default=None,
        help="Minimum tag-center distance used to form shelf rows (default: 35).",
    )
    parser.add_argument(
        "--sale-shelf-image-y-ratio",
        type=float,
        default=None,
        help="Image-height share used to form shelf rows (default: 0.035).",
    )
    parser.add_argument(
        "--sale-baseline-min-confidence",
        type=float,
        default=None,
        help="Minimum dominant-color share required for an image baseline (default: 0.60).",
    )
    parser.add_argument(
        "--sale-baseline-min-tags",
        type=int,
        default=None,
        help="Minimum non-promotional tags required to infer the image baseline (default: 3).",
    )
    parser.add_argument(
        "--sale-regular-tag-color",
        choices=["white", "yellow", "red"],
        default=None,
        help="Override automatic regular-tag color inference for a known store format.",
    )
    parser.add_argument("--hide-unknown-products", action="store_true", help="Do not draw rejected products.")
    parser.add_argument("--check-env", action="store_true", help="Print dependency/artifact availability and exit.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UnifiedPipelineConfig:
    """Apply explicitly supplied CLI values to the runtime configuration.

    Input:
        args: Parsed command-line values returned by ``parse_args``.
    Output:
        A pipeline configuration containing defaults plus CLI overrides.
    """
    config_kwargs = {}
    if args.product_memory_bank:
        # Supplying a path bypasses default folder discovery.
        config_kwargs["product_memory_bank_path"] = args.product_memory_bank
    config = UnifiedPipelineConfig(**config_kwargs)

    # Apply general paths and detector/product-recognition settings.
    config.device = args.device
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.shelf_detector:
        config.shelf_detector_path = args.shelf_detector
    if args.price_tag_detector:
        config.price_tag_detector_path = args.price_tag_detector
    if args.product_threshold:
        config.product_threshold_path = args.product_threshold
    if args.product_model_name:
        config.product_model_name = args.product_model_name
    if args.product_conf is not None:
        config.product_detector_conf = args.product_conf
    if args.price_tag_conf is not None:
        config.price_tag_detector_conf = args.price_tag_conf
    config.apply_product_nms = not args.no_product_nms

    # Apply text-extraction settings.
    config.run_ocr = not args.no_ocr
    if args.ocr_version:
        config.ocr_version = args.ocr_version
    if args.ocr_language:
        config.ocr_language = args.ocr_language
    if args.ocr_min_confidence is not None:
        config.ocr_min_confidence = args.ocr_min_confidence
    if args.ocr_max_prices is not None:
        config.ocr_max_prices = args.ocr_max_prices
    if args.ocr_upscale is not None:
        config.ocr_upscale = args.ocr_upscale

    # Apply promotion-classification settings.
    config.run_sale_classification = not args.no_sale_classification
    if args.sale_vertical_aspect_ratio is not None:
        config.sale_vertical_aspect_ratio = args.sale_vertical_aspect_ratio
    if args.sale_large_area_ratio is not None:
        config.sale_large_area_ratio = args.sale_large_area_ratio
    if args.sale_red_pixel_ratio is not None:
        config.sale_red_pixel_ratio = args.sale_red_pixel_ratio
    if args.sale_yellow_pixel_ratio is not None:
        config.sale_yellow_pixel_ratio = args.sale_yellow_pixel_ratio
    if args.sale_shelf_min_y_threshold_px is not None:
        config.sale_shelf_min_y_threshold_px = args.sale_shelf_min_y_threshold_px
    if args.sale_shelf_image_y_ratio is not None:
        config.sale_shelf_image_y_ratio = args.sale_shelf_image_y_ratio
    if args.sale_baseline_min_confidence is not None:
        config.sale_baseline_min_confidence = args.sale_baseline_min_confidence
    if args.sale_baseline_min_tags is not None:
        config.sale_baseline_min_tags = args.sale_baseline_min_tags
    if args.sale_regular_tag_color is not None:
        config.sale_regular_tag_color = args.sale_regular_tag_color
    config.draw_unknown_products = not args.hide_unknown_products
    return config


def main() -> None:
    """Run the CLI environment check or shelf-image pipeline.

    Input:
        None. Runtime options are read from the process command line.
    Output:
        None. Status and result locations are printed to standard output.
    """
    args = parse_args()
    config = build_config(args)

    # The environment check is intentionally model-free and exits immediately.
    if args.check_env:
        print(environment_report(config).to_string(index=False))
        return
    if not args.input:
        raise ValueError("Provide --input, or use --check-env.")

    # Load the pipeline once, process all resolved images, and summarize outputs.
    pipeline = ShelfPipeline(config)
    results = pipeline.process(args.input)
    print(f"Processed {len(results)} image(s).")
    for result in results:
        print(f"{result.image_path.name}")
        print(f"  annotated: {result.annotated_image_path}")
        print(f"  products: {len(result.products_df)} rows")
        print(f"  price_tags: {len(result.price_tags_df)} rows")


if __name__ == "__main__":
    main()
