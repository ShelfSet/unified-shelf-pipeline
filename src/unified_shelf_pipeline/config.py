"""Runtime configuration and project paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    """Resolve the repository root containing runtime folders.

    Input:
        None.
    Output:
        The absolute project-root path derived from this module location.
    """
    return Path(__file__).resolve().parents[2]


def discover_product_memory_bank_path(directory: Optional[Path] = None) -> Path:
    """Resolve the only product memory-bank pickle in its artifact folder.

    Input:
        directory: Optional folder override used by tests or integrations.
    Output:
        The sole ``.pkl`` file in the selected product memory-bank folder.
    """
    memory_bank_dir = directory or (
        project_root() / "artifacts" / "product_memory_bank"
    )
    candidates = sorted(
        (path for path in memory_bank_dir.glob("*.pkl") if path.is_file()),
        key=lambda path: path.name.casefold(),
    )

    if not candidates:
        raise FileNotFoundError(
            "No product memory-bank .pkl file found in "
            f"{memory_bank_dir.resolve()}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            "Expected exactly one product memory-bank .pkl file in "
            f"{memory_bank_dir.resolve()}, found {len(candidates)}: {names}"
        )
    return candidates[0].resolve()


@dataclass
class UnifiedPipelineConfig:
    """Paths and behavior settings for the unified shelf pipeline."""

    shelf_detector_path: Path = (
        project_root() / "artifacts" / "shelf_detector" / "shelf_server.pt"
    )
    price_tag_detector_path: Path = (
        project_root() / "artifacts" / "price_tag_detector" / "best.pt"
    )
    product_memory_bank_path: Path = field(
        default_factory=discover_product_memory_bank_path
    )
    product_threshold_path: Path = (
        project_root() / "artifacts" / "product_thresholds" / "thresholds.csv"
    )
    product_model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    output_dir: Path = project_root() / "outputs"
    device: str = "auto"
    product_detector_conf: float = 0.25
    price_tag_detector_conf: float = 0.25
    product_detector_iou: float = 0.25
    product_top_k: int = 5
    use_global_product_threshold: bool = False
    apply_product_nms: bool = True
    run_ocr: bool = True
    ocr_version: str = "PP-OCRv6"
    ocr_language: str = "pt"
    ocr_min_confidence: float = 0.40
    ocr_max_prices: int = 3
    ocr_upscale: float = 3.0
    run_sale_classification: bool = True
    sale_vertical_aspect_ratio: float = 1.20
    sale_large_area_ratio: float = 1.80
    sale_red_pixel_ratio: float = 0.10
    sale_yellow_pixel_ratio: float = 0.25
    sale_shelf_min_y_threshold_px: int = 35
    sale_shelf_image_y_ratio: float = 0.035
    sale_baseline_min_confidence: float = 0.60
    sale_baseline_min_tags: int = 3
    sale_regular_tag_color: Optional[str] = None
    draw_unknown_products: bool = True
    save_crops: bool = True
