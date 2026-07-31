"""Environment and artifact diagnostics for the command-line interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import UnifiedPipelineConfig


def environment_report(
    config: Optional[UnifiedPipelineConfig] = None,
) -> pd.DataFrame:
    """Inspect required packages and artifact paths without loading models.

    Input:
        config: Optional artifact configuration. Defaults are used when omitted.
    Output:
        A table with check kind, dependency or artifact name, and availability.
    """
    config = config or UnifiedPipelineConfig()
    checks = [
        ("python_package", "numpy", importlib.util.find_spec("numpy") is not None),
        ("python_package", "pandas", importlib.util.find_spec("pandas") is not None),
        ("python_package", "cv2", importlib.util.find_spec("cv2") is not None),
        (
            "python_package",
            "ultralytics",
            importlib.util.find_spec("ultralytics") is not None,
        ),
        ("python_package", "faiss", importlib.util.find_spec("faiss") is not None),
        (
            "python_package",
            "sentence_transformers",
            importlib.util.find_spec("sentence_transformers") is not None,
        ),
        (
            "python_package",
            "paddleocr",
            importlib.util.find_spec("paddleocr") is not None,
        ),
        ("python_package", "paddle", importlib.util.find_spec("paddle") is not None),
        (
            "artifact",
            str(config.shelf_detector_path),
            Path(config.shelf_detector_path).is_file(),
        ),
        (
            "artifact",
            str(config.price_tag_detector_path),
            Path(config.price_tag_detector_path).is_file(),
        ),
        (
            "artifact",
            str(config.product_memory_bank_path),
            Path(config.product_memory_bank_path).is_file(),
        ),
        (
            "artifact",
            str(config.product_threshold_path),
            Path(config.product_threshold_path).is_file(),
        ),
    ]
    return pd.DataFrame(checks, columns=["kind", "name", "available"])
