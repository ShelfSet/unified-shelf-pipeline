"""Shared data structures used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class PipelineResult:
    """Outputs produced for one shelf image."""

    image_path: Path
    image_output_dir: Path
    annotated_image_path: Path
    products_df: pd.DataFrame
    price_tags_df: pd.DataFrame


@dataclass
class Detection:
    """One detector result in pixel coordinates."""

    index: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: Optional[int] = None


@dataclass
class OcrItem:
    """One OCR text result and its optional bounding box."""

    text: str
    confidence: float
    bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class PriceCandidate:
    """A normalized price candidate assembled from OCR results."""

    price: str
    value: float
    confidence: float
    area: float
    x_center: float
    source_text: str
    y_center: float = 0.0
    source_index: int = 0
    match_index: int = 0
    format_quality: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None
    component_ids: tuple[str, ...] = ()
