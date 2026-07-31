"""Public API for the unified shelf image pipeline.

The package exposes a Python API that returns one product DataFrame and one
price-tag DataFrame per input image, plus annotated output images on disk.
"""

from .pipeline import PipelineResult, ShelfPipeline, UnifiedPipelineConfig
from .shelf_analysis.promotion_classification import (
    SaleClassification,
    SaleClassifier,
    SaleClassifierConfig,
    SaleTag,
)

__all__ = [
    "PipelineResult",
    "SaleClassification",
    "SaleClassifier",
    "SaleClassifierConfig",
    "SaleTag",
    "ShelfPipeline",
    "UnifiedPipelineConfig",
]
