"""Product alignment and price-tag promotion analysis."""

from .product_alignment import align_price_tags_to_products
from .promotion_classification import (
    SaleClassification,
    SaleClassifier,
    SaleClassifierConfig,
    SaleTag,
)

__all__ = [
    "SaleClassification",
    "SaleClassifier",
    "SaleClassifierConfig",
    "SaleTag",
    "align_price_tags_to_products",
]
