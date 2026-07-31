"""Public API for extracting prices from price-tag images."""

from .text_extraction import PaddlePriceExtractor
from .price_parsing import (
    flatten_ocr_result,
    price_candidates_from_ocr,
    select_price_candidates,
)

__all__ = [
    "PaddlePriceExtractor",
    "flatten_ocr_result",
    "price_candidates_from_ocr",
    "select_price_candidates",
]
