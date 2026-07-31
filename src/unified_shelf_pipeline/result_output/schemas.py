"""Define persisted result columns and price serialization helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..shelf_analysis.product_alignment import (
    PRICE_TAG_ALIGNMENT_COLUMNS,
    PRODUCT_ALIGNMENT_COLUMNS,
)
from ..shelf_analysis.promotion_classification import (
    SALE_CLASSIFICATION_COLUMNS,
)


PRODUCT_COLUMNS = [
    "source_image",
    "product_index",
    "detector_confidence",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "crop_file",
    "final_pred",
    "is_rejected",
    "best_label",
    "best_score",
    "effective_score_threshold",
    "threshold_offset",
    "second_label",
    "second_score",
    "margin",
    "reject_reasons",
    "nearest_price_tag_index",
    "nearest_price",
    "nearest_price_value",
    "nearest_secondary_price",
    "nearest_secondary_price_value",
    "nearest_tertiary_price",
    "nearest_tertiary_price_value",
    "nearest_prices",
    "nearest_price_values",
    "nearest_sale_classification",
    "nearest_is_promo",
    "nearest_sale_reason",
    "nearest_tag_color",
    "price_tag_distance_px",
    *PRODUCT_ALIGNMENT_COLUMNS,
]

PRICE_TAG_COLUMNS = [
    "source_image",
    "tag_index",
    "detector_confidence",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    *SALE_CLASSIFICATION_COLUMNS,
    "primary_price",
    "primary_value",
    "primary_price_confidence",
    "secondary_price",
    "secondary_value",
    "secondary_price_confidence",
    "tertiary_price",
    "tertiary_value",
    "tertiary_price_confidence",
    "price_count",
    "prices",
    "price_values",
    "crop_file",
    "ocr_error",
    "ocr_text",
    *PRICE_TAG_ALIGNMENT_COLUMNS,
]


def has_value(value) -> bool:
    """Check whether a tabular value is present and non-blank.

    Input:
        value: Scalar value intended for persisted output.
    Output:
        ``True`` when the value is neither missing nor blank.
    """
    return value is not None and pd.notna(value) and str(value).strip() != ""


def join_price_values(values: Sequence[object]) -> Optional[str]:
    """Serialize numeric price values into the persisted audit format.

    Input:
        values: Ordered numeric price values, optionally containing blanks.
    Output:
        Slash-separated values with two decimals, or ``None`` when empty.
    """
    parts = [
        f"{float(value):.2f}"
        for value in values
        if has_value(value)
    ]
    return " / ".join(parts) if parts else None


def join_prices(prices: Sequence[object]) -> Optional[str]:
    """Serialize display prices into the persisted audit format.

    Input:
        prices: Ordered display-price values, optionally containing blanks.
    Output:
        Slash-separated display prices, or ``None`` when empty.
    """
    parts = [
        str(price).strip()
        for price in prices
        if has_value(price)
    ]
    return " / ".join(parts) if parts else None


def price_count(prices: Sequence[object]) -> int:
    """Count present price values.

    Input:
        prices: Price values that may include missing entries.
    Output:
        Number of non-blank price values.
    """
    return sum(1 for price in prices if has_value(price))
