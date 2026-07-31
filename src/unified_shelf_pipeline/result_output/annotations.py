"""Render product, price-tag, and alignment annotations.

Annotation functions operate on copies of the source image. They draw resolved
product-to-tag connections first, followed by labeled product and price-tag
boxes whose colors communicate unknown products and promotional tags.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..image_processing.geometry import bbox_center
from ..image_processing.image_io import import_cv2
from .schemas import has_value


def draw_label(
    image,
    bbox: Sequence[int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a colored bounding box and compact text label.

    Input:
        image: Mutable destination image in BGR array format.
        bbox: Integer coordinates ordered as ``x1, y1, x2, y2``.
        label: Text displayed above or inside the box.
        color: OpenCV BGR drawing color.
    Output:
        None. The supplied image is modified in place.
    """
    cv2 = import_cv2()
    x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = label[:80]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    text_y = max(text_height + baseline + 4, y1)
    cv2.rectangle(
        image,
        (x1, text_y - text_height - baseline - 4),
        (x1 + text_width + 6, text_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 3, text_y - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_center_connection(
    image,
    product_row,
    price_tag_row,
    color: tuple[int, int, int],
) -> None:
    """Draw a line between resolved product and price-tag centers.

    Input:
        image: Mutable destination image in BGR array format.
        product_row: Product table row containing bounding-box coordinates.
        price_tag_row: Price-tag row containing bounding-box coordinates.
        color: OpenCV BGR drawing color.
    Output:
        None. The supplied image is modified in place.
    """
    cv2 = import_cv2()
    product_center = bbox_center(
        (
            product_row["bbox_xmin"],
            product_row["bbox_ymin"],
            product_row["bbox_xmax"],
            product_row["bbox_ymax"],
        )
    )
    tag_center = bbox_center(
        (
            price_tag_row["bbox_xmin"],
            price_tag_row["bbox_ymin"],
            price_tag_row["bbox_xmax"],
            price_tag_row["bbox_ymax"],
        )
    )
    product_point = (
        int(round(product_center[0])),
        int(round(product_center[1])),
    )
    tag_point = (
        int(round(tag_center[0])),
        int(round(tag_center[1])),
    )
    cv2.line(
        image,
        product_point,
        tag_point,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image, product_point, 5, color, -1, cv2.LINE_AA)
    cv2.circle(image, tag_point, 5, color, -1, cv2.LINE_AA)


def build_annotated_image(
    image,
    products_df: pd.DataFrame,
    price_tags_df: pd.DataFrame,
    *,
    draw_unknown_products: bool,
):
    """Build an annotated copy of one source image.

    Input:
        image: Source image in BGR array format.
        products_df: Final product results and alignment fields.
        price_tags_df: Final price-tag results and alignment fields.
        draw_unknown_products: Whether rejected products should be visible.
    Output:
        A copied image containing connections, product boxes, and tag boxes.
    """
    annotated = image.copy()

    # Draw resolved relationships beneath boxes and labels to preserve legibility.
    product_by_index = {
        int(row["product_index"]): row
        for _, row in products_df.iterrows()
        if has_value(row.get("product_index"))
    }
    for _, row in price_tags_df.iterrows():
        product_index = row.get("aligned_product_index")
        if not has_value(product_index):
            continue
        product_row = product_by_index.get(int(product_index))
        if product_row is not None:
            draw_center_connection(
                annotated,
                product_row,
                row,
                (255, 220, 20),
            )

    # Draw known and optionally unknown product recognition results.
    for _, row in products_df.iterrows():
        if bool(row.get("is_rejected")) and not draw_unknown_products:
            continue
        label = row.get("final_pred") or row.get("best_label") or "product"
        is_unknown = bool(row.get("is_rejected")) or str(label) == "__Unknown__"
        score = row.get("best_score")
        label_text = f"P{int(row['product_index'])}: {label}"
        if pd.notna(score):
            label_text = f"{label_text} {float(score):.2f}"
        draw_label(
            annotated,
            (
                row["bbox_xmin"],
                row["bbox_ymin"],
                row["bbox_xmax"],
                row["bbox_ymax"],
            ),
            label_text,
            (30, 30, 220) if is_unknown else (45, 170, 60),
        )

    # Draw price and promotion evidence for every detected tag.
    for _, row in price_tags_df.iterrows():
        price = (
            row.get("prices")
            if has_value(row.get("prices"))
            else "not found"
        )
        is_promo = (
            has_value(row.get("is_promo"))
            and bool(row.get("is_promo"))
        )
        sale_classification = row.get("sale_classification")
        sale_label = (
            f" {str(sale_classification).upper()}"
            if has_value(sale_classification)
            else ""
        )
        draw_label(
            annotated,
            (
                row["bbox_xmin"],
                row["bbox_ymin"],
                row["bbox_xmax"],
                row["bbox_ymax"],
            ),
            f"T{int(row['tag_index'])}{sale_label}: {price}",
            (30, 30, 220) if is_promo else (20, 135, 245),
        )
    return annotated
