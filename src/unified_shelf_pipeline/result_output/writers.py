"""Persist per-image and combined pipeline results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..image_processing.image_io import import_cv2
from ..models import PipelineResult


def write_image_outputs(
    image_output_dir: Path,
    annotated,
    products_df: pd.DataFrame,
    price_tags_df: pd.DataFrame,
) -> Path:
    """Persist one image's annotation and result tables.

    Input:
        image_output_dir: Existing per-image destination directory.
        annotated: Annotated image in BGR array format.
        products_df: Final product result table.
        price_tags_df: Final price-tag result table.
    Output:
        Path of the written annotated image.
    """
    cv2 = import_cv2()
    annotated_path = image_output_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    products_df.to_csv(image_output_dir / "products.csv", index=False)
    price_tags_df.to_csv(image_output_dir / "price_tags.csv", index=False)
    return annotated_path


def write_combined_outputs(
    output_dir: Path,
    results: Sequence[PipelineResult],
) -> None:
    """Persist combined product and price-tag tables across images.

    Input:
        output_dir: Root destination for combined CSV files.
        results: Completed per-image pipeline results.
    Output:
        None. Non-empty result collections are concatenated and written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exclude empty per-image tables so concatenation always has valid inputs.
    product_frames = [
        result.products_df
        for result in results
        if not result.products_df.empty
    ]
    tag_frames = [
        result.price_tags_df
        for result in results
        if not result.price_tags_df.empty
    ]
    if product_frames:
        pd.concat(product_frames, ignore_index=True).to_csv(
            output_dir / "all_products.csv",
            index=False,
        )
    if tag_frames:
        pd.concat(tag_frames, ignore_index=True).to_csv(
            output_dir / "all_price_tags.csv",
            index=False,
        )
