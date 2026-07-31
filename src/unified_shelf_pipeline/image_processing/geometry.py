"""Provide bounding-box geometry helpers for image processing."""

from __future__ import annotations

from typing import Sequence


def clamp_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Round and constrain a bounding box to image dimensions.

    Input:
        bbox: Coordinates ordered as ``x1, y1, x2, y2``.
        width: Maximum horizontal image coordinate.
        height: Maximum vertical image coordinate.
    Output:
        Integer coordinates clipped to the supplied image bounds.
    """
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def bbox_area(bbox: Sequence[float]) -> float:
    """Calculate non-negative bounding-box area.

    Input:
        bbox: Coordinates ordered as ``x1, y1, x2, y2``.
    Output:
        Box area in squared pixels, or zero for an inverted dimension.
    """
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float(x2) - float(x1)) * max(
        0.0,
        float(y2) - float(y1),
    )


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    """Calculate the center coordinates of a bounding box.

    Input:
        bbox: Coordinates ordered as ``x1, y1, x2, y2``.
    Output:
        Floating-point ``(x, y)`` center coordinates.
    """
    x1, y1, x2, y2 = bbox[:4]
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
