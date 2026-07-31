"""Parse OCR text and geometry into normalized price candidates.

The parser accepts the common PaddleOCR result shapes, preserves text-box
positions, recognizes explicit decimal prices, and conservatively reconstructs
prices split across multiple OCR boxes. Candidate selection removes overlapping
interpretations and returns prices in visual reading order.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

from ..image_processing.geometry import bbox_area, bbox_center
from ..models import OcrItem, PriceCandidate


MAX_PRICE_VALUE = 200.0
DEFAULT_OCR_MIN_CONFIDENCE = 0.40
DEFAULT_OCR_MAX_PRICES = 3
PRICE_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[,\.]\s*(\d{2})")
COMPACT_PRICE_RE = re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
CURRENCY_PREFIX_RE = re.compile(r"^\s*R\s*(?:\$|S)?\s*", re.IGNORECASE)
PRICE_WHOLE_FRAGMENT_RE = re.compile(
    r"^\s*(?:R\s*(?:\$|S)?\s*)?(\d{1,3})\s*[,\.]\s*$",
    re.IGNORECASE,
)
PLAIN_WHOLE_FRAGMENT_RE = re.compile(
    r"^\s*(?:R\s*(?:\$|S)?\s*)?(\d{1,3})\s*$",
    re.IGNORECASE,
)
PRICE_CENTS_FRAGMENT_RE = re.compile(r"^\s*(\d{2})\s*$")
DECIMAL_SEPARATOR_FRAGMENT_RE = re.compile(r"^\s*[,\.]\s*$")
CURRENCY_FRAGMENT_RE = re.compile(r"^\s*R\s*(?:\$|S)\s*$", re.IGNORECASE)
UNIT_FRAGMENT_RE = re.compile(
    r"^\s*(?:g|gr|kg|mg|ml|cl|l|lt|litro|litros|un|und|unid|cm|mm|oz|lb)"
    r"s?\.?\s*$",
    re.IGNORECASE,
)


def points_to_bbox(
    points,
) -> Optional[tuple[float, float, float, float]]:
    """Convert polygon points to an axis-aligned bounding box.

    Input:
        points: Array-like polygon coordinates returned by OCR.
    Output:
        ``(x1, y1, x2, y2)`` bounds, or ``None`` for invalid coordinates.
    """
    try:
        array = np.asarray(points, dtype="float32")
        if array.ndim == 2 and array.shape[1] >= 2:
            return (
                float(array[:, 0].min()),
                float(array[:, 1].min()),
                float(array[:, 0].max()),
                float(array[:, 1].max()),
            )
    except Exception:
        return None
    return None


def flatten_ocr_result(raw) -> List[OcrItem]:
    """Flatten common PaddleOCR response formats into shared OCR items.

    Input:
        raw: Nested dictionary, list, or tuple returned by PaddleOCR.
    Output:
        Text, confidence, and optional bounding box for every recognized item.
    """
    items: List[OcrItem] = []

    def walk(value):
        """Recursively collect recognized text from one nested OCR value.

        Input:
            value: Current nested PaddleOCR response value.
        Output:
            None. Normalized items are appended to the enclosing ``items`` list.
        """
        if value is None:
            return

        # PaddleOCR 3.x commonly exposes parallel text, score, and box arrays.
        if isinstance(value, dict):
            texts = value.get("rec_texts")
            scores = value.get("rec_scores")
            boxes = value.get("rec_polys")
            if boxes is None:
                boxes = value.get("rec_boxes")
            if texts is not None and scores is not None:
                for index, text in enumerate(texts):
                    box = (
                        boxes[index]
                        if boxes is not None and index < len(boxes)
                        else None
                    )
                    items.append(
                        OcrItem(
                            str(text),
                            float(scores[index]),
                            points_to_bbox(box),
                        )
                    )
                return
            for child in value.values():
                walk(child)
            return
        # Earlier PaddleOCR versions return ``[polygon, [text, score]]`` rows.
        if isinstance(value, (list, tuple)):
            if (
                len(value) == 2
                and isinstance(value[1], (list, tuple))
                and len(value[1]) >= 2
                and isinstance(value[1][0], str)
            ):
                items.append(
                    OcrItem(
                        value[1][0],
                        float(value[1][1]),
                        points_to_bbox(value[0]),
                    )
                )
                return
            for child in value:
                walk(child)

    walk(raw)
    return items


def normalize_price(parts: tuple[str, str]) -> tuple[str, float]:
    """Normalize whole and cents fragments into display and numeric values.

    Input:
        parts: Whole-number and two-digit cents strings.
    Output:
        Comma-formatted display price and floating-point numeric value.
    """
    whole, cents = parts
    price = f"{int(whole)},{cents}"
    return price, float(f"{int(whole)}.{cents}")


def _candidate_position(
    item: OcrItem,
    text: str,
    start: int,
    end: int,
) -> tuple[float, float, float]:
    """Estimate one text match's center and area within its OCR item.

    Input:
        item: OCR item containing the matched text.
        text: Full normalized text represented by the OCR item.
        start: Inclusive match-character offset.
        end: Exclusive match-character offset.
    Output:
        Candidate ``x`` center, ``y`` center, and bounding-box area.
    """
    candidate_bbox = _candidate_bbox(item, text, start, end)
    if candidate_bbox is None:
        return 0.0, 0.0, 1.0
    x_center, y_center = bbox_center(candidate_bbox)
    return x_center, y_center, bbox_area(candidate_bbox)


def _candidate_bbox(
    item: OcrItem,
    text: str,
    start: int,
    end: int,
) -> Optional[tuple[float, float, float, float]]:
    """Estimate a substring bounding box within one OCR text box.

    Input:
        item: OCR item containing the substring.
        text: Full text represented by the OCR item.
        start: Inclusive substring-character offset.
        end: Exclusive substring-character offset.
    Output:
        Estimated substring bounds, full item bounds for blank text, or ``None``.
    """
    if item.bbox is None:
        return None
    x1, y1, x2, y2 = item.bbox
    if not text:
        return item.bbox
    text_length = max(1.0, float(len(text)))
    match_x1 = x1 + (x2 - x1) * (float(start) / text_length)
    match_x2 = x1 + (x2 - x1) * (float(end) / text_length)
    return match_x1, y1, match_x2, y2


def _compact_price_text(text: str) -> Optional[str]:
    """Recover a digit-only price token after conservative currency cleanup.

    Input:
        text: Recognized OCR text that may omit a decimal separator.
    Output:
        Compact price digits, or ``None`` when the text is not price-like.
    """
    stripped = CURRENCY_PREFIX_RE.sub("", text.strip())
    if re.fullmatch(r"\d{1,3}\s+\d{2}", stripped):
        return re.sub(r"\s", "", stripped)
    if COMPACT_PRICE_RE.fullmatch(stripped):
        return stripped
    return None


def _bbox_height(bbox: Sequence[float]) -> float:
    """Calculate a positive height used by spatial heuristics.

    Input:
        bbox: Coordinates ordered as ``x1, y1, x2, y2``.
    Output:
        Bounding-box height with a minimum value of one pixel.
    """
    return max(1.0, float(bbox[3]) - float(bbox[1]))


def _bbox_iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Calculate intersection over union for two bounding boxes.

    Input:
        first: First ``x1, y1, x2, y2`` box.
        second: Second ``x1, y1, x2, y2`` box.
    Output:
        Intersection-over-union ratio between zero and one.
    """
    intersection_x = max(
        0.0,
        min(float(first[2]), float(second[2]))
        - max(float(first[0]), float(second[0])),
    )
    intersection_y = max(
        0.0,
        min(float(first[3]), float(second[3]))
        - max(float(first[1]), float(second[1])),
    )
    intersection = intersection_x * intersection_y
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / max(1.0, union)


def _vertical_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Measure vertical overlap relative to the shorter box.

    Input:
        first: First ``x1, y1, x2, y2`` box.
        second: Second ``x1, y1, x2, y2`` box.
    Output:
        Vertical overlap ratio between zero and one.
    """
    overlap = max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    denominator = max(
        1.0,
        min(first[3] - first[1], second[3] - second[1]),
    )
    return overlap / denominator


def _has_nearby_unit_marker(
    item_index: int,
    items: Sequence[OcrItem],
) -> bool:
    """Detect a nearby unit suffix such as OCR output ``600`` plus ``g``.

    Input:
        item_index: Index of the numeric OCR item being evaluated.
        items: All OCR items detected on the same price tag.
    Output:
        ``True`` when a spatially adjacent measurement unit is present.
    """
    item = items[item_index]
    if item.bbox is None:
        return False
    height = _bbox_height(item.bbox)
    for marker_index, marker in enumerate(items):
        if marker_index == item_index or marker.bbox is None:
            continue
        if UNIT_FRAGMENT_RE.fullmatch(marker.text) is None:
            continue
        horizontal_gap = marker.bbox[0] - item.bbox[2]
        if not (-0.15 * height <= horizontal_gap <= 0.85 * height):
            continue
        center_delta = abs(
            bbox_center(item.bbox)[1] - bbox_center(marker.bbox)[1]
        )
        if (
            _vertical_overlap_ratio(item.bbox, marker.bbox) >= 0.25
            or center_delta <= 0.55 * height
        ):
            return True
    return False


def _max_numeric_item_height(items: Sequence[OcrItem]) -> float:
    """Find the largest OCR box height containing a digit.

    Input:
        items: OCR items detected on the same price tag.
    Output:
        Maximum numeric-item height, defaulting to one pixel.
    """
    heights = [
        _bbox_height(item.bbox)
        for item in items
        if item.bbox is not None and re.search(r"\d", item.text)
    ]
    return max(heights, default=1.0)


def _allow_compact_candidate(
    item_index: int,
    items: Sequence[OcrItem],
    max_numeric_height: float,
) -> bool:
    """Decide whether compact digits can safely represent a price.

    Input:
        item_index: Index of the candidate OCR item.
        items: All OCR items detected on the same price tag.
        max_numeric_height: Largest numeric OCR-box height on the tag.
    Output:
        ``True`` when currency, spacing, size, and unit checks allow recovery.
    """
    item = items[item_index]
    text = item.text.strip()
    if CURRENCY_PREFIX_RE.match(text) or re.fullmatch(
        r"\d{1,3}\s+\d{2}",
        text,
    ):
        return True
    if _has_nearby_unit_marker(item_index, items):
        return False
    if item.bbox is None:
        return False
    return _bbox_height(item.bbox) >= 0.55 * max(1.0, max_numeric_height)


def _nearby_fragment_indices(
    left: OcrItem,
    right: OcrItem,
    items: Sequence[OcrItem],
    pattern: re.Pattern,
    min_confidence: float,
) -> List[int]:
    """Find separator-like OCR fragments between whole and cents boxes.

    Input:
        left: OCR item containing the whole-number component.
        right: OCR item containing the cents component.
        items: All OCR items detected on the same price tag.
        pattern: Regular expression that accepted fragments must match.
        min_confidence: Minimum confidence accepted for a fragment.
    Output:
        Indices of spatially compatible OCR fragments.
    """
    if left.bbox is None or right.bbox is None:
        return []
    height = _bbox_height(left.bbox)
    x_min = left.bbox[2] - 0.25 * height
    x_max = right.bbox[0] + 0.35 * height
    y_min = min(left.bbox[1], right.bbox[1]) - 0.30 * height
    y_max = max(left.bbox[3], right.bbox[3]) + 0.30 * height
    matches = []
    for index, item in enumerate(items):
        if item is left or item is right or item.bbox is None:
            continue
        if (
            item.confidence < min_confidence
            or pattern.fullmatch(item.text) is None
        ):
            continue
        center_x, center_y = bbox_center(item.bbox)
        if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
            matches.append(index)
    return matches


def _nearby_currency_indices(
    left: OcrItem,
    items: Sequence[OcrItem],
    min_confidence: float,
) -> List[int]:
    """Find currency-marker OCR items immediately left of a number.

    Input:
        left: OCR item containing the whole-number component.
        items: All OCR items detected on the same price tag.
        min_confidence: Minimum confidence accepted for a currency marker.
    Output:
        Indices of spatially compatible currency markers.
    """
    if left.bbox is None:
        return []
    height = _bbox_height(left.bbox)
    matches = []
    for index, item in enumerate(items):
        if item is left or item.bbox is None:
            continue
        if (
            item.confidence < min_confidence
            or CURRENCY_FRAGMENT_RE.fullmatch(item.text) is None
        ):
            continue
        horizontal_gap = left.bbox[0] - item.bbox[2]
        center_delta = abs(
            bbox_center(left.bbox)[1] - bbox_center(item.bbox)[1]
        )
        if (
            -0.20 * height <= horizontal_gap <= 1.25 * height
            and center_delta <= 0.75 * height
        ):
            matches.append(index)
    return matches


def _fragment_pair_geometry(left: OcrItem, right: OcrItem) -> bool:
    """Check whether whole and cents OCR boxes form a plausible price.

    Input:
        left: Candidate whole-number OCR item.
        right: Candidate cents OCR item.
    Output:
        ``True`` when horizontal distance, height, and vertical overlap agree.
    """
    if left.bbox is None or right.bbox is None:
        return False
    left_height = _bbox_height(left.bbox)
    right_height = _bbox_height(right.bbox)
    horizontal_gap = right.bbox[0] - left.bbox[2]
    if not (-0.35 * left_height <= horizontal_gap <= 1.20 * left_height):
        return False
    if right_height > 1.40 * left_height:
        return False
    return (
        right.bbox[1] <= left.bbox[3] + 0.30 * left_height
        and right.bbox[3] >= left.bbox[1] - 0.30 * left_height
    )


def _price_candidates_from_fragments(
    items: Sequence[OcrItem],
    min_confidence: float,
) -> List[PriceCandidate]:
    """Assemble prices split across whole, separator, and cents boxes.

    Input:
        items: OCR items detected on one price tag.
        min_confidence: Minimum confidence for numeric components.
    Output:
        Normalized candidates with combined geometry and component identities.
    """
    candidates: List[PriceCandidate] = []

    # Treat each plausible whole-number item as the left side of a price.
    for left_index, left in enumerate(items):
        separated_whole_match = PRICE_WHOLE_FRAGMENT_RE.fullmatch(left.text)
        plain_whole_match = PLAIN_WHOLE_FRAGMENT_RE.fullmatch(left.text)
        whole_match = separated_whole_match or plain_whole_match
        if (
            whole_match is None
            or left.bbox is None
            or left.confidence < min_confidence
        ):
            continue
        if _has_nearby_unit_marker(left_index, items):
            continue
        # Pair the whole number only with spatially compatible cents items.
        for right_index, right in enumerate(items):
            cents_match = PRICE_CENTS_FRAGMENT_RE.fullmatch(right.text)
            if (
                right_index == left_index
                or cents_match is None
                or right.bbox is None
                or right.confidence < min_confidence
                or not _fragment_pair_geometry(left, right)
            ):
                continue

            # Require decimal, currency, or superscript-size evidence.
            separator_indices = _nearby_fragment_indices(
                left,
                right,
                items,
                DECIMAL_SEPARATOR_FRAGMENT_RE,
                max(0.15, min_confidence * 0.50),
            )
            currency_indices = _nearby_currency_indices(
                left,
                items,
                min_confidence,
            )
            has_separator = (
                separated_whole_match is not None or bool(separator_indices)
            )
            has_currency = (
                CURRENCY_PREFIX_RE.match(left.text.strip()) is not None
                or bool(currency_indices)
            )
            cents_are_smaller = (
                _bbox_height(right.bbox) <= 0.82 * _bbox_height(left.bbox)
            )
            if not (has_separator or has_currency or cents_are_smaller):
                continue

            # Normalize valid values and combine every contributing OCR box.
            price, value = normalize_price(
                (whole_match.group(1), cents_match.group(1))
            )
            if not 0.0 < value < MAX_PRICE_VALUE:
                continue
            bbox = (
                min(left.bbox[0], right.bbox[0]),
                min(left.bbox[1], right.bbox[1]),
                max(left.bbox[2], right.bbox[2]),
                max(left.bbox[3], right.bbox[3]),
            )
            x_center, y_center = bbox_center(bbox)
            component_indices = [
                left_index,
                right_index,
                *separator_indices,
                *currency_indices,
            ]
            unique_indices = list(dict.fromkeys(component_indices))
            source_text = "".join(
                items[index].text
                for index in sorted(
                    unique_indices,
                    key=lambda index: bbox_center(items[index].bbox)[0],
                )
            )
            candidates.append(
                PriceCandidate(
                    price,
                    value,
                    min(left.confidence, right.confidence),
                    bbox_area(bbox),
                    x_center,
                    source_text,
                    y_center=y_center,
                    source_index=min(unique_indices),
                    format_quality=3 if has_separator or has_currency else 2,
                    bbox=bbox,
                    component_ids=tuple(
                        f"item:{index}" for index in unique_indices
                    ),
                )
            )
    return candidates


def price_candidates_from_ocr(
    items: Sequence[OcrItem],
    min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
) -> List[PriceCandidate]:
    """Parse plausible price tokens while preserving spatial positions.

    Input:
        items: Normalized OCR items detected on one price tag.
        min_confidence: Minimum confidence accepted for a numeric component.
    Output:
        Explicit, compact, and split-fragment price interpretations.
    """
    candidates: List[PriceCandidate] = []
    max_numeric_height = _max_numeric_item_height(items)

    # Parse explicit decimal prices from each sufficiently confident OCR item.
    for source_index, item in enumerate(items):
        if float(item.confidence) < float(min_confidence):
            continue
        text = item.text.strip()
        normalized_text = re.sub(
            r"(?<=\d)[,\.]\s*[,\.](?=\s*\d{2}(?:\D|$))",
            ",",
            text,
        )
        explicit_matches = list(PRICE_RE.finditer(normalized_text))
        for match_index, match in enumerate(explicit_matches):
            price, value = normalize_price((match.group(1), match.group(2)))
            if 0.0 < value < MAX_PRICE_VALUE:
                x_center, y_center, area = _candidate_position(
                    item,
                    normalized_text,
                    match.start(),
                    match.end(),
                )
                candidate_bbox = _candidate_bbox(
                    item,
                    normalized_text,
                    match.start(),
                    match.end(),
                )
                candidates.append(
                    PriceCandidate(
                        price,
                        value,
                        item.confidence,
                        area,
                        x_center,
                        text,
                        y_center=y_center,
                        source_index=source_index,
                        match_index=match_index,
                        format_quality=3,
                        bbox=candidate_bbox,
                        component_ids=(
                            f"item:{source_index}:match:{match_index}",
                        ),
                    )
                )

        # Recover missing decimal separators only when geometry looks price-like.
        compact_text = None if explicit_matches else _compact_price_text(text)
        if compact_text is not None and _allow_compact_candidate(
            source_index,
            items,
            max_numeric_height,
        ):
            whole = compact_text[:-2]
            cents = compact_text[-2:]
            price, value = normalize_price((whole, cents))
            if 0.0 < value < MAX_PRICE_VALUE:
                x_center, y_center, area = _candidate_position(
                    item,
                    text,
                    0,
                    len(text),
                )
                candidate_bbox = _candidate_bbox(item, text, 0, len(text))
                candidates.append(
                    PriceCandidate(
                        price,
                        value,
                        item.confidence,
                        area,
                        x_center,
                        text,
                        y_center=y_center,
                        source_index=source_index,
                        format_quality=1,
                        bbox=candidate_bbox,
                        component_ids=(f"item:{source_index}",),
                    )
                )

    # Add prices whose whole and cents components were recognized separately.
    candidates.extend(
        _price_candidates_from_fragments(items, float(min_confidence))
    )
    return candidates


def _candidate_rank(candidate: PriceCandidate) -> tuple[int, float, float]:
    """Build the quality key used to prioritize price interpretations.

    Input:
        candidate: Parsed price interpretation.
    Output:
        Format-quality, area, and confidence ranking tuple.
    """
    return candidate.format_quality, candidate.area, candidate.confidence


def _candidates_conflict(
    first: PriceCandidate,
    second: PriceCandidate,
) -> bool:
    """Determine whether two candidates represent overlapping evidence.

    Input:
        first: First parsed price candidate.
        second: Second parsed price candidate.
    Output:
        ``True`` when candidates share OCR components or substantially overlap.
    """
    if (
        first.component_ids
        and second.component_ids
        and set(first.component_ids).intersection(second.component_ids)
    ):
        return True
    if (
        first.bbox is not None
        and second.bbox is not None
        and _bbox_iou(first.bbox, second.bbox) >= 0.60
    ):
        return True
    return False


def _same_price_row(
    first: PriceCandidate,
    second: PriceCandidate,
) -> bool:
    """Determine whether two candidates belong to the same visual row.

    Input:
        first: First parsed price candidate.
        second: Second parsed price candidate.
    Output:
        ``True`` when vertical overlap or center distance indicates one row.
    """
    if first.bbox is None or second.bbox is None:
        return False
    if _vertical_overlap_ratio(first.bbox, second.bbox) >= 0.35:
        return True
    center_delta = abs(first.y_center - second.y_center)
    return center_delta <= 0.30 * max(
        _bbox_height(first.bbox),
        _bbox_height(second.bbox),
    )


def _visual_price_order(
    candidates: Sequence[PriceCandidate],
) -> List[PriceCandidate]:
    """Order candidates left-to-right within top-to-bottom rows.

    Input:
        candidates: Selected, non-conflicting price candidates.
    Output:
        Candidates in visual reading order, with unpositioned values last.
    """
    # Separate candidates that can and cannot participate in geometric rows.
    positioned = [
        candidate for candidate in candidates if candidate.bbox is not None
    ]
    unpositioned = [
        candidate for candidate in candidates if candidate.bbox is None
    ]
    # Cluster positioned candidates into visually overlapping text rows.
    rows: List[List[PriceCandidate]] = []
    for candidate in sorted(
        positioned,
        key=lambda item: (item.y_center, item.x_center),
    ):
        matching_row = next(
            (
                row
                for row in rows
                if any(
                    _same_price_row(candidate, member)
                    for member in row
                )
            ),
            None,
        )
        if matching_row is None:
            rows.append([candidate])
        else:
            matching_row.append(candidate)

    # Sort rows top-to-bottom and candidates left-to-right within each row.
    rows.sort(
        key=lambda row: min(
            member.bbox[1]
            for member in row
            if member.bbox is not None
        )
    )
    ordered = [
        candidate
        for row in rows
        for candidate in sorted(
            row,
            key=lambda item: (
                item.x_center,
                item.y_center,
                item.source_index,
            ),
        )
    ]
    ordered.extend(
        sorted(
            unpositioned,
            key=lambda item: (item.source_index, item.match_index),
        )
    )
    return ordered


def select_price_candidates(
    candidates: Sequence[PriceCandidate],
    max_prices: int = DEFAULT_OCR_MAX_PRICES,
) -> List[PriceCandidate]:
    """Choose the strongest non-overlapping candidates in visual order.

    Input:
        candidates: All price interpretations found on one tag.
        max_prices: Maximum number of prices to return, from one to three.
    Output:
        Quality-selected candidates ordered as they appear on the tag.
    """
    if not 1 <= int(max_prices) <= 3:
        raise ValueError("max_prices must be between 1 and 3.")
    # Prefer explicit, visually prominent, high-confidence interpretations.
    ranked = sorted(candidates, key=_candidate_rank, reverse=True)
    selected: List[PriceCandidate] = []
    for candidate in ranked:
        if any(
            _candidates_conflict(candidate, existing)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(max_prices):
            break
    # Quality selection is independent of the final visual reading order.
    return _visual_price_order(selected)
