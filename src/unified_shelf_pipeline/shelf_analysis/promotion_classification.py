"""Classify shelf price tags as promotional or regular.

Color and geometry need different comparison scopes:

* the regular tag color is inferred across the image;
* unusual tag size is measured within a shelf row;
* explicit offer words from OCR provide independent promotional evidence.

This module has no model-loading, file-iteration, or output side effects so it
can be reused by the unified pipeline and tested independently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Sequence
import unicodedata

import numpy as np


DEFAULT_OFFER_KEYWORDS = (
    "OFERTA",
    "PROMOCAO",
    "DESCONTO",
    "LIQUIDACAO",
    "APROVEITE",
)
FUZZY_OFFER_KEYWORDS = {"APROVEITE"}
COLOR_PROMO_RANK = {"white": 0, "yellow": 1, "red": 2}

SALE_CLASSIFICATION_COLUMNS = [
    "sale_classification",
    "is_promo",
    "sale_reason",
    "sale_color_reason",
    "sale_geometry_reason",
    "sale_keyword_match",
    "sale_baseline_color",
    "sale_baseline_confidence",
    "sale_baseline_source",
    "tag_color",
    "tag_red_ratio",
    "tag_yellow_ratio",
    "tag_background_lab_b",
    "tag_background_saturation",
    "tag_aspect_ratio",
    "tag_area",
    "tag_area_to_median_ratio",
    "tag_area_to_shelf_median_ratio",
    "sale_shelf_index",
]


@dataclass(frozen=True)
class SaleClassifierConfig:
    """Thresholds used by :class:`SaleClassifier`."""

    vertical_aspect_ratio: float = 1.20
    vertical_min_shelf_area_ratio: float = 0.65
    large_area_ratio: float = 1.80
    red_pixel_ratio: float = 0.10
    yellow_pixel_ratio: float = 0.25
    neutral_lab_b_max: float = 12.0
    neutral_saturation_max: float = 45.0
    yellow_lab_b_min: float = 45.0
    yellow_saturation_min: float = 90.0
    neutral_anchor_min_tags: int = 2
    shelf_min_y_threshold_px: int = 35
    shelf_image_y_ratio: float = 0.035
    baseline_min_confidence: float = 0.60
    baseline_min_tags: int = 3
    regular_tag_color: str | None = None
    offer_keywords: tuple[str, ...] = DEFAULT_OFFER_KEYWORDS

    def __post_init__(self) -> None:
        """Validate classifier thresholds immediately after construction.

        Input:
            None. Values are read from this immutable configuration instance.
        Output:
            None. Invalid settings raise ``ValueError``.
        """
        # Validate geometry thresholds.
        if self.vertical_aspect_ratio <= 0:
            raise ValueError("vertical_aspect_ratio must be greater than zero.")
        if not 0.0 <= self.vertical_min_shelf_area_ratio:
            raise ValueError(
                "vertical_min_shelf_area_ratio cannot be negative."
            )
        if self.large_area_ratio <= 0:
            raise ValueError("large_area_ratio must be greater than zero.")
        # Validate color proportions and background-measurement limits.
        if not 0.0 <= self.red_pixel_ratio <= 1.0:
            raise ValueError("red_pixel_ratio must be between 0 and 1.")
        if not 0.0 <= self.yellow_pixel_ratio <= 1.0:
            raise ValueError("yellow_pixel_ratio must be between 0 and 1.")
        if self.neutral_saturation_max < 0:
            raise ValueError("neutral_saturation_max cannot be negative.")
        if self.yellow_saturation_min < 0:
            raise ValueError("yellow_saturation_min cannot be negative.")
        if self.neutral_anchor_min_tags < 1:
            raise ValueError("neutral_anchor_min_tags must be at least 1.")
        # Validate shelf grouping, baseline inference, and configured labels.
        if self.shelf_min_y_threshold_px < 0:
            raise ValueError("shelf_min_y_threshold_px cannot be negative.")
        if self.shelf_image_y_ratio < 0:
            raise ValueError("shelf_image_y_ratio cannot be negative.")
        if not 0.0 <= self.baseline_min_confidence <= 1.0:
            raise ValueError("baseline_min_confidence must be between 0 and 1.")
        if self.baseline_min_tags < 1:
            raise ValueError("baseline_min_tags must be at least 1.")
        if (
            self.regular_tag_color is not None
            and self.regular_tag_color not in COLOR_PROMO_RANK
        ):
            raise ValueError("regular_tag_color must be white, yellow, red, or None.")
        if not self.offer_keywords:
            raise ValueError("offer_keywords cannot be empty.")


@dataclass(frozen=True)
class SaleTag:
    """Minimal detected-tag input needed by the classifier."""

    tag_index: int
    bbox: tuple[int, int, int, int]
    ocr_text: str | None = None


@dataclass(frozen=True)
class ColorEvidence:
    """Measured color features for one detected price-tag crop."""

    color: str
    red_ratio: float
    yellow_ratio: float
    background_lab_b: float
    background_saturation: float


@dataclass(frozen=True)
class SaleClassification:
    """Promotion decision and auditable evidence for one price tag."""

    tag_index: int
    sale_classification: str
    is_promo: bool | None
    sale_reason: str
    sale_color_reason: str
    sale_geometry_reason: str | None
    sale_keyword_match: str | None
    sale_baseline_color: str | None
    sale_baseline_confidence: float
    sale_baseline_source: str
    tag_color: str
    tag_red_ratio: float
    tag_yellow_ratio: float
    tag_background_lab_b: float
    tag_background_saturation: float
    tag_aspect_ratio: float
    tag_area: int
    tag_area_to_median_ratio: float | None
    tag_area_to_shelf_median_ratio: float | None
    sale_shelf_index: int

    def to_row(self) -> dict:
        """Convert classification evidence into persisted output fields.

        Input:
            None. Values are read from this classification instance.
        Output:
            Dictionary containing every ``SALE_CLASSIFICATION_COLUMNS`` field.
        """
        return {
            column: getattr(self, column)
            for column in SALE_CLASSIFICATION_COLUMNS
        }


def _import_cv2():
    """Import OpenCV with an actionable dependency error.

    Input:
        None.
    Output:
        The imported ``cv2`` module.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for sale color classification. "
            "Install opencv-python in the active environment."
        ) from exc
    return cv2


def detect_color_evidence(
    crop_bgr,
    *,
    red_pixel_ratio: float = 0.10,
    yellow_pixel_ratio: float = 0.25,
    yellow_lab_b_min: float = 45.0,
    yellow_saturation_min: float = 90.0,
) -> ColorEvidence:
    """Classify a tag crop and retain robust paper-background measurements.

    Raw yellow-pixel ratios are highly sensitive to shadows and warm shelf
    lighting. The median color of the brighter half of the inner tag is a much
    more stable approximation of the tag paper, because it excludes most dark
    text and borders.

    Input:
        crop_bgr: Price-tag crop in BGR array format.
        red_pixel_ratio: Minimum red-pixel share for a red classification.
        yellow_pixel_ratio: Minimum yellow-pixel share for yellow evidence.
        yellow_lab_b_min: Minimum LAB yellow-axis background value.
        yellow_saturation_min: Minimum HSV background saturation for yellow.
    Output:
        Inferred paper color plus raw and background color measurements.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return ColorEvidence("white", 0.0, 0.0, 0.0, 0.0)

    # Analyze the inner crop so shelf background and detector borders contribute
    # less to the color decision.
    cv2 = _import_cv2()
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    crop_height, crop_width = hsv.shape[:2]
    inner_hsv = hsv[
        int(crop_height * 0.20):int(crop_height * 0.80),
        int(crop_width * 0.15):int(crop_width * 0.85),
    ]
    if inner_hsv.size == 0:
        inner_hsv = hsv
        inner_bgr = crop_bgr
    else:
        inner_bgr = crop_bgr[
            int(crop_height * 0.20):int(crop_height * 0.80),
            int(crop_width * 0.15):int(crop_width * 0.85),
        ]
    total_pixels = max(1, inner_hsv.shape[0] * inner_hsv.shape[1])

    # Measure strongly colored red and yellow pixels in HSV space.
    red_low = cv2.inRange(
        inner_hsv,
        np.array([0, 70, 50], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )
    red_high = cv2.inRange(
        inner_hsv,
        np.array([160, 70, 50], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(red_low, red_high)
    yellow_mask = cv2.inRange(
        inner_hsv,
        np.array([11, 60, 80], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )

    red_ratio = float(np.count_nonzero(red_mask)) / total_pixels
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / total_pixels

    # Approximate paper color from the brighter half, excluding dark text.
    inner_lab = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2LAB)
    flat_lab = inner_lab.reshape(-1, 3).astype(np.float32)
    flat_hsv = inner_hsv.reshape(-1, 3).astype(np.float32)
    lightness_cutoff = float(np.percentile(flat_lab[:, 0], 55))
    background_mask = flat_lab[:, 0] >= lightness_cutoff
    if not np.any(background_mask):
        background_mask = np.ones(len(flat_lab), dtype=bool)
    background_lab_b = (
        float(np.median(flat_lab[background_mask, 2])) - 128.0
    )
    background_saturation = float(
        np.median(flat_hsv[background_mask, 1])
    )

    # Require both yellow coverage and paper-background evidence for yellow tags.
    if red_ratio > red_pixel_ratio:
        color = "red"
    elif (
        yellow_ratio > yellow_pixel_ratio
        and background_lab_b >= yellow_lab_b_min
        and background_saturation >= yellow_saturation_min
    ):
        color = "yellow"
    else:
        color = "white"
    return ColorEvidence(
        color,
        red_ratio,
        yellow_ratio,
        background_lab_b,
        background_saturation,
    )


def detect_color(
    crop_bgr,
    *,
    red_pixel_ratio: float = 0.10,
    yellow_pixel_ratio: float = 0.25,
    yellow_lab_b_min: float = 45.0,
    yellow_saturation_min: float = 90.0,
) -> str:
    """Return only the inferred color from full tag-color evidence.

    Input:
        crop_bgr: Price-tag crop in BGR array format.
        red_pixel_ratio: Minimum red-pixel share for a red classification.
        yellow_pixel_ratio: Minimum yellow-pixel share for yellow evidence.
        yellow_lab_b_min: Minimum LAB yellow-axis background value.
        yellow_saturation_min: Minimum HSV background saturation for yellow.
    Output:
        One of ``white``, ``yellow``, or ``red``.
    """
    return detect_color_evidence(
        crop_bgr,
        red_pixel_ratio=red_pixel_ratio,
        yellow_pixel_ratio=yellow_pixel_ratio,
        yellow_lab_b_min=yellow_lab_b_min,
        yellow_saturation_min=yellow_saturation_min,
    ).color


def group_tags_by_shelves(
    tags: Sequence[SaleTag],
    y_threshold: int = 40,
) -> list[list[SaleTag]]:
    """Group tags by neighboring vertical centers.

    Input:
        tags: Detected price tags to group into shelf rows.
        y_threshold: Maximum center difference between adjacent row members.
    Output:
        Shelf groups ordered from top to bottom.
    """
    if not tags:
        return []

    def center_y(tag: SaleTag) -> float:
        """Calculate one tag's vertical center.

        Input:
            tag: Price tag with integer bounding-box coordinates.
        Output:
            Floating-point center ``y`` coordinate.
        """
        return (tag.bbox[1] + tag.bbox[3]) / 2.0

    sorted_tags = sorted(tags, key=lambda tag: (center_y(tag), tag.tag_index))
    shelves = []
    current_shelf = [sorted_tags[0]]
    for tag in sorted_tags[1:]:
        if abs(center_y(tag) - center_y(current_shelf[-1])) <= y_threshold:
            current_shelf.append(tag)
        else:
            shelves.append(current_shelf)
            current_shelf = [tag]
    shelves.append(current_shelf)
    return shelves


def _normalize_text(text: str | None) -> str:
    """Normalize OCR text for exact and fuzzy keyword matching.

    Input:
        text: Optional OCR text from a price tag.
    Output:
        Uppercase, accent-free text containing alphanumeric tokens.
    """
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^A-Z0-9]+", " ", without_accents.upper()).strip()


def _find_offer_keyword(
    text: str | None,
    keywords: Sequence[str],
) -> str | None:
    """Find exact or narrowly fuzzy promotional keywords in OCR text.

    Input:
        text: Optional OCR text from a price tag.
        keywords: Promotional keywords accepted by the classifier.
    Output:
        Normalized matching keyword, or ``None`` when no match is found.
    """
    normalized_text = _normalize_text(text)
    text_tokens = normalized_text.split()
    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword)
        if re.search(rf"\b{re.escape(normalized_keyword)}\b", normalized_text):
            return normalized_keyword
        # OCR often changes one or two characters in long offer words. Keep
        # fuzzy matching deliberately narrow so ordinary short product words
        # cannot become promotional evidence.
        if normalized_keyword in FUZZY_OFFER_KEYWORDS:
            max_distance = 2 if len(normalized_keyword) >= 8 else 1
            for token in text_tokens:
                if (
                    abs(len(token) - len(normalized_keyword)) <= max_distance
                    and _edit_distance_with_limit(
                        token,
                        normalized_keyword,
                        max_distance,
                    )
                    <= max_distance
                ):
                    return normalized_keyword
    return None


def _edit_distance_with_limit(
    left: str,
    right: str,
    limit: int,
) -> int:
    """Calculate bounded Levenshtein edit distance.

    Input:
        left: First normalized token.
        right: Second normalized token.
        limit: Maximum distance worth calculating exactly.
    Output:
        Edit distance, or ``limit + 1`` once the limit must be exceeded.
    """
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        row_minimum = row_index
        for column_index, right_char in enumerate(right, start=1):
            current_value = min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1]
                + (left_char != right_char),
            )
            current.append(current_value)
            row_minimum = min(row_minimum, current_value)
        if row_minimum > limit:
            return limit + 1
        previous = current
    return previous[-1]


class SaleClassifier:
    """Classify price tags using global color and shelf-local geometry."""

    def __init__(self, config: SaleClassifierConfig | None = None):
        """Store validated promotion-classification settings.

        Input:
            config: Optional classifier thresholds and keyword configuration.
        Output:
            None. Default settings are created when configuration is omitted.
        """
        self.config = config or SaleClassifierConfig()

    def _infer_baseline(self, states: Sequence[dict]) -> tuple[str | None, float, str]:
        """Infer the store's regular tag color from eligible image tags.

        Input:
            states: Per-tag color, geometry, keyword, and area evidence.
        Output:
            Baseline color, confidence, and source description. The color is
            ``None`` when the image has insufficient support.
        """
        if self.config.regular_tag_color is not None:
            return self.config.regular_tag_color, 1.0, "configured"

        baseline_pool = [
            state
            for state in states
            if state["geometry_reason"] is None
            and state["keyword_match"] is None
            and state["area"] > 0
        ]
        if not baseline_pool:
            return None, 0.0, "insufficient_support"

        if len(baseline_pool) < self.config.baseline_min_tags:
            return None, 0.0, "insufficient_support"

        neutral_flags = [
            state["evidence"].background_lab_b
            <= self.config.neutral_lab_b_max
            and state["evidence"].background_saturation
            <= self.config.neutral_saturation_max
            for state in baseline_pool
        ]
        promotional_color_flags = [
            state["evidence"].color in {"yellow", "red"}
            for state in baseline_pool
        ]
        neutral_count = sum(neutral_flags)
        promotional_color_count = sum(promotional_color_flags)
        explained_count = sum(
            is_neutral or is_promotional
            for is_neutral, is_promotional in zip(
                neutral_flags,
                promotional_color_flags,
            )
        )
        confidence = explained_count / len(baseline_pool)

        # A neutral paper cluster is the stable anchor. It lets a yellow cluster
        # remain promotional even when yellow cards outnumber regular tags.
        if (
            neutral_count >= self.config.neutral_anchor_min_tags
            and promotional_color_count > 0
            and confidence >= self.config.baseline_min_confidence
        ):
            return "white", confidence, "image_mixed_palette_white_anchor"

        # Without a neutral anchor there is no reliable evidence that color
        # variation represents two tag types rather than lighting. Be
        # conservative and treat the dominant observed paper color as the
        # store's single regular palette.
        counts = Counter(
            state["evidence"].color
            for state in baseline_pool
        )
        candidate = max(
            counts,
            key=lambda color: (
                counts[color],
                COLOR_PROMO_RANK[color],
            ),
        )
        candidate_count = counts[candidate]
        return (
            candidate,
            candidate_count / len(baseline_pool),
            "image_single_palette",
        )

    def classify(
        self,
        image_bgr,
        tags: Sequence[SaleTag],
    ) -> dict[int, SaleClassification]:
        """Classify detected price tags as promotional, regular, or uncertain.

        Input:
            image_bgr: Full shelf image in BGR array format.
            tags: Unique detected tags with bounding boxes and optional OCR text.
        Output:
            Mapping from tag index to classification and auditable evidence.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("A non-empty BGR image is required.")
        if not tags:
            return {}
        if len({tag.tag_index for tag in tags}) != len(tags):
            raise ValueError("Sale tag indices must be unique.")

        # Measure per-tag geometry, color, and independent keyword evidence.
        image_height, image_width = image_bgr.shape[:2]
        states: dict[int, dict] = {}
        for tag in tags:
            x1, y1, x2, y2 = tag.bbox
            x1 = max(0, min(int(x1), image_width))
            x2 = max(0, min(int(x2), image_width))
            y1 = max(0, min(int(y1), image_height))
            y2 = max(0, min(int(y2), image_height))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            area = width * height
            aspect_ratio = height / max(1, width)
            evidence = detect_color_evidence(
                image_bgr[y1:y2, x1:x2],
                red_pixel_ratio=self.config.red_pixel_ratio,
                yellow_pixel_ratio=self.config.yellow_pixel_ratio,
                yellow_lab_b_min=self.config.yellow_lab_b_min,
                yellow_saturation_min=self.config.yellow_saturation_min,
            )
            states[tag.tag_index] = {
                "tag": tag,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "evidence": evidence,
                "keyword_match": _find_offer_keyword(
                    tag.ocr_text,
                    self.config.offer_keywords,
                ),
                "geometry_reason": None,
                "shelf_index": None,
                "shelf_area_ratio": None,
            }

        # Form shelf rows and detect row-local size or orientation outliers.
        image_median_area = float(
            np.median([state["area"] for state in states.values()])
        )
        y_threshold = max(
            self.config.shelf_min_y_threshold_px,
            int(image_height * self.config.shelf_image_y_ratio),
        )
        shelves = group_tags_by_shelves(tags, y_threshold=y_threshold)
        for shelf_index, shelf in enumerate(shelves):
            shelf_states = [states[tag.tag_index] for tag in shelf]
            shelf_median_area = float(
                np.median([state["area"] for state in shelf_states])
            )
            for state in shelf_states:
                state["shelf_index"] = shelf_index
                state["shelf_area_ratio"] = (
                    float(state["area"]) / shelf_median_area
                    if shelf_median_area > 0
                    else None
                )
                is_vertical = (
                    state["aspect_ratio"] > self.config.vertical_aspect_ratio
                    and state["shelf_area_ratio"] is not None
                    and state["shelf_area_ratio"]
                    >= self.config.vertical_min_shelf_area_ratio
                )
                is_large = (
                    state["shelf_area_ratio"] is not None
                    and state["shelf_area_ratio"] > self.config.large_area_ratio
                )
                if is_vertical and is_large:
                    state["geometry_reason"] = (
                        "vertical_and_shelf_size_outlier"
                    )
                elif is_vertical:
                    state["geometry_reason"] = "vertical_shape"
                elif is_large:
                    state["geometry_reason"] = "shelf_size_outlier"

        # Infer the regular paper color only after excluding obvious promotions.
        baseline_color, baseline_confidence, baseline_source = (
            self._infer_baseline(list(states.values()))
        )
        color_contrast_is_reliable = baseline_source in {
            "configured",
            "image_mixed_palette_white_anchor",
        }

        # Apply keyword, geometry, and reliable color evidence in priority order.
        classifications = {}
        for tag_index, state in states.items():
            evidence = state["evidence"]
            reliable_red_contrast = (
                baseline_color is not None
                and evidence.color == "red"
                and baseline_color != "red"
            )
            if baseline_color is None:
                color_reason = "insufficient_image_color_baseline"
            elif (
                baseline_source == "image_single_palette"
                and not reliable_red_contrast
            ):
                color_reason = "single_image_palette_no_reliable_contrast"
            elif evidence.color == baseline_color:
                color_reason = "matches_image_base_color"
            elif COLOR_PROMO_RANK[evidence.color] > COLOR_PROMO_RANK[baseline_color]:
                color_reason = (
                    f"{evidence.color}_vs_image_{baseline_color}"
                )
            else:
                color_reason = (
                    f"{evidence.color}_not_more_promotional_than_"
                    f"image_{baseline_color}"
                )

            if state["keyword_match"] is not None:
                sale_classification = "promo"
                is_promo = True
                sale_reason = "offer_keyword"
            elif state["geometry_reason"] is not None:
                sale_classification = "promo"
                is_promo = True
                sale_reason = state["geometry_reason"]
            elif (
                baseline_color is not None
                and (
                    color_contrast_is_reliable
                    or reliable_red_contrast
                )
                and COLOR_PROMO_RANK[evidence.color]
                > COLOR_PROMO_RANK[baseline_color]
            ):
                sale_classification = "promo"
                is_promo = True
                sale_reason = color_reason
            elif baseline_color is None:
                sale_classification = "uncertain"
                is_promo = None
                sale_reason = "insufficient_color_baseline"
            elif baseline_source == "image_single_palette":
                sale_classification = "regular"
                is_promo = False
                sale_reason = "image_single_palette"
            else:
                sale_classification = "regular"
                is_promo = False
                sale_reason = "image_base_color"

            # Persist raw measurements alongside the final decision.
            area_to_image_median = (
                float(state["area"]) / image_median_area
                if image_median_area > 0
                else None
            )
            classifications[tag_index] = SaleClassification(
                tag_index=tag_index,
                sale_classification=sale_classification,
                is_promo=is_promo,
                sale_reason=sale_reason,
                sale_color_reason=color_reason,
                sale_geometry_reason=state["geometry_reason"],
                sale_keyword_match=state["keyword_match"],
                sale_baseline_color=baseline_color,
                sale_baseline_confidence=baseline_confidence,
                sale_baseline_source=baseline_source,
                tag_color=evidence.color,
                tag_red_ratio=evidence.red_ratio,
                tag_yellow_ratio=evidence.yellow_ratio,
                tag_background_lab_b=evidence.background_lab_b,
                tag_background_saturation=evidence.background_saturation,
                tag_aspect_ratio=state["aspect_ratio"],
                tag_area=state["area"],
                tag_area_to_median_ratio=area_to_image_median,
                tag_area_to_shelf_median_ratio=state["shelf_area_ratio"],
                sale_shelf_index=state["shelf_index"],
            )
        return classifications
