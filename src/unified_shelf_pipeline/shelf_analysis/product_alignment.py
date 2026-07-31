"""Align detected price tags with recognized products on a shelf.

Alignment is shelf-local and one-to-one. The module first builds plausible
product candidates above each tag, applies recognition-aware corrections for
unknown products, groups tags and products into shelf rows, and then uses
dynamic programming to produce non-crossing assignments. Both output tables
retain detailed audit fields explaining every decision.
"""

from __future__ import annotations

from collections import Counter
from math import hypot
from typing import Optional, Sequence

import pandas as pd


UNKNOWN_LABEL = "__Unknown__"
CONSENSUS_NEIGHBOR_LIMIT = 6
CONSENSUS_MIN_VOTES = 2
CONSENSUS_MIN_SHARE = 0.60
SAME_ROW_MIN_VERTICAL_OVERLAP = 0.50
DIRECT_MATCH_MIN_HORIZONTAL_OVERLAP = 0.50
TAG_ROW_MIN_VERTICAL_OVERLAP = 0.50
ONE_TO_ONE_MAX_HORIZONTAL_OFFSET_RATIO = 0.85
ONE_TO_ONE_MAX_VERTICAL_GAP_RATIO = 3.00
ONE_TO_ONE_UNMATCHED_COST = 0.85
ONE_TO_ONE_VERTICAL_COST_WEIGHT = 0.10
ONE_TO_ONE_CORRECTION_BONUS = 0.75

PRODUCT_ALIGNMENT_COLUMNS = [
    "aligned_price_tag_indices",
    "aligned_price_tag_count",
]

PRICE_TAG_ALIGNMENT_COLUMNS = [
    "closest_product_index",
    "closest_product_label",
    "closest_product_score",
    "closest_product_is_unknown",
    "closest_product_is_rejected",
    "closest_product_distance_px",
    "preferred_product_index",
    "preferred_product_label",
    "preferred_product_distance_px",
    "alignment_shelf_row_index",
    "alignment_horizontal_offset_ratio",
    "alignment_vertical_gap_ratio",
    "alignment_match_cost",
    "alignment_one_to_one_reassigned",
    "alignment_one_to_one_rejection_count",
    "aligned_product_index",
    "aligned_product_label",
    "aligned_product_score",
    "aligned_product_is_unknown",
    "aligned_product_is_rejected",
    "alignment_corrected",
    "alignment_correction_reason",
    "alignment_consensus_label",
    "alignment_consensus_votes",
    "alignment_consensus_total",
    "alignment_consensus_share",
    "alignment_consensus_product_indices",
    "alignment_distance_px",
    "alignment_dx_px",
    "alignment_dy_px",
    "alignment_product_center_x",
    "alignment_product_center_y",
    "alignment_price_tag_center_x",
    "alignment_price_tag_center_y",
    "alignment_status",
]

PRODUCT_PRICE_COLUMNS = [
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
]


def _has_value(value) -> bool:
    """Check whether a tabular value is present and non-blank.

    Input:
        value: Scalar value read from a DataFrame row.
    Output:
        ``True`` when the value is neither missing nor blank.
    """
    return value is not None and pd.notna(value) and str(value).strip() != ""


def _is_rejected(value) -> bool:
    """Normalize a stored product-rejection flag to Boolean form.

    Input:
        value: Boolean, string, numeric, or missing rejection value.
    Output:
        Normalized rejection state; missing values are treated as ``False``.
    """
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _bbox_from_row(row) -> tuple[float, float, float, float]:
    """Read floating-point bounding-box coordinates from a table row.

    Input:
        row: Product or price-tag row containing persisted box columns.
    Output:
        Coordinates ordered as ``x1, y1, x2, y2``.
    """
    return (
        float(row["bbox_xmin"]),
        float(row["bbox_ymin"]),
        float(row["bbox_xmax"]),
        float(row["bbox_ymax"]),
    )


def _bbox_center(row) -> tuple[float, float]:
    """Calculate a product or tag row's bounding-box center.

    Input:
        row: Product or price-tag row containing persisted box columns.
    Output:
        Floating-point ``(x, y)`` center coordinates.
    """
    x1, y1, x2, y2 = _bbox_from_row(row)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _product_label(row) -> Optional[str]:
    """Read the preferred recognition label from a product row.

    Input:
        row: Product result row.
    Output:
        Final or best recognition label, or ``None`` when unavailable.
    """
    label = row.get("final_pred") or row.get("best_label")
    return str(label) if _has_value(label) else None


def _is_unknown_product(row, unknown_label: str) -> bool:
    """Check whether a product row carries the configured unknown label.

    Input:
        row: Product result row.
        unknown_label: Recognition label used for unknown products.
    Output:
        ``True`` when the row resolves to the unknown label.
    """
    return _product_label(row) == unknown_label


def _candidate_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """Retain only product rows with complete bounding-box coordinates.

    Input:
        products_df: Product results that may contain incomplete rows.
    Output:
        A filtered DataFrame eligible for geometric alignment.
    """
    if products_df.empty:
        return products_df

    mask = []
    for _, row in products_df.iterrows():
        has_bbox = all(_has_value(row.get(column)) for column in ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"])
        mask.append(has_bbox)
    return products_df.loc[mask]


def _candidate_tuple(row_idx, product, tag_cx: float, tag_cy: float) -> Optional[dict]:
    """Build geometric evidence for a product positioned above a tag.

    Input:
        row_idx: Product DataFrame index used for later updates.
        product: Candidate product row.
        tag_cx: Price-tag center ``x`` coordinate.
        tag_cy: Price-tag center ``y`` coordinate.
    Output:
        Candidate geometry dictionary, or ``None`` if the product is not above.
    """
    product_cx, product_cy = _bbox_center(product)
    dy = tag_cy - product_cy
    if dy <= 0:
        return None
    dx = tag_cx - product_cx
    return {
        "row_idx": row_idx,
        "product": product,
        "product_cx": product_cx,
        "product_cy": product_cy,
        "dx": dx,
        "dy": dy,
        "distance": hypot(dx, dy),
    }


def _known_candidate(candidate: dict, unknown_label: str) -> bool:
    """Check whether a candidate has an accepted known-product label.

    Input:
        candidate: Product candidate and its geometry.
        unknown_label: Recognition label used for unknown products.
    Output:
        ``True`` when the candidate is labeled, known, and not rejected.
    """
    product = candidate["product"]
    label = _product_label(product)
    return _has_value(label) and label != unknown_label and not _is_rejected(product.get("is_rejected"))


def _vertical_overlap_ratio(first_product, second_product) -> float:
    """Measure vertical overlap relative to the shorter bounding box.

    Input:
        first_product: First product or tag table row.
        second_product: Second product or tag table row.
    Output:
        Vertical overlap ratio, or zero for invalid box heights.
    """
    _, first_y1, _, first_y2 = _bbox_from_row(first_product)
    _, second_y1, _, second_y2 = _bbox_from_row(second_product)
    overlap = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    shorter_height = min(first_y2 - first_y1, second_y2 - second_y1)
    return overlap / shorter_height if shorter_height > 0 else 0.0


def _horizontal_overlap_ratio(first_bbox, second_bbox) -> float:
    """Measure horizontal overlap relative to the narrower bounding box.

    Input:
        first_bbox: First product or tag table row.
        second_bbox: Second product or tag table row.
    Output:
        Horizontal overlap ratio, or zero for invalid box widths.
    """
    first_x1, _, first_x2, _ = _bbox_from_row(first_bbox)
    second_x1, _, second_x2, _ = _bbox_from_row(second_bbox)
    overlap = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    shorter_width = min(first_x2 - first_x1, second_x2 - second_x1)
    return overlap / shorter_width if shorter_width > 0 else 0.0


def _same_row_candidates(
    candidates: Sequence[dict],
    anchor_candidate: dict,
    min_vertical_overlap: float,
) -> list[dict]:
    """Find the anchor's vertically connected product shelf row.

    Input:
        candidates: Distance-ordered product candidates.
        anchor_candidate: Candidate defining the initial shelf row.
        min_vertical_overlap: Overlap required to connect two row members.
    Output:
        Connected row candidates in their original distance order.
    """
    row_members = [anchor_candidate]
    remaining = list(candidates)

    while remaining:
        newly_connected = []
        still_remaining = []
        for candidate in remaining:
            if any(
                _vertical_overlap_ratio(member["product"], candidate["product"]) >= min_vertical_overlap
                for member in row_members
            ):
                newly_connected.append(candidate)
            else:
                still_remaining.append(candidate)
        if not newly_connected:
            break
        row_members.extend(newly_connected)
        remaining = still_remaining

    row_indices = {candidate["row_idx"] for candidate in row_members}
    return [candidate for candidate in candidates if candidate["row_idx"] in row_indices]


def _directly_above_known_candidate(
    candidates: Sequence[dict],
    tag,
    unknown_label: str,
    min_horizontal_overlap: float,
) -> Optional[dict]:
    """Find the nearest known product directly above a price tag.

    Input:
        candidates: Distance-ordered candidates on the anchor shelf row.
        tag: Price-tag table row.
        unknown_label: Recognition label used for unknown products.
        min_horizontal_overlap: Required product/tag horizontal overlap.
    Output:
        The first qualifying candidate, or ``None``.
    """
    return next(
        (
            candidate
            for candidate in candidates
            if _known_candidate(candidate, unknown_label)
            and _horizontal_overlap_ratio(tag, candidate["product"]) >= min_horizontal_overlap
        ),
        None,
    )


def _consensus_correction(
    candidates: Sequence[dict],
    unknown_label: str,
    neighbor_limit: int,
    min_votes: int,
    min_share: float,
) -> Optional[dict]:
    """Find a dominant known label among the nearest product candidates.

    Input:
        candidates: Distance-ordered candidates on one product row.
        unknown_label: Recognition label used for unknown products.
        neighbor_limit: Maximum number of nearby candidates to inspect.
        min_votes: Minimum matching labels required for correction.
        min_share: Minimum share of known neighbors required for correction.
    Output:
        Consensus evidence and representative candidate, or ``None``.
    """
    # Restrict voting to nearby candidates with accepted recognition labels.
    nearest_neighbors = list(candidates[:neighbor_limit])
    known_neighbors = [candidate for candidate in nearest_neighbors if _known_candidate(candidate, unknown_label)]
    if not known_neighbors:
        return None

    labels = [_product_label(candidate["product"]) for candidate in known_neighbors]
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return None

    # Require an unambiguous label with sufficient votes and vote share.
    ranked = counts.most_common()
    label, votes = ranked[0]
    second_votes = ranked[1][1] if len(ranked) > 1 else 0
    total = len(known_neighbors)
    share = votes / max(total, 1)
    if votes < min_votes or share < min_share or votes <= second_votes:
        return None

    # Return the nearest representative plus every supporting product index.
    representative = next(
        candidate
        for candidate in known_neighbors
        if _product_label(candidate["product"]) == label
    )
    product_indices = [
        int(candidate["product"]["product_index"])
        for candidate in known_neighbors
        if _product_label(candidate["product"]) == label
    ]
    return {
        "candidate": representative,
        "label": label,
        "votes": votes,
        "total": total,
        "share": share,
        "product_indices": product_indices,
    }


def _tag_indices_text(indices: Sequence[int]) -> Optional[str]:
    """Serialize aligned tag indices for persisted product output.

    Input:
        indices: Ordered price-tag indices.
    Output:
        Slash-separated indices, or ``None`` when the sequence is empty.
    """
    return " / ".join(str(int(idx)) for idx in indices) if indices else None


def _select_alignment_candidate(
    candidates: Sequence[dict],
    tag,
    unknown_label: str,
    neighbor_limit: int,
    min_votes: int,
    min_share: float,
    same_row_min_vertical_overlap: float,
    direct_match_min_horizontal_overlap: float,
) -> dict:
    """Select a preferred product using recognition-aware corrections.

    Input:
        candidates: Distance-ordered product candidates above the tag.
        tag: Price-tag table row being aligned.
        unknown_label: Recognition label used for unknown products.
        neighbor_limit: Maximum number of neighbors used for correction.
        min_votes: Minimum label votes required for consensus correction.
        min_share: Minimum known-neighbor share required for consensus.
        same_row_min_vertical_overlap: Overlap used to restrict correction to
            the anchor's product row.
        direct_match_min_horizontal_overlap: Required overlap for a directly
            aligned known product.
    Output:
        Preferred candidate plus optional correction evidence and reason.
    """
    # Start with the geometrically closest product above the tag.
    closest_candidate = candidates[0]
    closest_product = closest_candidate["product"]
    closest_is_unknown = _is_unknown_product(closest_product, unknown_label)
    closest_is_rejected = _is_rejected(closest_product.get("is_rejected"))

    final_candidate = closest_candidate
    correction = None
    correction_reason = None
    # Correct unknown/rejected matches without leaving the same shelf row.
    if closest_is_unknown or closest_is_rejected:
        same_row_candidates = _same_row_candidates(
            candidates[1:],
            closest_candidate,
            same_row_min_vertical_overlap,
        )
        nearest_same_row_candidates = same_row_candidates[:neighbor_limit]
        direct_candidate = _directly_above_known_candidate(
            nearest_same_row_candidates,
            tag,
            unknown_label,
            direct_match_min_horizontal_overlap,
        )
        if direct_candidate is not None:
            correction = {"candidate": direct_candidate}
            correction_reason = "directly_above_known"
        else:
            correction = _consensus_correction(
                nearest_same_row_candidates,
                unknown_label,
                neighbor_limit,
                min_votes,
                min_share,
            )
            if correction is not None:
                correction_reason = "neighbor_consensus"
        if correction is not None:
            final_candidate = correction["candidate"]

    return {
        "candidate": final_candidate,
        "correction": correction,
        "correction_reason": correction_reason,
    }


def _candidate_product_index(candidate: dict) -> int:
    """Read a candidate's stable product index.

    Input:
        candidate: Product candidate dictionary.
    Output:
        Integer product index used by persisted result tables.
    """
    return int(candidate["product"]["product_index"])


def _connected_tag_rows(tag_records: Sequence[dict], min_vertical_overlap: float) -> list[list[dict]]:
    """Group price tags into connected horizontal shelf-edge bands.

    Input:
        tag_records: Price-tag records containing original table rows.
        min_vertical_overlap: Overlap required to connect two tag boxes.
    Output:
        Tag groups ordered from the top shelf row to the bottom.
    """
    remaining = list(tag_records)
    groups = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            still_remaining = []
            for record in remaining:
                if any(
                    _vertical_overlap_ratio(record["tag"], member["tag"]) >= min_vertical_overlap
                    for member in group
                ):
                    group.append(record)
                    changed = True
                else:
                    still_remaining.append(record)
            remaining = still_remaining
        groups.append(group)

    return sorted(
        groups,
        key=lambda group: sum(_bbox_center(record["tag"])[1] for record in group) / len(group),
    )


def _assign_products_to_tag_rows(candidate_products: pd.DataFrame, tag_rows: Sequence[Sequence[dict]]) -> dict:
    """Assign every candidate product to the nearest tag row below it.

    Input:
        candidate_products: Products with complete bounding boxes.
        tag_rows: Connected price-tag shelf rows.
    Output:
        Mapping from tag-row index to product DataFrame rows.
    """
    row_centers = [
        sum(_bbox_center(record["tag"])[1] for record in group) / len(group)
        for group in tag_rows
    ]
    products_by_row = {row_index: [] for row_index in range(len(tag_rows))}
    for product_row_idx, product in candidate_products.iterrows():
        _, product_cy = _bbox_center(product)
        rows_below = [
            (row_center - product_cy, row_index)
            for row_index, row_center in enumerate(row_centers)
            if row_center > product_cy
        ]
        if not rows_below:
            continue
        _, row_index = min(rows_below)
        products_by_row[row_index].append((product_row_idx, product))
    return products_by_row


def _horizontal_offset_ratio(candidate: dict, tag) -> float:
    """Normalize product/tag center offset by their wider bounding box.

    Input:
        candidate: Product candidate and its tag-relative geometry.
        tag: Price-tag table row.
    Output:
        Non-negative horizontal offset ratio.
    """
    product_x1, _, product_x2, _ = _bbox_from_row(candidate["product"])
    tag_x1, _, tag_x2, _ = _bbox_from_row(tag)
    width_scale = max(product_x2 - product_x1, tag_x2 - tag_x1, 1.0)
    return abs(float(candidate["dx"])) / width_scale


def _vertical_gap_ratio(candidate: dict, tag) -> float:
    """Normalize the product-bottom to tag-center vertical gap.

    Input:
        candidate: Product candidate and its tag-relative geometry.
        tag: Price-tag table row.
    Output:
        Non-negative vertical gap measured in product heights.
    """
    _, product_y1, _, product_y2 = _bbox_from_row(candidate["product"])
    _, tag_cy = _bbox_center(tag)
    product_height = max(product_y2 - product_y1, 1.0)
    return max(0.0, tag_cy - product_y2) / product_height


def _ordered_match_cost(
    candidate: dict,
    tag,
    preferred_decision: Optional[dict],
    vertical_cost_weight: float,
    correction_bonus: float,
) -> float:
    """Calculate the assignment cost for one product/tag pair.

    Input:
        candidate: Product candidate and its geometry.
        tag: Price-tag table row.
        preferred_decision: Optional recognition-aware preferred match.
        vertical_cost_weight: Contribution of normalized vertical distance.
        correction_bonus: Cost reduction for a preferred corrected match.
    Output:
        Pair cost where lower values represent better assignments.
    """
    horizontal_cost = _horizontal_offset_ratio(candidate, tag)
    cost = horizontal_cost + vertical_cost_weight * _vertical_gap_ratio(candidate, tag)

    if (
        preferred_decision is not None
        and preferred_decision["correction"] is not None
        and _candidate_product_index(preferred_decision["candidate"]) == _candidate_product_index(candidate)
    ):
        cost -= correction_bonus
    return cost


def _monotonic_row_assignment(
    records: Sequence[dict],
    row_products: Sequence[tuple],
    max_horizontal_offset_ratio: float,
    max_vertical_gap_ratio: float,
    unmatched_cost: float,
    vertical_cost_weight: float,
    correction_bonus: float,
) -> dict[int, tuple[dict, float]]:
    """Find a minimum-cost non-crossing partial match for one shelf row.

    Input:
        records: Tag records assigned to the shelf row.
        row_products: Product DataFrame rows assigned to the same shelf row.
        max_horizontal_offset_ratio: Maximum horizontal pairing distance.
        max_vertical_gap_ratio: Maximum vertical pairing distance.
        unmatched_cost: Cost of leaving one tag without a product.
        vertical_cost_weight: Contribution of vertical distance to pair cost.
        correction_bonus: Cost reduction for a preferred corrected pairing.
    Output:
        Mapping from tag sequence to matched candidate and pair cost.
    """
    # Left-to-right ordering makes every dynamic-programming match non-crossing.
    ordered_records = sorted(
        records,
        key=lambda record: (
            _bbox_center(record["tag"])[0],
            int(record["tag"]["tag_index"]),
            int(record["sequence"]),
        ),
    )
    ordered_products = sorted(
        row_products,
        key=lambda item: (_bbox_center(item[1])[0], int(item[1]["product_index"])),
    )
    record_count = len(ordered_records)
    product_count = len(ordered_products)
    costs = [[float("inf")] * (product_count + 1) for _ in range(record_count + 1)]
    actions = [[None] * (product_count + 1) for _ in range(record_count + 1)]

    # Initialize boundary costs for skipped products and unmatched tags.
    for product_offset in range(product_count + 1):
        costs[0][product_offset] = 0.0
        if product_offset:
            actions[0][product_offset] = "skip_product"
    for record_offset in range(1, record_count + 1):
        costs[record_offset][0] = record_offset * unmatched_cost
        actions[record_offset][0] = "skip_tag"

    # Precompute only pairs that satisfy horizontal and vertical limits.
    pair_candidates = {}
    pair_costs = {}
    for record_offset, record in enumerate(ordered_records, start=1):
        candidates_by_index = {
            _candidate_product_index(candidate): candidate
            for candidate in record["row_candidates"]
        }
        for product_offset, (_, product) in enumerate(ordered_products, start=1):
            product_index = int(product["product_index"])
            candidate = candidates_by_index.get(product_index)
            if candidate is None:
                continue
            if _horizontal_offset_ratio(candidate, record["tag"]) > max_horizontal_offset_ratio:
                continue
            if _vertical_gap_ratio(candidate, record["tag"]) > max_vertical_gap_ratio:
                continue
            pair_candidates[(record_offset, product_offset)] = candidate
            pair_costs[(record_offset, product_offset)] = _ordered_match_cost(
                candidate,
                record["tag"],
                record.get("preferred_decision"),
                vertical_cost_weight,
                correction_bonus,
            )

    # Choose the cheapest skip or match action for every prefix combination.
    for record_offset in range(1, record_count + 1):
        for product_offset in range(1, product_count + 1):
            options = [
                (costs[record_offset - 1][product_offset] + unmatched_cost, 2, "skip_tag"),
                (costs[record_offset][product_offset - 1], 1, "skip_product"),
            ]
            pair_key = (record_offset, product_offset)
            if pair_key in pair_candidates:
                options.append(
                    (
                        costs[record_offset - 1][product_offset - 1] + pair_costs[pair_key],
                        0,
                        "match",
                    )
                )
            best_cost, _, best_action = min(options, key=lambda option: (option[0], option[1]))
            costs[record_offset][product_offset] = best_cost
            actions[record_offset][product_offset] = best_action

    # Backtrack through the action table to recover the selected assignments.
    matches = {}
    record_offset = record_count
    product_offset = product_count
    while record_offset > 0 or product_offset > 0:
        action = actions[record_offset][product_offset]
        if action == "match":
            record = ordered_records[record_offset - 1]
            pair_key = (record_offset, product_offset)
            matches[int(record["sequence"])] = (pair_candidates[pair_key], pair_costs[pair_key])
            record_offset -= 1
            product_offset -= 1
        elif action == "skip_tag":
            record_offset -= 1
        elif action == "skip_product":
            product_offset -= 1
        else:
            break
    return matches


def align_price_tags_to_products(
    products_df: pd.DataFrame,
    price_tags_df: pd.DataFrame,
    unknown_label: str = UNKNOWN_LABEL,
    neighbor_limit: int = CONSENSUS_NEIGHBOR_LIMIT,
    min_votes: int = CONSENSUS_MIN_VOTES,
    min_share: float = CONSENSUS_MIN_SHARE,
    same_row_min_vertical_overlap: float = SAME_ROW_MIN_VERTICAL_OVERLAP,
    direct_match_min_horizontal_overlap: float = DIRECT_MATCH_MIN_HORIZONTAL_OVERLAP,
    tag_row_min_vertical_overlap: float = TAG_ROW_MIN_VERTICAL_OVERLAP,
    max_horizontal_offset_ratio: float = ONE_TO_ONE_MAX_HORIZONTAL_OFFSET_RATIO,
    max_vertical_gap_ratio: float = ONE_TO_ONE_MAX_VERTICAL_GAP_RATIO,
    unmatched_cost: float = ONE_TO_ONE_UNMATCHED_COST,
    vertical_cost_weight: float = ONE_TO_ONE_VERTICAL_COST_WEIGHT,
    correction_bonus: float = ONE_TO_ONE_CORRECTION_BONUS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align each price tag to the best product above it.

    The raw match is price-tag centric: every price tag first finds the closest
    detected product whose bbox center is above the tag center. If that closest
    product is unknown/rejected, the next nearest products on the same shelf row
    are checked. Vertically overlapping product boxes form a connected shelf
    row, allowing stacked detections to belong to the anchor's row. A nearby
    known product horizontally aligned with the tag is preferred; otherwise a
    dominant known label can correct the final alignment.

    Final matches are one-to-one and shelf-local. Price tags are grouped into
    shelf-edge rows, and every product is assigned to the nearest tag row below
    it. Within each row, dynamic programming finds a minimum-cost left-to-right
    partial matching. This makes crossing connections impossible and prevents
    conflict resolution from jumping to a product on another shelf.

    Input:
        products_df: Product detections and recognition evidence.
        price_tags_df: Price-tag detections, extracted prices, and promotion
            evidence.
        unknown_label: Recognition label used for unknown products.
        neighbor_limit: Maximum correction neighbors inspected per tag.
        min_votes: Minimum same-label votes required for consensus correction.
        min_share: Minimum vote share required for consensus correction.
        same_row_min_vertical_overlap: Product overlap used to define one row.
        direct_match_min_horizontal_overlap: Product/tag overlap required for a
            directly-above correction.
        tag_row_min_vertical_overlap: Tag overlap used to form shelf-edge rows.
        max_horizontal_offset_ratio: Maximum product/tag horizontal distance.
        max_vertical_gap_ratio: Maximum product/tag vertical gap.
        unmatched_cost: Dynamic-programming cost of leaving a tag unmatched.
        vertical_cost_weight: Vertical-distance contribution to assignment cost.
        correction_bonus: Cost reduction for recognition-aware preferred pairs.
    Output:
        Copies of the product and price-tag tables populated with bidirectional
        alignment fields and audit evidence.
    """
    # Work on copies so callers retain their original stage-level tables.
    products_out = products_df.copy()
    tags_out = price_tags_df.copy()

    # Initialize every alignment column so early exits preserve a stable schema.
    for column in PRODUCT_ALIGNMENT_COLUMNS + PRODUCT_PRICE_COLUMNS:
        products_out[column] = None
    for column in PRICE_TAG_ALIGNMENT_COLUMNS:
        tags_out[column] = None

    if products_out.empty or tags_out.empty:
        return products_out, tags_out

    candidate_products = _candidate_products(products_out)
    tag_records = []

    # Build every geometrically valid product-above-tag candidate list.
    for sequence, (tag_row_idx, tag) in enumerate(tags_out.iterrows()):
        tag_cx, tag_cy = _bbox_center(tag)
        candidates = []

        for product_row_idx, product in candidate_products.iterrows():
            candidate = _candidate_tuple(product_row_idx, product, tag_cx, tag_cy)
            if candidate is not None:
                candidates.append(candidate)

        candidates = sorted(candidates, key=lambda item: item["distance"])

        tags_out.at[tag_row_idx, "alignment_price_tag_center_x"] = tag_cx
        tags_out.at[tag_row_idx, "alignment_price_tag_center_y"] = tag_cy

        tag_records.append(
            {
                "sequence": sequence,
                "tag_row_idx": tag_row_idx,
                "tag": tag,
                "candidates": candidates,
                "row_candidates": [],
                "preferred_decision": None,
                "match": None,
            }
        )

        # Persist the raw closest candidate independently of final assignment.
        if candidates:
            closest_candidate = candidates[0]
            closest_product = closest_candidate["product"]
            closest_label = _product_label(closest_product)
            closest_is_unknown = _is_unknown_product(closest_product, unknown_label)
            closest_is_rejected = _is_rejected(closest_product.get("is_rejected"))
            tags_out.at[tag_row_idx, "closest_product_index"] = int(closest_product["product_index"])
            tags_out.at[tag_row_idx, "closest_product_label"] = closest_label
            tags_out.at[tag_row_idx, "closest_product_score"] = closest_product.get("best_score")
            tags_out.at[tag_row_idx, "closest_product_is_unknown"] = closest_is_unknown
            tags_out.at[tag_row_idx, "closest_product_is_rejected"] = closest_is_rejected
            tags_out.at[tag_row_idx, "closest_product_distance_px"] = float(closest_candidate["distance"])
        else:
            tags_out.at[tag_row_idx, "alignment_status"] = "no_product_above"

    # Restrict matching to connected shelf rows before one-to-one assignment.
    tag_rows = _connected_tag_rows(tag_records, tag_row_min_vertical_overlap)
    products_by_row = _assign_products_to_tag_rows(candidate_products, tag_rows)
    for shelf_row_index, records in enumerate(tag_rows):
        row_products = products_by_row[shelf_row_index]
        row_product_indices = {int(product["product_index"]) for _, product in row_products}
        for record in records:
            tag_row_idx = record["tag_row_idx"]
            tags_out.at[tag_row_idx, "alignment_shelf_row_index"] = shelf_row_index
            record["row_candidates"] = [
                candidate
                for candidate in record["candidates"]
                if _candidate_product_index(candidate) in row_product_indices
            ]
            if not record["row_candidates"]:
                if record["candidates"]:
                    tags_out.at[tag_row_idx, "alignment_status"] = "no_product_in_shelf_row"
                continue

            preferred_decision = _select_alignment_candidate(
                record["row_candidates"],
                record["tag"],
                unknown_label,
                neighbor_limit,
                min_votes,
                min_share,
                same_row_min_vertical_overlap,
                direct_match_min_horizontal_overlap,
            )
            record["preferred_decision"] = preferred_decision
            preferred_candidate = preferred_decision["candidate"]
            preferred_product = preferred_candidate["product"]
            tags_out.at[tag_row_idx, "preferred_product_index"] = int(preferred_product["product_index"])
            tags_out.at[tag_row_idx, "preferred_product_label"] = _product_label(preferred_product)
            tags_out.at[tag_row_idx, "preferred_product_distance_px"] = float(preferred_candidate["distance"])

        # Resolve row conflicts with a non-crossing one-to-one assignment.
        matches = _monotonic_row_assignment(
            records,
            row_products,
            max_horizontal_offset_ratio,
            max_vertical_gap_ratio,
            unmatched_cost,
            vertical_cost_weight,
            correction_bonus,
        )
        for record in records:
            record["match"] = matches.get(int(record["sequence"]))

    # Persist final tag-side evidence and the nearest tag fields on products.
    aligned_tags_by_product: dict[int, list[tuple[int, float]]] = {}
    for record in tag_records:
        tag_row_idx = record["tag_row_idx"]
        tag = record["tag"]
        match = record["match"]
        preferred_decision = record["preferred_decision"]
        if match is None:
            if preferred_decision is not None:
                tags_out.at[tag_row_idx, "alignment_one_to_one_reassigned"] = True
                tags_out.at[tag_row_idx, "alignment_one_to_one_rejection_count"] = sum(
                    1
                    for candidate in record["row_candidates"]
                    if _horizontal_offset_ratio(candidate, tag) <= max_horizontal_offset_ratio
                    and _vertical_gap_ratio(candidate, tag) <= max_vertical_gap_ratio
                )
                tags_out.at[tag_row_idx, "alignment_status"] = "no_available_product_for_one_to_one"
            continue

        final_candidate, match_cost = match
        preferred_candidate = preferred_decision["candidate"]
        preferred_product_index = _candidate_product_index(preferred_candidate)
        final_product_index = _candidate_product_index(final_candidate)
        if final_product_index == preferred_product_index:
            correction = preferred_decision["correction"]
            correction_reason = preferred_decision["correction_reason"]
        else:
            correction = None
            correction_reason = None
        product_row_idx = final_candidate["row_idx"]
        product = final_candidate["product"]
        product_cx = final_candidate["product_cx"]
        product_cy = final_candidate["product_cy"]
        best_distance = final_candidate["distance"]
        best_dx = final_candidate["dx"]
        best_dy = final_candidate["dy"]
        product_index = int(product["product_index"])
        tag_index = int(tag["tag_index"])
        product_label = _product_label(product)
        product_is_unknown = _is_unknown_product(product, unknown_label)
        product_is_rejected = _is_rejected(product.get("is_rejected"))

        tags_out.at[tag_row_idx, "alignment_one_to_one_reassigned"] = product_index != preferred_product_index
        tags_out.at[tag_row_idx, "alignment_one_to_one_rejection_count"] = sum(
            1
            for candidate in record["row_candidates"]
            if _ordered_match_cost(
                candidate,
                tag,
                preferred_decision,
                vertical_cost_weight,
                correction_bonus,
            ) < match_cost
            and _horizontal_offset_ratio(candidate, tag) <= max_horizontal_offset_ratio
            and _vertical_gap_ratio(candidate, tag) <= max_vertical_gap_ratio
        )
        tags_out.at[tag_row_idx, "alignment_horizontal_offset_ratio"] = _horizontal_offset_ratio(final_candidate, tag)
        tags_out.at[tag_row_idx, "alignment_vertical_gap_ratio"] = _vertical_gap_ratio(final_candidate, tag)
        tags_out.at[tag_row_idx, "alignment_match_cost"] = float(match_cost)
        tags_out.at[tag_row_idx, "aligned_product_index"] = product_index
        tags_out.at[tag_row_idx, "aligned_product_label"] = product_label
        tags_out.at[tag_row_idx, "aligned_product_score"] = product.get("best_score")
        tags_out.at[tag_row_idx, "aligned_product_is_unknown"] = product_is_unknown
        tags_out.at[tag_row_idx, "aligned_product_is_rejected"] = product_is_rejected
        tags_out.at[tag_row_idx, "alignment_corrected"] = correction is not None
        if correction is not None:
            tags_out.at[tag_row_idx, "alignment_correction_reason"] = correction_reason
        if correction_reason == "neighbor_consensus":
            tags_out.at[tag_row_idx, "alignment_consensus_label"] = correction["label"]
            tags_out.at[tag_row_idx, "alignment_consensus_votes"] = correction["votes"]
            tags_out.at[tag_row_idx, "alignment_consensus_total"] = correction["total"]
            tags_out.at[tag_row_idx, "alignment_consensus_share"] = correction["share"]
            tags_out.at[tag_row_idx, "alignment_consensus_product_indices"] = _tag_indices_text(correction["product_indices"])
        tags_out.at[tag_row_idx, "alignment_distance_px"] = float(best_distance)
        tags_out.at[tag_row_idx, "alignment_dx_px"] = float(best_dx)
        tags_out.at[tag_row_idx, "alignment_dy_px"] = float(best_dy)
        tags_out.at[tag_row_idx, "alignment_product_center_x"] = product_cx
        tags_out.at[tag_row_idx, "alignment_product_center_y"] = product_cy
        if correction_reason == "directly_above_known":
            status = "corrected_directly_above_known"
        elif correction is not None:
            status = "corrected_neighbor_consensus"
        elif product_is_unknown:
            status = "aligned_unknown"
        elif product_is_rejected:
            status = "aligned_rejected"
        else:
            status = "aligned_known"
        tags_out.at[tag_row_idx, "alignment_status"] = status

        aligned_tags_by_product.setdefault(product_index, []).append((tag_index, float(best_distance)))

        current_best = products_out.at[product_row_idx, "price_tag_distance_px"]
        if current_best is None or pd.isna(current_best) or float(best_distance) < float(current_best):
            products_out.at[product_row_idx, "nearest_price_tag_index"] = tag_index
            products_out.at[product_row_idx, "nearest_price"] = tag.get("primary_price")
            products_out.at[product_row_idx, "nearest_price_value"] = tag.get("primary_value")
            products_out.at[product_row_idx, "nearest_secondary_price"] = tag.get("secondary_price")
            products_out.at[product_row_idx, "nearest_secondary_price_value"] = tag.get("secondary_value")
            products_out.at[product_row_idx, "nearest_tertiary_price"] = tag.get("tertiary_price")
            products_out.at[product_row_idx, "nearest_tertiary_price_value"] = tag.get("tertiary_value")
            products_out.at[product_row_idx, "nearest_prices"] = tag.get("prices")
            products_out.at[product_row_idx, "nearest_price_values"] = tag.get("price_values")
            products_out.at[product_row_idx, "nearest_sale_classification"] = tag.get("sale_classification")
            products_out.at[product_row_idx, "nearest_is_promo"] = tag.get("is_promo")
            products_out.at[product_row_idx, "nearest_sale_reason"] = tag.get("sale_reason")
            products_out.at[product_row_idx, "nearest_tag_color"] = tag.get("tag_color")
            products_out.at[product_row_idx, "price_tag_distance_px"] = float(best_distance)

    # Summarize all aligned tag indices for every product.
    for product_row_idx, product in products_out.iterrows():
        product_index = int(product["product_index"])
        aligned_tags = sorted(aligned_tags_by_product.get(product_index, []), key=lambda item: item[1])
        tag_indices = [tag_index for tag_index, _ in aligned_tags]
        products_out.at[product_row_idx, "aligned_price_tag_indices"] = _tag_indices_text(tag_indices)
        products_out.at[product_row_idx, "aligned_price_tag_count"] = len(tag_indices)

    return products_out, tags_out
