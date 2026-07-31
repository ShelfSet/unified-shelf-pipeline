"""Unit tests for product-to-price-tag alignment."""

import unittest

import pandas as pd

from unified_shelf_pipeline.shelf_analysis.product_alignment import (
    align_price_tags_to_products,
)


def product(product_index, label, bbox, *, rejected=False):
    """Build one product-row fixture for alignment tests.

    Input:
        product_index: Stable product identifier.
        label: Recognition label stored on the row.
        bbox: Integer product bounding-box coordinates.
        rejected: Whether recognition rejected the product.
    Output:
        Dictionary matching the alignment product schema.
    """
    x1, y1, x2, y2 = bbox
    return {
        "product_index": product_index,
        "bbox_xmin": x1,
        "bbox_ymin": y1,
        "bbox_xmax": x2,
        "bbox_ymax": y2,
        "final_pred": label,
        "best_label": label,
        "best_score": 0.9,
        "is_rejected": rejected,
    }


def price_tag(bbox, tag_index=0):
    """Build one price-tag row fixture for alignment tests.

    Input:
        bbox: Integer tag bounding-box coordinates.
        tag_index: Stable price-tag identifier.
    Output:
        Dictionary containing tag geometry fields.
    """
    x1, y1, x2, y2 = bbox
    return {
        "tag_index": tag_index,
        "bbox_xmin": x1,
        "bbox_ymin": y1,
        "bbox_xmax": x2,
        "bbox_ymax": y2,
    }


class PriceTagAlignmentTests(unittest.TestCase):
    """Verify shelf-local, one-to-one product and price-tag alignment."""

    def test_one_to_one_reassigns_farther_tag_to_next_product(self):
        """Verify the closer tag keeps a contested product.

        Input:
            None. Product and tag fixtures are created inside the test.
        Output:
            None. Assertions validate one-to-one reassignment.
        """
        products = pd.DataFrame(
            [
                product(0, "known-a", (70, 100, 110, 200)),
                product(1, "known-b", (110, 100, 150, 200)),
            ]
        )
        # Put the farther tag first to verify that DataFrame order does not let
        # it retain product 0 when the closer tag needs the same product.
        tags = pd.DataFrame(
            [
                price_tag((95, 300, 115, 320), tag_index=1),
                price_tag((80, 300, 100, 320), tag_index=0),
            ]
        )

        aligned_products, aligned_tags = align_price_tags_to_products(products, tags)

        tag_0 = aligned_tags.loc[aligned_tags["tag_index"] == 0].iloc[0]
        tag_1 = aligned_tags.loc[aligned_tags["tag_index"] == 1].iloc[0]
        self.assertEqual(tag_0["preferred_product_index"], 0)
        self.assertEqual(tag_0["aligned_product_index"], 0)
        self.assertFalse(tag_0["alignment_one_to_one_reassigned"])
        self.assertEqual(tag_1["preferred_product_index"], 0)
        self.assertEqual(tag_1["aligned_product_index"], 1)
        self.assertTrue(tag_1["alignment_one_to_one_reassigned"])
        self.assertEqual(aligned_products["aligned_price_tag_count"].tolist(), [1, 1])
        self.assertEqual(aligned_tags["aligned_product_index"].nunique(), 2)

    def test_one_to_one_leaves_tag_unmatched_when_products_are_exhausted(self):
        """Verify excess tags remain unmatched when products are exhausted.

        Input:
            None. Product and tag fixtures are created inside the test.
        Output:
            None. Assertions validate the unmatched status.
        """
        products = pd.DataFrame([product(0, "known-a", (90, 100, 110, 200))])
        tags = pd.DataFrame(
            [
                price_tag((105, 300, 125, 320), tag_index=1),
                price_tag((90, 300, 110, 320), tag_index=0),
            ]
        )

        aligned_products, aligned_tags = align_price_tags_to_products(products, tags)

        tag_0 = aligned_tags.loc[aligned_tags["tag_index"] == 0].iloc[0]
        tag_1 = aligned_tags.loc[aligned_tags["tag_index"] == 1].iloc[0]
        self.assertEqual(tag_0["aligned_product_index"], 0)
        self.assertTrue(pd.isna(tag_1["aligned_product_index"]))
        self.assertEqual(tag_1["alignment_status"], "no_available_product_for_one_to_one")
        self.assertTrue(tag_1["alignment_one_to_one_reassigned"])
        self.assertEqual(aligned_products.iloc[0]["aligned_price_tag_count"], 1)

    def test_one_to_one_does_not_fall_back_to_previous_shelf_row(self):
        """Verify conflict resolution never enters a previous shelf row.

        Input:
            None. Multi-row product and tag fixtures are created internally.
        Output:
            None. Assertions validate shelf-local assignment.
        """
        products = pd.DataFrame(
            [
                product(0, "lower-row", (0, 200, 100, 300)),
                product(1, "upper-row", (10, 0, 110, 100)),
            ]
        )
        tags = pd.DataFrame(
            [
                price_tag((40, 140, 60, 160), tag_index=2),
                price_tag((40, 340, 60, 360), tag_index=0),
                price_tag((45, 340, 65, 360), tag_index=1),
            ]
        )

        _, aligned_tags = align_price_tags_to_products(products, tags)

        upper_tag = aligned_tags.loc[aligned_tags["tag_index"] == 2].iloc[0]
        lower_tag_0 = aligned_tags.loc[aligned_tags["tag_index"] == 0].iloc[0]
        lower_tag_1 = aligned_tags.loc[aligned_tags["tag_index"] == 1].iloc[0]
        self.assertEqual(upper_tag["aligned_product_index"], 1)
        self.assertEqual(lower_tag_0["aligned_product_index"], 0)
        self.assertTrue(pd.isna(lower_tag_1["aligned_product_index"]))
        self.assertEqual(lower_tag_1["alignment_status"], "no_available_product_for_one_to_one")

    def test_one_to_one_preserves_left_to_right_order(self):
        """Verify assignments cannot create crossing center connections.

        Input:
            None. Product and tag fixtures are created inside the test.
        Output:
            None. Assertions validate left-to-right monotonic matching.
        """
        products = pd.DataFrame(
            [
                product(0, "left", (40, 100, 240, 200)),
                product(1, "right", (170, 100, 370, 200)),
            ]
        )
        tags = pd.DataFrame(
            [
                price_tag((130, 300, 170, 320), tag_index=1),
                price_tag((100, 300, 140, 320), tag_index=0),
            ]
        )

        _, aligned_tags = align_price_tags_to_products(products, tags)

        left_tag = aligned_tags.loc[aligned_tags["tag_index"] == 0].iloc[0]
        right_tag = aligned_tags.loc[aligned_tags["tag_index"] == 1].iloc[0]
        self.assertEqual(left_tag["aligned_product_index"], 0)
        self.assertEqual(right_tag["aligned_product_index"], 1)
        self.assertLess(left_tag["alignment_product_center_x"], right_tag["alignment_product_center_x"])

    def test_large_vertical_gap_is_not_forced_into_a_match(self):
        """Verify a tag beyond the vertical limit remains unmatched.

        Input:
            None. Distant product and tag fixtures are created internally.
        Output:
            None. Assertions validate distance-based rejection.
        """
        products = pd.DataFrame([product(0, "previous-row", (50, 0, 150, 100))])
        tags = pd.DataFrame([price_tag((90, 490, 110, 510), tag_index=0)])

        aligned_products, aligned_tags = align_price_tags_to_products(products, tags)

        result = aligned_tags.iloc[0]
        self.assertTrue(pd.isna(result["aligned_product_index"]))
        self.assertEqual(result["alignment_status"], "no_available_product_for_one_to_one")
        self.assertEqual(aligned_products.iloc[0]["aligned_price_tag_count"], 0)

    def test_consensus_does_not_use_products_from_row_above_anchor(self):
        """Verify consensus correction stays on the anchor product row.

        Input:
            None. Multi-row recognition fixtures are created internally.
        Output:
            None. Assertions validate row-restricted consensus.
        """
        products = pd.DataFrame(
            [
                product(0, "__Unknown__", (90, 100, 110, 200)),
                product(1, "known-a", (80, 0, 120, 80)),
                product(2, "known-a", (120, 0, 160, 80)),
            ]
        )
        tags = pd.DataFrame([price_tag((90, 300, 110, 320))])

        _, aligned_tags = align_price_tags_to_products(products, tags)

        result = aligned_tags.iloc[0]
        self.assertEqual(result["closest_product_index"], 0)
        self.assertEqual(result["aligned_product_index"], 0)
        self.assertFalse(result["alignment_corrected"])
        self.assertEqual(result["alignment_status"], "aligned_unknown")

    def test_consensus_still_corrects_to_known_product_on_anchor_row(self):
        """Verify same-row known labels can correct an unknown product.

        Input:
            None. Recognition and geometry fixtures are created internally.
        Output:
            None. Assertions validate consensus correction evidence.
        """
        products = pd.DataFrame(
            [
                product(0, "__Unknown__", (90, 100, 110, 200)),
                product(1, "known-a", (105, 110, 145, 190)),
                product(2, "known-a", (55, 110, 95, 190)),
                product(3, "known-b", (150, 110, 190, 190)),
                product(4, "known-b", (80, 0, 120, 80)),
            ]
        )
        tags = pd.DataFrame([price_tag((90, 300, 110, 320))])

        _, aligned_tags = align_price_tags_to_products(products, tags)

        result = aligned_tags.iloc[0]
        self.assertEqual(result["closest_product_index"], 0)
        self.assertEqual(result["aligned_product_index"], 1)
        self.assertTrue(result["alignment_corrected"])
        self.assertEqual(result["alignment_consensus_votes"], 2)
        self.assertEqual(result["alignment_consensus_total"], 3)
        self.assertEqual(result["alignment_status"], "corrected_neighbor_consensus")

    def test_known_product_directly_above_beats_distant_label_consensus(self):
        """Verify direct horizontal evidence takes priority over consensus.

        Input:
            None. Competing product candidates are created internally.
        Output:
            None. Assertions validate the preferred correction reason.
        """
        products = pd.DataFrame(
            [
                product(88, "__Unknown__", (1799, 3085, 2107, 3162), rejected=True),
                product(47, "__Unknown__", (1822, 3045, 2094, 3088), rejected=True),
                product(59, "__Unknown__", (1822, 2990, 2094, 3047), rejected=True),
                product(54, "direct-product", (1820, 2908, 2104, 2988)),
                product(70, "distant-majority", (1479, 3004, 1764, 3066)),
                product(82, "distant-majority", (1447, 2846, 1798, 3161)),
                product(0, "distant-majority", (468, 2811, 761, 3168)),
            ]
        )
        tags = pd.DataFrame([price_tag((1861, 3168, 2020, 3239))])

        _, aligned_tags = align_price_tags_to_products(products, tags)

        result = aligned_tags.iloc[0]
        self.assertEqual(result["closest_product_index"], 88)
        self.assertEqual(result["aligned_product_index"], 54)
        self.assertEqual(result["alignment_correction_reason"], "directly_above_known")
        self.assertEqual(result["alignment_status"], "corrected_directly_above_known")

    def test_three_prices_are_propagated_to_the_aligned_product(self):
        """Verify all extracted prices propagate to the aligned product.

        Input:
            None. Product and multi-price tag fixtures are created internally.
        Output:
            None. Assertions validate primary, secondary, and tertiary fields.
        """
        products = pd.DataFrame([product(0, "known-a", (80, 100, 120, 200))])
        tag = price_tag((85, 250, 125, 280))
        tag.update(
            {
                "primary_price": "6,59",
                "primary_value": 6.59,
                "secondary_price": "6,59",
                "secondary_value": 6.59,
                "tertiary_price": "6,99",
                "tertiary_value": 6.99,
                "prices": "6,59 / 6,59 / 6,99",
                "price_values": "6.59 / 6.59 / 6.99",
                "sale_classification": "promo",
                "is_promo": True,
                "sale_reason": "red_vs_white",
                "tag_color": "red",
            }
        )

        aligned_products, _ = align_price_tags_to_products(products, pd.DataFrame([tag]))

        result = aligned_products.iloc[0]
        self.assertEqual(result["nearest_price"], "6,59")
        self.assertEqual(result["nearest_secondary_price"], "6,59")
        self.assertEqual(result["nearest_tertiary_price"], "6,99")
        self.assertEqual(result["nearest_prices"], "6,59 / 6,59 / 6,99")
        self.assertEqual(result["nearest_sale_classification"], "promo")
        self.assertTrue(result["nearest_is_promo"])
        self.assertEqual(result["nearest_sale_reason"], "red_vs_white")
        self.assertEqual(result["nearest_tag_color"], "red")


if __name__ == "__main__":
    unittest.main()
