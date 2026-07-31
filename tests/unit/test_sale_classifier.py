"""Unit tests for heuristic sale classification."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

from unified_shelf_pipeline.cli import build_config, parse_args
from unified_shelf_pipeline.shelf_analysis.promotion_classification import (
    SaleClassifier,
    SaleTag,
)


YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def paint(image, bbox, color):
    """Fill one rectangular image region with a test color.

    Input:
        image: Mutable test image array.
        bbox: Integer ``x1, y1, x2, y2`` coordinates.
        color: BGR color assigned to the region.
    Output:
        None. The test image is modified in place.
    """
    x1, y1, x2, y2 = bbox
    image[y1:y2, x1:x2] = color


class SaleClassifierTests(unittest.TestCase):
    """Verify promotion classification from color, geometry, and OCR text."""

    def test_image_white_baseline_marks_homogeneous_yellow_shelf_as_promo(self):
        """Verify a white baseline makes homogeneous yellow tags promotional.

        Input:
            None. A controlled multi-color shelf image is created internally.
        Output:
            None. Assertions validate baseline and promotion decisions.
        """
        image = np.full((360, 560, 3), 255, dtype=np.uint8)
        white_boxes = [
            (20, 20, 80, 50),
            (100, 20, 160, 50),
            (180, 20, 240, 50),
            (20, 100, 80, 130),
            (100, 100, 160, 130),
            (180, 100, 240, 130),
        ]
        yellow_boxes = [
            (20, 220, 80, 250),
            (100, 220, 160, 250),
            (180, 220, 240, 250),
        ]
        for bbox in yellow_boxes:
            paint(image, bbox, YELLOW)
        tags = [
            SaleTag(index, bbox)
            for index, bbox in enumerate(white_boxes + yellow_boxes)
        ]

        results = SaleClassifier().classify(image, tags)

        self.assertTrue(all(results[index].is_promo for index in range(6, 9)))
        self.assertTrue(
            all(
                results[index].sale_reason == "yellow_vs_image_white"
                for index in range(6, 9)
            )
        )
        self.assertTrue(all(not results[index].is_promo for index in range(6)))
        self.assertEqual(results[0].sale_baseline_color, "white")
        self.assertAlmostEqual(results[0].sale_baseline_confidence, 1.0)
        self.assertEqual(
            results[0].sale_baseline_source,
            "image_mixed_palette_white_anchor",
        )

    def test_yellow_majority_does_not_become_the_regular_baseline(self):
        """Verify a neutral anchor prevents yellow-majority baseline drift.

        Input:
            None. White and yellow tag fixtures are created internally.
        Output:
            None. Assertions validate the inferred regular color.
        """
        image = np.full((240, 800, 3), 255, dtype=np.uint8)
        white_boxes = [
            (20, 40, 80, 70),
            (100, 40, 160, 70),
            (180, 40, 240, 70),
        ]
        yellow_boxes = [
            (20 + index * 90, 150, 90 + index * 90, 190)
            for index in range(7)
        ]
        for bbox in yellow_boxes:
            paint(image, bbox, YELLOW)
        tags = [
            SaleTag(index, bbox)
            for index, bbox in enumerate(white_boxes + yellow_boxes)
        ]

        results = SaleClassifier().classify(image, tags)

        self.assertEqual(results[0].sale_baseline_color, "white")
        self.assertTrue(all(not results[index].is_promo for index in range(3)))
        self.assertTrue(
            all(results[index].is_promo for index in range(3, 10))
        )

    def test_all_yellow_image_uses_yellow_as_regular_baseline(self):
        """Verify a single yellow palette is treated as regular.

        Input:
            None. A uniformly yellow tag image is created internally.
        Output:
            None. Assertions validate baseline and regular classifications.
        """
        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 30, 80, 60),
            (100, 30, 160, 60),
            (180, 30, 240, 60),
            (20, 180, 80, 210),
            (100, 180, 160, 210),
            (180, 180, 240, 210),
        ]
        for bbox in boxes:
            paint(image, bbox, YELLOW)

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertTrue(all(result.sale_classification == "regular" for result in results.values()))
        self.assertTrue(all(result.is_promo is False for result in results.values()))
        self.assertTrue(all(result.sale_baseline_color == "yellow" for result in results.values()))

    def test_shelf_local_area_avoids_perspective_false_positives(self):
        """Verify shelf-local area baselines handle perspective changes.

        Input:
            None. Different-sized shelf rows are created internally.
        Output:
            None. Assertions validate absence of false size promotions.
        """
        image = np.full((400, 700, 3), 255, dtype=np.uint8)
        large_row = [
            (20, 40, 140, 80),
            (160, 40, 280, 80),
            (300, 40, 420, 80),
            (440, 40, 560, 80),
        ]
        small_row = [
            (20, 260, 90, 290),
            (110, 260, 180, 290),
            (200, 260, 270, 290),
            (290, 260, 360, 290),
        ]
        boxes = large_row + small_row
        for bbox in boxes:
            paint(image, bbox, YELLOW)

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertGreater(results[0].tag_area_to_median_ratio, 1.35)
        self.assertEqual(results[0].tag_area_to_shelf_median_ratio, 1.0)
        self.assertTrue(all(result.is_promo is False for result in results.values()))

    def test_shelf_size_outlier_and_vertical_tag_are_promotional(self):
        """Verify substantial shelf-size and vertical outliers are promotional.

        Input:
            None. Geometry outlier fixtures are created inside the test.
        Output:
            None. Assertions validate geometry-based promotion reasons.
        """
        image = np.full((260, 650, 3), 255, dtype=np.uint8)
        boxes = [
            (10, 100, 60, 120),
            (80, 100, 130, 120),
            (150, 100, 200, 120),
            (230, 80, 430, 140),
            (470, 85, 490, 135),
        ]
        tags = [SaleTag(index, bbox) for index, bbox in enumerate(boxes)]

        results = SaleClassifier().classify(image, tags)

        self.assertEqual(results[3].sale_reason, "shelf_size_outlier")
        self.assertGreater(results[3].tag_area_to_shelf_median_ratio, 1.35)
        self.assertTrue(results[3].is_promo)
        self.assertEqual(results[4].sale_reason, "vertical_shape")
        self.assertTrue(results[4].is_promo)
        self.assertEqual(results[3].sale_shelf_index, results[0].sale_shelf_index)

    def test_offer_keyword_is_independent_promotional_evidence(self):
        """Verify an offer keyword independently marks a tag promotional.

        Input:
            None. OCR text fixtures include an explicit offer keyword.
        Output:
            None. Assertions validate keyword evidence and final decision.
        """
        image = np.full((180, 500, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 80, 80, 110),
            (100, 80, 160, 110),
            (180, 80, 240, 110),
            (260, 80, 320, 110),
            (340, 80, 400, 110),
        ]
        for bbox in boxes:
            paint(image, bbox, YELLOW)
        tags = [
            SaleTag(index, bbox, "OFERTA | 4,19" if index == 0 else "6,29")
            for index, bbox in enumerate(boxes)
        ]

        results = SaleClassifier().classify(image, tags)

        self.assertTrue(results[0].is_promo)
        self.assertEqual(results[0].sale_reason, "offer_keyword")
        self.assertEqual(results[0].sale_keyword_match, "OFERTA")
        self.assertTrue(all(results[index].is_promo is False for index in range(1, 5)))
        self.assertEqual(results[1].sale_baseline_color, "yellow")

    def test_long_offer_keyword_tolerates_small_ocr_errors(self):
        """Verify approved long keywords tolerate small OCR errors.

        Input:
            None. OCR text contains a narrowly misspelled offer word.
        Output:
            None. Assertions validate fuzzy keyword matching.
        """
        image = np.full((180, 500, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 80, 80, 110),
            (100, 80, 160, 110),
            (180, 80, 240, 110),
        ]
        tags = [
            SaleTag(
                index,
                bbox,
                "NO APP | APROYLITE | 4,40" if index == 0 else "4,89",
            )
            for index, bbox in enumerate(boxes)
        ]

        results = SaleClassifier().classify(image, tags)

        self.assertTrue(results[0].is_promo)
        self.assertEqual(results[0].sale_reason, "offer_keyword")
        self.assertEqual(results[0].sale_keyword_match, "APROVEITE")
        self.assertTrue(all(not results[index].is_promo for index in (1, 2)))

    def test_product_word_similar_to_desconto_is_not_fuzzy_matched(self):
        """Verify ordinary product text is not broadly fuzzy-matched.

        Input:
            None. OCR text resembles but does not equal an offer keyword.
        Output:
            None. Assertions validate the absence of keyword evidence.
        """
        image = np.full((180, 500, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 80, 80, 110),
            (100, 80, 160, 110),
            (180, 80, 240, 110),
        ]
        tags = [
            SaleTag(
                index,
                bbox,
                "BISCONTO MARIA LIANE | 4,99" if index == 0 else "4,99",
            )
            for index, bbox in enumerate(boxes)
        ]

        results = SaleClassifier().classify(image, tags)

        self.assertTrue(all(not result.is_promo for result in results.values()))
        self.assertTrue(
            all(result.sale_keyword_match is None for result in results.values())
        )

    def test_red_outliers_remain_promotional_in_a_warm_single_palette(self):
        """Verify red remains reliable contrast against a warm palette.

        Input:
            None. Warm regular tags and a red outlier are created internally.
        Output:
            None. Assertions validate red promotion evidence.
        """
        image = np.full((220, 700, 3), 255, dtype=np.uint8)
        boxes = [
            (20 + index * 100, 80, 100 + index * 100, 120)
            for index in range(6)
        ]
        warm_yellow = (20, 190, 210)
        red = (20, 20, 210)
        for bbox in boxes[:4]:
            paint(image, bbox, warm_yellow)
        for bbox in boxes[4:]:
            paint(image, bbox, red)

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertTrue(all(not results[index].is_promo for index in range(4)))
        self.assertTrue(all(results[index].is_promo for index in (4, 5)))
        self.assertTrue(
            all(
                results[index].sale_reason == "red_vs_image_yellow"
                for index in (4, 5)
            )
        )

    def test_balanced_distinct_white_and_yellow_palette_is_not_ambiguous(self):
        """Verify distinct white and yellow clusters produce a stable baseline.

        Input:
            None. Balanced color-cluster fixtures are created internally.
        Output:
            None. Assertions validate unambiguous color classification.
        """
        image = np.full((180, 500, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 80, 80, 110),
            (100, 80, 160, 110),
            (180, 80, 240, 110),
            (260, 80, 320, 110),
        ]
        paint(image, boxes[2], YELLOW)
        paint(image, boxes[3], YELLOW)

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertTrue(all(not results[index].is_promo for index in (0, 1)))
        self.assertTrue(all(results[index].is_promo for index in (2, 3)))
        self.assertEqual(results[0].sale_baseline_color, "white")

    def test_warm_single_palette_is_not_split_by_shadows(self):
        """Verify shadows do not split one warm paper palette.

        Input:
            None. Warm tags with lighting variation are created internally.
        Output:
            None. Assertions validate a single regular palette.
        """
        image = np.full((220, 700, 3), 255, dtype=np.uint8)
        boxes = [
            (20 + index * 100, 80, 100 + index * 100, 120)
            for index in range(6)
        ]
        pale_green = (125, 205, 190)
        dark_warm = (45, 105, 100)
        for bbox in boxes[:4]:
            paint(image, bbox, pale_green)
        for bbox in boxes[4:]:
            paint(image, bbox, dark_warm)

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertTrue(all(not result.is_promo for result in results.values()))
        self.assertTrue(
            all(
                result.sale_baseline_source == "image_single_palette"
                for result in results.values()
            )
        )

    def test_small_vertical_detection_fragment_is_not_promotional(self):
        """Verify a small vertical detector fragment is not promoted.

        Input:
            None. A small narrow tag fragment is created inside the test.
        Output:
            None. Assertions validate the minimum area safeguard.
        """
        image = np.full((220, 700, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 100, 70, 120),
            (100, 100, 150, 120),
            (180, 100, 230, 120),
            (270, 95, 280, 125),
        ]

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertLess(results[3].tag_area_to_shelf_median_ratio, 0.65)
        self.assertIsNone(results[3].sale_geometry_reason)
        self.assertFalse(results[3].is_promo)

    def test_substantial_but_normal_area_difference_is_not_a_size_promotion(self):
        """Verify normal within-shelf area variation remains regular.

        Input:
            None. Moderately different tag sizes are created internally.
        Output:
            None. Assertions validate the size-outlier threshold.
        """
        image = np.full((220, 700, 3), 255, dtype=np.uint8)
        boxes = [
            (20, 100, 120, 140),
            (150, 100, 250, 140),
            (280, 100, 380, 140),
            (410, 100, 580, 140),
        ]

        results = SaleClassifier().classify(
            image,
            [SaleTag(index, bbox) for index, bbox in enumerate(boxes)],
        )

        self.assertAlmostEqual(
            results[3].tag_area_to_shelf_median_ratio,
            1.70,
        )
        self.assertIsNone(results[3].sale_geometry_reason)
        self.assertFalse(results[3].is_promo)

    def test_cli_exposes_sale_classification_configuration(self):
        """Verify CLI options expose promotion-classification thresholds.

        Input:
            None. The process argument list is patched by the test.
        Output:
            None. Assertions validate configuration mapping.
        """
        with patch.object(
            sys,
            "argv",
            [
                "shelf-pipeline",
                "--no-sale-classification",
                "--sale-vertical-aspect-ratio",
                "1.4",
                "--sale-large-area-ratio",
                "1.6",
                "--sale-red-pixel-ratio",
                "0.2",
                "--sale-yellow-pixel-ratio",
                "0.3",
                "--sale-shelf-min-y-threshold-px",
                "42",
                "--sale-shelf-image-y-ratio",
                "0.04",
                "--sale-baseline-min-confidence",
                "0.7",
                "--sale-baseline-min-tags",
                "4",
                "--sale-regular-tag-color",
                "yellow",
            ],
        ):
            config = build_config(parse_args())

        self.assertFalse(config.run_sale_classification)
        self.assertEqual(config.sale_vertical_aspect_ratio, 1.4)
        self.assertEqual(config.sale_large_area_ratio, 1.6)
        self.assertEqual(config.sale_red_pixel_ratio, 0.2)
        self.assertEqual(config.sale_yellow_pixel_ratio, 0.3)
        self.assertEqual(config.sale_shelf_min_y_threshold_px, 42)
        self.assertEqual(config.sale_shelf_image_y_ratio, 0.04)
        self.assertEqual(config.sale_baseline_min_confidence, 0.7)
        self.assertEqual(config.sale_baseline_min_tags, 4)
        self.assertEqual(config.sale_regular_tag_color, "yellow")

if __name__ == "__main__":
    unittest.main()
