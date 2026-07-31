"""Unit tests for OCR normalization and price extraction."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

from unified_shelf_pipeline.cli import build_config, parse_args
from unified_shelf_pipeline.pipeline import (
    OcrItem,
    PaddlePriceExtractor,
    UnifiedPipelineConfig,
    price_candidates_from_ocr,
    select_price_candidates,
)


def legacy_ocr_row(text, confidence, bbox):
    """Build one legacy PaddleOCR response row.

    Input:
        text: Recognized text fixture.
        confidence: OCR confidence fixture.
        bbox: Integer text bounding-box coordinates.
    Output:
        Legacy polygon, text, and confidence response structure.
    """
    x1, y1, x2, y2 = bbox
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return [points, [text, confidence]]


class FakeOcr:
    """Return one deterministic OCR response."""

    def __init__(self, rows):
        """Store legacy OCR rows returned by predictions.

        Input:
            rows: OCR row fixtures.
        Output:
            None.
        """
        self.rows = rows

    def predict(self, _image):
        """Return the configured OCR rows.

        Input:
            _image: Ignored image fixture.
        Output:
            Nested response matching PaddleOCR prediction output.
        """
        return [self.rows]


class SequencedFakeOcr:
    """Return deterministic OCR responses in call order."""

    def __init__(self, responses):
        """Store the ordered OCR response fixtures.

        Input:
            responses: OCR rows returned by successive predictions.
        Output:
            None.
        """
        self.responses = responses
        self.call_count = 0

    def predict(self, _image):
        """Return the next configured OCR response.

        Input:
            _image: Ignored image fixture.
        Output:
            Nested response matching PaddleOCR prediction output.
        """
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return [response]


class PriceOcrTests(unittest.TestCase):
    """Verify price-text extraction, parsing, ordering, and configuration."""

    def test_defaults_use_ppocr_v6_for_portuguese_and_three_prices(self):
        """Verify default OCR settings match the target price tags.

        Input:
            None. Defaults are read from the pipeline configuration.
        Output:
            None. Assertions validate version, language, and price count.
        """
        config = UnifiedPipelineConfig()

        self.assertEqual(config.ocr_version, "PP-OCRv6")
        self.assertEqual(config.ocr_language, "pt")
        self.assertEqual(config.ocr_max_prices, 3)
        self.assertEqual(config.ocr_upscale, 3.0)

    def test_cli_can_override_ocr_settings(self):
        """Verify CLI options override every exposed OCR setting.

        Input:
            None. The process argument list is patched by the test.
        Output:
            None. Assertions validate the mapped configuration.
        """
        with patch.object(
            sys,
            "argv",
            [
                "shelf-pipeline",
                "--ocr-version",
                "PP-OCRv5",
                "--ocr-language",
                "en",
                "--ocr-min-confidence",
                "0.6",
                "--ocr-max-prices",
                "2",
                "--ocr-upscale",
                "2",
            ],
        ):
            config = build_config(parse_args())

        self.assertEqual(config.ocr_version, "PP-OCRv5")
        self.assertEqual(config.ocr_language, "en")
        self.assertEqual(config.ocr_min_confidence, 0.6)
        self.assertEqual(config.ocr_max_prices, 2)
        self.assertEqual(config.ocr_upscale, 2.0)

    def test_extract_preserves_three_spatially_ordered_prices_and_duplicates(self):
        """Verify extraction preserves three visual prices including duplicates.

        Input:
            None. A deterministic OCR engine supplies positioned prices.
        Output:
            None. Assertions validate values and visual ordering.
        """
        extractor = PaddlePriceExtractor(min_confidence=0.4, max_prices=3)
        extractor._ocr = FakeOcr(
            [
                legacy_ocr_row("6,99", 0.95, (200, 10, 250, 40)),
                legacy_ocr_row("6,59", 0.92, (20, 10, 70, 40)),
                legacy_ocr_row("6,59", 0.90, (110, 10, 160, 40)),
            ]
        )

        result = extractor.extract(None)

        self.assertEqual(result["primary_price"], "6,59")
        self.assertEqual(result["secondary_price"], "6,59")
        self.assertEqual(result["tertiary_price"], "6,99")
        self.assertIsNone(result["ocr_error"])

    def test_compact_currency_and_repeated_punctuation_are_constrained(self):
        """Verify conservative recovery of compact and noisy price text.

        Input:
            None. OCR item fixtures contain compact and repeated separators.
        Output:
            None. Assertions validate accepted price interpretations.
        """
        items = [
            OcrItem("R1020", 0.9, (0, 0, 40, 20)),
            OcrItem("RS875", 0.9, (50, 0, 90, 20)),
            OcrItem("8.,99", 0.9, (100, 0, 140, 20)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["10,20", "8,75", "8,99"])

    def test_low_confidence_and_out_of_range_values_are_rejected(self):
        """Verify confidence and value limits reject implausible prices.

        Input:
            None. OCR item fixtures cover invalid confidence and value ranges.
        Output:
            None. Assertions validate the remaining candidates.
        """
        items = [
            OcrItem("4,10", 0.39, (0, 0, 40, 20)),
            OcrItem("200,00", 0.99, (50, 0, 100, 20)),
            OcrItem("4,59", 0.95, (110, 0, 150, 20)),
        ]

        selected = select_price_candidates(
            price_candidates_from_ocr(items, min_confidence=0.4),
            max_prices=3,
        )

        self.assertEqual([candidate.price for candidate in selected], ["4,59"])

    def test_adjacent_decimal_and_cents_fragments_are_joined(self):
        """Verify adjacent whole and cents OCR fragments form one price.

        Input:
            None. Positioned OCR fragments are created inside the test.
        Output:
            None. Assertions validate reconstructed price text.
        """
        items = [
            OcrItem("1,69", 0.96, (10, 10, 50, 30)),
            OcrItem("1.", 0.77, (70, 10, 84, 30)),
            OcrItem("69", 0.99, (82, 10, 105, 30)),
            OcrItem("1,79", 0.98, (125, 10, 165, 30)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["1,69", "1,69", "1,79"])

    def test_large_whole_and_superscript_cents_are_joined_without_separator(self):
        """Verify superscript cents can recover a missing separator.

        Input:
            None. Differently sized OCR boxes are created internally.
        Output:
            None. Assertions validate reconstructed price text.
        """
        items = [
            OcrItem("600", 0.98, (5, 5, 42, 17)),
            OcrItem("g", 0.97, (44, 5, 51, 17)),
            OcrItem("5", 0.93, (20, 35, 80, 150)),
            OcrItem("95", 0.91, (85, 45, 120, 88)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["5,95"])

    def test_separate_low_comma_joins_whole_and_superscript_cents(self):
        """Verify a separate low comma joins whole and cents fragments.

        Input:
            None. Whole, separator, and cents OCR fixtures are created.
        Output:
            None. Assertions validate reconstructed price text.
        """
        items = [
            OcrItem("5", 0.93, (20, 35, 80, 150)),
            OcrItem(",", 0.30, (78, 105, 86, 132)),
            OcrItem("95", 0.91, (85, 45, 120, 88)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["5,95"])
        self.assertEqual(selected[0].format_quality, 3)

    def test_overlapping_large_whole_and_cents_boxes_are_joined(self):
        """Verify overlapping whole and cents boxes can form one price.

        Input:
            None. Overlapping OCR fragment fixtures are created internally.
        Output:
            None. Assertions validate reconstructed price text.
        """
        items = [
            OcrItem("5", 0.99, (39, 115, 414, 618)),
            OcrItem("95", 0.99, (287, 121, 576, 359)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["5,95"])

    def test_zero_result_retries_a_lower_price_focused_crop(self):
        """Verify extraction retries a focused crop after an empty result.

        Input:
            None. Sequenced OCR responses simulate initial failure and recovery.
        Output:
            None. Assertions validate retry count and recovered price.
        """
        extractor = PaddlePriceExtractor(min_confidence=0.4, max_prices=3, upscale=1.0)
        extractor._ocr = SequencedFakeOcr(
            [
                [legacy_ocr_row("BISCOITO", 0.99, (5, 5, 80, 20))],
                [
                    legacy_ocr_row("5", 0.99, (20, 20, 70, 90)),
                    legacy_ocr_row("95", 0.99, (60, 25, 95, 50)),
                ],
            ]
        )

        result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(result["primary_price"], "5,95")
        self.assertEqual(extractor._ocr.call_count, 2)

    def test_split_weight_unit_is_not_converted_to_a_price(self):
        """Verify numeric weight plus unit text is not treated as a price.

        Input:
            None. OCR fixtures contain a split measurement and unit.
        Output:
            None. Assertions validate that no price is selected.
        """
        items = [
            OcrItem("600", 0.98, (5, 5, 42, 17)),
            OcrItem("g", 0.97, (44, 5, 51, 17)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual(selected, [])

    def test_vertically_stacked_prices_are_ordered_top_to_bottom(self):
        """Verify vertically stacked prices use top-to-bottom order.

        Input:
            None. Positioned OCR price fixtures are created internally.
        Output:
            None. Assertions validate visual ordering.
        """
        items = [
            OcrItem("2,99", 0.96, (10, 80, 60, 110)),
            OcrItem("3,19", 0.95, (20, 10, 70, 40)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["3,19", "2,99"])

    def test_mixed_rows_use_row_major_price_order(self):
        """Verify mixed price rows use row-major visual order.

        Input:
            None. Multi-row OCR fixtures are created inside the test.
        Output:
            None. Assertions validate top-to-bottom and left-to-right ordering.
        """
        items = [
            OcrItem("5,99", 0.96, (100, 10, 150, 40)),
            OcrItem("4,99", 0.95, (20, 10, 70, 40)),
            OcrItem("3,99", 0.94, (10, 70, 60, 100)),
        ]

        selected = select_price_candidates(price_candidates_from_ocr(items), max_prices=3)

        self.assertEqual([candidate.price for candidate in selected], ["4,99", "5,99", "3,99"])

    def test_invalid_extractor_constraints_fail_fast(self):
        """Verify invalid extractor limits raise configuration errors.

        Input:
            None. Invalid constructor values are supplied directly.
        Output:
            None. Assertions validate immediate ``ValueError`` failures.
        """
        with self.assertRaises(ValueError):
            PaddlePriceExtractor(max_prices=4)
        with self.assertRaises(ValueError):
            PaddlePriceExtractor(min_confidence=1.1)
        with self.assertRaises(ValueError):
            PaddlePriceExtractor(upscale=5.0)


if __name__ == "__main__":
    unittest.main()
