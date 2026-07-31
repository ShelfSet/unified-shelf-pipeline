"""Unit tests for CLI configuration mapping."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unified_shelf_pipeline.cli import build_config, parse_args
from unified_shelf_pipeline.product_recognition.inference import (
    load_thresholds,
)


class CliTests(unittest.TestCase):
    """Verify command-line options map to direct runtime artifact paths."""

    def test_artifact_options_accept_one_file_path_each(self):
        """Verify each artifact option accepts one direct file path.

        Input:
            None. The test patches the process argument list.
        Output:
            None. Assertions validate the resulting configuration paths.
        """
        with patch.object(
            sys,
            "argv",
            [
                "shelf-pipeline",
                "--shelf-detector",
                "models/shelf.pt",
                "--price-tag-detector",
                "models/price-tags.pt",
                "--product-memory-bank",
                "models/products.pkl",
                "--product-threshold",
                "models/thresholds.csv",
            ],
        ):
            config = build_config(parse_args())

        self.assertEqual(config.shelf_detector_path, Path("models/shelf.pt"))
        self.assertEqual(config.price_tag_detector_path, Path("models/price-tags.pt"))
        self.assertEqual(config.product_memory_bank_path, Path("models/products.pkl"))
        self.assertEqual(config.product_threshold_path, Path("models/thresholds.csv"))

    def test_product_threshold_accepts_a_direct_csv_path(self):
        """Verify threshold loading accepts a direct CSV artifact path.

        Input:
            None. The test creates a temporary threshold file.
        Output:
            None. Assertions validate global and class threshold values.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            threshold_path = Path(temp_dir) / "thresholds.csv"
            threshold_path.write_text(
                "label,global_threshold,selected_threshold\n"
                "product-a,0.71,0.66\n",
                encoding="utf-8",
            )

            global_threshold, class_thresholds = load_thresholds(str(threshold_path))

        self.assertEqual(global_threshold, 0.71)
        self.assertEqual(class_thresholds, {"product-a": 0.66})


if __name__ == "__main__":
    unittest.main()
