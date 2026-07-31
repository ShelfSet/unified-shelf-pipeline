"""Unit tests for runtime artifact-path discovery."""

import tempfile
import unittest
from pathlib import Path

from unified_shelf_pipeline.config import discover_product_memory_bank_path


class ProductMemoryBankDiscoveryTests(unittest.TestCase):
    """Verify deterministic discovery of the default memory-bank artifact."""

    def test_discovers_the_only_pickle_regardless_of_filename(self):
        """Verify one pickle is selected while unrelated files are ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            expected = directory / "product_memory_bank_latest_version.pkl"
            expected.touch()
            (directory / "README.txt").touch()

            actual = discover_product_memory_bank_path(directory)

        self.assertEqual(actual, expected.resolve())

    def test_rejects_a_folder_without_a_pickle(self):
        """Verify a missing default artifact produces an actionable error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                FileNotFoundError,
                "No product memory-bank .pkl file found",
            ):
                discover_product_memory_bank_path(Path(temp_dir))

    def test_rejects_multiple_pickles(self):
        """Verify ambiguous defaults are rejected instead of chosen silently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "bank_a.pkl").touch()
            (directory / "bank_b.pkl").touch()

            with self.assertRaisesRegex(
                RuntimeError,
                "Expected exactly one product memory-bank .pkl file",
            ):
                discover_product_memory_bank_path(directory)


if __name__ == "__main__":
    unittest.main()
