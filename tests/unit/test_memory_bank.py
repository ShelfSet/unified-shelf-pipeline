"""Unit tests for inference-only memory-bank artifact loading."""

import pickle
import tempfile
import unittest
from pathlib import Path

from unified_shelf_pipeline.product_recognition.memory_bank import (
    load_memory_bank,
)


class MemoryBankTests(unittest.TestCase):
    """Verify inference-only memory-bank artifact loading."""

    def test_loads_one_memory_bank_artifact_path(self):
        """Verify one valid artifact is normalized for runtime inference.

        Input:
            None. The test writes a temporary trusted pickle.
        Output:
            None. Assertions validate required normalized payload fields.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "products.pkl"
            with artifact_path.open("wb") as file:
                pickle.dump(
                    {
                        "memory_items": [
                            {
                                "label": "product-a",
                                "image_path": "product-a.jpg",
                            }
                        ],
                        "memory_embeddings": None,
                    },
                    file,
                )

            payload = load_memory_bank(artifact_path)

        self.assertEqual(payload["memory_items"][0]["label"], "product-a")
        self.assertEqual(payload["memory_bank_path"], artifact_path.resolve())
        self.assertEqual(payload["metadata"], {})

    def test_rejects_artifact_without_memory_items(self):
        """Verify artifacts missing memory items are rejected.

        Input:
            None. The test writes an invalid temporary pickle.
        Output:
            None. Assertions validate the raised schema error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "invalid.pkl"
            with artifact_path.open("wb") as file:
                pickle.dump({"memory_embeddings": None}, file)

            with self.assertRaisesRegex(ValueError, "missing memory_items"):
                load_memory_bank(artifact_path)


if __name__ == "__main__":
    unittest.main()
