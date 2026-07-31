"""Unit tests for global product-threshold policy."""

import unittest
from unittest.mock import patch

from unified_shelf_pipeline.pipeline import ShelfPipeline, UnifiedPipelineConfig


class GlobalProductThresholdTests(unittest.TestCase):
    """Verify the optional global-only product-threshold policy."""

    def test_global_policy_removes_class_threshold_overrides(self):
        """Verify global policy clears class-specific threshold overrides.

        Input:
            None. Recognizer loading is replaced with a controlled fixture.
        Output:
            None. Assertions validate loader arguments and threshold mutation.
        """
        recognizer = {
            "global_score_threshold": 0.71,
            "class_score_thresholds": {"product-a": 0.63},
        }
        config = UnifiedPipelineConfig(use_global_product_threshold=True)

        with patch(
            (
                "unified_shelf_pipeline.product_recognition."
                "inference.load_recognizer"
            ),
            return_value=recognizer,
        ) as load_recognizer:
            loaded = ShelfPipeline(config)._load_product_recognizer()

        load_recognizer.assert_called_once_with(
            memory_bank_path=config.product_memory_bank_path,
            threshold_path=config.product_threshold_path,
            model_name=config.product_model_name,
            device=config.device,
            top_k=config.product_top_k,
        )
        self.assertEqual(loaded["global_score_threshold"], 0.71)
        self.assertEqual(loaded["class_score_thresholds"], {})


if __name__ == "__main__":
    unittest.main()
