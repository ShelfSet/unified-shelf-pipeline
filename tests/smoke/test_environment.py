"""Smoke tests for packaged defaults and bundled artifacts."""

import unittest

from unified_shelf_pipeline.config import UnifiedPipelineConfig
from unified_shelf_pipeline.diagnostics import environment_report
from unified_shelf_pipeline.product_recognition.inference import (
    load_thresholds,
)
from unified_shelf_pipeline.product_recognition.memory_bank import (
    load_memory_bank,
)


class EnvironmentSmokeTests(unittest.TestCase):
    """Verify packaged defaults and bundled inference artifacts."""

    def test_default_artifacts_are_available(self):
        """Verify all four configured runtime artifacts exist.

        Input:
            None. The default project configuration supplies artifact paths.
        Output:
            None. Assertions validate the environment report.
        """
        report = environment_report(UnifiedPipelineConfig())
        artifacts = report.loc[report["kind"] == "artifact"]

        self.assertEqual(len(artifacts), 4)
        self.assertTrue(artifacts["available"].all())

    def test_recognition_artifacts_are_inference_ready(self):
        """Verify bundled recognition artifacts contain usable runtime data.

        Input:
            None. Default memory-bank and threshold artifacts are loaded.
        Output:
            None. Assertions validate embeddings and threshold values.
        """
        config = UnifiedPipelineConfig()
        memory_bank = load_memory_bank(config.product_memory_bank_path)
        global_threshold, class_thresholds = load_thresholds(
            config.product_threshold_path
        )

        self.assertIsNotNone(memory_bank["memory_embeddings"])
        self.assertEqual(
            len(memory_bank["memory_embeddings"]),
            len(memory_bank["memory_items"]),
        )
        self.assertGreater(global_threshold, 0.0)
        self.assertTrue(class_thresholds)


if __name__ == "__main__":
    unittest.main()
