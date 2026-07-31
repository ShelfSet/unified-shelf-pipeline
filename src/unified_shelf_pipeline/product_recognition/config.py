"""Default artifact paths and settings for product recognition."""

from ..config import project_root


DEFAULT_THRESHOLD_PATH = str(
    project_root()
    / "artifacts"
    / "product_thresholds"
    / "thresholds.csv"
)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_UNKNOWN_LABEL = "__Unknown__"
DEFAULT_TOP_K = 5
