"""Load trusted product-memory-bank artifacts for inference."""

from __future__ import annotations

import pickle
from pathlib import Path


def load_memory_bank(memory_bank_path: Path | str) -> dict:
    """Load and normalize one trusted memory-bank artifact.

    Input:
        memory_bank_path: Direct path to an inference-ready pickle artifact.
    Output:
        Memory-bank payload with required runtime keys and resolved source path.
    """
    artifact_path = Path(memory_bank_path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Product memory-bank artifact not found: {artifact_path}"
        )

    with artifact_path.open("rb") as file:
        payload = pickle.load(file)

    if "memory_items" not in payload:
        raise ValueError(
            f"Invalid memory-bank artifact at {artifact_path}: "
            "missing memory_items."
        )

    for key in [
        "metadata",
        "memory_embeddings",
        "memory_labels",
        "memory_paths",
    ]:
        payload.setdefault(key, None if key != "metadata" else {})
    payload["memory_bank_path"] = artifact_path
    return payload
