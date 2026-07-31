"""Run inference-only nearest-neighbor product recognition.

This module is the main integration surface for backend inference. Runtime code
loads `load_recognizer(...)` once at service startup, keeps the returned runtime
dictionary in memory, and passes it to `predict_image(...)` for incoming
requests.

Product images are embedded into the same vector space as the memory bank, FAISS
retrieves the nearest memory images, and neighbor scores are averaged by product
label. The final accept/reject decision is score-threshold based. If the best
class score is below its threshold, the result becomes the unknown label instead
of a product label.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import faiss
import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K, DEFAULT_UNKNOWN_LABEL
from .embeddings import extract_embedding_for_image, load_embedding_model
from .memory_bank import load_memory_bank


def build_faiss_index(memory_embeddings: np.ndarray):
    """Build a cosine-similarity FAISS index from normalized memory embeddings.

    Input:
        memory_embeddings: Float32 matrix with shape `[num_items, embedding_dim]`.
    Output:
        FAISS inner-product index ready for nearest-neighbor search.
    """
    if memory_embeddings.ndim != 2:
        raise ValueError("memory_embeddings must have shape [N, D].")

    index = faiss.IndexFlatIP(memory_embeddings.shape[1])
    index.add(memory_embeddings.astype("float32"))
    return index


def aggregate_topk_predictions(
    query_embedding: np.ndarray,
    index,
    memory_labels: Sequence[str],
    memory_paths: Sequence[str],
    top_k: int,
) -> dict:
    """Retrieve top-k neighbors and aggregate their scores by product label.

    Input:
        query_embedding: Single normalized embedding vector.
        index: FAISS index built from memory embeddings.
        memory_labels: Labels aligned with index rows.
        memory_paths: Image paths aligned with index rows.
        top_k: Maximum neighbors to retrieve.
    Output:
        Prediction details including best label, score, margin, and neighbors.
    """
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must have shape [D].")

    # Retrieve the strongest available memory-bank neighbors.
    top_k = min(int(top_k), int(index.ntotal))
    raw_vals, idxs = index.search(query_embedding.astype("float32")[None, :], top_k)
    raw_vals = raw_vals[0]
    idxs = idxs[0]

    # Preserve individual neighbors while collecting scores by product label.
    neighbors = []
    class_scores = defaultdict(list)
    for score, idx in zip(raw_vals, idxs):
        if idx < 0:
            continue
        label = memory_labels[idx]
        path = memory_paths[idx]
        score = float(score)
        neighbors.append({"label": label, "score": score, "path": path})
        class_scores[label].append(score)

    if not class_scores:
        raise ValueError("No neighbors were retrieved from the FAISS index.")

    # Rank labels by their mean neighbor score and calculate runner-up margin.
    ranked_classes = sorted(
        ((label, float(np.mean(scores))) for label, scores in class_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    best_label, best_score = ranked_classes[0]
    second_label = ranked_classes[1][0] if len(ranked_classes) > 1 else None
    second_score = ranked_classes[1][1] if len(ranked_classes) > 1 else None
    margin = None if second_score is None else float(best_score - second_score)

    return {
        "best_label": best_label,
        "best_score": best_score,
        "second_label": second_label,
        "second_score": second_score,
        "margin": margin,
        "ranked_classes": ranked_classes,
        "neighbors": neighbors,
    }


def load_thresholds(
    threshold_path: Optional[Path | str],
    default_score_threshold: float = 0.72,
) -> tuple[float, dict[str, float]]:
    """Load current-format global and per-class thresholds.

    Input:
        threshold_path: Path to thresholds.csv, or None.
        default_score_threshold: Fallback threshold when no CSV is supplied.
    Output:
        Global score threshold and label-to-threshold mapping.
    """
    if threshold_path is None:
        return default_score_threshold, {}

    threshold_table_path = Path(threshold_path).expanduser().resolve()
    if not threshold_table_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_table_path}")

    threshold_df = pd.read_csv(threshold_table_path)
    required_cols = {"label", "global_threshold", "selected_threshold"}
    missing_cols = required_cols.difference(threshold_df.columns)
    if missing_cols:
        raise ValueError(
            f"Threshold file {threshold_table_path} is missing columns: {sorted(missing_cols)}"
        )
    if threshold_df.empty:
        raise ValueError(f"Threshold file {threshold_table_path} is empty.")

    # In the current MVP format every class row repeats the same global
    # threshold and carries its own deployable selected threshold.
    score_threshold = float(threshold_df.iloc[0]["global_threshold"])
    class_thresholds = dict(zip(threshold_df["label"], threshold_df["selected_threshold"]))

    return score_threshold, class_thresholds


def load_recognizer(
    memory_bank_path: Path | str,
    threshold_path: Optional[Path | str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    top_k: int = DEFAULT_TOP_K,
    unknown_label: str = DEFAULT_UNKNOWN_LABEL,
) -> dict:
    """Load all runtime state needed for product recognition.

    Input:
        memory_bank_path: Path to the inference-ready memory-bank pickle.
        threshold_path: Optional path to thresholds.csv.
        model_name: Embedding model id or local model path.
        device: `auto`, `cpu`, or `cuda`.
        top_k: Number of memory neighbors used per prediction.
        unknown_label: Output label when a prediction is rejected.
    Output:
        Plain dictionary containing model, FAISS index, labels, paths, and thresholds.
    """
    # Load the trusted artifact and require precomputed inference embeddings.
    payload = load_memory_bank(memory_bank_path)
    memory_embeddings = payload.get("memory_embeddings")
    if memory_embeddings is None:
        raise ValueError(
            "Memory-bank artifact does not contain memory_embeddings. "
            "Provide an inference-ready artifact from the dedicated "
            "memory-bank project."
        )

    # Prefer explicit aligned arrays, falling back to the canonical item records.
    memory_items = payload["memory_items"]
    memory_labels = payload.get("memory_labels") or [item["label"] for item in memory_items]
    memory_paths = payload.get("memory_paths") or [item["image_path"] for item in memory_items]

    if len(memory_embeddings) != len(memory_labels) or len(memory_embeddings) != len(memory_paths):
        raise ValueError("Memory embeddings, labels, and paths are not aligned.")

    # Load runtime dependencies once and assemble the reusable recognizer state.
    score_threshold, class_thresholds = load_thresholds(threshold_path)
    model = load_embedding_model(model_name, device)

    return {
        "memory_embeddings": memory_embeddings,
        "memory_labels": list(memory_labels),
        "memory_paths": list(memory_paths),
        "model": model,
        "top_k": top_k,
        "unknown_label": unknown_label,
        "global_score_threshold": score_threshold,
        "class_score_thresholds": class_thresholds,
        "index": build_faiss_index(memory_embeddings),
    }


def predict_embedding(recognizer: dict, query_embedding: np.ndarray) -> dict:
    """Classify one embedding and apply the unknown threshold policy.

    Input:
        recognizer: Runtime dictionary returned by `load_recognizer`.
        query_embedding: Single normalized embedding vector.
    Output:
        Prediction dictionary with `final_pred`, scores, thresholds, and neighbors.
    """
    out = aggregate_topk_predictions(
        query_embedding=query_embedding,
        index=recognizer["index"],
        memory_labels=recognizer["memory_labels"],
        memory_paths=recognizer["memory_paths"],
        top_k=recognizer["top_k"],
    )
    effective_score_threshold = recognizer["class_score_thresholds"].get(
        out["best_label"],
        recognizer["global_score_threshold"],
    )

    # Rejection is explicit and inspectable: the best neighbor class is still
    # returned in `best_label`, while `final_pred` becomes the unknown label.
    reject_reasons = []
    if out["best_score"] < effective_score_threshold:
        reject_reasons.append("low_score")

    is_rejected = bool(reject_reasons)
    return {
        "final_pred": recognizer["unknown_label"] if is_rejected else out["best_label"],
        "is_rejected": is_rejected,
        "reject_reasons": reject_reasons,
        "effective_score_threshold": effective_score_threshold,
        "threshold_offset": effective_score_threshold - recognizer["global_score_threshold"],
        **out,
    }


def predict_image(recognizer: dict, image_path: str) -> dict:
    """Classify one image file.

    Input:
        recognizer: Runtime dictionary returned by `load_recognizer`.
        image_path: Path to the image to classify.
    Output:
        Prediction dictionary including the original image path.
    """
    embedding = extract_embedding_for_image(
        image_path=image_path,
        model=recognizer["model"],
    )
    result = predict_embedding(recognizer, embedding)
    result["image_path"] = image_path
    return result


def predict_images(recognizer: dict, image_paths: Iterable[str]) -> pd.DataFrame:
    """Classify multiple image files and return a result table.

    Input:
        recognizer: Runtime dictionary returned by `load_recognizer`.
        image_paths: Iterable of image file paths.
    Output:
        DataFrame with one prediction row per image.
    """
    rows = [predict_image(recognizer, path) for path in image_paths]
    return pd.DataFrame(rows)


def list_image_paths(image_dir: str) -> list[str]:
    """List supported image files recursively under a directory.

    Input:
        image_dir: Root folder to scan.
    Output:
        Sorted list of supported image file paths.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(image_dir)
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_exts
    )
