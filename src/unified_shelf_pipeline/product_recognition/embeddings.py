"""Load embedding models and extract image embeddings for inference.

Every returned embedding is converted to float32 and L2-normalized. That means a
FAISS inner-product index behaves like cosine similarity, which keeps retrieval
fast while avoiding separate cosine-distance logic.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def resolve_device(device: str = "auto") -> str:
    """Resolve the requested runtime device to cpu or cuda.

    Input:
        device: `auto`, `cpu`, or `cuda`.
    Output:
        Concrete device string used by torch and sentence-transformers.
    """
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_status_lines(requested_device: str = "auto") -> List[str]:
    """Build human-readable CUDA status lines for CLI diagnostics.

    Input:
        requested_device: Device argument requested by the user.
    Output:
        Lines describing torch CUDA visibility and selected runtime device.
    """
    resolved_device = resolve_device(requested_device)
    lines = [
        f"Requested device: {requested_device}",
        f"Resolved device: {resolved_device}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        current_idx = torch.cuda.current_device()
        lines.extend(
            [
                f"CUDA device count: {torch.cuda.device_count()}",
                f"CUDA current device: {current_idx}",
                f"CUDA device name: {torch.cuda.get_device_name(current_idx)}",
                f"CUDA runtime version: {torch.version.cuda}",
            ]
        )
    return lines


def load_embedding_model(model_name: str, device: str = "auto") -> SentenceTransformer:
    """Load the image embedding model on the selected device.

    Input:
        model_name: SentenceTransformer model id or local model path.
        device: `auto`, `cpu`, or `cuda`.
    Output:
        Loaded SentenceTransformer model.
    """
    resolved_device = resolve_device(device)
    # CUDA uses bfloat16 for lower memory pressure; outputs are converted back to
    # float32 before storage/search so FAISS receives a stable dtype.
    return SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16 if resolved_device == "cuda" else torch.float32,
        },
        device=resolved_device,
    )


def load_image_rgb(path: str) -> Image.Image:
    """Open an image file and convert it to RGB.

    Input:
        path: Image file path.
    Output:
        PIL image in RGB mode.
    """
    return Image.open(path).convert("RGB")


@torch.inference_mode()
def extract_embeddings_batch(
    images: List[Image.Image],
    model: SentenceTransformer,
) -> np.ndarray:
    """Embed a batch of PIL images and return normalized float32 vectors.

    Input:
        images: PIL images already loaded in memory.
        model: Loaded SentenceTransformer embedding model.
    Output:
        NumPy array with shape `[num_images, embedding_dim]`.
    """
    if not images:
        raise ValueError("extract_embeddings_batch received an empty image list.")

    # SentenceTransformer handles image preprocessing through the model's
    # processor. We hide its internal progress bar and expose our own batch-level
    # tqdm in `extract_embeddings_for_items`.
    emb = model.encode(
        images,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    emb = emb.to(torch.float32)

    # Normalization is part of the contract with FAISS IndexFlatIP.
    emb = F.normalize(emb, p=2, dim=1)

    return emb.detach().cpu().numpy().astype("float32")


def extract_embeddings_for_items(
    items: Sequence[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
    desc: str = "Embedding images",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Embed item records and return embeddings with aligned labels and paths.

    Input:
        items: Records containing `image_path` and `label`.
        model: Loaded SentenceTransformer embedding model.
        batch_size: Number of images embedded per model call.
        desc: Progress bar label shown while embedding.
    Output:
        Embedding array plus label and path lists in the same row order.
    """
    if len(items) == 0:
        return np.empty((0, 0), dtype="float32"), [], []
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    all_embeddings = []
    all_labels = []
    all_paths = []

    batch_starts = range(0, len(items), batch_size)
    for start_idx in tqdm(batch_starts, desc=desc, unit="batch"):
        batch_items = items[start_idx:start_idx + batch_size]
        batch_images = []
        batch_labels = []
        batch_paths = []

        # Keep labels and paths in the exact same order as the embeddings. The
        # memory cache and FAISS index rely on this row alignment.
        for item in batch_items:
            batch_images.append(load_image_rgb(item["image_path"]))
            batch_labels.append(item["label"])
            batch_paths.append(item["image_path"])

        batch_embeddings = extract_embeddings_batch(
            images=batch_images,
            model=model,
        )
        all_embeddings.append(batch_embeddings)
        all_labels.extend(batch_labels)
        all_paths.extend(batch_paths)

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    return embeddings, all_labels, all_paths


def extract_embedding_for_image(
    image_path: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """Embed one image file and return a single normalized vector.

    Input:
        image_path: Image file path.
        model: Loaded SentenceTransformer embedding model.
    Output:
        One NumPy vector with shape `[embedding_dim]`.
    """
    embeddings = extract_embeddings_batch(
        images=[load_image_rgb(image_path)],
        model=model,
    )
    return embeddings[0]
