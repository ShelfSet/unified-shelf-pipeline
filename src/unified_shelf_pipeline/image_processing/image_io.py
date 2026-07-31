"""Discover, load, crop, and persist pipeline images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .geometry import clamp_bbox


VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".heic",
    ".heif",
}


def resolve_path(value: Path | str) -> Path:
    """Expand a user path and resolve it to an absolute path.

    Input:
        value: Path object or path string to normalize.
    Output:
        Expanded absolute path.
    """
    return Path(value).expanduser().resolve()


def list_image_paths(input_path: Path | str) -> List[Path]:
    """Resolve one image or discover supported images below a directory.

    Input:
        input_path: Image file or directory to search recursively.
    Output:
        Sorted absolute paths for supported image files.
    """
    path = resolve_path(input_path)
    if path.is_file():
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def safe_stem(path: Path | str) -> str:
    """Convert a path stem into a filesystem-safe output name.

    Input:
        path: Source path whose filename stem should be normalized.
    Output:
        Stem containing only alphanumeric characters, hyphens, and underscores.
    """
    stem = Path(path).stem
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in stem
    )


def import_cv2():
    """Import OpenCV with an actionable dependency error.

    Input:
        None.
    Output:
        The imported ``cv2`` module.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required. Install opencv-python in the active environment."
        ) from exc
    return cv2


def load_image_bgr(image_path: Path | str):
    """Load a regular or HEIC image into OpenCV BGR format.

    Input:
        image_path: Path to a supported source image.
    Output:
        A NumPy image array with channels ordered as BGR.
    """
    cv2 = import_cv2()
    path = resolve_path(image_path)

    # HEIC files require Pillow decoding before conversion to OpenCV format.
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener  # type: ignore
        except ImportError as exc:
            raise ImportError("HEIC input requires pillow-heif and pillow.") from exc
        register_heif_opener()
        pil_image = Image.open(path)
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.asarray(pil_image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def crop_image(image_bgr, bbox: Sequence[int]):
    """Crop an image using integer pixel coordinates.

    Input:
        image_bgr: Source image array in BGR format.
        bbox: Crop coordinates ordered as ``x1, y1, x2, y2``.
    Output:
        Array view containing the requested image region.
    """
    x1, y1, x2, y2 = bbox[:4]
    return image_bgr[int(y1) : int(y2), int(x1) : int(x2)]


def crop_price_tag(image_bgr, bbox: Sequence[int]):
    """Crop a price tag with padding that remains inside the image.

    Input:
        image_bgr: Source image array in BGR format.
        bbox: Detected tag coordinates ordered as ``x1, y1, x2, y2``.
    Output:
        Padded tag crop that preserves digits close to detector boundaries.
    """
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    tag_width = max(1.0, x2 - x1)
    tag_height = max(1.0, y2 - y1)
    padded_bbox = clamp_bbox(
        (
            x1 - max(2.0, 0.02 * tag_width),
            y1 - max(2.0, 0.04 * tag_height),
            x2 + max(2.0, 0.02 * tag_width),
            y2 + max(2.0, 0.04 * tag_height),
        ),
        width=width,
        height=height,
    )
    return crop_image(image_bgr, padded_bbox)


def save_crop(crop_bgr, path: Path) -> None:
    """Persist an image crop and create its parent directory when needed.

    Input:
        crop_bgr: Image crop in BGR array format.
        path: Destination image path.
    Output:
        None. The crop is written to the destination path.
    """
    cv2 = import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop_bgr)
