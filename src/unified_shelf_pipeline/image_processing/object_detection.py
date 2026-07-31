"""Run YOLO object detection and de-duplicate product boxes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .geometry import clamp_bbox
from .image_io import resolve_path
from ..models import Detection


def custom_nms(
    detections: Sequence[Detection],
    iou_threshold: float,
) -> List[Detection]:
    """Apply class-agnostic non-maximum suppression to detector results.

    Input:
        detections: Candidate boxes with confidence scores.
        iou_threshold: Maximum allowed overlap with a higher-scoring box.
    Output:
        Filtered detections reindexed in retained confidence order.
    """
    if not detections:
        return []

    # Sort all boxes by detector confidence before overlap suppression.
    boxes = np.asarray([detection.bbox for detection in detections], dtype="float32")
    scores = np.asarray(
        [detection.confidence for detection in detections],
        dtype="float32",
    )
    order = scores.argsort()[::-1]
    keep: List[int] = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    # Retain the best remaining box and discard excessive overlaps.
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_x1 = np.maximum(x1[index], x1[remaining])
        intersection_y1 = np.maximum(y1[index], y1[remaining])
        intersection_x2 = np.minimum(x2[index], x2[remaining])
        intersection_y2 = np.minimum(y2[index], y2[remaining])
        intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
        intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
        intersection = intersection_width * intersection_height
        union = areas[index] + areas[remaining] - intersection
        iou = np.divide(intersection, np.maximum(union, 1e-9))
        order = remaining[iou <= iou_threshold]

    # Reindex retained detections so downstream crop names remain contiguous.
    kept = [detections[index] for index in keep]
    return [
        Detection(
            index=index,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
        )
        for index, detection in enumerate(kept)
    ]


class YoloDetector:
    """Lazily load and run an Ultralytics YOLO detector."""

    def __init__(
        self,
        model_path: Path | str,
        device: Optional[str] = None,
    ):
        """Store model settings without loading the YOLO weights.

        Input:
            model_path: Path to the Ultralytics YOLO model artifact.
            device: Optional runtime device; ``auto`` delegates selection.
        Output:
            None. The detector remains unloaded until its first prediction.
        """
        self.model_path = resolve_path(model_path)
        self.device = None if device in (None, "auto") else device
        self._model = None

    def _load(self):
        """Load and cache the configured YOLO model.

        Input:
            None. The model path comes from detector initialization.
        Output:
            The cached Ultralytics YOLO model.
        """
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Ultralytics is required. Install ultralytics in the active "
                    "environment."
                ) from exc
            if not self.model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image_bgr, conf: float) -> List[Detection]:
        """Run YOLO and normalize valid boxes to the shared detection schema.

        Input:
            image_bgr: Source image array in BGR format.
            conf: Minimum detector confidence accepted by YOLO.
        Output:
            Valid, image-bounded detections in model result order.
        """
        model = self._load()
        arguments = {"conf": conf, "verbose": False}
        if self.device:
            arguments["device"] = self.device
        result = model.predict(image_bgr, **arguments)[0]
        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Move model tensors to CPU and preserve class identifiers when present.
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = (
            result.boxes.cls.cpu().numpy()
            if result.boxes.cls is not None
            else [None] * len(confidences)
        )
        height, width = image_bgr.shape[:2]
        # Clamp boxes to source dimensions and reject empty regions.
        for index, (box, score, class_id) in enumerate(
            zip(boxes, confidences, class_ids)
        ):
            bbox = clamp_bbox(box, width=width, height=height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            detections.append(
                Detection(
                    index=index,
                    bbox=bbox,
                    confidence=float(score),
                    class_id=None if class_id is None else int(class_id),
                )
            )
        return detections
