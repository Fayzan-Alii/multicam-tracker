"""Detector interface. Every detector sees a FrameContext, so ground-truth-driven detectors
(oracle, public) and image-driven ones (YOLO) are interchangeable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from multicam_tracker.data.base import GTFrame
from multicam_tracker.types import Detections


@dataclass
class FrameContext:
    sequence: str
    camera: int
    frame: int
    image_fn: Callable[[], np.ndarray]
    gt: GTFrame | None = None
    public: Detections | None = None
    _image: np.ndarray | None = field(default=None, repr=False)

    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = self.image_fn()
        return self._image


class Detector(Protocol):
    def detect(self, ctx: FrameContext) -> Detections: ...


class Embedder(Protocol):
    def embed(self, ctx: FrameContext, boxes: np.ndarray) -> np.ndarray: ...


def filter_detections(boxes, scores, min_height_px: float, image_size) -> Detections:
    """Clip boxes to the image and drop degenerate or too-short ones."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    H, W = image_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    keep = (w > 0) & (h > 0) & (h >= min_height_px)
    return Detections(boxes=boxes[keep], scores=np.asarray(scores, dtype=np.float64)[keep])
