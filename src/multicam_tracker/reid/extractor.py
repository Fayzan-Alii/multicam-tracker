"""Appearance embeddings for detections, from a trained re-ID checkpoint."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from multicam_tracker.config import ReIDConfig
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.device import resolve_device
from multicam_tracker.reid.model import load_checkpoint
from multicam_tracker.reid.transforms import test_transform


def crop_boxes(image_bgr: np.ndarray, boxes: np.ndarray) -> list[Image.Image]:
    """RGB crops for each box, clipped to the image and at least one pixel in each dimension."""
    H, W = image_bgr.shape[:2]
    rgb = image_bgr[:, :, ::-1]
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        x1 = int(np.clip(np.floor(x1), 0, W - 1))
        y1 = int(np.clip(np.floor(y1), 0, H - 1))
        x2 = int(np.clip(np.ceil(x2), x1 + 1, W))
        y2 = int(np.clip(np.ceil(y2), y1 + 1, H))
        crops.append(Image.fromarray(np.ascontiguousarray(rgb[y1:y2, x1:x2])))
    return crops


class TorchReIDEmbedder:
    def __init__(self, cfg: ReIDConfig):
        if not cfg.checkpoint:
            raise ValueError("reid.checkpoint is empty; train one with `multicam-tracker train-reid` first")
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model, meta = load_checkpoint(cfg.checkpoint, self.device)
        self.transform = test_transform(meta.get("input_size", cfg.input_size))
        self.dim = meta.get("feat_dim")

    @torch.inference_mode()
    def embed(self, ctx: FrameContext, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, self.dim or 0))
        crops = crop_boxes(ctx.image(), boxes)
        out = []
        for start in range(0, len(crops), self.cfg.batch_size):
            batch = torch.stack([self.transform(c) for c in crops[start:start + self.cfg.batch_size]])
            out.append(self.model(batch.to(self.device)).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)
