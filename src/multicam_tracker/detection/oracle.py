"""Ground-truth-driven stand-ins for the detector and the re-ID model.

They let the tracking and fusion stages be tested, tuned and demonstrated on synthetic data,
and give an upper bound ("perfect detector") row in ablation tables.
"""

from __future__ import annotations

import numpy as np

from multicam_tracker.config import DetectorConfig, ReIDConfig
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.tracking.assignment import linear_assignment
from multicam_tracker.tracking.costs import iou_matrix, normalize_rows
from multicam_tracker.types import Detections


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([abs(int(k)) for k in keys])


class OracleDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

    def detect(self, ctx: FrameContext) -> Detections:
        gt = ctx.gt
        if gt is None or len(gt.ids) == 0:
            return Detections.empty()
        rng = _rng(self.cfg.seed, ctx.camera, ctx.frame)
        keep = rng.random(len(gt.ids)) >= self.cfg.oracle_miss_rate
        boxes = gt.boxes[keep] + rng.normal(0.0, self.cfg.oracle_noise_px, size=(int(keep.sum()), 4))
        scores = rng.uniform(0.5, 1.0, size=len(boxes))
        return Detections(boxes=boxes, scores=scores)


class OracleEmbedder:
    """Each identity gets a fixed random unit vector; observations add isotropic noise with
    expected norm `oracle_noise`. Boxes are tied to identities by IoU with the GT boxes."""

    def __init__(self, cfg: ReIDConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed

    def _identity_vector(self, pid: int) -> np.ndarray:
        return normalize_rows(_rng(self.seed, 7919, pid).normal(size=self.cfg.oracle_dim))

    def embed(self, ctx: FrameContext, boxes: np.ndarray) -> np.ndarray:
        d = self.cfg.oracle_dim
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        rng = _rng(self.seed, 104729, ctx.camera, ctx.frame)
        out = normalize_rows(rng.normal(size=(len(boxes), d)))  # unmatched boxes: random vectors
        gt = ctx.gt
        if gt is not None and len(gt.ids) and len(boxes):
            matches, _, _ = linear_assignment(1.0 - iou_matrix(boxes, gt.boxes), 0.7)
            for bi, gi in matches:
                out[bi] = self._identity_vector(int(gt.ids[gi]))
        noise = rng.normal(size=(len(boxes), d)) * (self.cfg.oracle_noise / np.sqrt(d))
        return normalize_rows(out + noise)
