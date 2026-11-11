"""Core data containers shared across detection, tracking and fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detections:
    """Struct-of-arrays for one camera frame, so cost matrices can be built without Python loops.

    boxes are [x1, y1, x2, y2] pixels. The optional fields are filled in by later stages:
    re-ID adds embeddings, ground projection adds world_xy / world_cov / ground_valid.
    """

    boxes: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray | None = None
    world_xy: np.ndarray | None = None
    world_cov: np.ndarray | None = None
    ground_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        n = len(self.boxes)
        for name in ("scores", "embeddings", "world_xy", "world_cov", "ground_valid"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"Detections.{name} has {len(value)} rows but there are {n} boxes")

    @classmethod
    def empty(cls) -> "Detections":
        return cls(boxes=np.zeros((0, 4)), scores=np.zeros(0))

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, idx) -> "Detections":
        def pick(a):
            return None if a is None else a[idx]

        return Detections(
            boxes=self.boxes[idx],
            scores=self.scores[idx],
            embeddings=pick(self.embeddings),
            world_xy=pick(self.world_xy),
            world_cov=pick(self.world_cov),
            ground_valid=pick(self.ground_valid),
        )


@dataclass
class TrackOutput:
    """One confirmed track observed in the current frame of one camera."""

    track_id: int
    box: np.ndarray
    score: float
    camera_id: int = 0
    world_xy: np.ndarray | None = None  # the detection's ground measurement this frame
    world_cov: np.ndarray | None = None
    embedding: np.ndarray | None = None  # smoothed track appearance


@dataclass
class GlobalTrackOutput:
    """One identity on the shared ground plane after cross-camera fusion."""

    global_id: int
    world_xy: np.ndarray
    world_cov: np.ndarray
    cameras: list[int] = field(default_factory=list)
    local: dict[int, int] = field(default_factory=dict)  # camera -> local track id
    boxes: dict[int, np.ndarray] = field(default_factory=dict)  # camera -> box
