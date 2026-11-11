"""On-disk caches for detections and embeddings.

Detection is by far the slowest stage, while tracking and fusion ablations only need its
output. Files are keyed by a hash of the producing configuration, so different detectors or
re-ID checkpoints can share one cache directory without ever being mixed up.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from multicam_tracker.detection.base import FrameContext
from multicam_tracker.types import Detections


def fingerprint_hash(fingerprint: dict) -> str:
    return hashlib.sha1(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:10]


class _FrameStore:
    """frame -> dict of arrays, one .npz per (sequence, camera), written on flush()."""

    def __init__(self, root, prefix: str, fingerprint: dict):
        self.root = Path(root)
        self.prefix = f"{prefix}_{fingerprint_hash(fingerprint)}"
        self.fingerprint = fingerprint
        self._data: dict[tuple[str, int], dict[int, dict]] = {}
        self._dirty: set[tuple[str, int]] = set()

    def _path(self, sequence: str, camera: int) -> Path:
        return self.root / f"{self.prefix}_{sequence}_c{camera}.npz"

    def _entry(self, sequence: str, camera: int) -> dict[int, dict]:
        key = (sequence, camera)
        if key not in self._data:
            path = self._path(sequence, camera)
            frames: dict[int, dict] = {}
            if path.exists():
                with np.load(path) as z:
                    offsets = z["offsets"]
                    # NPZ members decompress on every access. Load each once so
                    # frame slices share one backing array per member.
                    arrays = {k: z[k] for k in z.files if k not in ("frames", "offsets")}
                    for i, f in enumerate(z["frames"]):
                        frames[int(f)] = {k: a[offsets[i]:offsets[i + 1]] for k, a in arrays.items()}
            self._data[key] = frames
        return self._data[key]

    def get(self, ctx: FrameContext) -> dict | None:
        return self._entry(ctx.sequence, ctx.camera).get(ctx.frame)

    def put(self, ctx: FrameContext, arrays: dict) -> None:
        self._entry(ctx.sequence, ctx.camera)[ctx.frame] = arrays
        self._dirty.add((ctx.sequence, ctx.camera))

    def flush(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{self.prefix}_meta.json").write_text(json.dumps(self.fingerprint, indent=2, default=str))
        for sequence, camera in sorted(self._dirty):
            frames = self._data[(sequence, camera)]
            order = sorted(frames)
            keys = list(frames[order[0]].keys()) if order else []
            counts = [len(frames[f][keys[0]]) if keys else 0 for f in order]
            payload = {"frames": np.array(order, dtype=np.int64),
                       "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)}
            for k in keys:
                payload[k] = np.concatenate([frames[f][k] for f in order])
            np.savez_compressed(self._path(sequence, camera), **payload)
        self._dirty.clear()


class CachedDetector:
    def __init__(self, inner, root, fingerprint: dict):
        self.inner = inner
        self.store = _FrameStore(root, "det", fingerprint)

    def detect(self, ctx: FrameContext) -> Detections:
        hit = self.store.get(ctx)
        if hit is not None:
            return Detections(boxes=hit["boxes"], scores=hit["scores"])
        dets = self.inner.detect(ctx)
        self.store.put(ctx, {"boxes": dets.boxes, "scores": dets.scores})
        return dets

    def flush(self) -> None:
        self.store.flush()


class CachedEmbedder:
    def __init__(self, inner, root, fingerprint: dict):
        self.inner = inner
        self.store = _FrameStore(root, "emb", fingerprint)

    def embed(self, ctx: FrameContext, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        hit = self.store.get(ctx)
        if hit is not None and hit["boxes"].shape == boxes.shape and np.allclose(hit["boxes"], boxes):
            return hit["embeddings"]
        if self.inner is None:
            raise KeyError(f"embeddings for {ctx.sequence} camera {ctx.camera} frame {ctx.frame} are not cached")
        emb = self.inner.embed(ctx, boxes)
        self.store.put(ctx, {"boxes": boxes, "embeddings": emb})
        return emb

    def flush(self) -> None:
        self.store.flush()
