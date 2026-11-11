"""Build Market-style re-ID crops using only the multi-view training split.

The final validation_fraction of training frames supplies query/gallery crops for
checkpoint selection. The dataset test split remains untouched for tracking evaluation.
Images are written immediately; only validation paths are retained in memory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from multicam_tracker.config import ConfigError
from multicam_tracker.data.multiview import load_multiview


def build_reid_crops(root, kind: str, out, min_height: int = 40, seed: int = 0,
                     validation_fraction: float = 0.2) -> dict:
    if not 0 < validation_fraction < 1:
        raise ConfigError("validation_fraction must be between 0 and 1 (exclusive)")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConfigError(f"crop output must be an empty directory: {out}; use a new output path")
    seq = load_multiview(root, kind, split="train")
    # At least two validation frames allow a query and gallery observation per camera.
    n_val = max(2, math.ceil(len(seq.frames) * validation_fraction))
    cut = len(seq.frames) - n_val
    if cut < 1:
        raise ConfigError("need at least one training frame and two validation frames; "
                          "provide more data or reduce validation_fraction")
    parts = {"train": seq.frames[:cut], "val": seq.frames[cut:]}
    for name in ("bounding_box_train", "query", "bounding_box_test"):
        (out / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts = {"train": 0, "val": 0}
    groups: dict[tuple[int, int], list[Path]] = {}
    for part, frames in parts.items():
        directory = out / ("bounding_box_train" if part == "train" else "bounding_box_test")
        for f in frames:
            for c in sorted(seq.cameras):
                gt = seq.gt_view(c, f)
                if len(gt.ids) == 0:
                    continue
                image = seq.image(c, f)
                H, W = image.shape[:2]
                for pid, (x1, y1, x2, y2) in zip(gt.ids, gt.boxes):
                    x1, x2 = int(max(0, x1)), int(min(W, x2))
                    y1, y2 = int(max(0, y1)), int(min(H, y2))
                    if y2 - y1 < min_height or x2 - x1 < 2:
                        continue
                    # Market reserves pid 0 for distractors.
                    pid, cam = int(pid) + 1, c + 1
                    index = counts[part]
                    path = directory / f"{pid:04d}_c{cam}s1_{f:06d}_{index:06d}.jpg"
                    if not cv2.imwrite(str(path), image[y1:y2, x1:x2]):
                        raise OSError(f"failed to write re-ID crop: {path}")
                    counts[part] += 1
                    if part == "val":
                        groups.setdefault((pid, cam), []).append(path)
                del image
    # Choose queries after writing, moving only files rather than retaining image arrays.
    for paths in groups.values():
        chosen = paths[int(rng.integers(len(paths)))]
        chosen.rename(out / "query" / chosen.name)
    (out / "split.json").write_text(json.dumps({
        "dataset": kind, "source_split": "train", "validation_fraction": validation_fraction,
        "seed": seed, "frames": parts, "counts": counts,
    }, indent=2))
    return counts
