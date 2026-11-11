"""Multi-view detection metrics on the ground plane (MODA / MODP, as used for WILDTRACK)."""

from __future__ import annotations

import numpy as np

from multicam_tracker.tracking.assignment import linear_assignment


def detection_metrics(gt: dict, det: dict, frames, radius: float = 0.5) -> dict:
    """gt/det: frame -> (N, 2) points in metres. A detection within `radius` of a GT point can
    match it (one-to-one, Hungarian per frame). MODP averages 1 - d / radius over matches."""
    tp = fp = fn = 0
    modp_sum = 0.0
    for f in frames:
        g = np.asarray(gt.get(f, np.zeros((0, 2))), float).reshape(-1, 2)
        d = np.asarray(det.get(f, np.zeros((0, 2))), float).reshape(-1, 2)
        if len(g) and len(d):
            dist = np.linalg.norm(g[:, None] - d[None], axis=-1)
            matches, _, _ = linear_assignment(dist, radius)
        else:
            matches = np.zeros((0, 2), dtype=int)
        tp += len(matches)
        fn += len(g) - len(matches)
        fp += len(d) - len(matches)
        if len(matches):
            modp_sum += float(np.sum(1.0 - dist[matches[:, 0], matches[:, 1]] / radius))
    n_gt = tp + fn
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "MODA": 1.0 - (fp + fn) / max(1, n_gt),
        "MODP": modp_sum / max(1, tp),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, n_gt),
    }
