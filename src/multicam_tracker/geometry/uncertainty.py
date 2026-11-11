"""Foot points and first-order propagation of their pixel uncertainty onto the ground plane.

A one-pixel error near the camera is centimetres on the ground; near the horizon it is
metres. Carrying the covariance lets the tracker and fusion weight observations properly
and gate with Mahalanobis distance instead of a hand-tuned radius.
"""

from __future__ import annotations

import numpy as np


def foot_points(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of (N, 4) [x1, y1, x2, y2] boxes."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])


def foot_covariance(boxes: np.ndarray, sigma_u: float, sigma_v: float, min_sigma_px: float) -> np.ndarray:
    """(N, 2, 2) diagonal pixel covariance: std proportional to box width (u) and height (v)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    su = np.maximum(sigma_u * (boxes[:, 2] - boxes[:, 0]), min_sigma_px)
    sv = np.maximum(sigma_v * (boxes[:, 3] - boxes[:, 1]), min_sigma_px)
    cov = np.zeros((len(boxes), 2, 2))
    cov[:, 0, 0] = su**2
    cov[:, 1, 1] = sv**2
    return cov


def propagate_to_ground(camera, uv: np.ndarray, cov_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map pixels and their covariances to the ground: Sigma_world = J Sigma_px J^T."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy, valid = camera.image_to_ground(uv)
    J = camera.image_to_ground_jacobian(uv)
    cov = J @ np.asarray(cov_px, dtype=np.float64) @ np.transpose(J, (0, 2, 1))
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    valid = valid & np.isfinite(cov).all(axis=(1, 2))
    return xy, cov, valid
