from __future__ import annotations

import numpy as np

from multicam_tracker.config import GroundConfig
from multicam_tracker.geometry.uncertainty import foot_covariance, foot_points, propagate_to_ground
from multicam_tracker.types import Detections


def project_to_ground(dets: Detections, camera, cfg: GroundConfig, roi) -> Detections:
    """Attach each detection's ground position and covariance. Detections whose foot point is
    above the horizon or outside the region of interest are marked ground_valid = False."""
    out = dets[np.arange(len(dets))]
    if len(dets) == 0:
        out.world_xy, out.world_cov, out.ground_valid = np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros(0, bool)
        return out
    cov_px = foot_covariance(dets.boxes, cfg.foot_sigma_u, cfg.foot_sigma_v, cfg.min_sigma_px)
    xy, cov, valid = propagate_to_ground(camera, foot_points(dets.boxes), cov_px)
    if cfg.foot_depth_offset_m and hasattr(camera, "center"):
        ray = xy - camera.center[:2]
        xy = xy + cfg.foot_depth_offset_m * ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
    if not cfg.propagate_uncertainty:
        cov = np.repeat(np.eye(2)[None] * cfg.fixed_sigma_m**2, len(dets), axis=0)
    x0, y0, x1, y1 = roi
    inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    out.world_xy, out.world_cov, out.ground_valid = xy, cov, valid & inside
    return out
