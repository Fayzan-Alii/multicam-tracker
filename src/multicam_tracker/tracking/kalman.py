"""Linear Kalman filter and the two motion spaces the tracker can run in.

ImageMotion is the ByteTrack/DeepSORT box model: state [cx, cy, a, h, and their velocities],
noise scaled by box height, time in frames. It is the only option without calibration (MOT17).

GroundMotion tracks foot points on the ground plane: state [x, y, vx, vy] in metres and
seconds, so velocity is physical, one model serves every camera, and measurement noise is the
per-detection covariance propagated from pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from multicam_tracker.config import MotionConfig


class KalmanFilter:
    @staticmethod
    def predict(mean, cov, F, Q):
        return F @ mean, F @ cov @ F.T + Q

    @staticmethod
    def update(mean, cov, z, H, R):
        S = H @ cov @ H.T + R
        chol = cho_factor(S, lower=True, check_finite=False)
        K = cho_solve(chol, H @ cov, check_finite=False).T
        y = z - H @ mean
        I_KH = np.eye(len(mean)) - K @ H
        # Joseph form keeps the covariance symmetric positive definite under round-off.
        return mean + K @ y, I_KH @ cov @ I_KH.T + K @ R @ K.T


def mahalanobis_sq(residuals: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each (N, d) residual under its own (N, d, d) covariance."""
    sol = np.linalg.solve(covs, residuals[..., None])[..., 0]
    return np.einsum("nd,nd->n", residuals, sol)


def box_to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    h = y2 - y1
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1) / h, h])


class GroundMotion:
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg

    def initiate(self, z: np.ndarray, R: np.ndarray):
        mean = np.array([z[0], z[1], 0.0, 0.0])
        cov = np.zeros((4, 4))
        cov[:2, :2] = self.measurement_noise(R)
        cov[2, 2] = cov[3, 3] = self.cfg.init_speed_std**2
        return mean, cov

    def predict(self, mean, cov, dt: float):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Discrete white-noise acceleration (Bar-Shalom): per axis sigma_a^2 [dt^4/4, dt^3/2; dt^3/2, dt^2].
        q = self.cfg.accel_noise**2
        Q = np.zeros((4, 4))
        for p, v in ((0, 2), (1, 3)):
            Q[p, p] = q * dt**4 / 4.0
            Q[p, v] = Q[v, p] = q * dt**3 / 2.0
            Q[v, v] = q * dt**2
        return KalmanFilter.predict(mean, cov, F, Q)

    def measurement_noise(self, R: np.ndarray) -> np.ndarray:
        R = np.array(R, dtype=np.float64)
        floor = self.cfg.min_meas_std**2
        idx = np.arange(2)
        R[idx, idx] = np.maximum(R[idx, idx], floor)
        return R

    def update(self, mean, cov, z, R):
        return KalmanFilter.update(mean, cov, np.asarray(z, float), self.H, self.measurement_noise(R))

    def gating_distance(self, mean, cov, zs: np.ndarray, Rs: np.ndarray) -> np.ndarray:
        zs = np.asarray(zs, dtype=np.float64).reshape(-1, 2)
        if len(zs) == 0:
            return np.zeros(0)
        P = cov[:2, :2]
        S = P[None] + np.stack([self.measurement_noise(R) for R in Rs])
        return mahalanobis_sq(zs - mean[:2], S)

    # Adapters used by the tracker, keyed on a Detections container.
    def initiate_from(self, dets, i):
        return self.initiate(dets.world_xy[i], dets.world_cov[i])

    def update_from(self, mean, cov, dets, i):
        return self.update(mean, cov, dets.world_xy[i], dets.world_cov[i])

    def motion_cost(self, means, covs, dets, gate_chi2: float) -> np.ndarray:
        """(T, N) squared Mahalanobis distance normalised by the gate; inf outside the gate."""
        cost = np.full((len(means), len(dets)), np.inf)
        if len(dets) == 0:
            return cost
        valid = dets.ground_valid if dets.ground_valid is not None else np.ones(len(dets), bool)
        idx = np.nonzero(valid)[0]
        for t, (mean, cov) in enumerate(zip(means, covs)):
            d2 = self.gating_distance(mean, cov, dets.world_xy[idx], dets.world_cov[idx])
            row = d2 / gate_chi2
            row[d2 > gate_chi2] = np.inf
            cost[t, idx] = row
        return cost

    def to_box(self, mean):
        return None


class ImageMotion:
    H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def __init__(self, cfg: MotionConfig, fps: float):
        self.cfg = cfg
        self.fps = fps

    def initiate(self, box: np.ndarray):
        z = box_to_xyah(np.asarray(box, float))
        h = z[3]
        wp, wv = self.cfg.std_weight_position, self.cfg.std_weight_velocity
        std = np.array([2 * wp * h, 2 * wp * h, 1e-2, 2 * wp * h, 10 * wv * h, 10 * wv * h, 1e-5, 10 * wv * h])
        return np.concatenate([z, np.zeros(4)]), np.diag(std**2)

    def predict(self, mean, cov, dt: float):
        steps = dt * self.fps
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * steps
        h = mean[3]
        wp, wv = self.cfg.std_weight_position, self.cfg.std_weight_velocity
        std = np.array([wp * h, wp * h, 1e-2, wp * h, wv * h, wv * h, 1e-5, wv * h])
        return KalmanFilter.predict(mean, cov, F, np.diag(std**2) * max(steps, 1e-6))

    def update(self, mean, cov, box):
        h = mean[3]
        wp = self.cfg.std_weight_position
        R = np.diag(np.array([wp * h, wp * h, 1e-1, wp * h]) ** 2)
        return KalmanFilter.update(mean, cov, box_to_xyah(np.asarray(box, float)), self.H, R)

    def to_box(self, mean) -> np.ndarray:
        cx, cy, a, h = mean[:4]
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    def initiate_from(self, dets, i):
        return self.initiate(dets.boxes[i])

    def update_from(self, mean, cov, dets, i):
        return self.update(mean, cov, dets.boxes[i])

    def motion_cost(self, means, covs, dets, gate_chi2: float) -> np.ndarray:
        from multicam_tracker.tracking.costs import iou_matrix

        if len(means) == 0:
            return np.zeros((0, len(dets)))
        boxes = np.stack([self.to_box(m) for m in means])
        return 1.0 - iou_matrix(boxes, dets.boxes)
