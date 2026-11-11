"""Pinhole camera with a ground plane at world Z = 0.

Conventions: X_cam = R @ X_world + t; pixels are (u, v) with v pointing down.
World units are metres throughout the codebase; dataset loaders convert on read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from multicam_tracker.geometry.homography import apply_homography, estimate_homography


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    image_size: tuple[int, int] = (1080, 1920)  # H, W
    name: str = ""

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        # Ground (x, y, 1) -> pixel: K [r1 r2 t]. Its inverse maps undistorted pixels to the ground.
        self.H_ground_to_image = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        self.H_image_to_ground = np.linalg.inv(self.H_ground_to_image)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist != 0))

    def project(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) world points -> (N, 2) pixels and a mask of points in front of the camera."""
        cam = np.asarray(world, dtype=np.float64) @ self.R.T + self.t
        in_front = cam[:, 2] > 1e-9
        if self.has_distortion:
            rvec, _ = cv2.Rodrigues(self.R)
            uv, _ = cv2.projectPoints(np.asarray(world, dtype=np.float64), rvec, self.t, self.K, self.dist)
            return uv.reshape(-1, 2), in_front
        with np.errstate(divide="ignore", invalid="ignore"):
            p = cam @ self.K.T
            return p[:, :2] / p[:, 2:], in_front

    def ground_to_image(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        uv, _ = self.project(np.hstack([xy, np.zeros((len(xy), 1))]))
        return uv

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if not self.has_distortion or len(uv) == 0:
            return uv
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
        src = uv.reshape(-1, 1, 2)
        if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
            out = cv2.undistortPointsIter(src, self.K, self.dist, None, self.K, criteria)
        else:  # OpenCV 5.x folded the iterative variant into undistortPoints
            out = cv2.undistortPoints(src, self.K, self.dist, R=None, P=self.K, criteria=criteria)
        return out.reshape(-1, 2)

    def image_to_ground(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Back-project pixels onto Z = 0. A pixel whose ray meets the plane behind the camera
        (anything above the horizon) is marked invalid."""
        uv = self.undistort(uv)
        xy, _ = apply_homography(self.H_image_to_ground, uv)
        world = np.hstack([xy, np.zeros((len(xy), 1))])
        depth = (world @ self.R.T + self.t)[:, 2]
        valid = np.isfinite(xy).all(axis=1) & (depth > 1e-9)
        return xy, valid

    def image_to_ground_jacobian(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2, 2) Jacobian d(x, y)/d(u, v) of the undistorted-pixel -> ground homography."""
        return homography_jacobian(self.H_image_to_ground, self.undistort(uv))


@dataclass
class HomographyCamera:
    """Ground-plane-only camera for footage with no full calibration: a single image->ground
    homography fitted from clicked correspondences. Pixels are assumed undistorted."""

    H_image_to_ground: np.ndarray
    image_size: tuple[int, int]
    front_sign: float = 1.0  # sign of the homogeneous scale for pixels below the horizon
    name: str = ""

    def __post_init__(self) -> None:
        self.H_image_to_ground = np.asarray(self.H_image_to_ground, dtype=np.float64)
        self.H_ground_to_image = np.linalg.inv(self.H_image_to_ground)

    @classmethod
    def from_correspondences(cls, image_pts, ground_pts, image_size, name: str = "") -> "HomographyCamera":
        H = estimate_homography(np.asarray(image_pts, float), np.asarray(ground_pts, float))
        _, w = apply_homography(H, np.asarray(image_pts, float))
        return cls(H, tuple(image_size), float(np.sign(np.median(w))), name)

    def image_to_ground(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy, w = apply_homography(self.H_image_to_ground, np.asarray(uv, dtype=np.float64).reshape(-1, 2))
        return xy, np.isfinite(xy).all(axis=1) & (np.sign(w) == self.front_sign)

    def ground_to_image(self, xy: np.ndarray) -> np.ndarray:
        uv, _ = apply_homography(self.H_ground_to_image, np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        return uv

    def image_to_ground_jacobian(self, uv: np.ndarray) -> np.ndarray:
        return homography_jacobian(self.H_image_to_ground, np.asarray(uv, dtype=np.float64).reshape(-1, 2))


def homography_jacobian(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Jacobian of p -> (h1.p / h3.p, h2.p / h3.p) at each (u, v)."""
    p = np.hstack([uv, np.ones((len(uv), 1))])
    num = p @ H.T
    w = num[:, 2]
    x = num[:, 0] / w
    y = num[:, 1] / w
    J = np.empty((len(uv), 2, 2))
    J[:, 0, 0] = (H[0, 0] - x * H[2, 0]) / w
    J[:, 0, 1] = (H[0, 1] - x * H[2, 1]) / w
    J[:, 1, 0] = (H[1, 0] - y * H[2, 0]) / w
    J[:, 1, 1] = (H[1, 1] - y * H[2, 1]) / w
    return J


def look_at(center, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose optical axis points from `center` to `target`, with world `up` appearing
    as image up. Rows are the camera's right, down and forward axes in world coordinates."""
    forward = np.asarray(target, float) - np.asarray(center, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])
