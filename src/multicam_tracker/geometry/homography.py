"""Homography estimation: normalised DLT plus RANSAC, written out rather than delegated to OpenCV."""

from __future__ import annotations

import numpy as np


def apply_homography(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map (N, 2) points through H. Returns (mapped points, homogeneous scale w)."""
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / w[:, None], w


def _normalising_transform(pts: np.ndarray) -> np.ndarray:
    # Hartley normalisation: centroid to the origin, mean distance sqrt(2).
    centroid = pts.mean(axis=0)
    mean_dist = np.linalg.norm(pts - centroid, axis=1).mean()
    s = np.sqrt(2.0) / max(mean_dist, 1e-12)
    return np.array([[s, 0.0, -s * centroid[0]], [0.0, s, -s * centroid[1]], [0.0, 0.0, 1.0]])


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares H with dst ~ H @ src from N >= 4 correspondences, scaled so H[2, 2] = 1."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("a homography needs at least 4 matching point pairs")
    Ts, Td = _normalising_transform(src), _normalising_transform(dst)
    s, _ = apply_homography(Ts, src)
    d, _ = apply_homography(Td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 3.0,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust H: sample minimal sets, keep the largest consensus, refit on its inliers."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        raise ValueError("a homography needs at least 4 matching point pairs")
    rng = np.random.default_rng(seed)
    best = np.zeros(n, dtype=bool)
    for _ in range(iterations):
        idx = rng.choice(n, size=4, replace=False)
        try:
            H = estimate_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        proj, w = apply_homography(H, src)
        err = np.linalg.norm(proj - dst, axis=1)
        inliers = np.isfinite(err) & (err < threshold) & (np.abs(w) > 1e-12)
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 4:
        raise ValueError("RANSAC found no consensus set of 4 or more inliers")
    H = estimate_homography(src[best], dst[best])
    proj, _ = apply_homography(H, src)
    inliers = np.linalg.norm(proj - dst, axis=1) < threshold
    return estimate_homography(src[inliers], dst[inliers]), inliers
