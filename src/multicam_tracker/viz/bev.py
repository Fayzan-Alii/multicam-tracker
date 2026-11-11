"""Bird's-eye-view map of the ground plane: grid, camera fields of view, identities with
2-sigma uncertainty ellipses and trails, and (optionally) ground-truth positions."""

from __future__ import annotations

import cv2
import numpy as np

from multicam_tracker.viz.colors import id_color

BACKGROUND = (32, 32, 32)


def covariance_ellipse(cov: np.ndarray, n_sigma: float = 2.0) -> tuple[np.ndarray, float]:
    """Full axis lengths (major, minor) and the major axis angle in degrees."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return n_sigma * np.sqrt(np.maximum(vals, 0.0)), angle


class BEVCanvas:
    def __init__(self, roi, px_per_m: float = 20.0):
        self.roi = tuple(float(v) for v in roi)
        x0, y0, x1, y1 = self.roi
        self.s = float(px_per_m)
        self.transpose = (y1 - y0) > (x1 - x0)  # draw the long side horizontally
        w_m, h_m = ((y1 - y0), (x1 - x0)) if self.transpose else ((x1 - x0), (y1 - y0))
        self.size = (int(round(h_m * self.s)), int(round(w_m * self.s)))

    def world_to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        x0, y0, x1, y1 = self.roi
        if self.transpose:  # top view rotated 90 degrees clockwise: +y right, +x down
            return np.column_stack([(xy[:, 1] - y0) * self.s, (xy[:, 0] - x0) * self.s])
        return np.column_stack([(xy[:, 0] - x0) * self.s, (y1 - xy[:, 1]) * self.s])  # +y up

    def _base(self, cameras) -> np.ndarray:
        H, W = self.size
        img = np.full((H, W, 3), BACKGROUND, np.uint8)
        x0, y0, x1, y1 = self.roi
        if cameras:
            overlay = img.copy()
            for cam in cameras.values():
                poly = self._fov(cam)
                if len(poly) >= 3:
                    cv2.fillPoly(overlay, [np.round(self.world_to_px(poly)).astype(np.int32)], (70, 60, 45))
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        for x in np.arange(np.ceil(x0), x1 + 1e-9):
            self._segment(img, (x, y0), (x, y1), (80, 80, 80) if x % 5 else (120, 120, 120))
        for y in np.arange(np.ceil(y0), y1 + 1e-9):
            self._segment(img, (x0, y), (x1, y), (80, 80, 80) if y % 5 else (120, 120, 120))
        H, W = self.size
        for cid, cam in (cameras or {}).items():
            # Cameras usually stand outside the tracked area: pin them to the border and put the
            # label on the inward side so it stays on the canvas.
            c = cam.center[:2]
            p = np.round(self.world_to_px(np.clip(c, [x0, y0], [x1, y1])[None])[0]).astype(int)
            p = np.clip(p, 0, [W - 1, H - 1])
            cv2.circle(img, tuple(int(v) for v in p), 6, (255, 255, 255), -1)
            label = f"C{cid + 1}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = int(np.clip(p[0] + (8 if p[0] < W / 2 else -8 - tw), 0, W - tw))
            ty = int(np.clip(p[1] + (8 + th if p[1] < H / 2 else -8), th, H - 1))
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def _segment(self, img, a, b, color):
        p = np.round(self.world_to_px(np.array([a, b]))).astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), color, 1)

    def _fov(self, cam, n: int = 30) -> np.ndarray:
        """Ground footprint of the image, clipped to the ROI."""
        H, W = cam.image_size
        t = np.linspace(0, 1, n)
        border = np.concatenate([
            np.column_stack([t * W, np.full(n, H - 1)]),
            np.column_stack([np.full(n, W - 1), (1 - t) * H]),
            np.column_stack([(1 - t) * W, np.zeros(n)]),
            np.column_stack([np.zeros(n), t * H]),
        ])
        xy, valid = cam.image_to_ground(border)
        x0, y0, x1, y1 = self.roi
        pad = 0.5 * max(x1 - x0, y1 - y0)
        xy = xy[valid]
        xy = xy[(np.abs(xy[:, 0] - (x0 + x1) / 2) < pad * 3) & (np.abs(xy[:, 1] - (y0 + y1) / 2) < pad * 3)]
        if len(xy) < 3:
            return xy
        c = cam.center[:2]
        ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
        return np.vstack([np.clip(c, [x0, y0], [x1, y1]), xy[np.argsort(ang)]])

    def render(self, tracks=(), cameras=None, trails=None, gt=None) -> np.ndarray:
        """tracks: iterable of (identity, xy, cov). trails: identity -> list of xy. gt: (N, 2)."""
        img = self._base(cameras)
        if gt is not None and len(gt):
            for u, v in np.round(self.world_to_px(gt)).astype(int):
                cv2.drawMarker(img, (u, v), (160, 160, 160), cv2.MARKER_TILTED_CROSS, 8, 1)
        for ident, pts in (trails or {}).items():
            if len(pts) >= 2:
                cv2.polylines(img, [np.round(self.world_to_px(np.asarray(pts))).astype(np.int32)], False,
                              id_color(ident), 1)
        for ident, xy, cov in tracks:
            color = id_color(ident)
            (u, v), = np.round(self.world_to_px(xy)).astype(int)
            axes, angle = covariance_ellipse(cov)
            # World angle (CCW from +x) -> OpenCV angle (clockwise from +u, since v points down).
            angle = 90.0 - angle if self.transpose else -angle
            cv2.ellipse(img, (u, v), (max(1, int(axes[0] * self.s)), max(1, int(axes[1] * self.s))),
                        angle, 0, 360, color, 1)
            cv2.circle(img, (u, v), 4, color, -1)
            cv2.putText(img, str(ident), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return img
