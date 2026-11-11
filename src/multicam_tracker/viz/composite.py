from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def compose(panels: list[np.ndarray], bev: np.ndarray, panel_width: int = 480, cols: int = 2) -> np.ndarray:
    """Tile camera panels (resized to panel_width) in a grid, with the map scaled to the grid
    height on the right."""
    resized = []
    for p in panels:
        h = int(round(p.shape[0] * panel_width / p.shape[1]))
        resized.append(cv2.resize(p, (panel_width, h), interpolation=cv2.INTER_AREA))
    ph = max(r.shape[0] for r in resized)
    rows = int(np.ceil(len(resized) / cols))
    grid = np.zeros((rows * ph, cols * panel_width, 3), np.uint8)
    for i, r in enumerate(resized):
        y, x = (i // cols) * ph, (i % cols) * panel_width
        grid[y:y + r.shape[0], x:x + panel_width] = r
    scale = grid.shape[0] / bev.shape[0]
    bev_r = cv2.resize(bev, (int(round(bev.shape[1] * scale)), grid.shape[0]), interpolation=cv2.INTER_AREA)
    return np.hstack([grid, bev_r])


class VideoWriter:
    def __init__(self, path, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer = None
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open a video writer for {self.path}")
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


def save_gif(frames: list[np.ndarray], path, fps: float = 5.0, width: int = 960) -> None:
    """Downscaled animated GIF (for README embeds)."""
    images = []
    for f in frames:
        h = int(round(f.shape[0] * width / f.shape[1]))
        images.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA),
                                                   cv2.COLOR_BGR2RGB)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
