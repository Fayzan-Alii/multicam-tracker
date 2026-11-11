"""Demo video: camera panels coloured by fused global identity beside the ground-plane map."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from multicam_tracker.config import OutputConfig
from multicam_tracker.viz.bev import BEVCanvas
from multicam_tracker.viz.composite import VideoWriter, compose, save_gif
from multicam_tracker.viz.overlay import draw_boxes, label_panel

MAX_GIF_FRAMES = 80


class DemoRenderer:
    def __init__(self, cfg: OutputConfig, seq, roi, out_dir):
        self.cfg = cfg
        self.seq = seq
        self.bev = BEVCanvas(roi, cfg.bev_px_per_m)
        fps = cfg.video_fps or seq.fps
        self.video = VideoWriter(out_dir / "demo.mp4", fps)
        self.gif_path = out_dir / "demo.gif"
        self.fps = fps
        self.trail_len = max(2, int(round(cfg.trail_s * seq.fps)))
        self.world_trails = defaultdict(lambda: deque(maxlen=self.trail_len))
        self.gif_frames = []
        n = len(seq.cameras)
        self.cols = 1 if n == 1 else min(4, int(np.ceil(n / 2)))

    def done(self) -> bool:
        return bool(self.cfg.max_video_frames) and self.video.frames_written >= self.cfg.max_video_frames

    def add(self, frame: int, images: dict, local: dict, global_tracks) -> None:
        if self.done():
            return
        panels = []
        for c in sorted(self.seq.cameras):
            if global_tracks is not None:
                items = [(o.global_id, o.boxes[c]) for o in global_tracks if c in o.boxes]
                prefix = "G"
            else:
                items = [(t.track_id, t.box) for t in local.get(c, [])]
                prefix = "L"
            panel = draw_boxes(images[c], items, prefix=prefix)
            panels.append(label_panel(panel, f"C{c + 1}  t={self.seq.timestamp(frame):.1f}s"))
        tracks = []
        if global_tracks is not None:
            for o in global_tracks:
                self.world_trails[o.global_id].append(o.world_xy)
                tracks.append((o.global_id, o.world_xy, o.world_cov))
        gt = self.seq.gt_world(frame)[1] if self.seq.has_gt else None
        bev = self.bev.render(tracks=tracks, cameras=self.seq.cameras,
                              trails={k: list(v) for k, v in self.world_trails.items()}, gt=gt)
        composite = compose(panels, bev, self.cfg.panel_width, self.cols)
        self.video.write(composite)
        if len(self.gif_frames) < MAX_GIF_FRAMES:
            self.gif_frames.append(composite)

    def close(self) -> None:
        self.video.close()
        if self.gif_frames:
            save_gif(self.gif_frames, self.gif_path, fps=self.fps, width=min(1280, self.gif_frames[0].shape[1]))


class SingleCameraRenderer:
    """Tracked boxes with identities on the original frames (MOT17-style sequences)."""

    def __init__(self, cfg: OutputConfig, out_dir, name: str, fps: float):
        self.cfg = cfg
        self.fps = cfg.video_fps or fps
        self.video = VideoWriter(out_dir / f"demo_{name}.mp4", self.fps)
        self.gif_path = out_dir / f"demo_{name}.gif"
        self.name = name
        self.gif_frames = []
        self.trails = defaultdict(lambda: deque(maxlen=max(2, int(round(cfg.trail_s * fps)))))

    def done(self) -> bool:
        return bool(self.cfg.max_video_frames) and self.video.frames_written >= self.cfg.max_video_frames

    def add(self, image, tracks, frame: int) -> None:
        if self.done():
            return
        for t in tracks:
            self.trails[t.track_id].append(((t.box[0] + t.box[2]) / 2, t.box[3]))
        live = {t.track_id for t in tracks}
        panel = draw_boxes(image, [(t.track_id, t.box) for t in tracks],
                           trails={k: list(v) for k, v in self.trails.items() if k in live})
        panel = label_panel(panel, f"{self.name}  frame {frame}")
        self.video.write(panel)
        if len(self.gif_frames) < MAX_GIF_FRAMES:
            self.gif_frames.append(panel)

    def close(self) -> None:
        self.video.close()
        if self.gif_frames:
            save_gif(self.gif_frames, self.gif_path, fps=self.fps, width=min(960, self.gif_frames[0].shape[1]))
