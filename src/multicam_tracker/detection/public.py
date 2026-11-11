from multicam_tracker.config import DetectorConfig
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.types import Detections


class PublicDetector:
    """The dataset's own detections (MOT17 det.txt): isolates tracker quality from detector quality."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

    def detect(self, ctx: FrameContext) -> Detections:
        if ctx.public is None:
            raise ValueError(f"{ctx.sequence} provides no public detections")
        return ctx.public[ctx.public.scores >= self.cfg.conf]
