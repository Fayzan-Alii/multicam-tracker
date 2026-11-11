"""Ultralytics YOLO person detector. Any Ultralytics detection weights work (yolov8*.pt,
yolo11*.pt, or your own fine-tuned best.pt); person is class 0 in COCO-trained models."""

from __future__ import annotations

from multicam_tracker.config import DetectorConfig
from multicam_tracker.detection.base import FrameContext, filter_detections
from multicam_tracker.device import resolve_device
from multicam_tracker.types import Detections


class YoloDetector:
    def __init__(self, cfg: DetectorConfig, person_class: int = 0):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("YOLO detection needs Ultralytics: python -m pip install -r requirements.txt") from exc
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.half = cfg.half and self.device.startswith("cuda")
        self.person_class = person_class
        self.model = YOLO(cfg.weights)

    def detect(self, ctx: FrameContext) -> Detections:
        image = ctx.image()
        # Ultralytics replaced the old ``half=True`` flag with ``quantize=16``.
        # Both request 16-bit floating-point inference on a supported GPU.
        kwargs = {"quantize": 16} if self.half else {}
        result = self.model.predict(
            image, classes=[self.person_class], conf=self.cfg.conf, iou=self.cfg.nms_iou,
            imgsz=self.cfg.imgsz, device=self.device, verbose=False, **kwargs,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return filter_detections(boxes, scores, self.cfg.min_height_px, image.shape[:2])
