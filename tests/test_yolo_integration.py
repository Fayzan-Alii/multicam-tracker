"""Runs the real Ultralytics model. Skipped without Ultralytics or without network access
for the one-time 5 MB weight download (cached in tests/.weights)."""

from pathlib import Path

import cv2
import pytest

ultralytics = pytest.importorskip("ultralytics")

from multicam_tracker.config import DetectorConfig  # noqa: E402
from multicam_tracker.detection.base import FrameContext  # noqa: E402
from multicam_tracker.detection.yolo import YoloDetector  # noqa: E402

WEIGHTS = Path(__file__).parent / ".weights" / "yolo11n.pt"


def test_yolo_finds_the_four_people_in_the_ultralytics_bus_image():
    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    try:
        det = YoloDetector(DetectorConfig(weights=str(WEIGHTS), imgsz=640, conf=0.25, device="cpu"))
    except Exception as exc:  # offline: weights cannot be fetched
        pytest.skip(f"could not load YOLO weights: {exc}")
    out = det.detect(FrameContext(sequence="bus", camera=0, frame=0, image_fn=lambda: image))
    assert len(out) == 4
    h, w = image.shape[:2]
    assert (out.boxes[:, [0, 2]] <= w).all() and (out.boxes[:, [1, 3]] <= h).all()
    assert ((out.boxes[:, 3] - out.boxes[:, 1]) > 300).all()  # people, not the bus
