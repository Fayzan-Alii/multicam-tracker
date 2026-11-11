"""ONNX Runtime re-ID embedder (CUDA provider when available). Same preprocessing as PyTorch."""

from __future__ import annotations

import numpy as np

from multicam_tracker.config import ReIDConfig
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.reid.extractor import crop_boxes
from multicam_tracker.reid.transforms import test_transform


class OnnxReIDEmbedder:
    def __init__(self, cfg: ReIDConfig):
        import onnxruntime as ort

        self.cfg = cfg
        available = ort.get_available_providers()
        want_cuda = cfg.device != "cpu" and "CUDAExecutionProvider" in available
        providers = (["CUDAExecutionProvider"] if want_cuda else []) + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(cfg.checkpoint, providers=providers)
        meta = self.session.get_modelmeta().custom_metadata_map
        size = [int(v) for v in meta["input_size"].split(",")] if "input_size" in meta else cfg.input_size
        self.transform = test_transform(size)
        self.dim = int(meta["feat_dim"]) if meta.get("feat_dim") else None

    def embed(self, ctx: FrameContext, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, self.dim or 0))
        crops = crop_boxes(ctx.image(), boxes)
        out = []
        for start in range(0, len(crops), self.cfg.batch_size):
            batch = np.stack([self.transform(c).numpy() for c in crops[start:start + self.cfg.batch_size]])
            out.append(self.session.run(None, {"images": batch.astype(np.float32)})[0])
        return np.concatenate(out).astype(np.float64)
