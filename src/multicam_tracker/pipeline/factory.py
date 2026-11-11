"""Build detectors and embedders from config, wrapped in caches when detector.cache_dir is set.

Models are constructed lazily, so a fully cached run never loads YOLO or the re-ID network."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

from multicam_tracker.config import AppConfig
from multicam_tracker.detection.cache import CachedDetector, CachedEmbedder


class _Lazy:
    def __init__(self, factory):
        self._factory = factory
        self._obj = None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._factory()
        return getattr(self._obj, name)


def _dataset_fingerprint(cfg: AppConfig) -> dict:
    fp = {"kind": cfg.dataset.kind, "root": str(Path(cfg.dataset.root).resolve()) if cfg.dataset.root else ""}
    if cfg.dataset.kind == "synthetic":
        fp["synthetic"] = dataclasses.asdict(cfg.dataset.synthetic)
    return fp


def _checkpoint_fingerprint(value: str) -> dict:
    """Hash local weights without loading a model, including same-size replacements."""
    path = Path(value)
    digest = None
    if path.is_file():
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def build_detector(cfg: AppConfig):
    d = cfg.detector
    if d.kind == "yolo":
        from multicam_tracker.detection.yolo import YoloDetector

        inner = _Lazy(lambda: YoloDetector(d))
    elif d.kind == "public":
        from multicam_tracker.detection.public import PublicDetector

        inner = PublicDetector(d)
    else:
        from multicam_tracker.detection.oracle import OracleDetector

        inner = OracleDetector(d)
    if not d.cache_dir:
        return inner
    fields = dataclasses.asdict(d)
    fields.pop("cache_dir")
    fields.pop("device")
    if d.kind == "yolo":
        fields["checkpoint"] = _checkpoint_fingerprint(d.weights)
    return CachedDetector(inner, d.cache_dir, {"dataset": _dataset_fingerprint(cfg), "detector": fields})


def build_embedder(cfg: AppConfig):
    r = cfg.reid
    if not r.enabled:
        return None
    if r.kind == "torch":
        from multicam_tracker.reid.extractor import TorchReIDEmbedder

        inner = _Lazy(lambda: TorchReIDEmbedder(r))
    elif r.kind == "onnx":
        from multicam_tracker.deploy.onnx_embedder import OnnxReIDEmbedder

        inner = _Lazy(lambda: OnnxReIDEmbedder(r))
    else:
        from multicam_tracker.detection.oracle import OracleEmbedder

        inner = OracleEmbedder(r, seed=cfg.detector.seed)
    if not cfg.detector.cache_dir:
        return inner
    fp = {"dataset": _dataset_fingerprint(cfg), "kind": r.kind, "input_size": r.input_size}
    if r.kind == "oracle":
        fp.update(dim=r.oracle_dim, noise=r.oracle_noise, seed=cfg.detector.seed)
    else:
        fp["checkpoint"] = _checkpoint_fingerprint(r.checkpoint)
    return CachedEmbedder(inner, cfg.detector.cache_dir, fp)
