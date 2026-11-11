"""YAML configuration: `_base_` inheritance, dotted CLI overrides, and typed dataclasses.

Every run is described by one YAML file. Files may inherit from others with a
`_base_` key (a path or list of paths, relative to the file), which is what makes
ablations cheap: each variant is a few lines on top of a shared base.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _choice(default: str, *choices: str) -> Any:
    return field(default=default, metadata={"choices": (default, *choices)})


@dataclass
class SyntheticConfig:
    num_cameras: int = 4
    num_people: int = 10
    num_frames: int = 60
    fps: float = 2.0
    area: list[float] = field(default_factory=lambda: [0.0, 0.0, 12.0, 12.0])
    image_size: list[int] = field(default_factory=lambda: [540, 960])  # H, W
    walk_speed: float = 1.2
    seed: int = 0


@dataclass
class DatasetConfig:
    kind: str = _choice("synthetic", "mot17", "wildtrack", "multiviewx")
    root: str = ""
    # Multi-view: all | train | test (MVDet protocol: first 90% / last 10% of frames).
    # MOT17: all | first_half | second_half of each sequence (ByteTrack val protocol).
    split: str = _choice("all", "train", "test", "first_half", "second_half")
    mot17_subset: str = _choice("train", "test")
    mot17_detector: str = _choice("FRCNN", "DPM", "SDP")
    sequences: list[str] = field(default_factory=list)
    cameras: list[int] = field(default_factory=list)
    fps: float = 0.0  # 0 = dataset default
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass
class DetectorConfig:
    kind: str = _choice("yolo", "public", "oracle")
    weights: str = "yolo11s.pt"
    conf: float = 0.1
    nms_iou: float = 0.7
    imgsz: int = 1280
    device: str = "auto"
    half: bool = True
    min_height_px: float = 0.0
    cache_dir: str = ""
    oracle_noise_px: float = 2.0
    oracle_miss_rate: float = 0.05
    seed: int = 0


@dataclass
class ReIDConfig:
    enabled: bool = False
    kind: str = _choice("torch", "onnx", "oracle")
    checkpoint: str = ""
    input_size: list[int] = field(default_factory=lambda: [256, 128])  # H, W
    batch_size: int = 64
    device: str = "auto"
    oracle_dim: int = 64
    oracle_noise: float = 0.25


@dataclass
class GroundConfig:
    roi: list[float] = field(default_factory=list)  # xmin, ymin, xmax, ymax (m); [] = dataset default
    foot_sigma_u: float = 0.08  # 1-sigma foot-point error as a fraction of box width
    foot_sigma_v: float = 0.04  # ... and of box height
    min_sigma_px: float = 1.5
    # Ablation switch: False replaces the propagated covariance with fixed_sigma_m^2 * I.
    propagate_uncertainty: bool = True
    fixed_sigma_m: float = 0.3
    # A box's bottom edge is the nearest foot, so its back-projection lands short of where the
    # person stands. Push it this far away from the camera; `multicam-tracker verify-calibration` measures it.
    foot_depth_offset_m: float = 0.0


@dataclass
class MotionConfig:
    space: str = _choice("image", "ground")
    # Ground plane, constant velocity in metres and seconds.
    accel_noise: float = 1.5  # 1-sigma white-noise acceleration, m/s^2
    init_speed_std: float = 1.5  # m/s
    min_meas_std: float = 0.05  # m
    # Image plane, ByteTrack/DeepSORT parameterisation relative to box height.
    std_weight_position: float = 1.0 / 20
    std_weight_velocity: float = 1.0 / 160


@dataclass
class TrackerConfig:
    motion: MotionConfig = field(default_factory=MotionConfig)
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    second_stage: bool = True
    n_init: int = 2
    max_lost_s: float = 1.0
    gate_chi2: float = 9.21  # 99% for 2 dof
    fuse_score: bool = False
    appearance_weight: float = 0.0  # 0 disables appearance in the association cost
    appearance_gate: float = 0.35  # max cosine distance for a match
    embedding_ema: float = 0.9
    assignment: str = _choice("scipy", "native")


@dataclass
class FusionConfig:
    enabled: bool = True
    motion: MotionConfig = field(default_factory=lambda: MotionConfig(space="ground"))
    link_gate_chi2: float = 13.82  # 99.9% for 2 dof: keep an existing local->global link
    gate_chi2: float = 9.21
    appearance_weight: float = 0.3
    appearance_gate: float = 0.4
    match_thresh: float = 0.9
    cluster_thresh: float = 0.8
    max_lost_s: float = 2.0
    merge_chi2: float = 2.0
    merge_appearance: float = 0.25
    merge_frames: int = 2  # consecutive agreeing frames before duplicates merge
    embedding_ema: float = 0.9
    assignment: str = _choice("scipy", "native")


@dataclass
class EvalConfig:
    enabled: bool = True
    iou_threshold: float = 0.5
    ground_track_radius_m: float = 1.0
    ground_det_radius_m: float = 0.5


@dataclass
class OutputConfig:
    dir: str = "runs/default"
    save_video: bool = False
    video_fps: float = 0.0  # 0 = dataset fps
    max_video_frames: int = 0  # 0 = all
    panel_width: int = 480
    bev_px_per_m: float = 20.0
    trail_s: float = 3.0


@dataclass
class AppConfig:
    name: str = "run"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Read a YAML file and resolve its `_base_` chain into one merged dict."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        merged = deep_merge(merged, load_config(path.parent / base))
    return deep_merge(merged, data)


def parse_overrides(items: list[str]) -> dict:
    """Turn ["a.b=1", "c=[1, 2]"] into {"a": {"b": 1}, "c": [1, 2]}; values are parsed as YAML."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key=value")
        key, raw = item.split("=", 1)
        node = out
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def _build(cls: type, data: Any, prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        where = ", ".join(f"{prefix}{k}" for k in unknown)
        raise ConfigError(f"unknown config key(s): {where}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        dotted = f"{prefix}{name}"
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, dotted + ".")
            continue
        choices = fields[name].metadata.get("choices")
        if choices and value not in choices:
            raise ConfigError(f"{dotted}: {value!r} is not one of {list(choices)}")
        kwargs[name] = value
    return cls(**kwargs)


def build_dataclass(cls: type, data: dict) -> Any:
    """Validate a dict into any config dataclass (unknown keys and bad choices raise ConfigError)."""
    return _build(cls, data, "")


def build_config(data: dict) -> AppConfig:
    return build_dataclass(AppConfig, data)


def config_from_file(path: str | Path, overrides: list[str] | None = None) -> AppConfig:
    data = load_config(path)
    if overrides:
        data = deep_merge(data, parse_overrides(overrides))
    return build_config(data)


def config_to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)
