"""Run a configured experiment end to end: data -> detect -> (embed) -> track -> (fuse) -> evaluate."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from multicam_tracker.config import AppConfig, ConfigError, config_to_dict
from multicam_tracker.data.mot17 import load_mot17, write_mot_results
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.eval.mot_metrics import combine
from multicam_tracker.fusion.associator import CrossCameraFusion
from multicam_tracker.pipeline.evaluate import evaluate_boxes, evaluate_multiview
from multicam_tracker.pipeline.factory import build_detector, build_embedder
from multicam_tracker.pipeline.ground import project_to_ground
from multicam_tracker.timing import NullTimer
from multicam_tracker.types import Detections
from multicam_tracker.tracking.tracker import SingleCameraTracker, make_motion

SUMMARY_KEYS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "MODA", "MODP", "precision", "recall")


def _jsonable(metrics):
    if isinstance(metrics, dict):
        return {k: _jsonable(v) for k, v in metrics.items() if not k.endswith("_arr")
                and not (isinstance(v, np.ndarray) and v.ndim > 0)}
    if isinstance(metrics, (np.floating, np.integer)):
        return metrics.item()
    return metrics


def _embed(embedder, ctx, dets, timer):
    if embedder is not None and len(dets):
        with timer.measure("embed"):
            dets.embeddings = embedder.embed(ctx, dets.boxes)
    return dets


def run_single_camera(cfg: AppConfig, sequences, detector, embedder, out_dir: Path, timer=None,
                      max_frames: int = 0) -> dict:
    timer = timer or NullTimer()
    metrics = {}
    for seq in sequences:
        if cfg.tracker.motion.space == "ground" and seq.camera is None:
            raise ConfigError(f"tracker.motion.space=ground needs camera calibration and {seq.name} has none; "
                              "use tracker.motion.space=image for MOT17")
        tracker = SingleCameraTracker(cfg.tracker, make_motion(cfg.tracker.motion, seq.fps))
        results = {}
        frames = seq.frames[:max_frames] if max_frames else seq.frames
        renderer = None
        if cfg.output.save_video:
            from multicam_tracker.pipeline.render import SingleCameraRenderer

            renderer = SingleCameraRenderer(cfg.output, out_dir, seq.name, seq.fps)
        for f in tqdm(frames, desc=seq.name, leave=False):
            public = seq.public.get(f) if seq.public is not None else None
            ctx = FrameContext(seq.name, 0, f, partial(seq.image, f), gt=seq.gt(f) if seq.has_gt else None,
                               public=public)
            with timer.measure("detect"):
                dets = detector.detect(ctx)
            dets = _embed(embedder, ctx, dets, timer)
            with timer.measure("track"):
                tracks = tracker.update(dets, seq.timestamp(f))
            results[f] = [(t.track_id, t.box, t.score) for t in tracks]
            if renderer and not renderer.done():
                renderer.add(ctx.image(), tracks, f)
        if renderer:
            renderer.close()
        write_mot_results(out_dir / "mot" / f"{seq.name}.txt", results)
        if cfg.eval.enabled and seq.has_gt:
            metrics[seq.name] = evaluate_boxes(seq.gt, results, frames, cfg.eval.iou_threshold, mot17_filter=True)
    if metrics:
        metrics["COMBINED"] = combine(list(metrics.values()))
    return metrics


def _write_global_csv(path: Path, global_results: dict) -> None:
    lines = ["frame,global_id,x,y,cov_xx,cov_xy,cov_yy,cameras\n"]
    for f in sorted(global_results):
        for o in global_results[f]:
            c = o.world_cov
            cams = ";".join(f"C{k + 1}" for k in o.cameras)
            lines.append(f"{f},{o.global_id},{o.world_xy[0]:.4f},{o.world_xy[1]:.4f},"
                         f"{c[0, 0]:.6f},{c[0, 1]:.6f},{c[1, 1]:.6f},{cams}\n")
    path.write_text("".join(lines))


def run_multi_camera(cfg: AppConfig, seq, detector, embedder, out_dir: Path, timer=None,
                     max_frames: int = 0, log=print) -> dict:
    timer = timer or NullTimer()
    if max_frames:
        seq.frames = seq.frames[:max_frames]
    roi = tuple(cfg.ground.roi) if cfg.ground.roi else tuple(seq.roi)
    cams = sorted(seq.cameras)
    trackers = {c: SingleCameraTracker(cfg.tracker, make_motion(cfg.tracker.motion, seq.fps), c) for c in cams}
    fusion = CrossCameraFusion(cfg.fusion) if cfg.fusion.enabled else None
    renderer = None
    if cfg.output.save_video:
        from multicam_tracker.pipeline.render import DemoRenderer

        renderer = DemoRenderer(cfg.output, seq, roi, out_dir)
    local_results = {c: {} for c in cams}
    global_results = {} if fusion else None
    rejected = {c: 0 for c in cams}
    unreadable = {c: 0 for c in cams}

    for f in tqdm(seq.frames, desc=seq.name, leave=False):
        t = seq.timestamp(f)
        frame_tracks, local_now, contexts = [], {}, {}
        for c in cams:
            ctx = FrameContext(seq.name, c, f, partial(seq.image, c, f), gt=seq.gt_view(c, f) if seq.has_gt else None)
            contexts[c] = ctx
            try:
                with timer.measure("detect"):
                    dets = detector.detect(ctx)
                dets = _embed(embedder, ctx, dets, timer)
            except FileNotFoundError as exc:
                # A missing or corrupt frame: no detections for this camera, so its tracks coast.
                if unreadable[c] == 0:
                    log(f"warning: {exc} (tracks coast; further failures are only counted)")
                unreadable[c] += 1
                dets = Detections.empty()
                ctx._image = np.zeros((*seq.image_size, 3), np.uint8)
            with timer.measure("ground"):
                dets = project_to_ground(dets, seq.cameras[c], cfg.ground, roi)
            rejected[c] += int((~dets.ground_valid).sum())
            with timer.measure("track"):
                tracks = trackers[c].update(dets, t)
            local_now[c] = tracks
            local_results[c][f] = [(tr.track_id, tr.box, tr.score) for tr in tracks]
            frame_tracks += tracks
        with timer.measure("fuse"):
            fused = fusion.update(frame_tracks, t) if fusion else None
        if fusion:
            global_results[f] = fused
        if renderer and not renderer.done():
            renderer.add(f, {c: contexts[c].image() for c in cams}, local_now, fused)
    if renderer:
        renderer.close()

    for c in cams:
        write_mot_results(out_dir / "mot" / f"{seq.name}_C{c + 1}.txt", local_results[c])
    if fusion:
        _write_global_csv(out_dir / "global_tracks.csv", global_results)
    diagnostics = {
        "rejected_ground_points": {f"C{c + 1}": n for c, n in rejected.items()},
        "unreadable_frames": {f"C{c + 1}": n for c, n in unreadable.items() if n},
    }
    if not (cfg.eval.enabled and seq.has_gt):
        return {"diagnostics": diagnostics}
    return {**evaluate_multiview(seq, local_results, global_results, cfg.eval), "diagnostics": diagnostics}


def load_multiview_dataset(cfg: AppConfig):
    d = cfg.dataset
    if d.kind == "synthetic":
        from multicam_tracker.data.synthetic import make_synthetic

        return make_synthetic(d.synthetic)
    from multicam_tracker.data.multiview import load_multiview

    return load_multiview(d.root, d.kind, split=d.split, cameras=d.cameras, fps=d.fps)


def run(cfg: AppConfig, log=print, timer=None, max_frames: int = 0) -> dict:
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
    detector, embedder = build_detector(cfg), build_embedder(cfg)
    t0 = time.time()
    if cfg.dataset.kind == "mot17":
        split = "all" if cfg.dataset.split in ("train", "test") else cfg.dataset.split
        sequences = load_mot17(cfg.dataset.root, cfg.dataset.mot17_subset, cfg.dataset.mot17_detector,
                               cfg.dataset.sequences, split)
        if not sequences:
            raise ConfigError(f"no MOT17 sequences matched under {cfg.dataset.root}")
        metrics = run_single_camera(cfg, sequences, detector, embedder, out_dir, timer, max_frames)
    else:
        metrics = run_multi_camera(cfg, load_multiview_dataset(cfg), detector, embedder, out_dir, timer, max_frames, log)
    for stage in (detector, embedder):
        if hasattr(stage, "flush"):
            stage.flush()
    metrics = _jsonable(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"[{cfg.name}] finished in {time.time() - t0:.1f}s -> {out_dir}")
    for section, values in metrics.items():
        if isinstance(values, dict) and any(k in values for k in SUMMARY_KEYS):
            shown = "  ".join(f"{k} {values[k]:.3f}" if isinstance(values[k], float) else f"{k} {values[k]}"
                              for k in SUMMARY_KEYS if k in values)
            log(f"  {section:18s} {shown}")
    diag = metrics.get("diagnostics")
    if diag:
        total = sum(diag["rejected_ground_points"].values())
        log(f"  detections rejected (foot point above horizon or outside ROI): {total}"
            f"  {diag['rejected_ground_points']}")
        if diag["unreadable_frames"]:
            log(f"  unreadable frames: {diag['unreadable_frames']}")
    return {"metrics": metrics, "out_dir": out_dir}
