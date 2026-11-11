"""Command-line entry point: `multicam-tracker <command> --help` for details."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from multicam_tracker.config import ConfigError, config_from_file

# Median foot-point reprojection error above which calibration is suspect. Convention errors are
# hundreds of pixels; the near-foot bias of real boxes alone is typically 5-20 px.
FAIL_PX = 50.0


def _cfg(args):
    return config_from_file(args.config, getattr(args, "set", None) or [])


def cmd_run(args) -> int:
    from multicam_tracker.pipeline.runner import run

    run(_cfg(args))
    return 0


def cmd_ablate(args) -> int:
    from multicam_tracker.tools.ablate import ablation_table

    table = ablation_table([config_from_file(c, args.set or []) for c in args.configs])
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
    return 0


def cmd_verify_calibration(args) -> int:
    from multicam_tracker.data.calibration_check import diagnose_conventions, ground_offsets, reprojection_errors
    from multicam_tracker.data.multiview import load_multiview

    seq = load_multiview(args.root, args.dataset)
    errors = reprojection_errors(seq, max_frames=args.frames)
    ok = True
    print(f"foot-point reprojection error (pixels) for {args.dataset} at {args.root}")
    for cam, e in sorted(errors.items()):
        if len(e) == 0:
            print(f"  C{cam + 1}: no annotated boxes")
            continue
        med, p90 = float(np.median(e)), float(np.percentile(e, 90))
        status = "OK" if med < FAIL_PX else "FAIL"
        ok &= status == "OK"
        print(f"  C{cam + 1}: median {med:7.2f}   p90 {p90:7.2f}   n={len(e):5d}   {status}")
    if ok:
        print("OK: calibration and ground-grid conventions are consistent with the annotations.")
        along, err = ground_offsets(seq, max_frames=args.frames)
        if len(along):
            offset = float(np.median(along))
            print(f"ground-plane error of box foot points: median {np.median(err):.3f} m, p90 {np.percentile(err, 90):.3f} m")
            print(f"true positions lie a median {offset:+.3f} m beyond the box bottom along the camera ray")
            print(f"  -> suggested setting: ground.foot_depth_offset_m: {max(0.0, offset):.2f}")
        return 0
    print("FAIL: large errors. Median error under alternative conventions (lower is better):")
    for name, err in diagnose_conventions(args.root, args.dataset):
        print(f"  {name:28s} {err:10.2f}")
    print("If an alternative is far better, the loader's convention in multicam_tracker/data/multiview.py needs changing;"
          " if only some cameras fail, check those calibration files.")
    return 1


def cmd_train_reid(args) -> int:
    from multicam_tracker.config import build_dataclass, deep_merge, load_config, parse_overrides
    from multicam_tracker.reid.train import ReIDTrainConfig, train

    data = deep_merge(load_config(args.config), parse_overrides(args.set or []))
    train(build_dataclass(ReIDTrainConfig, data))
    return 0


def cmd_eval_reid(args) -> int:
    from multicam_tracker.data.market1501 import load_market_style
    from multicam_tracker.device import resolve_device
    from multicam_tracker.reid.evaluate import evaluate_model
    from multicam_tracker.reid.model import load_checkpoint
    from multicam_tracker.reid.transforms import test_transform

    device = resolve_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)
    metrics = evaluate_model(model, load_market_style(args.data_root), test_transform(meta["input_size"]), device)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_build_reid_crops(args) -> int:
    from multicam_tracker.tools.reid_crops import build_reid_crops

    print(build_reid_crops(args.root, args.dataset, args.out, args.min_height,
                           validation_fraction=args.validation_fraction))
    return 0


def cmd_prepare_yolo(args) -> int:
    from multicam_tracker.tools.prepare_yolo import prepare_mot17, prepare_multiview

    if args.dataset == "mot17":
        counts = prepare_mot17(args.root, args.out, copy=args.copy)
    else:
        counts = prepare_multiview(args.root, args.dataset, args.out, copy=args.copy)
    print(f"{counts} images -> {Path(args.out) / 'data.yaml'}")
    return 0


def cmd_train_detector(args) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("detector training needs Ultralytics: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    YOLO(args.weights).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                             device=args.device, project=args.project, name=args.name, workers=args.workers,
                             patience=args.patience, cache=args.cache_images, amp=True)
    return 0


def cmd_export_onnx(args) -> int:
    from multicam_tracker.deploy.export import export_reid_onnx

    print(f"wrote {export_reid_onnx(args.checkpoint, args.out, args.opset)} (verified against PyTorch)")
    return 0


def cmd_trackeval(args) -> int:
    from multicam_tracker.tools.trackeval_bridge import run_trackeval

    cfg = _cfg(args)
    if cfg.dataset.kind != "mot17":
        print("error: trackeval scores MOT17 runs; multi-view metrics come from `multicam-tracker run`", file=sys.stderr)
        return 2
    split = "all" if cfg.dataset.split in ("train", "test") else cfg.dataset.split
    try:
        metrics = run_trackeval(cfg.dataset.root, Path(cfg.output.dir) / "mot", Path(cfg.output.dir) / "trackeval",
                                split, cfg.dataset.mot17_detector, cfg.dataset.sequences)
    except ImportError:
        print("error: install TrackEval: python -m pip install -r requirements.txt",
              file=sys.stderr)
        return 2
    print(json.dumps(metrics, indent=2))
    (Path(cfg.output.dir) / "metrics_trackeval.json").write_text(json.dumps(metrics, indent=2))
    return 0


def cmd_benchmark(args) -> int:
    from multicam_tracker.tools.benchmark import benchmark

    benchmark(_cfg(args), max_frames=args.frames)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multicam-tracker", description="Multi-camera people tracking with ground-plane fusion")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", required=True, help="YAML config (may use _base_)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. tracker.max_lost_s=2")

    p = sub.add_parser("run", help="track (and evaluate) one configured experiment")
    with_config(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ablate", help="run several configs and print a Markdown comparison table")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override applied to every config")
    p.add_argument("--out", help="also write the table to this file")
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("verify-calibration", help="check a multi-view dataset's calibration against its boxes")
    p.add_argument("--dataset", choices=["wildtrack", "multiviewx"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--frames", type=int, default=50, help="number of frames to sample")
    p.set_defaults(fn=cmd_verify_calibration)

    p = sub.add_parser("train-reid", help="train a re-ID model (Bag of Tricks recipe)")
    with_config(p)
    p.set_defaults(fn=cmd_train_reid)

    p = sub.add_parser("eval-reid", help="Rank-k / mAP of a re-ID checkpoint on a Market-style dataset")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_eval_reid)

    p = sub.add_parser("build-reid-crops", help="Market-style re-ID dataset from WILDTRACK / MultiviewX")
    p.add_argument("--dataset", choices=["wildtrack", "multiviewx"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-height", type=int, default=40)
    p.add_argument("--validation-fraction", type=float, default=0.2,
                   help="final fraction of training frames reserved for re-ID validation (default: 0.2)")
    p.set_defaults(fn=cmd_build_reid_crops)

    p = sub.add_parser("prepare-yolo", help="convert a dataset to YOLO detection format")
    p.add_argument("--dataset", choices=["mot17", "wildtrack", "multiviewx"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    p.set_defaults(fn=cmd_prepare_yolo)

    p = sub.add_parser("train-detector", help="fine-tune an Ultralytics YOLO person detector")
    p.add_argument("--data", required=True, help="data.yaml from prepare-yolo")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--cache-images", action="store_true", help="cache decoded images in RAM")
    p.add_argument("--project", default="runs/detector")
    p.add_argument("--name", default="finetune")
    p.set_defaults(fn=cmd_train_detector)

    p = sub.add_parser("export-onnx", help="export a re-ID checkpoint to ONNX and verify it")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(fn=cmd_export_onnx)

    p = sub.add_parser("trackeval", help="score a finished MOT17 run with the official TrackEval")
    with_config(p)
    p.set_defaults(fn=cmd_trackeval)

    p = sub.add_parser("benchmark", help="per-stage latency on the first N frames")
    with_config(p)
    p.add_argument("--frames", type=int, default=100)
    p.set_defaults(fn=cmd_benchmark)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
