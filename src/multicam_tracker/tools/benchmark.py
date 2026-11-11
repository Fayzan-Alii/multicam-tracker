"""Per-stage latency of the pipeline on the first N frames of a configured dataset."""

from __future__ import annotations

import dataclasses
import time

from multicam_tracker.pipeline.runner import run
from multicam_tracker.timing import StageTimer

STAGES = ("detect", "embed", "ground", "track", "fuse")


def benchmark(cfg, max_frames: int = 100, log=print) -> dict:
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, enabled=False),
                              output=dataclasses.replace(cfg.output, save_video=False))
    timer = StageTimer()
    t0 = time.perf_counter()
    run(cfg, log=lambda *a: None, timer=timer, max_frames=max_frames)
    total = time.perf_counter() - t0
    out = timer.summary(STAGES)
    frames = max(1, max_frames)
    out["total"] = {"calls": frames, "mean_ms": 1000.0 * total / frames, "p95_ms": 0.0}
    log(f"{'stage':8s} {'calls':>6s} {'mean ms':>9s} {'p95 ms':>9s}")
    for stage, v in out.items():
        log(f"{stage:8s} {v['calls']:6d} {v['mean_ms']:9.2f} {v['p95_ms']:9.2f}")
    log(f"throughput ~ {1000.0 / max(out['total']['mean_ms'], 1e-9):.1f} frames/s (all cameras, incl. data loading)")
    return out
