from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class StageTimer:
    """Collects wall-clock durations per named stage."""

    def __init__(self):
        self.samples = defaultdict(list)

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples[stage].append(time.perf_counter() - t0)

    def summary(self, stages) -> dict:
        out = {}
        for stage in stages:
            s = np.asarray(self.samples.get(stage, []), dtype=np.float64) * 1000.0
            out[stage] = {"calls": int(len(s)), "mean_ms": float(s.mean()) if len(s) else 0.0,
                          "p95_ms": float(np.percentile(s, 95)) if len(s) else 0.0}
        return out


class NullTimer:
    @contextmanager
    def measure(self, stage: str):
        yield
