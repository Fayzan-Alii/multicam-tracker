"""Parity with the official TrackEval implementation on randomised sequences.

Skipped unless TrackEval is installed:  python -m pip install -r requirements.txt
"""

import numpy as np
import pytest

import scipy.optimize  # noqa: F401  (import before restoring the aliases TrackEval needs)

if not hasattr(np, "float"):
    np.float = float
    np.int = int
trackeval = pytest.importorskip("trackeval")

from multicam_tracker.eval.mot_metrics import SequenceData, combine, compute_all  # noqa: E402


def random_sequence(rng):
    n_gt = int(rng.integers(1, 7))
    tr_of = {g: g + 100 for g in range(n_gt)}
    nxt = 1000
    seq = SequenceData([], [], [])
    for t in range(int(rng.integers(5, 30))):
        g = np.sort(rng.choice(n_gt, size=int(rng.integers(0, n_gt + 1)), replace=False)).astype(int)
        tids = []
        for gi in g:
            if rng.random() < 0.1:
                tr_of[gi] = nxt
                nxt += 1
            if rng.random() < 0.85:
                tids.append(tr_of[gi])
        tids += list(range(5000 + 10 * t, 5000 + 10 * t + int(rng.integers(0, 3))))
        tids = np.array(tids, dtype=int)
        sim = rng.uniform(0, 0.3, size=(len(g), len(tids)))
        for i, gi in enumerate(g):
            sim[i, tids == tr_of[gi]] = rng.uniform(0.3, 1.0)
        seq.gt_ids.append(g)
        seq.tr_ids.append(tids)
        seq.similarity.append(sim)
    return seq


def to_trackeval(seq):
    data = {"similarity_scores": seq.similarity, "num_timesteps": len(seq.gt_ids)}
    for ours, theirs, count in (("gt_ids", "gt_ids", "num_gt_ids"), ("tr_ids", "tracker_ids", "num_tracker_ids")):
        ids = getattr(seq, ours)
        uniq = np.unique(np.concatenate(ids))
        lut = {v: k for k, v in enumerate(uniq)}
        data[theirs] = [np.array([lut[v] for v in x], dtype=int) for x in ids]
        data[count] = len(uniq)
    data["num_gt_dets"] = sum(map(len, seq.gt_ids))
    data["num_tracker_dets"] = sum(map(len, seq.tr_ids))
    return data


def test_matches_trackeval_per_sequence_and_combined():
    rng = np.random.default_rng(7)
    metrics = {"HOTA": trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               "CLEAR": trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               "Identity": trackeval.metrics.Identity({"PRINT_CONFIG": False})}
    keys = {"HOTA": ["HOTA", "DetA", "AssA", "LocA"], "CLEAR": ["MOTA", "MOTP", "IDSW", "Frag", "MT", "ML"],
            "Identity": ["IDF1", "IDTP"]}
    ours_all, ref_all = [], {name: [] for name in metrics}
    for _ in range(50):
        seq = random_sequence(rng)
        ours = compute_all(seq)
        ours_all.append(ours)
        data = to_trackeval(seq)
        for name, metric in metrics.items():
            ref = metric.eval_sequence(data)
            ref_all[name].append(ref)
            for key in keys[name]:
                assert float(np.mean(ref[key])) == pytest.approx(float(ours[key]), abs=1e-9), key
    comb = combine(ours_all)
    for name, metric in metrics.items():
        ref = metric.combine_sequences(dict(enumerate(ref_all[name])))
        for key in keys[name]:
            assert float(np.mean(ref[key])) == pytest.approx(float(comb[key]), abs=1e-9), key
