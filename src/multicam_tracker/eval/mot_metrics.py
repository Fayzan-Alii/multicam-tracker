"""CLEAR MOT, Identity (IDF1) and HOTA, following TrackEval's definitions.

Metrics are computed from per-timestep similarity matrices, so the same code scores image
boxes (IoU similarity) and ground-plane points (1 - distance / radius). Implemented here so
ablations run offline in seconds; publish numbers from the official TrackEval as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_tracker.tracking.costs import iou_matrix

EPS = np.finfo(float).eps
ALPHAS = np.arange(0.05, 0.99, 0.05)
MOT17_PEDESTRIAN = 1
MOT17_DISTRACTORS = (2, 7, 8, 12)  # person on vehicle, static person, distractor, reflection


@dataclass
class SequenceData:
    gt_ids: list[np.ndarray]
    tr_ids: list[np.ndarray]
    similarity: list[np.ndarray]


def _remap(ids: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    unique = np.unique(np.concatenate([np.asarray(i, dtype=np.int64) for i in ids])) if ids else np.zeros(0, int)
    lookup = {int(v): k for k, v in enumerate(unique)}
    return [np.array([lookup[int(v)] for v in i], dtype=int) for i in ids], len(unique)


def clear_metrics(seq: SequenceData, threshold: float = 0.5) -> dict:
    gt_ids, n_gt = _remap(seq.gt_ids)
    tr_ids, n_tr = _remap(seq.tr_ids)
    tp = fn = fp = idsw = 0
    motp_sum = 0.0
    gt_count = np.zeros(n_gt)
    gt_matched = np.zeros(n_gt)
    frag_count = np.zeros(n_gt)
    prev_tr = np.full(n_gt, np.nan)  # last tracker id ever matched to each gt (for IDSW)
    # Tracker id matched at the last non-empty timestep. Like TrackEval, timesteps with no GT
    # or no tracker boxes leave it untouched.
    prev_step_tr = np.full(n_gt, np.nan)

    for g, t, sim in zip(gt_ids, tr_ids, seq.similarity):
        if len(g) == 0:
            fp += len(t)
            continue
        gt_count[g] += 1
        if len(t) == 0:
            fn += len(g)
            continue
        score = 1000.0 * (t[None, :] == prev_step_tr[g][:, None]) + sim
        score[sim < threshold - EPS] = 0
        rows, cols = linear_sum_assignment(-score)
        ok = score[rows, cols] > EPS
        rows, cols = rows[ok], cols[ok]
        mg, mt = g[rows], t[cols]
        prev = prev_tr[mg]
        idsw += int(np.sum(~np.isnan(prev) & (mt != prev)))
        gt_matched[mg] += 1
        frag_count[mg[np.isnan(prev_step_tr[mg])]] += 1
        prev_tr[mg] = mt
        prev_step_tr[:] = np.nan
        prev_step_tr[mg] = mt
        tp += len(mg)
        fn += len(g) - len(mg)
        fp += len(t) - len(mg)
        motp_sum += float(sim[rows, cols].sum())

    present = gt_count > 0
    ratio = gt_matched[present] / gt_count[present]
    mt_ = int(np.sum(ratio > 0.8))
    pt = int(np.sum((ratio >= 0.2) & (ratio <= 0.8)))
    out = {
        "CLR_TP": tp, "CLR_FN": fn, "CLR_FP": fp, "IDSW": idsw, "MOTP_sum": motp_sum,
        "MT": mt_, "PT": pt, "ML": int(present.sum()) - mt_ - pt,
        "Frag": int(np.sum(frag_count[frag_count > 0] - 1)),
    }
    return _finish_clear(out)


def _finish_clear(r: dict) -> dict:
    n = r["CLR_TP"] + r["CLR_FN"]
    r["MOTA"] = (r["CLR_TP"] - r["CLR_FP"] - r["IDSW"]) / max(1, n)
    r["MOTP"] = r["MOTP_sum"] / max(1, r["CLR_TP"])
    return r


def identity_metrics(seq: SequenceData, threshold: float = 0.5) -> dict:
    gt_ids, n_gt = _remap(seq.gt_ids)
    tr_ids, n_tr = _remap(seq.tr_ids)
    potential = np.zeros((n_gt, n_tr))
    gt_count = np.zeros(n_gt)
    tr_count = np.zeros(n_tr)
    for g, t, sim in zip(gt_ids, tr_ids, seq.similarity):
        if len(g) and len(t):
            gi, ti = np.nonzero(sim >= threshold - EPS)
            potential[g[gi], t[ti]] += 1
        gt_count[g] += 1
        tr_count[t] += 1
    # Min-cost bipartite matching over whole trajectories, with dummies for unmatched ids.
    n = n_gt + n_tr
    fp_mat = np.zeros((n, n))
    fn_mat = np.zeros((n, n))
    fp_mat[n_gt:, :n_tr] = 1e10
    fn_mat[:n_gt, n_tr:] = 1e10
    for i in range(n_gt):
        fn_mat[i, :n_tr] = gt_count[i]
        fn_mat[i, n_tr + i] = gt_count[i]
    for j in range(n_tr):
        fp_mat[:n_gt, j] = tr_count[j]
        fp_mat[j + n_gt, j] = tr_count[j]
    fn_mat[:n_gt, :n_tr] -= potential
    fp_mat[:n_gt, :n_tr] -= potential
    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    idfn = int(fn_mat[rows, cols].sum())
    idfp = int(fp_mat[rows, cols].sum())
    idtp = int(gt_count.sum() - idfn)
    return _finish_identity({"IDTP": idtp, "IDFN": idfn, "IDFP": idfp})


def _finish_identity(r: dict) -> dict:
    tp, fn, fp = r["IDTP"], r["IDFN"], r["IDFP"]
    r["IDF1"] = tp / max(1, tp + 0.5 * fp + 0.5 * fn)
    r["IDR"] = tp / max(1, tp + fn)
    r["IDP"] = tp / max(1, tp + fp)
    return r


def hota_metrics(seq: SequenceData) -> dict:
    gt_ids, n_gt = _remap(seq.gt_ids)
    tr_ids, n_tr = _remap(seq.tr_ids)
    A = len(ALPHAS)
    tp, fn, fp, loc = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    potential = np.zeros((n_gt, n_tr))
    gt_count = np.zeros((n_gt, 1))
    tr_count = np.zeros((1, n_tr))
    for g, t, sim in zip(gt_ids, tr_ids, seq.similarity):
        if len(g) and len(t):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            mask = denom > EPS
            sim_iou[mask] = sim[mask] / denom[mask]
            potential[g[:, None], t[None, :]] += sim_iou
        gt_count[g] += 1
        tr_count[0, t] += 1
    global_score = potential / np.maximum(gt_count + tr_count - potential, EPS)
    matches = np.zeros((A, n_gt, n_tr))
    for g, t, sim in zip(gt_ids, tr_ids, seq.similarity):
        if len(g) == 0:
            fp += len(t)
            continue
        if len(t) == 0:
            fn += len(g)
            continue
        rows, cols = linear_sum_assignment(-(global_score[g[:, None], t[None, :]] * sim))
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            tp[a] += len(r)
            fn[a] += len(g) - len(r)
            fp[a] += len(t) - len(r)
            if len(r):
                loc[a] += sim[r, c].sum()
                matches[a, g[r], t[c]] += 1
    ass_a, ass_re, ass_pr = np.zeros(A), np.zeros(A), np.zeros(A)
    for a in range(A):
        mc = matches[a]
        ass_a[a] = np.sum(mc * (mc / np.maximum(1, gt_count + tr_count - mc))) / max(1, tp[a])
        ass_re[a] = np.sum(mc * (mc / np.maximum(1, gt_count))) / max(1, tp[a])
        ass_pr[a] = np.sum(mc * (mc / np.maximum(1, tr_count))) / max(1, tp[a])
    return _finish_hota({
        "HOTA_TP": tp, "HOTA_FN": fn, "HOTA_FP": fp, "AssA_arr": ass_a, "AssRe_arr": ass_re,
        "AssPr_arr": ass_pr, "LocA_arr": np.maximum(1e-10, loc) / np.maximum(1e-10, tp),
    })


def _finish_hota(r: dict) -> dict:
    tp, fn, fp = r["HOTA_TP"], r["HOTA_FN"], r["HOTA_FP"]
    det_a = tp / np.maximum(1, tp + fn + fp)
    r["HOTA_arr"] = np.sqrt(det_a * r["AssA_arr"])
    r["DetA_arr"] = det_a
    r["DetRe_arr"] = tp / np.maximum(1, tp + fn)
    r["DetPr_arr"] = tp / np.maximum(1, tp + fp)
    for key in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr"):
        r[key] = float(np.mean(r[f"{key}_arr"]))
    return r


def compute_all(seq: SequenceData, threshold: float = 0.5) -> dict:
    return {**clear_metrics(seq, threshold), **identity_metrics(seq, threshold), **hota_metrics(seq)}


def combine(results: list[dict]) -> dict:
    """Pool per-sequence results the way TrackEval's COMBINED row does."""
    out: dict = {}
    for key in ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "MOTP_sum",
                "IDTP", "IDFN", "IDFP"):
        out[key] = sum(r[key] for r in results)
    for key in ("HOTA_TP", "HOTA_FN", "HOTA_FP"):
        out[key] = np.sum([r[key] for r in results], axis=0)
    tp_total = np.maximum(1e-10, out["HOTA_TP"])
    for key in ("AssA_arr", "AssRe_arr", "AssPr_arr", "LocA_arr"):
        out[key] = np.sum([r[key] * r["HOTA_TP"] for r in results], axis=0) / tp_total
    return _finish_hota(_finish_identity(_finish_clear(out)))


def sequence_from_boxes(gt: dict, tr: dict, frames) -> SequenceData:
    """gt/tr: frame -> (ids (N,), boxes (N, 4)). Missing frames count as empty."""
    return _sequence(gt, tr, frames, lambda a, b: iou_matrix(a, b), 4)


def sequence_from_points(gt: dict, tr: dict, frames, radius: float) -> SequenceData:
    """gt/tr: frame -> (ids (N,), points (N, 2)); similarity = max(0, 1 - distance / radius)."""

    def sim(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return np.clip(1.0 - d / radius, 0.0, None)

    return _sequence(gt, tr, frames, sim, 2)


def _sequence(gt, tr, frames, sim_fn, dim) -> SequenceData:
    empty = (np.zeros(0, int), np.zeros((0, dim)))
    out = SequenceData([], [], [])
    for f in frames:
        g_ids, g_x = gt.get(f, empty)
        t_ids, t_x = tr.get(f, empty)
        out.gt_ids.append(np.asarray(g_ids, dtype=int))
        out.tr_ids.append(np.asarray(t_ids, dtype=int))
        if len(g_ids) and len(t_ids):
            out.similarity.append(sim_fn(np.asarray(g_x, float), np.asarray(t_x, float)))
        else:
            out.similarity.append(np.zeros((len(g_ids), len(t_ids))))
    return out


def filter_mot17_frame(gt_boxes, gt_classes, gt_consider, tr_boxes) -> tuple[np.ndarray, np.ndarray]:
    """TrackEval's MOT17 preprocessing for one frame. Tracker boxes that match (IoU >= 0.5)
    a distractor-class GT box are dropped rather than counted as false positives; only
    pedestrian GT with the consider flag set is scored. Returns boolean keep masks."""
    keep_tr = np.ones(len(tr_boxes), dtype=bool)
    if len(gt_boxes) and len(tr_boxes):
        sim = iou_matrix(gt_boxes, tr_boxes)
        sim[sim < 0.5 - EPS] = 0
        rows, cols = linear_sum_assignment(-sim)
        ok = sim[rows, cols] > EPS
        rows, cols = rows[ok], cols[ok]
        keep_tr[cols[np.isin(np.asarray(gt_classes)[rows], MOT17_DISTRACTORS)]] = False
    keep_gt = (np.asarray(gt_consider) != 0) & (np.asarray(gt_classes) == MOT17_PEDESTRIAN)
    return keep_gt, keep_tr
