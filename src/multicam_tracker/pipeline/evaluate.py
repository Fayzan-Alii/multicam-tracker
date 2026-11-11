"""Glue between pipeline outputs and the metric implementations."""

from __future__ import annotations

import numpy as np

from multicam_tracker.eval.ground_plane import detection_metrics
from multicam_tracker.eval.mot_metrics import combine, compute_all, filter_mot17_frame, sequence_from_boxes, sequence_from_points

GROUND_MATCH_EPS = 1e-6  # ground-plane similarity is 1 - d / radius: any d < radius is a match


def _tracks_to_arrays(tracks):
    if not tracks:
        return np.zeros(0, int), np.zeros((0, 4))
    return np.array([t[0] for t in tracks], dtype=int), np.array([t[1] for t in tracks], dtype=np.float64)


def evaluate_boxes(gt_fn, results: dict, frames, threshold: float, mot17_filter: bool = False) -> dict:
    """gt_fn(frame) -> GTFrame; results: frame -> [(id, box, score)]."""
    gt_map, tr_map = {}, {}
    for f in frames:
        gt = gt_fn(f)
        tr_ids, tr_boxes = _tracks_to_arrays(results.get(f, []))
        keep_gt = np.ones(len(gt.ids), bool)
        keep_tr = np.ones(len(tr_ids), bool)
        if mot17_filter and gt.classes is not None:
            keep_gt, keep_tr = filter_mot17_frame(gt.boxes, gt.classes, gt.consider, tr_boxes)
        gt_map[f] = (gt.ids[keep_gt], gt.boxes[keep_gt])
        tr_map[f] = (tr_ids[keep_tr], tr_boxes[keep_tr])
    return compute_all(sequence_from_boxes(gt_map, tr_map, frames), threshold)


def evaluate_multiview(seq, local_results: dict, global_results: dict | None, cfg_eval) -> dict:
    cams = sorted(local_results)
    per_cam = {c: evaluate_boxes(lambda f, c=c: seq.gt_view(c, f), local_results[c], seq.frames, cfg_eval.iou_threshold)
               for c in cams}
    metrics = {"local": combine(list(per_cam.values())), "per_camera": {f"C{c + 1}": m for c, m in per_cam.items()}}
    if global_results is None:
        return metrics

    gt_pts, tr_pts, gt_xy, det_xy = {}, {}, {}, {}
    for f in seq.frames:
        ids, xy = seq.gt_world(f)
        gt_pts[f], gt_xy[f] = (ids, xy), xy
        g = global_results.get(f, [])
        tr_pts[f] = (np.array([o.global_id for o in g], dtype=int),
                     np.array([o.world_xy for o in g], dtype=np.float64).reshape(-1, 2))
        det_xy[f] = tr_pts[f][1]
    metrics["ground_detection"] = detection_metrics(gt_xy, det_xy, seq.frames, cfg_eval.ground_det_radius_m)
    metrics["ground_tracking"] = compute_all(
        sequence_from_points(gt_pts, tr_pts, seq.frames, cfg_eval.ground_track_radius_m), GROUND_MATCH_EPS)

    # Cross-camera identity: every (frame, camera) is a timestep, GT ids are the dataset's global
    # person ids and tracker ids are fused global ids, so one person keeping one id across views
    # is exactly what IDF1 rewards.
    keys, gt_boxes, tr_boxes = [], {}, {}
    for f in seq.frames:
        for c in cams:
            keys.append((f, c))
            gt = seq.gt_view(c, f)
            gt_boxes[(f, c)] = (gt.ids, gt.boxes)
            in_view = [o for o in global_results.get(f, []) if c in o.boxes]
            tr_boxes[(f, c)] = (np.array([o.global_id for o in in_view], dtype=int),
                                np.array([o.boxes[c] for o in in_view], dtype=np.float64).reshape(-1, 4))
    metrics["cross_camera"] = compute_all(sequence_from_boxes(gt_boxes, tr_boxes, keys), cfg_eval.iou_threshold)
    return metrics
