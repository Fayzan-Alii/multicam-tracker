import numpy as np
import pytest

from multicam_tracker.eval.ground_plane import detection_metrics
from multicam_tracker.eval.mot_metrics import (
    SequenceData,
    clear_metrics,
    filter_mot17_frame,
    combine,
    compute_all,
    hota_metrics,
    identity_metrics,
    sequence_from_boxes,
    sequence_from_points,
)


def seq(gt, tr, sims):
    """gt/tr: per-frame id lists; sims: per-frame (n_gt, n_tr) similarity lists."""
    return SequenceData(
        gt_ids=[np.array(g, dtype=int) for g in gt],
        tr_ids=[np.array(t, dtype=int) for t in tr],
        similarity=[np.array(s, dtype=float).reshape(len(g), len(t)) for g, t, s in zip(gt, tr, sims)],
    )


def perfect(n_frames=4):
    return seq([[1, 2]] * n_frames, [[7, 8]] * n_frames, [[[1.0, 0.0], [0.0, 1.0]]] * n_frames)


def test_perfect_tracking_scores_one_everywhere():
    m = compute_all(perfect())
    for key in ("MOTA", "MOTP", "IDF1", "HOTA", "DetA", "AssA"):
        assert m[key] == pytest.approx(1.0), key
    assert m["IDSW"] == 0 and m["MT"] == 2


def test_id_switch_is_counted_once():
    # One person over 4 frames; the tracker changes ID after frame 2.
    s = seq([[1]] * 4, [[5], [5], [6], [6]], [[[1.0]]] * 4)
    m = clear_metrics(s)
    assert m["CLR_TP"] == 4 and m["IDSW"] == 1
    assert m["MOTA"] == pytest.approx(0.75)


def test_split_trajectory_identity_and_hota():
    s = seq([[1]] * 4, [[5], [5], [6], [6]], [[[1.0]]] * 4)
    idm = identity_metrics(s)
    # Best 1:1 pairing covers two frames: IDTP=2, IDFN=2, IDFP=2 -> IDF1 = 2 / (2 + 1 + 1).
    assert (idm["IDTP"], idm["IDFN"], idm["IDFP"]) == (2, 2, 2)
    assert idm["IDF1"] == pytest.approx(0.5)
    h = hota_metrics(s)
    assert h["DetA"] == pytest.approx(1.0)
    assert h["AssA"] == pytest.approx(0.5)
    assert h["HOTA"] == pytest.approx(np.sqrt(0.5))


def test_misses_and_false_positives():
    # Frame 0: gt 1 matched; frame 1: gt 1 missed and tracker 9 is a FP (no overlap).
    s = seq([[1], [1]], [[5], [9]], [[[0.9]], [[0.0]]])
    m = clear_metrics(s)
    assert (m["CLR_TP"], m["CLR_FN"], m["CLR_FP"]) == (1, 1, 1)
    assert m["MOTA"] == pytest.approx(0.0)
    assert m["MOTP"] == pytest.approx(0.9)


def test_match_below_threshold_is_not_a_true_positive():
    m = clear_metrics(seq([[1]], [[5]], [[[0.4]]]), threshold=0.5)
    assert (m["CLR_TP"], m["CLR_FN"], m["CLR_FP"]) == (0, 1, 1)


def test_clear_prefers_continuing_previous_match():
    # In frame 1 the swapped pairing has higher total IoU (0.95 + 0.55 > 0.6 + 0.6), but both
    # existing pairs are still valid matches, so CLEAR keeps them and counts no switch.
    s = seq([[1, 2], [1, 2]], [[5, 6], [5, 6]], [[[0.9, 0.0], [0.0, 0.9]], [[0.6, 0.95], [0.55, 0.6]]])
    m = clear_metrics(s)
    assert m["IDSW"] == 0 and m["CLR_TP"] == 4


def test_hota_localisation_sweeps_alpha():
    # Constant IoU 0.6: matched for alpha <= 0.6 (0.05..0.60 = 12 of 19 thresholds).
    s = seq([[1]] * 3, [[5]] * 3, [[[0.6]]] * 3)
    h = hota_metrics(s)
    assert h["DetA"] == pytest.approx(12 / 19)
    # TrackEval defines LocA as 1 at thresholds with no true positives.
    assert h["LocA"] == pytest.approx((12 * 0.6 + 7 * 1.0) / 19)


def test_empty_frames_count_misses_and_false_positives():
    s = seq([[1], []], [[], [5]], [np.zeros((1, 0)), np.zeros((0, 1))])
    m = compute_all(s)
    assert m["CLR_FN"] == 1 and m["CLR_FP"] == 1
    assert m["HOTA"] == pytest.approx(0.0)


def test_combine_pools_counts_across_sequences():
    a = compute_all(seq([[1]] * 4, [[5], [5], [6], [6]], [[[1.0]]] * 4))
    b = compute_all(perfect(4))
    both = combine([a, b])
    assert both["CLR_TP"] == 12 and both["IDSW"] == 1
    assert both["MOTA"] == pytest.approx((12 - 1) / 12)
    assert both["IDF1"] == pytest.approx(2 * (2 + 8) / (2 * (2 + 8) + 2 + 2))


def test_sequence_from_boxes_builds_iou_similarity():
    gt = {0: (np.array([1]), np.array([[0.0, 0, 10, 10]]))}
    tr = {0: (np.array([3]), np.array([[5.0, 0, 15, 10]])), 1: (np.array([3]), np.array([[0.0, 0, 1, 1]]))}
    s = sequence_from_boxes(gt, tr, frames=[0, 1])
    np.testing.assert_allclose(s.similarity[0], [[1 / 3]])
    assert s.gt_ids[1].size == 0 and s.similarity[1].shape == (0, 1)


def test_sequence_from_points_uses_distance_radius():
    gt = {0: (np.array([1, 2]), np.array([[0.0, 0.0], [10.0, 0.0]]))}
    tr = {0: (np.array([4]), np.array([[0.25, 0.0]]))}
    s = sequence_from_points(gt, tr, frames=[0], radius=1.0)
    np.testing.assert_allclose(s.similarity[0], [[0.75], [0.0]])


def test_ground_detection_metrics():
    gt = {0: np.array([[0.0, 0.0], [5.0, 5.0]]), 1: np.array([[1.0, 1.0]])}
    det = {0: np.array([[0.1, 0.0], [9.0, 9.0]]), 1: np.zeros((0, 2))}
    m = detection_metrics(gt, det, frames=[0, 1], radius=0.5)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 1, 2)
    assert m["MODA"] == pytest.approx(1 - 3 / 3)
    assert m["MODP"] == pytest.approx(1 - 0.1 / 0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(1 / 3)


def test_mot17_filter_drops_distractor_matches_and_ignored_gt():
    gt_ids = np.array([1, 2, 3])
    gt_boxes = np.array([[0.0, 0, 10, 10], [50.0, 0, 60, 10], [100.0, 0, 110, 10]])
    gt_classes = np.array([1, 8, 1])  # 8 = distractor
    gt_consider = np.array([1, 1, 0])
    tr_ids = np.array([7, 8, 9])
    tr_boxes = np.array([[0.0, 0, 10, 10], [50.0, 0, 60, 10], [200.0, 0, 210, 10]])
    keep_gt, keep_tr = filter_mot17_frame(gt_boxes, gt_classes, gt_consider, tr_boxes)
    assert gt_ids[keep_gt].tolist() == [1]
    assert tr_ids[keep_tr].tolist() == [7, 9]  # 8 sat on a distractor; 9 is an honest FP


def test_fragmentation_follows_trackeval_across_empty_timesteps():
    # TrackEval keeps the previous-match state through timesteps where either side is empty, so
    # tracked -> (no tracker boxes at all) -> tracked is one continuous segment, while
    # tracked -> (tracker present but not matching) -> tracked is a fragmentation.
    through_empty = seq([[1], [1], [1]], [[5], [], [5]], [[[1.0]], np.zeros((1, 0)), [[1.0]]])
    through_miss = seq([[1], [1], [1]], [[5], [9], [5]], [[[1.0]], [[0.0]], [[1.0]]])
    assert clear_metrics(through_empty)["Frag"] == 0
    assert clear_metrics(through_miss)["Frag"] == 1
