import numpy as np
import pytest

from multicam_tracker.tracking.costs import cosine_distance, fuse_score, iou_matrix
from multicam_tracker.types import Detections


def test_iou_of_half_overlapping_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0 / 3.0, 1.0, 0.0]])


def test_iou_with_empty_side():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_cosine_distance_normalises_inputs():
    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    b = np.array([[5.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(cosine_distance(a, b), [[0.0, 1 - np.sqrt(0.5)], [1.0, 1 - np.sqrt(0.5)]])


def test_fuse_score_discounts_low_confidence():
    np.testing.assert_allclose(fuse_score(np.array([[0.2, 1.0]]), np.array([0.5, 0.9])), [[0.6, 1.0]])


def test_detections_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="scores"):
        Detections(boxes=np.zeros((2, 4)), scores=np.zeros(3))


def test_detection_subset_keeps_fields_aligned():
    dets = Detections(
        boxes=np.arange(12.0).reshape(3, 4),
        scores=np.array([0.1, 0.2, 0.3]),
        embeddings=np.eye(3),
        world_xy=np.arange(6.0).reshape(3, 2),
        world_cov=np.repeat(np.eye(2)[None], 3, 0) * np.array([1.0, 2.0, 3.0])[:, None, None],
        ground_valid=np.array([True, False, True]),
    )
    sub = dets[np.array([2, 0])]
    np.testing.assert_allclose(sub.scores, [0.3, 0.1])
    np.testing.assert_allclose(sub.boxes[0], [8, 9, 10, 11])
    np.testing.assert_allclose(sub.embeddings, [[0, 0, 1], [1, 0, 0]])
    np.testing.assert_allclose(sub.world_cov[:, 0, 0], [3.0, 1.0])
    assert sub.ground_valid.tolist() == [True, True]
