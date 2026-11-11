import numpy as np
import pytest

from multicam_tracker.data.calibration_check import diagnose_conventions, ground_offsets, reprojection_errors
from multicam_tracker.data.market1501 import load_market_style, parse_market_name
from multicam_tracker.data.mot17 import load_mot17
from multicam_tracker.data.multiview import SPECS, load_multiview
from multicam_tracker.data.synthetic import make_synthetic
from multicam_tracker.config import SyntheticConfig

from builders import write_market, write_multiview, write_mot17

GT_ROWS = [
    (1, 1, 10, 20, 30, 40, 1, 1, 1.0),
    (1, 2, 50, 5, 10, 20, 0, 7, 0.5),  # static person, not considered
    (2, 1, 12, 20, 30, 40, 1, 1, 0.8),
    (2, 3, 0, 0, 5, 5, 1, 8, 1.0),  # distractor
]
DET_ROWS = [(2, -1, 11, 19, 30, 41, 0.95, -1, -1, -1), (2, -1, 40, 1, 8, 8, 0.3, -1, -1, -1)]


@pytest.fixture
def mot_root(tmp_path):
    write_mot17(tmp_path, gt_rows=GT_ROWS, det_rows=DET_ROWS)
    write_mot17(tmp_path, detector="DPM", gt_rows=GT_ROWS)
    return tmp_path


def test_mot17_sequence_metadata_and_images(mot_root):
    (seq,) = load_mot17(mot_root, detector="FRCNN")
    assert seq.name == "MOT17-09-FRCNN" and seq.fps == 30.0
    assert seq.frames == [1, 2, 3, 4]
    assert seq.image(3).shape == (48, 64, 3) and int(seq.image(3)[0, 0, 0]) == 30
    assert seq.timestamp(4) - seq.timestamp(1) == pytest.approx(0.1)


def test_mot17_ground_truth_is_converted_to_corner_boxes(mot_root):
    (seq,) = load_mot17(mot_root)
    gt = seq.gt(1)
    assert gt.ids.tolist() == [1, 2]
    np.testing.assert_allclose(gt.boxes[0], [10, 20, 40, 60])
    assert gt.classes.tolist() == [1, 7] and gt.consider.tolist() == [1, 0]
    np.testing.assert_allclose(gt.visibility, [1.0, 0.5])


def test_mot17_public_detections(mot_root):
    (seq,) = load_mot17(mot_root)
    dets = seq.public_detections(2)
    np.testing.assert_allclose(dets.boxes[0], [11, 19, 41, 60])
    np.testing.assert_allclose(dets.scores, [0.95, 0.3])
    assert len(seq.public_detections(1)) == 0


@pytest.mark.parametrize("split,frames", [("first_half", [1, 2, 3]), ("second_half", [4]), ("all", [1, 2, 3, 4])])
def test_mot17_half_splits_follow_bytetrack(mot_root, split, frames):
    (seq,) = load_mot17(mot_root, split=split)
    assert seq.frames == frames


def test_mot17_sequence_filter(mot_root):
    assert load_mot17(mot_root, sequences=["MOT17-02"]) == []
    assert len(load_mot17(mot_root, sequences=["MOT17-09"])) == 1


def test_mot17_missing_root_explains_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_mot17(tmp_path / "nope")


def test_wildtrack_position_ids_decode_to_metres():
    # pos = gx + 480 * gy ; x = -3 + 0.025 gx ; y = -9 + 0.025 gy
    np.testing.assert_allclose(SPECS["wildtrack"].position_to_world(40 + 480 * 100), [-2.0, -6.5])
    np.testing.assert_allclose(SPECS["multiviewx"].position_to_world(7 + 1000 * 3), [0.175, 0.075])


@pytest.fixture
def wildtrack_root(tmp_path):
    people = {f: [(1, 40 + 480 * (200 + f)), (99, 300 + 480 * 900)] for f in range(0, 50, 5)}
    cams = write_multiview(tmp_path, "wildtrack", people)
    return tmp_path, cams


def test_wildtrack_calibration_is_loaded_in_metres(wildtrack_root):
    root, cams = wildtrack_root
    seq = load_multiview(root, "wildtrack")
    assert sorted(seq.cameras) == [0, 1]
    np.testing.assert_allclose(seq.cameras[0].t, cams[0].t, atol=1e-6)
    np.testing.assert_allclose(seq.cameras[1].R, cams[1].R, atol=1e-6)
    np.testing.assert_allclose(seq.cameras[0].K[0, 0], 1000.0)


def test_wildtrack_frames_timestamps_and_world_ground_truth(wildtrack_root):
    root, _ = wildtrack_root
    seq = load_multiview(root, "wildtrack")
    assert seq.frames == list(range(0, 50, 5)) and seq.fps == 2.0
    assert seq.timestamp(10) == pytest.approx(1.0)
    ids, xy = seq.gt_world(5)
    assert ids.tolist() == [1, 99]
    np.testing.assert_allclose(xy[0], [-3.0 + 0.025 * 40, -9.0 + 0.025 * 205])


def test_wildtrack_invisible_views_are_skipped(wildtrack_root):
    root, _ = wildtrack_root
    seq = load_multiview(root, "wildtrack")
    assert seq.gt_view(0, 0).ids.tolist() == [1, 99]
    assert seq.gt_view(1, 0).ids.tolist() == [1]


def test_wildtrack_test_split_is_last_ten_percent(wildtrack_root):
    root, _ = wildtrack_root
    assert load_multiview(root, "wildtrack", split="test").frames == [45]
    assert load_multiview(root, "wildtrack", split="train").frames == list(range(0, 45, 5))


def test_multiview_camera_subset(wildtrack_root):
    root, _ = wildtrack_root
    assert sorted(load_multiview(root, "wildtrack", cameras=[1]).cameras) == [1]


def test_multiviewx_reads_opencv_matrix_extrinsics(tmp_path):
    cams = write_multiview(tmp_path, "multiviewx", {0: [(1, 400 + 1000 * 300)], 1: [(1, 410 + 1000 * 300)]})
    seq = load_multiview(tmp_path, "multiviewx")
    np.testing.assert_allclose(seq.cameras[1].t, cams[1].t, atol=1e-6)
    assert seq.timestamp(1) == pytest.approx(0.5)


def test_calibration_check_reports_small_error_for_consistent_data(wildtrack_root):
    root, _ = wildtrack_root
    errors = reprojection_errors(load_multiview(root, "wildtrack"))
    assert set(errors) == {0, 1}
    assert max(float(np.median(e)) for e in errors.values()) < 0.01


def test_calibration_diagnosis_ranks_the_true_convention_first(wildtrack_root):
    root, _ = wildtrack_root
    ranking = diagnose_conventions(root, "wildtrack")
    assert ranking[0][0] == "as-is"
    assert ranking[0][1] < 0.01 < ranking[1][1]


def test_market_filename_parsing():
    assert parse_market_name("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_market_name("-1_c3s2_000001_00.jpg") == (-1, 3)


def test_market_split_relabels_train_and_drops_junk(tmp_path):
    write_market(tmp_path, {
        "bounding_box_train": ["0007_c3s2_000001_01.jpg", "0002_c1s1_000451_03.jpg", "0002_c2s1_000301_01.jpg"],
        "query": ["0002_c1s1_000100_00.jpg"],
        "bounding_box_test": ["0002_c2s1_000200_00.jpg", "0000_c1s1_000002_00.jpg", "-1_c1s1_000001_00.jpg"],
    })
    data = load_market_style(tmp_path)
    assert sorted((s.label, s.pid) for s in data["train"]) == [(0, 2), (0, 2), (1, 7)]
    assert [s.pid for s in data["query"]] == [2]
    assert sorted(s.pid for s in data["gallery"]) == [0, 2]  # junk (-1) removed, distractor (0) kept


def test_synthetic_boxes_are_consistent_with_world_positions():
    seq = make_synthetic(SyntheticConfig(num_cameras=3, num_people=6, num_frames=8, seed=3))
    assert len(seq.cameras) == 3 and len(seq.frames) == 8
    checked = 0
    for f in seq.frames:
        ids, xy = seq.gt_world(f)
        pos = dict(zip(ids.tolist(), xy))
        for cam_id, cam in seq.cameras.items():
            gt = seq.gt_view(cam_id, f)
            if len(gt.ids) == 0:
                continue
            foot = np.column_stack([(gt.boxes[:, 0] + gt.boxes[:, 2]) / 2, gt.boxes[:, 3]])
            back, valid = cam.image_to_ground(foot)
            assert valid.all()
            np.testing.assert_allclose(back, np.stack([pos[i] for i in gt.ids.tolist()]), atol=1e-6)
            checked += len(gt.ids)
    assert checked > 20


def test_synthetic_is_deterministic_and_renders_images():
    a = make_synthetic(SyntheticConfig(seed=5, num_frames=3))
    b = make_synthetic(SyntheticConfig(seed=5, num_frames=3))
    np.testing.assert_allclose(a.gt_world(a.frames[-1])[1], b.gt_world(b.frames[-1])[1])
    img = a.image(0, a.frames[0])
    assert img.shape == (540, 960, 3) and img.dtype == np.uint8


def test_ground_offsets_are_zero_for_exact_foot_points(wildtrack_root):
    root, _ = wildtrack_root
    along, err = ground_offsets(load_multiview(root, "wildtrack"))
    assert abs(float(np.median(along))) < 1e-3 and float(np.median(err)) < 1e-3


def test_ground_offsets_measure_the_near_foot_bias(tmp_path):
    people = {f: [(1, 40 + 480 * (200 + f)), (2, 300 + 480 * 900)] for f in range(0, 50, 5)}
    write_multiview(tmp_path, "wildtrack", people, foot_bias_m=0.25)
    along, _ = ground_offsets(load_multiview(tmp_path, "wildtrack"))
    assert float(np.median(along)) == pytest.approx(0.25, abs=0.01)


def test_camera_without_calibration_is_skipped_with_a_warning(wildtrack_root):
    root, _ = wildtrack_root
    (root / "calibrations" / "extrinsic" / "extr_CVLab2.xml").unlink()
    with pytest.warns(UserWarning, match="C2"):
        seq = load_multiview(root, "wildtrack")
    assert sorted(seq.cameras) == [0]
