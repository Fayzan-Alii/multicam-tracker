import json

import cv2
import pytest

from multicam_tracker.config import ConfigError, build_config
from multicam_tracker.pipeline.runner import run

from builders import write_mot17


def synthetic_cfg(tmp_path, **sections):
    base = {
        "name": "synth",
        "dataset": {"kind": "synthetic", "synthetic": {"num_cameras": 4, "num_people": 8, "num_frames": 40, "seed": 1}},
        "detector": {"kind": "oracle", "oracle_noise_px": 1.5, "oracle_miss_rate": 0.03},
        "reid": {"enabled": True, "kind": "oracle", "oracle_noise": 0.25},
        "tracker": {"motion": {"space": "ground"}, "appearance_weight": 0.3},
        "output": {"dir": str(tmp_path / "out")},
    }
    for key, value in sections.items():
        base[key] = {**base.get(key, {}), **value}
    return build_config(base)


def test_multi_camera_pipeline_tracks_synthetic_people_end_to_end(tmp_path):
    result = run(synthetic_cfg(tmp_path))
    m = result["metrics"]
    assert m["ground_detection"]["MODA"] > 0.9
    assert m["ground_tracking"]["IDF1"] > 0.9
    assert m["cross_camera"]["IDF1"] > 0.85
    assert m["local"]["IDF1"] > 0.9
    out = tmp_path / "out"
    assert json.loads((out / "metrics.json").read_text())["ground_tracking"]["IDF1"] == pytest.approx(
        m["ground_tracking"]["IDF1"])
    assert (out / "global_tracks.csv").read_text().startswith("frame,global_id,x,y")
    assert len(list((out / "mot").glob("*.txt"))) == 4


def test_ground_plane_tracking_beats_image_plane_at_low_frame_rate(tmp_path):
    ground = run(synthetic_cfg(tmp_path / "g", reid={"enabled": False}, tracker={"appearance_weight": 0.0}))
    image = run(synthetic_cfg(tmp_path / "i", reid={"enabled": False},
                              tracker={"appearance_weight": 0.0, "motion": {"space": "image"}}))
    assert ground["metrics"]["local"]["IDF1"] > image["metrics"]["local"]["IDF1"] + 0.2


def test_disabling_fusion_skips_global_metrics(tmp_path):
    result = run(synthetic_cfg(tmp_path, fusion={"enabled": False}))
    assert "ground_tracking" not in result["metrics"] and "local" in result["metrics"]


def test_detection_cache_reproduces_results(tmp_path):
    cache = str(tmp_path / "cache")
    first = run(synthetic_cfg(tmp_path / "a", detector={"cache_dir": cache}))
    second = run(synthetic_cfg(tmp_path / "b", detector={"cache_dir": cache}))
    assert first["metrics"]["ground_tracking"]["IDF1"] == second["metrics"]["ground_tracking"]["IDF1"]
    assert any(p.name.startswith("det_") for p in (tmp_path / "cache").iterdir())


def test_video_output_is_written(tmp_path):
    cfg = synthetic_cfg(tmp_path, output={"save_video": True, "max_video_frames": 3, "panel_width": 240})
    run(cfg)
    cap = cv2.VideoCapture(str(tmp_path / "out" / "demo.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert (tmp_path / "out" / "demo.gif").exists()


GT = [(f, pid, 10 + 3 * f + 60 * pid, 20, 30, 80, 1, 1, 1.0) for f in range(1, 9) for pid in (1, 2)]


def mot_cfg(tmp_path, **tracker):
    return build_config({
        "name": "mot",
        "dataset": {"kind": "mot17", "root": str(tmp_path / "MOT17")},
        "detector": {"kind": "public", "conf": 0.1},
        "tracker": tracker,
        "output": {"dir": str(tmp_path / "out")},
    })


def test_single_camera_pipeline_scores_perfect_detections_perfectly(tmp_path):
    dets = [(f, -1, x, y, w, h, 0.9, -1, -1, -1) for f, _, x, y, w, h, *_ in GT]
    write_mot17(tmp_path / "MOT17", n_frames=8, size=(200, 300), gt_rows=GT, det_rows=dets)
    result = run(mot_cfg(tmp_path))
    m = result["metrics"]["COMBINED"]
    assert m["MOTA"] == pytest.approx(1.0) and m["IDF1"] == pytest.approx(1.0) and m["HOTA"] > 0.99
    lines = (tmp_path / "out" / "mot" / "MOT17-09-FRCNN.txt").read_text().splitlines()
    assert len(lines) == 16 and lines[0].startswith("1,")


def test_ground_motion_without_calibration_is_a_config_error(tmp_path):
    write_mot17(tmp_path / "MOT17", n_frames=2, gt_rows=GT[:2], det_rows=[])
    with pytest.raises(ConfigError, match="calibration"):
        run(mot_cfg(tmp_path, motion={"space": "ground"}))


def test_single_camera_video_is_written(tmp_path):
    dets = [(f, -1, x, y, w, h, 0.9, -1, -1, -1) for f, _, x, y, w, h, *_ in GT]
    write_mot17(tmp_path / "MOT17", n_frames=8, size=(200, 300), gt_rows=GT, det_rows=dets)
    cfg = mot_cfg(tmp_path)
    cfg.output.save_video = True
    run(cfg)
    cap = cv2.VideoCapture(str(tmp_path / "out" / "demo_MOT17-09-FRCNN.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8


def test_rejected_ground_points_and_unreadable_frames_are_counted_not_fatal(tmp_path):
    from multicam_tracker.data.synthetic import make_synthetic
    from multicam_tracker.detection.oracle import OracleDetector
    from multicam_tracker.pipeline.runner import run_multi_camera

    cfg = synthetic_cfg(tmp_path, reid={"enabled": False}, ground={"roi": [0.0, 0.0, 6.0, 12.0]})
    seq = make_synthetic(cfg.dataset.synthetic)
    real = seq.image_fn

    def flaky(cam, frame):
        if (cam, frame) == (1, 5):
            raise FileNotFoundError("corrupt frame")
        return real(cam, frame)

    seq.image_fn = flaky

    class ImageReadingDetector(OracleDetector):  # like YOLO, it needs the pixels
        def detect(self, ctx):
            ctx.image()
            return super().detect(ctx)

    metrics = run_multi_camera(cfg, seq, ImageReadingDetector(cfg.detector), None, tmp_path / "o")
    diag = metrics["diagnostics"]
    assert diag["unreadable_frames"] == {"C2": 1}
    assert diag["rejected_ground_points"]["C1"] > 0  # people walking in the half outside the ROI
    assert metrics["ground_tracking"]["IDF1"] > 0.5
