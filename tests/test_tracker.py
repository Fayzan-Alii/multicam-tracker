import numpy as np

from multicam_tracker.config import MotionConfig, TrackerConfig
from multicam_tracker.tracking.kalman import GroundMotion, ImageMotion
from multicam_tracker.tracking.tracker import SingleCameraTracker, make_motion
from multicam_tracker.types import Detections


def box_at(cx, cy, w=40.0, h=100.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def image_tracker(**overrides):
    cfg = TrackerConfig(**overrides)
    return SingleCameraTracker(cfg, ImageMotion(cfg.motion, fps=30.0))


def run(tracker, frames, fps=30.0):
    """frames: list of lists of (gt_id, Detections-row kwargs). Returns {frame: {gt_id: track_id}}."""
    out = {}
    for f, rows in enumerate(frames):
        if rows:
            dets = Detections(
                boxes=np.array([r["box"] for _, r in rows]),
                scores=np.array([r.get("score", 0.9) for _, r in rows]),
                embeddings=np.array([r["emb"] for _, r in rows]) if "emb" in rows[0][1] else None,
                world_xy=np.array([r["xy"] for _, r in rows]) if "xy" in rows[0][1] else None,
                world_cov=np.array([np.eye(2) * 0.05**2 for _ in rows]) if "xy" in rows[0][1] else None,
                ground_valid=np.ones(len(rows), bool) if "xy" in rows[0][1] else None,
            )
        else:
            dets = Detections.empty()
        tracks = tracker.update(dets, timestamp=f / fps)
        by_box = {tuple(np.round(t.box, 6)): t.track_id for t in tracks}
        out[f] = {gid: by_box.get(tuple(np.round(r["box"], 6))) for gid, r in rows}
        out[f] = {g: t for g, t in out[f].items() if t is not None}
    return out


def ids_for(result, gid):
    return {m[gid] for m in result.values() if gid in m}


def walkers(n_frames, speed=3.0, gap=None, low=None):
    frames = []
    for f in range(n_frames):
        rows = []
        if gap is None or not (gap[0] <= f < gap[1]):
            score = 0.3 if low is not None and low[0] <= f < low[1] else 0.9
            rows.append(("a", {"box": box_at(100 + speed * f, 200), "score": score}))
        rows.append(("b", {"box": box_at(400 - speed * f, 400)}))
        frames.append(rows)
    return frames


def test_parallel_walkers_keep_distinct_stable_ids():
    result = run(image_tracker(), walkers(30))
    a, b = ids_for(result, "a"), ids_for(result, "b")
    assert len(a) == 1 and len(b) == 1 and a != b
    assert all(len(m) == 2 for m in result.values())


def test_short_occlusion_keeps_identity():
    result = run(image_tracker(max_lost_s=1.0), walkers(40, gap=(10, 20)))
    assert len(ids_for(result, "a")) == 1


def test_occlusion_longer_than_max_lost_starts_a_new_identity():
    result = run(image_tracker(max_lost_s=0.2), walkers(40, gap=(10, 20)))
    assert len(ids_for(result, "a")) == 2


def test_second_stage_keeps_low_score_detections_on_the_track():
    with_second = run(image_tracker(second_stage=True), walkers(30, low=(10, 15)))
    without = run(image_tracker(second_stage=False), walkers(30, low=(10, 15)))
    assert all("a" in with_second[f] for f in range(10, 15))
    assert not any("a" in without[f] for f in range(10, 15))
    assert len(ids_for(with_second, "a")) == 1


def test_new_object_after_first_frame_needs_n_init_hits():
    frames = [[("a", {"box": box_at(100, 100)})] for _ in range(6)]
    for f in range(2, 6):
        frames[f].append(("b", {"box": box_at(400 + f, 400)}))
    result = run(image_tracker(n_init=3), frames)
    assert "a" in result[0]  # objects on the very first frame activate immediately
    assert "b" not in result[2] and "b" not in result[3]
    assert "b" in result[4]


def test_single_frame_false_positive_never_reaches_output():
    frames = [[("a", {"box": box_at(100, 100)})] for _ in range(5)]
    frames[2].append(("fp", {"box": box_at(500, 500)}))
    result = run(image_tracker(n_init=2), frames)
    assert not ids_for(result, "fp")


def test_low_score_detection_never_starts_a_track():
    frames = [[("a", {"box": box_at(100, 100), "score": 0.4})] for _ in range(5)]
    assert not ids_for(run(image_tracker(), frames), "a")


def ground_walkers(n_frames):
    # Two people 3 m apart walking at 1.4 m/s, sampled at 2 fps: 0.7 m per frame. Their boxes move
    # 80 px per frame with 40 px width, so consecutive boxes never overlap.
    frames = []
    for f in range(n_frames):
        frames.append([
            ("a", {"box": box_at(100 + 80 * f, 300), "xy": [0.7 * f, 0.0]}),
            ("b", {"box": box_at(100 + 80 * f, 600), "xy": [0.7 * f, 3.0]}),
        ])
    return frames


def test_ground_plane_motion_survives_low_frame_rate():
    cfg = TrackerConfig(motion=MotionConfig(space="ground"))
    ground = run(SingleCameraTracker(cfg, GroundMotion(cfg.motion)), ground_walkers(12), fps=2.0)
    assert len(ids_for(ground, "a")) == 1 and len(ids_for(ground, "b")) == 1
    assert all("a" in ground[f] and "b" in ground[f] for f in range(12))


def test_image_plane_motion_loses_people_at_low_frame_rate():
    # Zero IoU between frames: the first-frame track goes lost and no replacement can confirm.
    cfg = TrackerConfig()
    image = run(SingleCameraTracker(cfg, ImageMotion(cfg.motion, fps=2.0)), ground_walkers(12), fps=2.0)
    assert sum("a" in image[f] for f in range(12)) < 3


def swap_scene():
    # Two people stand 1 m apart, vanish for 2 s, and reappear nearly swapped (0.1 m and 0.9 m).
    ea, eb = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    frames = []
    for _ in range(4):
        frames.append([("a", {"box": box_at(100, 300), "xy": [0.0, 0.0], "emb": ea}),
                       ("b", {"box": box_at(300, 300), "xy": [1.0, 0.0], "emb": eb})])
    frames += [[] for _ in range(4)]
    for _ in range(3):
        frames.append([("a", {"box": box_at(290, 300), "xy": [0.9, 0.0], "emb": ea}),
                       ("b", {"box": box_at(110, 300), "xy": [0.1, 0.0], "emb": eb})])
    return frames


def ground_tracker(**overrides):
    cfg = TrackerConfig(motion=MotionConfig(space="ground"), max_lost_s=5.0, **overrides)
    return SingleCameraTracker(cfg, GroundMotion(cfg.motion))


def test_motion_alone_swaps_identities_after_occlusion():
    result = run(ground_tracker(appearance_weight=0.0), swap_scene(), fps=2.0)
    assert result[0]["a"] == result[10]["b"]


def test_appearance_restores_identities_after_occlusion():
    result = run(ground_tracker(appearance_weight=0.8), swap_scene(), fps=2.0)
    assert result[0]["a"] == result[10]["a"] and result[0]["b"] == result[10]["b"]


def test_ground_tracker_ignores_detections_without_a_ground_point():
    tracker = ground_tracker()
    dets = Detections(boxes=np.array([box_at(100, 100)]), scores=np.array([0.9]),
                      world_xy=np.array([[np.nan, np.nan]]), world_cov=np.array([np.eye(2)]),
                      ground_valid=np.array([False]))
    assert tracker.update(dets, 0.0) == []


def test_make_motion_selects_space():
    assert isinstance(make_motion(MotionConfig(space="ground"), fps=5.0), GroundMotion)
    assert isinstance(make_motion(MotionConfig(space="image"), fps=5.0), ImageMotion)


def test_score_fusion_does_not_disable_low_confidence_rescue():
    # A prolonged confidence dip should not expire a continuously visible person.
    # With score fusion, confidence 0.3 has cost >= 0.7 even at perfect overlap,
    # so a second-stage threshold of 0.5 used to reject every such detection.
    frames = walkers(40, low=(10, 30))
    result = run(image_tracker(fuse_score=True, high_thresh=0.6,
                               new_track_thresh=0.7, max_lost_s=0.2), frames)
    assert all("a" in result[f] for f in range(10, 30))
    assert len(ids_for(result, "a")) == 1


def test_score_fusion_still_applies_to_high_confidence_matching():
    tracker = image_tracker(fuse_score=True, match_thresh=0.8, high_thresh=0.6)
    # Shift enough that overlap alone passes 0.8, but score-fused cost does not.
    frames = [[("a", {"box": box_at(100, 200)})],
              [("a", {"box": box_at(122, 200), "score": 0.6})]]
    result = run(tracker, frames)
    assert "a" in result[0]
    assert "a" not in result[1]
