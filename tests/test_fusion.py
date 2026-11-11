import numpy as np

from multicam_tracker.config import FusionConfig
from multicam_tracker.fusion.associator import CrossCameraFusion, fuse_measurements
from multicam_tracker.types import TrackOutput

E = {n: v for n, v in zip("abcd", np.eye(4))}


def obs(cam, local_id, xy, emb="a", std=0.1):
    return TrackOutput(track_id=local_id, box=np.array([0.0, 0, 10, 30]), score=0.9, camera_id=cam,
                       world_xy=np.array(xy, float), world_cov=np.eye(2) * std**2, embedding=E[emb])


def fusion(**overrides):
    return CrossCameraFusion(FusionConfig(**overrides))


def ids(out):
    return sorted(o.global_id for o in out)


def test_information_fusion_is_inverse_variance_weighted():
    z, R = fuse_measurements([np.array([0.0, 0.0]), np.array([1.0, 0.0])],
                             [np.diag([1.0, 1.0]), np.diag([3.0, 1.0])])
    np.testing.assert_allclose(z, [0.25, 0.0])
    np.testing.assert_allclose(R, np.diag([0.75, 0.5]))


def test_one_person_seen_by_two_cameras_gets_one_global_id():
    out = fusion().update([obs(0, 1, [2.0, 3.0]), obs(1, 7, [2.1, 3.05])], timestamp=0.0)
    assert len(out) == 1
    assert out[0].cameras == [0, 1] and out[0].local == {0: 1, 1: 7}


def test_fused_position_is_more_certain_than_either_view():
    out = fusion().update([obs(0, 1, [2.0, 3.0], std=0.2), obs(1, 7, [2.1, 3.0], std=0.2)], timestamp=0.0)
    assert np.trace(out[0].world_cov) < np.trace(np.eye(2) * 0.2**2)
    np.testing.assert_allclose(out[0].world_xy, [2.05, 3.0], atol=1e-9)


def test_distant_people_stay_separate():
    out = fusion().update([obs(0, 1, [0.0, 0.0], "a"), obs(1, 2, [8.0, 8.0], "b")], timestamp=0.0)
    assert len(out) == 2


def test_two_tracks_from_one_camera_are_never_merged():
    out = fusion().update([obs(0, 1, [0.0, 0.0], "a"), obs(0, 2, [0.05, 0.0], "a")], timestamp=0.0)
    assert len(out) == 2


def test_global_ids_are_stable_over_time():
    f = fusion()
    first = f.update([obs(0, 1, [0.0, 0.0], "a"), obs(1, 5, [0.0, 0.05], "a"),
                      obs(0, 2, [5.0, 5.0], "b")], timestamp=0.0)
    gid = {o.local[0]: o.global_id for o in first}
    for k in range(1, 6):
        t = 0.5 * k
        out = f.update([obs(0, 1, [0.5 * k, 0.0], "a"), obs(1, 5, [0.5 * k, 0.05], "a"),
                        obs(0, 2, [5.0, 5.0 - 0.3 * k], "b")], timestamp=t)
        assert {o.local[0]: o.global_id for o in out} == gid


def test_identity_survives_a_camera_losing_the_person():
    f = fusion()
    (start,) = f.update([obs(0, 1, [0.0, 0.0]), obs(1, 5, [0.0, 0.0])], timestamp=0.0)
    (only_cam1,) = f.update([obs(1, 5, [0.5, 0.0])], timestamp=0.5)
    # Camera 0 re-acquires the person under a brand-new local track id.
    (both,) = f.update([obs(0, 9, [1.0, 0.0]), obs(1, 5, [1.0, 0.0])], timestamp=1.0)
    assert start.global_id == only_cam1.global_id == both.global_id
    assert both.local == {0: 9, 1: 5}


def test_person_moving_between_camera_views_keeps_identity():
    f = fusion()
    (a,) = f.update([obs(0, 1, [0.0, 0.0])], timestamp=0.0)
    (b,) = f.update([obs(0, 1, [0.6, 0.0]), obs(1, 3, [0.62, 0.0])], timestamp=0.5)
    (c,) = f.update([obs(1, 3, [1.2, 0.0])], timestamp=1.0)
    assert a.global_id == b.global_id == c.global_id


def test_broken_link_is_dropped_when_local_track_jumps():
    # A local ID switch: camera 0's track 1 suddenly reports a position 6 m away.
    f = fusion()
    f.update([obs(0, 1, [0.0, 0.0], "a"), obs(1, 5, [0.0, 0.0], "a")], timestamp=0.0)
    out = f.update([obs(0, 1, [6.0, 0.0], "b"), obs(1, 5, [0.1, 0.0], "a")], timestamp=0.5)
    assert len(out) == 2
    by_cam = {tuple(o.cameras): o.global_id for o in out}
    assert by_cam[(1,)] != by_cam[(0,)]


def test_global_track_is_dropped_after_max_lost():
    f = fusion(max_lost_s=1.0)
    (first,) = f.update([obs(0, 1, [0.0, 0.0], "a")], timestamp=0.0)
    f.update([], timestamp=1.5)
    (again,) = f.update([obs(0, 4, [0.0, 0.0], "a")], timestamp=2.0)
    assert again.global_id != first.global_id


def test_appearance_prevents_joining_a_different_identity():
    f = fusion(appearance_weight=0.5, appearance_gate=0.4)
    out = f.update([obs(0, 1, [0.0, 0.0], "a"), obs(1, 2, [0.1, 0.0], "b")], timestamp=0.0)
    assert len(out) == 2


def test_duplicate_global_tracks_are_merged():
    # Two cameras first see the same person 1.2 m apart (bad foot points), so they start as two
    # identities; once their estimates agree the younger one is merged into the older.
    f = fusion(gate_chi2=9.21, merge_chi2=2.0)
    first = f.update([obs(0, 1, [0.0, 0.0], std=0.2), obs(1, 2, [1.2, 0.0], std=0.2)], timestamp=0.0)
    assert len(first) == 2
    older = min(o.global_id for o in first)
    out = []
    for k in range(1, 4):
        out = f.update([obs(0, 1, [0.0, 0.0], std=0.2), obs(1, 2, [0.05, 0.0], std=0.2)], timestamp=0.5 * k)
    assert ids(out) == [older]
    assert out[0].local == {0: 1, 1: 2}


def test_identity_never_takes_two_observations_from_one_camera():
    # Camera 0 spawns a duplicate local track right next to an already-linked one.
    f = fusion()
    f.update([obs(0, 1, [0.0, 0.0])], timestamp=0.0)
    out = f.update([obs(0, 1, [0.0, 0.0]), obs(0, 2, [0.05, 0.0])], timestamp=0.5)
    assert len(out) == 2
    assert all(len(o.cameras) == len(set(o.cameras)) for o in out)
