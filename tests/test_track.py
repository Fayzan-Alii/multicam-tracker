import numpy as np

from multicam_tracker.config import MotionConfig
from multicam_tracker.tracking.kalman import ImageMotion
from multicam_tracker.tracking.track import Track, TrackState
from multicam_tracker.types import Detections

MOTION = ImageMotion(MotionConfig(space="image"), fps=10.0)


def dets(box, emb=None):
    return Detections(boxes=np.array([box]), scores=np.array([0.9]), embeddings=None if emb is None else np.array([emb]))


def new_track(n_init=3, emb=None):
    return Track.create(1, MOTION, dets([0, 0, 10, 30], emb), 0, timestamp=0.0, n_init=n_init)


def step(track, t, box=(0, 0, 10, 30), emb=None, ema=0.9):
    track.predict(MOTION, t)
    track.update(MOTION, dets(list(box), emb), 0, t, ema, update_embedding=True)


def test_tentative_track_confirms_after_n_init_hits():
    track = new_track(n_init=3)
    assert track.state is TrackState.TENTATIVE
    step(track, 0.1)
    assert track.state is TrackState.TENTATIVE
    step(track, 0.2)
    assert track.state is TrackState.CONFIRMED


def test_missed_tentative_track_is_deleted():
    track = new_track()
    track.predict(MOTION, 0.1)
    track.mark_missed(0.1, max_lost_s=10.0)
    assert track.state is TrackState.DELETED


def test_confirmed_track_goes_lost_then_reactivates():
    track = new_track(n_init=1)
    assert track.state is TrackState.CONFIRMED
    track.predict(MOTION, 0.1)
    track.mark_missed(0.1, max_lost_s=1.0)
    assert track.state is TrackState.LOST
    step(track, 0.2)
    assert track.state is TrackState.CONFIRMED


def test_lost_track_is_deleted_after_max_lost():
    track = new_track(n_init=1)
    for t in (0.1, 0.5, 1.0):
        track.predict(MOTION, t)
        track.mark_missed(t, max_lost_s=1.0)
    assert track.state is TrackState.LOST  # exactly 1.0 s since the last update
    track.predict(MOTION, 1.1)
    track.mark_missed(1.1, max_lost_s=1.0)
    assert track.state is TrackState.DELETED


def test_embedding_is_an_exponential_moving_average_on_the_unit_sphere():
    track = new_track(n_init=1, emb=[1.0, 0.0])
    step(track, 0.1, emb=[0.0, 2.0], ema=0.75)
    expected = np.array([0.75, 0.25]) / np.linalg.norm([0.75, 0.25])
    np.testing.assert_allclose(track.embedding, expected)
