import numpy as np
import pytest

from multicam_tracker.config import DetectorConfig, GroundConfig, ReIDConfig
from multicam_tracker.data.base import GTFrame
from multicam_tracker.detection.base import FrameContext, filter_detections
from multicam_tracker.detection.cache import CachedDetector, CachedEmbedder
from multicam_tracker.detection.oracle import OracleDetector, OracleEmbedder
from multicam_tracker.detection.public import PublicDetector
from multicam_tracker.geometry.camera import Camera
from multicam_tracker.pipeline.ground import project_to_ground
from multicam_tracker.types import Detections


def ctx(frame=0, camera=0, gt=None, public=None, image=None):
    return FrameContext(sequence="seq", camera=camera, frame=frame, image_fn=lambda: image, gt=gt, public=public)


GT = GTFrame(ids=np.array([1, 2]), boxes=np.array([[10.0, 10, 30, 80], [100.0, 20, 130, 110]]))


def test_filter_clips_to_image_and_drops_short_boxes():
    dets = filter_detections(np.array([[-5.0, 10, 20, 60], [0.0, 0, 10, 5], [90.0, 90, 120, 130]]),
                             np.array([0.9, 0.8, 0.7]), min_height_px=10, image_size=(100, 110))
    np.testing.assert_allclose(dets.boxes, [[0, 10, 20, 60], [90, 90, 110, 100]])
    np.testing.assert_allclose(dets.scores, [0.9, 0.7])


def test_oracle_without_noise_returns_ground_truth():
    det = OracleDetector(DetectorConfig(kind="oracle", oracle_noise_px=0.0, oracle_miss_rate=0.0))
    out = det.detect(ctx(gt=GT))
    np.testing.assert_allclose(out.boxes, GT.boxes)
    assert (out.scores >= 0.5).all()


def test_oracle_is_deterministic_per_frame_and_camera():
    det = OracleDetector(DetectorConfig(kind="oracle", oracle_noise_px=3.0))
    a = det.detect(ctx(frame=4, camera=1, gt=GT))
    b = det.detect(ctx(frame=4, camera=1, gt=GT))
    c = det.detect(ctx(frame=5, camera=1, gt=GT))
    np.testing.assert_allclose(a.boxes, b.boxes)
    assert not np.allclose(a.boxes, c.boxes)


def test_oracle_misses_everything_at_full_miss_rate():
    det = OracleDetector(DetectorConfig(kind="oracle", oracle_miss_rate=1.0))
    assert len(det.detect(ctx(gt=GT))) == 0


def test_public_detector_applies_confidence_threshold():
    public = Detections(boxes=np.array([[0.0, 0, 10, 30], [5.0, 5, 15, 40]]), scores=np.array([0.9, 0.05]))
    out = PublicDetector(DetectorConfig(kind="public", conf=0.1)).detect(ctx(public=public))
    np.testing.assert_allclose(out.scores, [0.9])


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, c):
        self.calls += 1
        return Detections(boxes=np.array([[c.frame, 0.0, c.frame + 10, 40]]), scores=np.array([0.5 + 0.1 * c.camera]))


def test_cache_reuses_detections_across_runs(tmp_path):
    inner = CountingDetector()
    first = CachedDetector(inner, tmp_path, fingerprint={"weights": "a"})
    for f in range(3):
        first.detect(ctx(frame=f, camera=1))
    first.flush()
    second = CachedDetector(CountingDetector(), tmp_path, fingerprint={"weights": "a"})
    out = second.detect(ctx(frame=2, camera=1))
    assert inner.calls == 3 and second.inner.calls == 0
    np.testing.assert_allclose(out.boxes, [[2, 0, 12, 40]])
    np.testing.assert_allclose(out.scores, [0.6])


def test_cache_keeps_different_detector_configurations_apart(tmp_path):
    first = CachedDetector(CountingDetector(), tmp_path, fingerprint={"weights": "a"})
    first.detect(ctx())
    first.flush()
    other = CachedDetector(CountingDetector(), tmp_path, fingerprint={"weights": "b"})
    other.detect(ctx())
    assert other.inner.calls == 1


def test_embedding_cache_without_model_reports_a_miss(tmp_path):
    with pytest.raises(KeyError, match="not cached"):
        CachedEmbedder(None, tmp_path, fingerprint={"reid": "x"}).embed(ctx(gt=GT), GT.boxes)


def test_embedding_cache_keys_on_boxes(tmp_path):
    oracle = OracleEmbedder(ReIDConfig(kind="oracle", oracle_dim=8, oracle_noise=0.0))
    cache = CachedEmbedder(oracle, tmp_path, fingerprint={"reid": "x"})
    first = cache.embed(ctx(gt=GT), GT.boxes)
    cache.flush()
    again = CachedEmbedder(None, tmp_path, fingerprint={"reid": "x"}).embed(ctx(gt=GT), GT.boxes)
    np.testing.assert_allclose(again, first)


def test_oracle_embeddings_separate_identities():
    emb = OracleEmbedder(ReIDConfig(kind="oracle", oracle_dim=64, oracle_noise=0.25))
    a = emb.embed(ctx(camera=0, gt=GT), GT.boxes)
    b = emb.embed(ctx(camera=3, frame=9, gt=GT), GT.boxes[::-1])
    same = float(a[0] @ b[1])
    different = float(a[0] @ b[0])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert same > 0.8 and different < 0.5


def horizontal_camera():
    K = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return Camera(K=K, R=R, t=np.array([0.0, 2.0, 0.0]), image_size=(480, 640))


def test_ground_projection_fills_world_fields_and_rejects_bad_points():
    dets = Detections(
        boxes=np.array([[310.0, 200, 330, 340], [310.0, 50, 330, 100], [310.0, 200, 330, 250]]),
        scores=np.ones(3),
    )
    out = project_to_ground(dets, horizontal_camera(), GroundConfig(), roi=(-5.0, 0.0, 5.0, 15.0))
    np.testing.assert_allclose(out.world_xy[0], [0.0, 2.0], atol=1e-9)
    # Foot above the horizon -> invalid; foot at v=250 lands 20 m away, outside the ROI -> invalid.
    assert out.ground_valid.tolist() == [True, False, False]
    assert out.world_cov.shape == (3, 2, 2)


def test_fixed_isotropic_noise_replaces_propagation_when_disabled():
    dets = Detections(boxes=np.array([[310.0, 200, 330, 340]]), scores=np.ones(1))
    cfg = GroundConfig(propagate_uncertainty=False, fixed_sigma_m=0.3)
    out = project_to_ground(dets, horizontal_camera(), cfg, roi=(-5.0, 0.0, 5.0, 15.0))
    np.testing.assert_allclose(out.world_cov[0], np.eye(2) * 0.09)
    np.testing.assert_allclose(out.world_xy[0], [0.0, 2.0], atol=1e-9)


def test_foot_depth_offset_pushes_ground_point_away_from_camera():
    dets = Detections(boxes=np.array([[310.0, 200, 330, 340]]), scores=np.ones(1))
    out = project_to_ground(dets, horizontal_camera(), GroundConfig(foot_depth_offset_m=0.25), roi=(-5.0, 0.0, 5.0, 15.0))
    np.testing.assert_allclose(out.world_xy[0], [0.0, 2.25], atol=1e-9)


def test_cache_loads_each_archive_array_once(tmp_path, monkeypatch):
    first = CachedDetector(CountingDetector(), tmp_path, fingerprint={"weights": "a"})
    for frame in range(12):
        first.detect(ctx(frame=frame))
    first.flush()
    with np.load(next(tmp_path.glob("*.npz"))) as archive:
        archive_type = type(archive)
    read = archive_type.__getitem__
    reads = {}

    def counted(self, key):
        reads[key] = reads.get(key, 0) + 1
        return read(self, key)

    monkeypatch.setattr(archive_type, "__getitem__", counted)
    second = CachedDetector(CountingDetector(), tmp_path, fingerprint={"weights": "a"})
    for frame in range(12):
        np.testing.assert_allclose(second.detect(ctx(frame=frame)).boxes,
                                   [[frame, 0, frame + 10, 40]])
    assert second.inner.calls == 0
    assert reads == {"frames": 1, "offsets": 1, "boxes": 1, "scores": 1}
    entries = second.store._data[("seq", 0)]
    assert len({id(entry["boxes"].base) for entry in entries.values()}) == 1


def test_detector_cache_invalidates_replaced_weights(tmp_path, monkeypatch):
    import os

    from multicam_tracker.config import build_config
    from multicam_tracker.detection import yolo
    from multicam_tracker.pipeline.factory import build_detector

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"first weights")
    stat = weights.stat()
    inner = CountingDetector()
    monkeypatch.setattr(yolo, "YoloDetector", lambda cfg: inner)
    cfg = build_config({"detector": {"kind": "yolo", "weights": str(weights),
                                     "cache_dir": str(tmp_path / "cache")}})
    first = build_detector(cfg)
    first.detect(ctx())
    first.flush()
    cached = build_detector(cfg)
    cached.detect(ctx())
    assert inner.calls == 1
    assert cached.inner._obj is None  # Cache hits do not instantiate the model.
    weights.write_bytes(b"other weights")  # Same length, same mtime, different model.
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = build_detector(cfg)
    changed.detect(ctx())
    assert inner.calls == 2
    assert changed.store.prefix != first.store.prefix


def test_embedder_cache_invalidates_replaced_weights(tmp_path):
    import os

    from multicam_tracker.config import build_config
    from multicam_tracker.pipeline.factory import build_embedder

    weights = tmp_path / "reid.pt"
    weights.write_bytes(b"first weights")
    stat = weights.stat()
    cfg = build_config({"detector": {"cache_dir": str(tmp_path / "cache")},
                        "reid": {"enabled": True, "checkpoint": str(weights)}})
    first = build_embedder(cfg)
    weights.write_bytes(b"other weights")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = build_embedder(cfg)
    assert first.store.prefix != second.store.prefix
    assert first.inner._obj is None and second.inner._obj is None
