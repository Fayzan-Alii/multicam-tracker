import numpy as np
import pytest
import torch
import yaml

from multicam_tracker.config import ConfigError, ReIDConfig, build_config
from multicam_tracker.data.market1501 import load_market_style, parse_market_name
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.reid.model import ReIDNet
from multicam_tracker.tools.ablate import ablation_table
from multicam_tracker.tools.benchmark import benchmark
from multicam_tracker.tools.prepare_yolo import prepare_mot17, prepare_multiview
from multicam_tracker.tools.reid_crops import build_reid_crops

from builders import write_mot17, write_multiview
from test_data import GT_ROWS


def test_mot17_yolo_labels_are_normalised_and_skip_distractors(tmp_path):
    write_mot17(tmp_path / "MOT17", gt_rows=GT_ROWS)
    out = tmp_path / "yolo"
    prepare_mot17(tmp_path / "MOT17", out)
    lines = (out / "labels" / "train" / "MOT17-09_000001.txt").read_text().splitlines()
    # Pedestrian x 10..40, y 20..60 in a 64x48 image: it runs off the bottom edge, so it is
    # clipped to y 20..48 before normalising. The static person (class 7) is a person too.
    assert lines[0] == "0 0.390625 0.708333 0.468750 0.583333"
    assert len(lines) == 2
    assert len((out / "labels" / "train" / "MOT17-09_000002.txt").read_text().splitlines()) == 1  # distractor dropped
    assert (out / "images" / "val" / "MOT17-09_000004.jpg").exists()
    assert not (out / "images" / "train" / "MOT17-09_000004.jpg").exists()
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "person"} and data["train"] == "images/train"


def test_multiview_yolo_labels_follow_the_train_test_split(tmp_path):
    people = {f: [(1, 40 + 480 * (200 + f)), (99, 300 + 480 * 900)] for f in range(0, 50, 5)}
    write_multiview(tmp_path / "wt", "wildtrack", people)
    out = tmp_path / "yolo"
    prepare_multiview(tmp_path / "wt", "wildtrack", out)
    train = sorted((out / "labels" / "train").glob("*.txt"))
    val = sorted((out / "labels" / "val").glob("*.txt"))
    assert len(train) == 2 * 9 and len(val) == 2 * 1
    assert len((out / "labels" / "train" / "wildtrack_C1_00000000.txt").read_text().splitlines()) == 2
    assert len((out / "labels" / "train" / "wildtrack_C2_00000000.txt").read_text().splitlines()) == 1
    values = np.array([[float(v) for v in line.split()[1:]] for p in train for line in p.read_text().splitlines()])
    assert ((values >= 0) & (values <= 1)).all()


def test_reid_crops_use_market_naming_and_split_query_gallery(tmp_path):
    people = {f: [(0, 40 + 480 * (200 + f)), (99, 300 + 480 * 900)] for f in range(0, 100, 5)}
    write_multiview(tmp_path / "wt", "wildtrack", people, image_size=(1080, 1920))
    out = tmp_path / "crops"
    counts = build_reid_crops(tmp_path / "wt", "wildtrack", out, min_height=10, seed=0)
    data = load_market_style(out)
    # personID 0 would read as a Market distractor, so ids are shifted by one.
    assert sorted({s.pid for s in data["train"]}) == [1, 100]
    assert {parse_market_name(p.name)[1] for p in (out / "bounding_box_train").iterdir()} == {1, 2}
    # Last 4 of 18 training frames supply validation; test frames 90 and 95 are untouched.
    assert sorted((s.pid, s.camid) for s in data["query"]) == [(1, 1), (1, 2), (100, 1)]
    assert len(data["gallery"]) == counts["val"] - 3
    assert counts == {"train": 14 * 3, "val": 4 * 3}
    from pathlib import Path

    def frames(samples):
        return {int(Path(s.path).stem.split("_")[2]) for s in samples}

    train_frames = frames(data["train"])
    val_frames = frames(data["query"] + data["gallery"])
    assert train_frames == set(range(0, 70, 5))
    assert val_frames == set(range(70, 90, 5))
    assert train_frames.isdisjoint(val_frames)
    import json

    manifest = json.loads((out / "split.json").read_text())
    assert manifest["source_split"] == "train"
    assert manifest["frames"] == {"train": sorted(train_frames), "val": sorted(val_frames)}


def reid_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = ReIDNet("resnet18", num_classes=4, pretrained=False).eval()
    meta = {**model.config, "input_size": [64, 32], "feat_dim": 512}
    path = tmp_path / "reid.pt"
    torch.save({"model": model.state_dict(), "meta": meta}, path)
    return path


def test_onnx_export_matches_pytorch_embeddings(tmp_path):
    pytest.importorskip("onnxruntime")
    from multicam_tracker.deploy.export import export_reid_onnx
    from multicam_tracker.deploy.onnx_embedder import OnnxReIDEmbedder
    from multicam_tracker.reid.extractor import TorchReIDEmbedder

    ckpt = reid_checkpoint(tmp_path)
    onnx_path = export_reid_onnx(ckpt, tmp_path / "reid.onnx")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (120, 90, 3), dtype=np.uint8)
    ctx = FrameContext(sequence="s", camera=0, frame=0, image_fn=lambda: image)
    boxes = np.array([[5.0, 5, 40, 100], [30.0, 10, 80, 110], [0.0, 0, 20, 30]])
    ref = TorchReIDEmbedder(ReIDConfig(checkpoint=str(ckpt), device="cpu")).embed(ctx, boxes)
    onnx = OnnxReIDEmbedder(ReIDConfig(kind="onnx", checkpoint=str(onnx_path), device="cpu"))
    np.testing.assert_allclose(onnx.embed(ctx, boxes), ref, atol=1e-4)
    np.testing.assert_allclose(onnx.embed(ctx, boxes[:1]), ref[:1], atol=1e-4)  # dynamic batch


def synthetic(tmp_path, name, **tracker):
    return build_config({
        "name": name,
        "dataset": {"kind": "synthetic", "synthetic": {"num_cameras": 2, "num_people": 4, "num_frames": 12}},
        "detector": {"kind": "oracle"},
        "tracker": {"motion": {"space": "ground"}, **tracker},
        "output": {"dir": str(tmp_path / name)},
    })


def test_ablation_table_has_one_row_per_config(tmp_path):
    table = ablation_table([synthetic(tmp_path, "ground"), synthetic(tmp_path, "no-second", second_stage=False)])
    lines = table.splitlines()
    assert lines[0].startswith("| run |") and "ground IDF1" in lines[0]
    assert lines[2].startswith("| ground |") and lines[3].startswith("| no-second |")
    import json

    idf1 = json.loads((tmp_path / "ground" / "metrics.json").read_text())["ground_tracking"]["IDF1"]
    assert f"{100 * idf1:.1f}" in lines[2]


def test_benchmark_reports_every_stage(tmp_path):
    cfg = synthetic(tmp_path, "bench")
    timings = benchmark(cfg, max_frames=4)
    assert set(timings) == {"detect", "embed", "ground", "track", "fuse", "total"}
    assert all(v["mean_ms"] >= 0 for v in timings.values())
    assert timings["detect"]["calls"] == 8  # 4 frames x 2 cameras


def test_reid_crops_release_source_images_as_they_are_written(tmp_path, monkeypatch):
    import weakref
    from types import SimpleNamespace

    from multicam_tracker.data.base import GTFrame
    from multicam_tracker.tools import reid_crops

    images = []

    def image(camera, frame):
        assert not any(ref() is not None for ref in images)
        result = np.zeros((20, 20, 3), dtype=np.uint8)
        images.append(weakref.ref(result))
        return result

    seq = SimpleNamespace(frames=list(range(20)), cameras={0: None, 1: None}, image=image,
                          gt_view=lambda c, f: GTFrame(ids=np.array([1]),
                                                       boxes=np.array([[1, 1, 15, 19]])))

    def load(root, kind, split):
        assert split == "train"
        return seq

    monkeypatch.setattr(reid_crops, "load_multiview", load)
    counts = build_reid_crops("unused", "wildtrack", tmp_path / "crops", min_height=1)
    assert counts == {"train": 32, "val": 8}
    assert len(images) == 40
    assert not any(ref() is not None for ref in images)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_reid_crops_reject_invalid_validation_fraction(tmp_path, fraction):
    with pytest.raises(ConfigError, match="validation_fraction"):
        build_reid_crops("unused", "wildtrack", tmp_path / "crops", validation_fraction=fraction)


def test_reid_crops_refuse_to_mix_existing_data(tmp_path):
    out = tmp_path / "crops"
    out.mkdir()
    marker = out / "old-test-crop.jpg"
    marker.write_bytes(b"existing")
    with pytest.raises(ConfigError, match="empty directory"):
        build_reid_crops("unused", "wildtrack", out)
    assert marker.read_bytes() == b"existing"


def test_reid_crops_report_image_write_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from multicam_tracker.data.base import GTFrame
    from multicam_tracker.tools import reid_crops

    seq = SimpleNamespace(frames=list(range(10)), cameras={0: None},
                          image=lambda c, f: np.zeros((20, 20, 3), dtype=np.uint8),
                          gt_view=lambda c, f: GTFrame(ids=np.array([1]),
                                                       boxes=np.array([[1, 1, 15, 19]])))
    monkeypatch.setattr(reid_crops, "load_multiview", lambda *a, **kw: seq)
    monkeypatch.setattr(reid_crops.cv2, "imwrite", lambda *a: False)
    with pytest.raises(OSError, match="failed to write re-ID crop"):
        build_reid_crops("unused", "wildtrack", tmp_path / "crops", min_height=1)
