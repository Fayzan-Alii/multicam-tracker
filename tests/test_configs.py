"""Every shipped config must validate, so a typo fails here rather than hours into a run."""

from pathlib import Path

import pytest

from multicam_tracker.config import build_dataclass, config_from_file, load_config
from multicam_tracker.reid.train import ReIDTrainConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TRACKING = sorted(p for p in CONFIGS.rglob("*.yaml") if "reid" not in p.parts and not p.name.startswith("_"))
REID = sorted((CONFIGS / "reid").glob("*.yaml"))


def test_there_are_configs_to_check():
    assert len(TRACKING) >= 10 and len(REID) >= 2


@pytest.mark.parametrize("path", TRACKING, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_tracking_config_validates(path):
    cfg = config_from_file(path)
    assert cfg.name and cfg.output.dir.startswith("runs/")


@pytest.mark.parametrize("path", REID, ids=lambda p: p.name)
def test_reid_config_validates(path):
    cfg = build_dataclass(ReIDTrainConfig, load_config(path))
    assert cfg.output_dir.startswith("runs/")


def test_ablation_names_are_unique():
    names = [config_from_file(p).name for p in (CONFIGS / "wildtrack" / "ablations").glob("*.yaml")]
    assert len(names) == len(set(names))


def test_shipped_synthetic_config_runs(tmp_path):
    from multicam_tracker.pipeline.runner import run

    cfg = config_from_file(CONFIGS / "synthetic.yaml",
                           ["dataset.synthetic.num_frames=8", f"output.dir={tmp_path}", "output.save_video=false"])
    assert run(cfg, log=lambda *a: None)["metrics"]["ground_tracking"]["IDF1"] > 0.8
