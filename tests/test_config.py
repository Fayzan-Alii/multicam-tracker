import pytest

from multicam_tracker.config import AppConfig, ConfigError, build_config, load_config, parse_overrides


def write(path, text):
    path.write_text(text)
    return path


def test_base_file_is_inherited_and_overridden(tmp_path):
    write(tmp_path / "base.yaml", "tracker:\n  max_lost_s: 1.0\n  n_init: 3\nname: base\n")
    child = write(tmp_path / "child.yaml", "_base_: base.yaml\ntracker:\n  max_lost_s: 2.5\n")

    data = load_config(child)

    assert data["tracker"] == {"max_lost_s": 2.5, "n_init": 3}
    assert data["name"] == "base"


def test_base_chain_resolves_relative_to_each_file(tmp_path):
    (tmp_path / "a").mkdir()
    write(tmp_path / "root.yaml", "name: root\noutput:\n  dir: runs/root\n")
    write(tmp_path / "a" / "mid.yaml", "_base_: ../root.yaml\nname: mid\n")
    leaf = write(tmp_path / "a" / "leaf.yaml", "_base_: mid.yaml\noutput:\n  save_video: true\n")

    data = load_config(leaf)

    assert data["name"] == "mid"
    assert data["output"] == {"dir": "runs/root", "save_video": True}


def test_unknown_key_reports_dotted_path():
    with pytest.raises(ConfigError, match=r"tracker\.motion\.acel_noise"):
        build_config({"tracker": {"motion": {"acel_noise": 2.0}}})


def test_partial_config_fills_defaults():
    cfg = build_config({"tracker": {"motion": {"space": "ground"}}})

    assert isinstance(cfg, AppConfig)
    assert cfg.tracker.motion.space == "ground"
    assert cfg.tracker.n_init == AppConfig().tracker.n_init


def test_invalid_choice_is_rejected():
    with pytest.raises(ConfigError, match="tracker.motion.space"):
        build_config({"tracker": {"motion": {"space": "sideways"}}})


def test_non_mapping_section_is_rejected():
    with pytest.raises(ConfigError, match="tracker"):
        build_config({"tracker": 3})


def test_overrides_parse_yaml_values():
    assert parse_overrides(["tracker.max_lost_s=2", "ground.roi=[0, 0, 5, 5]", "name=exp one"]) == {
        "tracker": {"max_lost_s": 2},
        "ground": {"roi": [0, 0, 5, 5]},
        "name": "exp one",
    }


def test_override_without_equals_is_rejected():
    with pytest.raises(ConfigError, match="key=value"):
        parse_overrides(["tracker.max_lost_s"])
