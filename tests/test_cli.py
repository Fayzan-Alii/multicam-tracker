import json


from multicam_tracker.cli import main

from builders import write_multiview


def test_run_command_applies_overrides(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("dataset:\n  kind: synthetic\n  synthetic:\n    num_frames: 6\n    num_cameras: 2\n"
                   "detector:\n  kind: oracle\ntracker:\n  motion:\n    space: ground\n")
    out = tmp_path / "out"
    assert main(["run", "--config", str(cfg), "--set", f"output.dir={out}", "--set", "name=cli"]) == 0
    assert "ground_tracking" in json.loads((out / "metrics.json").read_text())


def test_verify_calibration_passes_on_consistent_data(tmp_path, capsys):
    people = {f: [(1, 40 + 480 * (200 + f))] for f in range(0, 20, 5)}
    write_multiview(tmp_path, "wildtrack", people)
    assert main(["verify-calibration", "--dataset", "wildtrack", "--root", str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_verify_calibration_fails_loudly_on_broken_calibration(tmp_path, capsys):
    people = {f: [(1, 40 + 480 * (200 + f))] for f in range(0, 20, 5)}
    write_multiview(tmp_path, "wildtrack", people)
    extr = tmp_path / "calibrations" / "extrinsic" / "extr_CVLab1.xml"
    text = extr.read_text()
    start = text.index("<tvec>") + len("<tvec>")
    end = text.index("</tvec>")
    values = [float(v) for v in text[start:end].split()]
    values[0] += 500.0  # camera C1 shifted 5 m sideways (WILDTRACK stores centimetres)
    extr.write_text(text[:start] + " ".join(f"{v:.6f}" for v in values) + text[end:])
    code = main(["verify-calibration", "--dataset", "wildtrack", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1 and "C1" in out and "FAIL" in out


def test_unknown_config_key_exits_with_message(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("trackr:\n  n_init: 3\n")
    assert main(["run", "--config", str(cfg)]) == 2
    assert "trackr" in capsys.readouterr().err
