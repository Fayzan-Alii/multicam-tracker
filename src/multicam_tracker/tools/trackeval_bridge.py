"""Score Multicam Tracker MOT17 results with the official TrackEval, for numbers you can publish.

Builds the folder layout TrackEval expects in a scratch directory (sequence GT, seqinfo.ini, a
seqmap, the tracker files) and runs its HOTA / CLEAR / Identity metrics with MOT17
preprocessing. For the val-half protocol the GT is trimmed to the evaluated frames.
Install: python -m pip install -r requirements.txt
"""

from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path

import numpy as np

from multicam_tracker.data.mot17 import load_mot17


def import_trackeval():
    import scipy.optimize  # noqa: F401  (must be imported before the aliases below)

    # TrackEval still uses aliases NumPy removed in 1.24.
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    # TrackEval imports every supported dataset and prints a warning when the
    # optional BURST mask dependency is absent. MOT17 does not use that package.
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval

    return trackeval


def run_trackeval(mot_root, results_dir, work_dir, split: str = "all", detector: str = "FRCNN",
                  sequences=()) -> dict:
    trackeval = import_trackeval()
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)
    gt_root, tr_root = work / "gt", work / "trackers" / "multicam_tracker" / "data"
    tr_root.mkdir(parents=True)
    names = []
    for seq in load_mot17(mot_root, "train", detector, sequences, split):
        result = Path(results_dir) / f"{seq.name}.txt"
        if not result.exists():
            continue
        names.append(seq.name)
        src = Path(mot_root) / "train" / seq.name
        dst = gt_root / seq.name / "gt"
        dst.mkdir(parents=True)
        shutil.copy2(src / "seqinfo.ini", gt_root / seq.name / "seqinfo.ini")
        keep = set(seq.frames)
        lines = [line for line in (src / "gt" / "gt.txt").read_text().splitlines()
                 if line.strip() and int(float(line.split(",")[0])) in keep]
        (dst / "gt.txt").write_text("\n".join(lines) + "\n")
        shutil.copy2(result, tr_root / result.name)
    if not names:
        raise FileNotFoundError(f"no result files in {results_dir} match MOT17 sequences under {mot_root}")
    seqmap = work / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(names) + "\n")

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                     "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                     "DISPLAY_LESS_PROGRESS": True})
    data_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    data_cfg.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(work / "trackers"),
                     "TRACKERS_TO_EVAL": ["multicam_tracker"], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
                     "SEQMAP_FILE": str(seqmap), "SKIP_SPLIT_FOL": True, "PRINT_CONFIG": False})
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    res, msg = trackeval.Evaluator(eval_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(data_cfg)], metrics)
    status = msg["MotChallenge2DBox"]["multicam_tracker"]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")
    combined = res["MotChallenge2DBox"]["multicam_tracker"]["COMBINED_SEQ"]["pedestrian"]
    hota, clear, ident = combined["HOTA"], combined["CLEAR"], combined["Identity"]
    return {
        "HOTA": float(np.mean(hota["HOTA"])), "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])), "LocA": float(np.mean(hota["LocA"])),
        "MOTA": float(clear["MOTA"]), "MOTP": float(clear["MOTP"]), "IDSW": int(clear["IDSW"]),
        "MT": int(clear["MT"]), "ML": int(clear["ML"]), "Frag": int(clear["Frag"]),
        "IDF1": float(ident["IDF1"]), "IDR": float(ident["IDR"]), "IDP": float(ident["IDP"]),
        "sequences": names,
    }
