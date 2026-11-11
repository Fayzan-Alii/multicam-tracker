"""The official-TrackEval bridge must agree exactly with Multicam Tracker's own evaluator, including MOT17
distractor preprocessing and the val-half GT trimming. Skipped without TrackEval."""

import numpy as np
import pytest

pytest.importorskip("scipy.optimize")
from multicam_tracker.tools.trackeval_bridge import import_trackeval  # noqa: E402

try:
    import_trackeval()
except ImportError:
    pytest.skip("TrackEval not installed", allow_module_level=True)

from multicam_tracker.data.mot17 import load_mot17, write_mot_results  # noqa: E402
from multicam_tracker.pipeline.evaluate import evaluate_boxes  # noqa: E402
from multicam_tracker.tools.trackeval_bridge import run_trackeval  # noqa: E402

from builders import write_mot17  # noqa: E402

N = 8
GT = [(f, 1, 10 + 4 * f, 20, 30, 80, 1, 1, 1.0) for f in range(1, N + 1)]
GT += [(f, 2, 150, 30, 30, 80, 1, 1, 1.0) for f in range(1, N + 1)]
GT += [(f, 3, 220, 40, 25, 60, 1, 8, 1.0) for f in range(1, N + 1)]  # distractor
GT += [(f, 4, 60, 100, 20, 50, 0, 7, 0.3) for f in range(1, N + 1)]  # static person, ignored


def tracker_output():
    results = {}
    for f in range(1, N + 1):
        rows = [(1 if f <= 5 else 9, np.array([10 + 4 * f, 20, 40 + 4 * f, 100.0]), 0.9)]  # ID switch at 6
        if f != 3:
            rows.append((2, np.array([151.0, 31, 181, 111]), 0.8))  # missed once
        rows.append((5, np.array([220.0, 40, 245, 100]), 0.7))  # sits on the distractor: dropped
        if f == 4:
            rows.append((7, np.array([270.0, 120, 290, 180]), 0.6))  # false positive
        results[f] = rows
    return results


@pytest.mark.parametrize("split", ["all", "second_half"])
def test_trackeval_bridge_matches_internal_evaluator(tmp_path, split):
    root = tmp_path / "MOT17"
    write_mot17(root, n_frames=N, size=(200, 300), gt_rows=GT)
    (seq,) = load_mot17(root, split=split)
    results = {f: r for f, r in tracker_output().items() if f in seq.frames}
    write_mot_results(tmp_path / "run" / "mot" / f"{seq.name}.txt", results)

    ours = evaluate_boxes(seq.gt, results, seq.frames, 0.5, mot17_filter=True)
    official = run_trackeval(root, tmp_path / "run" / "mot", tmp_path / "te", split=split)
    for key in ("HOTA", "MOTA", "IDF1", "DetA", "AssA"):
        assert official[key] == pytest.approx(ours[key], abs=1e-9), key
    assert official["IDSW"] == ours["IDSW"]
