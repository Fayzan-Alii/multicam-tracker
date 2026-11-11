# MOT17 occlusion investigation (2026-09-29)

Same fine-tuned YOLO, cached detections, seven videos, second-half validation frames. These are validation results, not independent test results.

The second matching pass incorrectly penalised low-confidence boxes with score fusion. Cost was 1 - IoU * confidence, so confidence 0.3 could never pass threshold 0.5, even at perfect overlap. The correction uses box overlap alone in this rescue pass; other passes retain score fusion.

| Variant | HOTA | MOTA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| original | 0.5330 | 0.5495 | 0.6342 | 203 | 663 |
| rescue_fix | 0.5336 | 0.5584 | 0.6341 | 222 | 440 |
| rescue_fix_memory2s | 0.5336 | 0.5619 | 0.6331 | 220 | 441 |
| rescue_fix_no_score | 0.5346 | 0.5599 | 0.6353 | 221 | 456 |
| rescue_fix_strict | 0.5349 | 0.5618 | 0.6351 | 220 | 457 |
| rescue_fix_lower_high | 0.5358 | 0.5488 | 0.6269 | 407 | 633 |

Original is the saved result before the correction; all other rows include it. memory2s changes max_lost_s to 2; no_score disables score fusion in all passes; strict sets second_match_thresh to 0.3; lower_high sets high_thresh to 0.4 and new_track_thresh to 0.5. Exact configurations are saved in each run directory.

Official TrackEval confirms correction-only results: HOTA 0.533645, MOTA 0.558360, IDF1 0.634054, IDSW 222, Frag 440. Interruptions fell from 663 to 440, but switches rose from 203 to 222. On MOT17-02, switches rose from 61 to 74. This does NOT solve identity switching. Keep the algorithm correction, but none of the parameter changes were adopted.

All 218 tests passed, including regression tests for low-confidence rescue and preserved high-confidence score fusion.

Reproduce from the project root with the environment activated:

```bash
python -m multicam_tracker run --config runs/mot17/occlusion_check/rescue_fix/config.yaml
python -m multicam_tracker trackeval --config runs/mot17/occlusion_check/rescue_fix/config.yaml
make mot17-occlusion-demo
```

Original preview: runs/mot17/finetuned_02_preview/demo_MOT17-02-FRCNN.mp4. Corrected preview: runs/mot17/occlusion_demo/demo_MOT17-02-FRCNN.mp4.

All per-sequence comparisons: runs/mot17/occlusion_check/comparison.json. Official corrected scores: runs/mot17/occlusion_check/rescue_fix/metrics_trackeval.json.

Appearance-based re-identification remains disabled. Next: evaluate a suitable trained appearance model to distinguish people at crossings. No Re-ID model was downloaded or trained during this investigation.
