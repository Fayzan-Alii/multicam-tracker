# Multicam Tracker: instruction manual

This manual takes you from a fresh machine to published results: setup, data, the
calibration check, training, evaluation, ablations, the demo video, and what to say about it
in an interview. Commands are run from the repository root unless a step says otherwise.

**New to a term?** Every technical word used here is explained in plain language in the
[glossary at the end](#appendix-glossary).

---

## Start here: your first hour

Do these in order. Each step says how to tell it worked; the last column points to the
section with the details.

| # | Step | Command or action | It worked if... | Section |
|---|---|---|---|---|
| 1 | Check the GPU is visible in WSL | `nvidia-smi` | it lists the RTX 2060 | 1.1 |
| 2 | Create an environment with CUDA PyTorch | `make setup`, then the GPU check | `torch.cuda.is_available()` prints `True` | 1.2, 1.3 |
| 3 | Install Multicam Tracker | `make setup` | it finishes without errors | 1.4 |
| 4 | Run the tests | `make test` | everything passes (a few may be skipped) | 1.5 |
| 5 | Smoke test, no data needed | `make smoke` | the numbers match section 1.5, and `runs/synthetic/demo.gif` shows labelled boxes beside a map | 1.5 |
| 6 | Start the MOT17 download | from motchallenge.net, about 5.5 GB | the download is running (it takes a while) | 2.1 |
| 7 | While it downloads | read sections 3 and 9 | you know what the output files and scores mean | 3, 9 |
| 8 | Get your first real result | unzip into `data/MOT17`, then `multicam-tracker run --config configs/mot17/public_frcnn.yaml` | `runs/mot17/public_frcnn/metrics.json` has a `COMBINED` section | 4.1 |

After this hour you have a working GPU setup, a passing test suite and your first real
tracking scores. Everything after that is in the milestone sections, in order.

---

## 0. Where things stand

**Done and tested.** Every component is written and covered by over 200 tests: config, camera
geometry, both Kalman motion models, the hand-written Hungarian solver, the ByteTrack-style
tracker, cross-camera fusion, re-ID training and evaluation, the MOTA/IDF1/HOTA metrics
(identical to official TrackEval on randomised checks), dataset loaders, visualisation, ONNX
export, and the CLI. The whole pipeline runs end to end on a synthetic 4-camera scene. The YOLO
wrapper was checked on a real image, and ONNX export was checked for numerical parity with
PyTorch.

**Left for you:**

1. Download the datasets and run the pipeline on real data. The loaders follow each dataset's
   documented format but have never read the real files. That is why step 2.2
   (`verify-calibration`) comes first: it tells you within a minute whether the WILDTRACK
   conventions are right.
2. Train the re-ID model (and optionally fine-tune YOLO) on your RTX 2060.
3. Put the numbers in the README tables.

**Suggested order** (each step leaves the repo in a state you can link to):

| Milestone | What you run | GPU time (rough) | Result you can show |
|---|---|---|---|
| M1 | MOT17: public detections, then YOLO | minutes | Hand-written tracker with official HOTA/MOTA/IDF1 |
| M2 | WILDTRACK: calibration check, ground-plane tracking | minutes | Bird's-eye map, image-vs-ground ablation |
| M3 | Train re-ID on Market-1501 | ~1–3 h | Rank-1/mAP, appearance ablation |
| M4 | Full fusion, all ablations, demo video | minutes | Seven-panel + map demo, full ablation table |
| M5 | ONNX export, latency benchmark (optional) | minutes | Per-stage latency table |

**The scores, in one line each.** They are percentages where higher is better (MOTA can even
go negative when mistakes outnumber people). Section 9 explains them properly.

- **MOTA**: starts at 100% and loses points for every missed person, false box and ID switch,
  relative to how many people there are. Mostly measures detection quality.
- **IDF1**: how consistently each person keeps the same ID over time. The main score for
  judging the tracking itself.
- **HOTA**: one balanced score combining "found the people" (**DetA**) and "kept their IDs"
  (**AssA**). The modern headline number.
- **MODA / MODP**: multi-camera detection on the ground: whether a position was found near
  each person (MODA), and how close it was (MODP).
- **Rank-1 / mAP**: re-ID quality: how often the correct person is the top match (Rank-1),
  and how well all correct matches rank overall (mAP).

---

## 1. Setup (Windows + WSL2 + RTX 2060 Max-Q)

### 1.1 GPU inside WSL2

Install the normal **Windows** NVIDIA driver. Do *not* install a Linux GPU driver inside WSL,
because the Windows driver exposes the GPU to WSL. Then check from the WSL shell:

```bash
nvidia-smi          # should list the RTX 2060 and a CUDA version
```

Two WSL-specific habits save a lot of pain:

- **Keep data on the Linux filesystem** (e.g. `~/work/multicam-tracker/data`), not under `/mnt/c/...`.
  Reading hundreds of thousands of small images across the Windows boundary is many times slower.
- WSL gets about half of the machine's RAM by default. If training processes get killed, lower
  `num_workers`, or raise the limit in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=...`).

### 1.2 Python environment

From the project directory, the shortest setup command is:

```bash
make setup
```

This creates `.venv` if needed, installs pip, installs CUDA 12.6 PyTorch, and installs Multicam Tracker
with the detection, ONNX, and testing packages. An existing `.venv` is reused. The Makefile
uses its Python directly, so commands such as `make test` need no environment activation.

If you prefer to perform setup manually with Python and pip, use the following steps.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
```

### 1.3 PyTorch with CUDA

For this WSL/NVIDIA setup:

```bash
python -m pip install --timeout 300 torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The check should print `CUDA: True` and your GPU's name. Other machines may need a different
PyTorch index. `make setup` already performs the installation command above.

### 1.4 Multicam Tracker

```bash
python -m pip install --timeout 300 -r requirements.txt
```

`make setup` also performs this step. `requirements.txt` lists the actual packages for
tracking, YOLO, ONNX export and inference, testing, and TrackEval. It also installs
Multicam Tracker in editable mode. The TrackEval source revision is pinned in that file.
The small `pyproject.toml` only configures local package installation and test discovery.

Use `make help` for short commands. For bare `multicam-tracker`, `pytest`, or `python` commands in the
rest of this manual, activate the environment with `source .venv/bin/activate`. You can also
write `python -m multicam_tracker` wherever this manual uses `multicam-tracker`.

Keep `pyproject.toml` and `requirements.txt` in version control. `.venv` stays ignored.
The requirements file selects packages but does not lock every dependency version.
ONNX Runtime currently uses CPU inference; GPU inference would need a separate dependency
change to `onnxruntime-gpu`.

### 1.5 Check the install

```bash
make test           # run tests, including official TrackEval comparisons
make smoke          # synthetic demo, no dataset download needed
```

The synthetic run should print numbers close to these (the scene is seeded, so they should
match almost exactly):

```
  local              HOTA 0.852  MOTA 0.943  IDF1 0.971  ...  IDSW 0
  ground_detection   MODA 1.000  MODP 0.958  precision 1.000  recall 1.000
  ground_tracking    HOTA 0.999  MOTA 1.000  IDF1 1.000  ...  IDSW 0
  cross_camera       HOTA 0.852  MOTA 0.943  IDF1 0.971  ...  IDSW 0
```

It also writes `runs/synthetic/demo.mp4` and `demo.gif`. Open them: four camera panels in
which each person has the same colour and `G<n>` label in every view, next to a top-down
map. These numbers use an *oracle* detector (ground-truth boxes plus noise), so they only
show that the plumbing works. They are not a result.

---

## 2. Datasets

Final layout (all paths are the config defaults; change `dataset.root` if yours differ):

```
data/
  MOT17/
    train/MOT17-02-FRCNN/{seqinfo.ini, img1/, gt/gt.txt, det/det.txt}   ... (7 sequences x 3 detectors)
    test/...
  Wildtrack/
    Image_subsets/C1 ... C7/00000000.png ...
    annotations_positions/00000000.json ...
    calibrations/intrinsic_zero/intr_CVLab1.xml ...   calibrations/extrinsic/extr_CVLab1.xml ...
  MultiviewX/                      (optional; same structure, 6 cameras, calibrations/intrinsic/)
  Market-1501-v15.09.15/
    bounding_box_train/  query/  bounding_box_test/
```

Download pages change over time. If a link is dead, search for the dataset name; what matters
is that the folder structure above matches.

### 2.1 MOT17

<https://motchallenge.net/data/MOT17/>. Download the full dataset zip (about 5.5 GB), unzip
into `data/MOT17`. Every training sequence appears three times (`-DPM`, `-FRCNN`, `-SDP`) with
identical images and different public detections. Multicam Tracker uses one copy (`dataset.mot17_detector`,
default `FRCNN`) so sequences are not counted three times.

### 2.2 WILDTRACK, then run the calibration check first

Download from the EPFL CVLab page (<https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>) and
unzip into `data/Wildtrack`. Before anything else, run:

```bash
multicam-tracker verify-calibration --dataset wildtrack --root data/Wildtrack
```

This projects every annotated ground position into each camera and measures the pixel
distance to the bottom-centre of that person's annotated box.

- **`OK` for all seven cameras** means the calibration files, the centimetre-to-metre
  conversion and the `positionID` grid convention are all being read correctly. Expect median
  errors of a few pixels up to around 20. That residual is real: a box's bottom edge is the
  nearest foot, not the centre of the person's footprint.
- It then prints a **suggested `ground.foot_depth_offset_m`**: how far, in metres, true
  positions lie beyond the back-projected box bottom. Put that value into
  `configs/wildtrack/base.yaml`. It moves each ground point back to the person's centre, so
  cameras on opposite sides of a person agree on where they stand. Measure the effect later
  with ablations; this is one of the more interesting knobs.
- **`FAIL`** prints the median error under alternative conventions (x/y swapped, units off by
  100, translation not rescaled). If one alternative is dramatically better, the convention in
  `src/multicam_tracker/data/multiview.py` (`SPECS["wildtrack"]`) needs that change. If only one camera
  fails, look at that camera's calibration files.

Do the same for MultiviewX if you use it (`--dataset multiviewx`).

WILDTRACK facts that shape the configs: 7 synchronised 1920×1080 cameras; 400 annotated frames
at **2 fps**; the standard split (from MVDet) is the first 90% of frames for training and the
**last 10% (40 frames) for testing**.

### 2.3 MultiviewX (optional)

<https://github.com/hou-yz/MultiviewX>: the download link is in its README. It is synthetic,
with the same format as WILDTRACK and 6 cameras. Useful as a second multi-view dataset in the
results table.

### 2.4 Market-1501

The original project page is often offline; copies are hosted on Kaggle (search
"Market-1501"). You need `bounding_box_train/` (751 identities), `query/` and
`bounding_box_test/`. File names look like `0002_c1s1_000451_03.jpg` (identity 2, camera 1).
Identity `-1` files are junk and are skipped automatically. Identity `0000` files are
distractors: they stay in the gallery, where they can only produce wrong matches, which is the
standard protocol.

---

## 3. How runs work

**Configs.** One YAML file describes one experiment. Files inherit with `_base_:` (a path
relative to the file) and override only what differs. `configs/_defaults.yaml` holds shared
tracker, fusion and eval settings. Any value can be overridden from the command line:

```bash
multicam-tracker run --config configs/wildtrack/base.yaml --set tracker.max_lost_s=2.0 --set name=lost2s --set output.dir=runs/wildtrack/lost2s
```

Unknown keys are errors (`trackr.n_init` fails immediately with the offending path), so a typo
never silently runs the default.

**Outputs** (`output.dir`, e.g. `runs/wildtrack/base/`):

| File | Contents |
|---|---|
| `config.yaml` | the fully resolved config that produced this run |
| `metrics.json` | every metric (sections described below) |
| `mot/*.txt` | per-sequence or per-camera tracks in MOTChallenge format |
| `global_tracks.csv` | multi-view only: frame, global id, x, y (m), covariance, contributing cameras |
| `demo.mp4`, `demo.gif` | when `output.save_video: true` |

**Metric sections** in `metrics.json`:

| Section | Meaning |
|---|---|
| `COMBINED` (MOT17) | all sequences pooled the way TrackEval's COMBINED row pools them |
| `local` | per-camera tracking in image space against per-view GT boxes, pooled over cameras |
| `per_camera` | the same, per camera |
| `ground_detection` | MODA / MODP / precision / recall of fused positions (0.5 m radius) |
| `ground_tracking` | HOTA / MOTA / IDF1 of fused identities on the ground plane (1 m radius) |
| `cross_camera` | IDF1 etc. where every (frame, camera) is a timestep and IDs are *global*: rewards one person keeping one ID across all views |
| `diagnostics` | per camera: detections rejected because their foot point was above the horizon or outside the ROI, and frames that could not be read (their tracks coast). A camera with far more rejections than the others usually means a calibration or ROI problem |

**Caches.** Detection is by far the slowest stage, so when `detector.cache_dir` is set the
boxes (and re-ID embeddings) are saved on the first run. Every later run with the same
detector settings reuses them, and never even loads YOLO. Cache files are keyed by a hash of
the detector configuration, so changing the weights or `imgsz` simply creates new cache files
next to the old ones. Delete `cache/` whenever you like; it only costs a re-run.

---

## 4. Milestone 1: single-camera tracking on MOT17

MOT17 has no camera calibration, so tracking here is image-space
(`tracker.motion.space: image`, ByteTrack's Kalman parameterisation). A config asking for
ground-plane motion on MOT17 stops with an error explaining why.

### 4.1 Tracker only, with public detections

```bash
multicam-tracker run --config configs/mot17/public_frcnn.yaml      # CPU is fine; ~1-2 min
```

Every tracker that uses these public FRCNN detections sees exactly the same boxes, so this
number measures association quality alone.

### 4.2 Full pipeline with YOLO

```bash
multicam-tracker run --config configs/mot17/yolo_coco.yaml
```

The first run downloads `yolo11s.pt` automatically and fills `cache/mot17/yolo11s_coco/`. The
COCO-pretrained model never saw MOT17, so evaluating on all training frames is legitimate. Any
Ultralytics weights work (`--set detector.weights=yolo11m.pt`); `yolo11m` at `imgsz 1280` still
fits comfortably in 6 GB for inference.

### 4.3 Official numbers (use these in the README)

```bash
multicam-tracker trackeval --config configs/mot17/public_frcnn.yaml     # after the run above
multicam-tracker trackeval --config configs/mot17/yolo_coco.yaml
```

This builds the folder layout TrackEval expects, runs the official HOTA/CLEAR/Identity code
with MOT17 preprocessing, and writes `metrics_trackeval.json`. TrackEval still uses `np.float`,
which NumPy removed in 1.24; the bridge restores the two aliases it needs, so no downgrade is
required. The built-in evaluator produces identical numbers (the test suite checks this), so
use the built-in one for quick iteration.

### 4.4 Fine-tune YOLO on MOT17 (optional)

Use the ByteTrack "half" protocol: train on the first half of every training sequence and
evaluate on the second halves.

```bash
multicam-tracker prepare-yolo --dataset mot17 --root data/MOT17 --out data/yolo_mot17
multicam-tracker train-detector --data data/yolo_mot17/data.yaml --weights yolo11s.pt --imgsz 1280 --batch 4 \
      --epochs 50 --project runs/detector --name mot17
multicam-tracker run --config configs/mot17/yolo_finetuned_val.yaml
multicam-tracker trackeval --config configs/mot17/yolo_finetuned_val.yaml
```

- 6 GB guidance: `yolo11s` at `--imgsz 1280 --batch 4` with AMP (on by default). On CUDA
  out-of-memory, try `--batch 2` or `--imgsz 960`.
- Pedestrians, people on vehicles and static people are all labelled `person`. Distractors and
  reflections are left unlabelled.
- Ultralytics writes to `runs/detector/mot17/weights/best.pt`, which is the path the config
  expects. If it adds a suffix (`mot172`), update `detector.weights`.
- Compare `yolo_finetuned_val` only with other second-half numbers. Evaluating a fine-tuned
  model on frames it trained on is not a valid result.

### 4.5 Demo clip

```bash
multicam-tracker run --config configs/mot17/yolo_coco.yaml --set output.save_video=true --set output.max_video_frames=300 \
          --set dataset.sequences=[MOT17-04] --set output.dir=runs/mot17/demo
```

This writes `demo_MOT17-04-FRCNN.mp4` and a GIF. MOT17-04 (a static camera over a busy
square) makes a clear clip.

**At the end of M1** the README can already show a hand-written tracker with official
HOTA/MOTA/IDF1 on public and private detections, plus a clip. That is a complete, honest
project to link while you continue.

---

## 5. Milestone 2: geometry and the ground plane (WILDTRACK)

```bash
multicam-tracker verify-calibration --dataset wildtrack --root data/Wildtrack    # section 2.2, if not done yet
multicam-tracker run --config configs/wildtrack/base.yaml
```

`base.yaml` detects with COCO YOLO at `imgsz 1280`, projects foot points to the ground with
propagated covariance, tracks each camera in metres, and fuses across cameras. It evaluates on
the **test split** (last 40 frames), the standard protocol. Use it whenever anything was
fine-tuned on WILDTRACK. With purely off-the-shelf models, `--set dataset.split=all` evaluates
on all 400 frames and gives much more statistically solid numbers. Say which one you report.

The key experiment of this milestone is decision 1, image-space vs world-space tracking:

```bash
multicam-tracker ablate --configs configs/wildtrack/ablations/01_image_space.yaml configs/wildtrack/ablations/02_ground_space.yaml
```

At 2 fps a walking person moves about 0.7 m between frames, often more than their box width
in the image, so IoU association should suffer badly in the `local` and `cross_camera`
columns. If it does not, that is worth understanding too (see section 9).

For people standing far from cameras, try `detector.imgsz: 1920` (native resolution). Small
distant people are the main source of missed detections on WILDTRACK.

---

## 6. Milestone 3: appearance re-ID

### 6.1 Train on Market-1501

```bash
multicam-tracker train-reid --config configs/reid/market1501_r50.yaml
```

This is the Bag of Tricks recipe: ImageNet ResNet-50, last stride 1, BNNeck, label-smoothed
cross-entropy plus batch-hard triplet loss, random erasing, 10-epoch warmup, and steps at 40
and 70 over 120 epochs. The paper reports about **94% Rank-1 / 86% mAP** on Market-1501. A
first run landing a couple of points below that is normal.

- **Memory:** batch 64 (P=16 identities × K=4 images) at 256×128 with AMP should fit in 6 GB.
  On out-of-memory, `--set p=8` (batch 32).
- **Time:** the log prints seconds per epoch. Expect very roughly 1–3 hours for 120 epochs on a
  laptop RTX 2060. For a faster first pass, `configs/reid/market1501_r18.yaml` (ResNet-18, 60
  epochs).
- **Evaluation** runs every 10 epochs; `best.pt` keeps the best mAP and `last.pt` the latest
  state.
- **Resuming** is automatic: rerun the same command and it continues from `last.pt`. This also
  makes Kaggle sessions (section 6.5) practical.

### 6.2 Evaluate a checkpoint

```bash
multicam-tracker eval-reid --checkpoint runs/reid/market_r50/best.pt --data-root data/Market-1501-v15.09.15
```

### 6.3 Use it for tracking

```bash
multicam-tracker run --config configs/wildtrack/reid.yaml        # re-ID in both the local trackers and fusion
multicam-tracker run --config configs/mot17/yolo_coco_reid.yaml
```

Appearance enters the association cost as `(1 - w)·motion + w·(cosine distance)` with
`w = appearance_weight`. Pairs whose appearance distance exceeds `appearance_gate` are ruled
out. Appearance can re-rank physically plausible pairs but never allows a physically
impossible match.

### 6.4 Fine-tune on WILDTRACK (closes the domain gap)

Market-1501 is a different campus, different cameras and much closer crops. Fine-tuning on
WILDTRACK identities, which are consistent across all seven views, usually helps:

```bash
multicam-tracker build-reid-crops --dataset wildtrack --root data/Wildtrack --out data/wildtrack_reid
multicam-tracker train-reid --config configs/reid/wildtrack_finetune.yaml
multicam-tracker run --config configs/wildtrack/reid.yaml --set reid.checkpoint=runs/reid/wildtrack_r50/best.pt \
          --set name=wildtrack-reid-ft --set output.dir=runs/wildtrack/reid_ft
```

Two cautions to state in your write-up:

- Both training and checkpoint-selection crops come only from the **dataset train split**.
  Its first 80% of frames supplies `bounding_box_train`; the last 20% supplies validation
  `query` and `bounding_box_test` folders (despite that conventional folder name, these are
  validation images). At least two frames are reserved for validation. Change the fraction
  with `--validation-fraction`; `split.json` records the actual frames. All camera views of
  a frame stay in the same partition. Tracking is evaluated on the untouched `split: test`,
  the default in the WILDTRACK configs.
- The same people may appear in training and validation (it is one continuous recording),
  so validation Rank-1/mAP is not comparable to Market-1501's disjoint-identity protocol.
  Report the final tracking metrics on the held-out test split.

Crops are written one image at a time to bound memory use. The output directory must be
empty: use a new path when rebuilding crops, and set `data_root` accordingly. Crops from
older versions used test frames for validation and must be regenerated; use a new training
`output_dir` as well so an old checkpoint is not resumed.

### 6.5 Training on Kaggle instead (optional)

Upload the repo and Market-1501 as Kaggle datasets and enable a GPU. `/kaggle/input` is
read-only, so copy the repo to the working directory first. Kaggle's PyTorch is already a
CUDA build, and the install keeps it. In a notebook:

```bash
cp -r /kaggle/input/<multicam-tracker-repo> /kaggle/working/multicam-tracker && cd /kaggle/working/multicam-tracker
python -m pip install -r requirements.txt
multicam-tracker train-reid --config configs/reid/market1501_r50.yaml \
  --set data_root=/kaggle/input/<market>/Market-1501-v15.09.15 --set output_dir=/kaggle/working/market_r50
```

If the session ends, start a new one with the previous `output_dir` contents restored and the
same command: it resumes. Download `best.pt` and point `reid.checkpoint` at it locally.

---

## 7. Milestone 4: full system, ablations, demo

### 7.1 The ablation table

```bash
multicam-tracker ablate --configs configs/wildtrack/ablations/0[1-5]_*.yaml --out runs/wildtrack/ablations.md
# after training re-ID, add row 6:
multicam-tracker ablate --configs configs/wildtrack/ablations/*.yaml --out runs/wildtrack/ablations.md
```

| Config | Tests |
|---|---|
| `01_image_space` | decision 1: pixel-space Kalman instead of metres |
| `02_ground_space` | the reference configuration |
| `03_fixed_noise` | decision 2: fixed 0.3 m isotropic noise instead of propagated covariance |
| `04_no_second_stage` | decision 3: no ByteTrack low-score association |
| `05_no_fusion` | no cross-camera fusion (only per-camera metrics) |
| `06_reid` | + appearance (needs a checkpoint) |

Detections are cached after the first config, so the whole table takes minutes. The output is
Markdown you can paste into the README. Add rows of your own the same way: a small YAML file
with `_base_: ../base.yaml` and the one setting it changes.

### 7.2 Upper bound: perfect detections

```bash
multicam-tracker run --config configs/wildtrack/oracle.yaml
```

This replaces YOLO with the ground-truth boxes (2 px noise, 5% dropped). The gap between this
and `base` is what a better detector could buy. The remaining gap to 100% is association and
geometry error. Showing you know where your errors come from is a strong interview point.

### 7.3 The demo video

```bash
multicam-tracker run --config configs/wildtrack/demo.yaml       # needs a re-ID checkpoint; or add --set reid.enabled=false
```

This renders all 400 frames: seven camera panels with boxes coloured and labelled by
**global** identity, beside the ground-plane map (camera fields of view, 2σ uncertainty
ellipses, trails, ground-truth crosses). It writes `runs/wildtrack/demo/demo.mp4` and a
README-sized `demo.gif`; copy the GIF to `docs/demo.gif` and uncomment the line near the top
of the README. `output.video_fps` sets the playback speed (4 = twice real time), and
`output.max_video_frames` limits the length.

### 7.4 MultiviewX

```bash
multicam-tracker verify-calibration --dataset multiviewx --root data/MultiviewX
multicam-tracker run --config configs/multiviewx/base.yaml
```

---

## 8. Milestone 5: deployment (optional)

```bash
multicam-tracker export-onnx --checkpoint runs/reid/market_r50/best.pt --out runs/reid/market_r50/reid.onnx
multicam-tracker run --config configs/wildtrack/reid.yaml --set reid.kind=onnx --set reid.checkpoint=runs/reid/market_r50/reid.onnx \
          --set detector.cache_dir="" --set name=wildtrack-onnx --set output.dir=runs/wildtrack/onnx
multicam-tracker benchmark --config configs/wildtrack/reid.yaml --set detector.cache_dir="" --frames 100
```

- The export uses a dynamic batch dimension and is checked numerically against PyTorch before
  the command reports success.
- ONNX Runtime uses CUDA when `onnxruntime-gpu` is installed, and CPU otherwise.
- `benchmark` prints mean and 95th-percentile milliseconds per stage (detect, embed, ground,
  track, fuse). Clear `cache_dir` as shown, or it measures cache lookups instead of the models.
- YOLO can be exported with Ultralytics (`yolo export model=yolo11s.pt format=onnx imgsz=1280`),
  and the `.onnx` file used directly as `detector.weights`.

A latency table (PyTorch vs ONNX, per stage, frames per second across 7 cameras) makes a good
closing section for the README.

---

## 9. Reading the numbers

- **MOTA** is dominated by detection errors (FP + FN), with ID switches a small term. Two
  trackers with the same detector have similar MOTA even when their identity handling is very
  different.
- **IDF1** measures how long each identity keeps the right ID. It is the metric that moves when
  association improves. For this project, IDF1 and **AssA** are the headline numbers.
- **HOTA** is the geometric mean of detection accuracy (DetA) and association accuracy (AssA),
  averaged over localisation thresholds. It is the modern single number to headline, and it
  shows *which* side is weak.
- **MODA / MODP** (ground plane) are detection-only: whether a fused position exists within
  0.5 m of every person. This is the metric multi-view detection papers report. MVDet (ECCV
  2020) is the usual reference; check its paper for the exact number before quoting it.
- **`cross_camera` IDF1** is specific to this project and is the best single measure of
  whether fusion works: one person, one ID, in every view.
- **Compare like with like:** the same split, the same detector, the same radius. The 40-frame
  WILDTRACK test split is short, so differences of a point or two are within noise. Say so
  rather than over-claim.

---

## 10. Tuning guide

| Symptom | Likely cause | Knobs |
|---|---|---|
| Same person split into two global IDs from different cameras | cross-view disagreement on ground position | `ground.foot_depth_offset_m` (from verify-calibration), `fusion.gate_chi2` up, `fusion.merge_chi2` up |
| Two nearby people merged into one global ID | gates too loose | `fusion.gate_chi2` / `cluster_thresh` down; enable re-ID; `fusion.appearance_gate` down |
| Local IDs switch when people cross | motion-only association | enable re-ID, `tracker.appearance_weight` 0.3 to 0.5 |
| Tracks die after short occlusions | lost too soon | `tracker.max_lost_s` up (2 fps: 1.5 s is only 3 frames) |
| Tracks die when people turn or stop suddenly | motion model too stiff | `tracker.motion.accel_noise` up (2 to 3), `tracker.gate_chi2` up |
| Many short false tracks | low-confidence starts | `tracker.new_track_thresh` up, `tracker.n_init` 3 |
| Distant people missed | small in the image | `detector.imgsz: 1920`, larger YOLO, fine-tune (section 4.4 with `--dataset wildtrack`) |
| Far-away positions jump around | large ground uncertainty | expected: check that ellipses in the demo grow with distance; `ground.foot_sigma_v` |

Change one thing at a time, and record every change as an ablation config rather than a
one-off `--set`, so the README table stays reproducible.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is False in WSL | Windows NVIDIA driver installed? `nvidia-smi` works in WSL? Does the PyTorch source index match your GPU (section 1.3)? |
| CUDA out of memory, re-ID training | `--set p=8`; keep `amp: true` |
| CUDA out of memory, YOLO training | `--batch 2`, `--imgsz 960` |
| Training killed without an error | WSL RAM limit: lower `num_workers`, raise the limit in `.wslconfig` |
| `verify-calibration` prints FAIL | read the alternative-convention ranking it prints (section 2.2) |
| `unknown config key(s): ...` | typo in YAML or `--set`; the message names the exact path |
| YOLO weights fail to download | download the `.pt` from the Ultralytics GitHub releases and set `detector.weights` to its path |
| `multicam-tracker trackeval` import error | `make setup` |
| Results unchanged after changing detector settings | not possible through the cache (it is keyed on those settings); check that `--set` targeted `detector.*` |
| Very slow first WILDTRACK run | that is detection filling the cache (7 × 400 full-HD frames); later runs are fast |
| Images load slowly | data is under `/mnt/c`; move it to the Linux filesystem |

---

## 12. Interview notes

**Pitch (30 seconds).** "Multicam Tracker tracks people across seven overlapping cameras and gives each
person one identity on a shared metric map. I wrote the tracker, the geometry and the metrics
myself. The metrics match the official TrackEval exactly. And every design decision has an
ablation behind it."

**The four decisions, and the evidence for each:**

1. *World-coordinate tracking.* Velocity in m/s is physical and camera-independent, and gates
   are in metres. Evidence: ablation 01 vs 02, plus the low-frame-rate regression tests. At
   2 fps a box moves further than its own width between frames, which is where IoU matching
   breaks.
2. *Uncertainty propagation.* Σ_world = J Σ_px Jᵀ, where J is the Jacobian of the
   image→ground homography (`geometry/camera.py`, `homography_jacobian`). Error grows roughly
   with the square of distance near the horizon. Gating is Mahalanobis, and fusion is
   inverse-covariance weighted, so a close camera outvotes a distant one. Evidence: ablation 03.
3. *Two-stage association.* Occluded people produce low-confidence boxes. Matching them only
   against existing tracks recovers them without letting them start false tracks. Evidence:
   ablation 04 and `tests/test_tracker.py`.
4. *Fuse tracks, not detections.* Sticky local→global links, per-camera assignment (an
   identity never takes two boxes from one camera), cannot-link clustering, and duplicate
   merging only after consecutive frames of agreement. Evidence: ablation 05, the
   `cross_camera` IDF1, and `tests/test_fusion.py`.

**Questions to expect:**

- *Why implement Hungarian yourself?* To understand it. It is the O(n³) potentials version,
  tested against SciPy on random problems, and SciPy is the default in production. Gating uses
  the "extended cost" trick from lapjv/ByteTrack: every row and column may pair with a dummy at
  threshold/2, so a real match is made only when it beats leaving both unmatched. Filtering
  after solving can let a costly pair block a valid cheaper one.
- *Why is your MOTA lower than paper X?* Check the detector first. MOTA is detection-dominated,
  and the oracle run shows how much comes from the detector.
- *What breaks?* The flat-ground assumption: stairs, ramps, and anyone whose feet are hidden
  (the box bottom is then not the foot). Crowds where feet are occluded. Calibration drift. The
  2 fps annotation rate. The 40-frame test split is small.
- *What would you do next?* Monocular depth, or head points plus a height prior, to handle
  hidden feet; learned multi-view fusion (MVDet-style) as a comparison to the geometric
  pipeline; auto-calibration from vanishing points for uncalibrated footage; TensorRT on
  embedded hardware.

**Be accurate about what is original:** the ByteTrack association scheme, the Bag of Tricks
recipe and the metric definitions are from the literature, and you reimplemented and verified
them. The ground-plane formulation, uncertainty propagation, fusion design, calibration check
and ablation study are the parts you designed.

---

## 13. Code map

| To change... | Look in |
|---|---|
| dataset formats and conventions | `src/multicam_tracker/data/multiview.py` (`SPECS`), `mot17.py`, `market1501.py` |
| camera model, back-projection, Jacobian | `src/multicam_tracker/geometry/camera.py`, `uncertainty.py` |
| motion models and Kalman filter | `src/multicam_tracker/tracking/kalman.py` |
| the association sequence | `src/multicam_tracker/tracking/tracker.py` (`SingleCameraTracker.update`) |
| cross-camera logic | `src/multicam_tracker/fusion/associator.py` |
| re-ID network and training | `src/multicam_tracker/reid/model.py`, `train.py` |
| what gets evaluated and how | `src/multicam_tracker/pipeline/evaluate.py`, `src/multicam_tracker/eval/` |
| the demo layout | `src/multicam_tracker/pipeline/render.py`, `src/multicam_tracker/viz/` |
| every tunable setting | `src/multicam_tracker/config.py` (defaults) and `configs/` |

Run `pytest` after every change. The suite runs in about 15 s and covers every module.

---

## Appendix: glossary

Grouped by topic. Each entry says what the term means *in this project*.

### Detection and tracking

- **Bounding box (box)**: the rectangle around a person in an image, stored as left, top,
  right and bottom pixel coordinates.
- **Detector**: the model that finds people in each frame and outputs boxes with confidence
  scores. Here it is YOLO.
- **Confidence score**: how sure the detector is (0 to 1) that a box contains a person.
- **Track, track ID**: one person followed over time, and the number assigned to them.
- **Local track / global ID**: a local track lives inside one camera. A global ID is the single
  identity that fusion gives a person across all cameras.
- **Association (matching)**: deciding, every frame, which new box belongs to which existing
  track. Multicam Tracker solves it with the **Hungarian algorithm**, which finds the pairing with the
  lowest total cost.
- **Cost, gate**: the cost says how unlikely a box–track pairing is. A gate is a hard limit
  beyond which a pairing is not allowed at all.
- **ID switch**: a person's ID changes, or two people swap IDs. The main failure trackers try
  to avoid.
- **Occlusion**: a person temporarily hidden behind someone or something.
- **ByteTrack, two-stage association**: match the confident boxes first, then try the
  low-confidence ones (often partly hidden people) against the tracks still unmatched.
- **Kalman filter**: predicts where each track will be in the next frame from its position and
  velocity, then blends that prediction with the new measurement according to how trustworthy
  each one is.
- **Re-ID (re-identification), appearance embedding**: a network that turns a crop of a person
  into a list of numbers (an embedding) so that crops of the same person are close together and
  different people are far apart. It lets the tracker tell people apart by how they look.
- **Cosine distance**: how different two embeddings are; 0 means the same.
- **Fusion**: combining the local tracks of all cameras into global identities.
- **Oracle detector**: a stand-in that uses the ground-truth boxes plus a little noise. Used for
  tests and for the upper-bound experiment, never reported as a result.

### Cameras and geometry

- **Calibration**: the numbers describing a camera. *Intrinsics* are its focal length and image
  centre; *extrinsics* are where it stands and which way it points.
- **Ground plane**: the floor, treated as flat at height 0. All positions are metres on it.
- **Foot point**: the bottom-centre of a person's box, taken as where they touch the ground.
- **Back-projection**: turning a pixel into a point on the ground using the calibration, the
  reverse of projecting a real-world point into the image.
- **Homography**: a 3×3 matrix that maps image pixels to ground points, and back, when the
  ground is flat.
- **Horizon**: the image row above which a pixel's line of sight never meets the ground. Foot
  points above it are rejected.
- **ROI (region of interest)**: the rectangle of ground, in metres, that is tracked. Anything
  outside it is ignored.
- **Bird's-eye view (BEV)**: the top-down map in the demo video.
- **Uncertainty, covariance**: how unsure a ground position is, in each direction. Drawn as the
  ellipses on the map; they grow for people far from the camera.
- **Jacobian**: how far a small pixel error moves the ground point. Used to turn pixel
  uncertainty into ground uncertainty.
- **Mahalanobis distance**: distance measured in units of uncertainty ("3 standard deviations
  away") rather than in metres, so an uncertain far-away observation is judged fairly.
- **Chi-square thresholds (the `*_chi2` settings)**: the gate values for Mahalanobis distance.
  9.21 means "allow pairings that are plausible with 99% probability"; 13.82 means 99.9%.

### Evaluation

- **Ground truth (GT)**: the human annotations that results are compared against.
- **Train / test split**: data used for training versus data held back for scoring. Scoring a
  model on the data it trained on inflates the result.
- **Val half**: the MOT17 convention of training on the first half of every sequence and
  scoring on the second half.
- **Ablation**: rerunning the system with one piece changed or removed, to measure what that
  piece contributes.
- **Upper bound**: the score with one component made perfect (for example the oracle
  detector), which shows how much there is to gain by improving it.
- **TrackEval**: the official evaluation code used by tracking benchmarks.
- **MOTA, IDF1, HOTA, DetA, AssA, MODA, MODP, Rank-1, mAP**: see the one-line definitions in
  section 0 and the fuller explanation in section 9.

### Training and deployment

- **Checkpoint (`.pt` file)**: saved model weights. `best.pt` is the best evaluation so far;
  `last.pt` is the most recent state, used to resume.
- **Epoch**: one full pass over the training data.
- **Fine-tuning**: continuing to train an already-trained model on new data, for example YOLO
  on MOT17.
- **AMP (mixed precision)**: training with 16-bit numbers where it is safe, which saves GPU
  memory and time.
- **OOM (out of memory)**: the GPU ran out of memory. Reduce the batch size or the image size.
- **Cache**: saved detections and embeddings, so later runs skip the slow models.
- **Config, override**: a config is the YAML file describing one experiment; an override
  (`--set key=value`) changes one setting from the command line.
- **ONNX, ONNX Runtime**: a portable model format, and the engine that runs it. Used for
  deployment.
- **Latency**: time per frame in milliseconds; lower is faster.
