# Common workflows. See MANUAL.md for what each step does and what to expect.
PYTHON ?= python3
PY ?= .venv/bin/python
.DEFAULT_GOAL := help

.PHONY: help setup install test smoke mot17-demo mot17-demo-score mot17-yolo-demo mot17-yolo-demo-score mot17-occlusion-demo \
        mot17-public mot17-yolo mot17-yolo-data mot17-yolo-train mot17-yolo-val mot17-trackeval \
        wildtrack-verify wildtrack wildtrack-ablate wildtrack-demo \
        reid-market reid-wildtrack-crops reid-wildtrack export bench

help:
	@echo "make setup             Create .venv and install packages with pip (CUDA 12.6)"
	@echo "make test              Run the tests"
	@echo "make smoke             Run the synthetic demo (no dataset needed)"
	@echo "make mot17-demo        Track one MOT17 clip and save a preview video"
	@echo "make mot17-demo-score  Score that clip with TrackEval"
	@echo "make mot17-yolo-demo   Run YOLO on the same clip and save a preview"
	@echo "make mot17-yolo-demo-score  Score the YOLO tracking result"
	@echo "make mot17-occlusion-demo  Preview the fine-tuned tracker after the rescue-stage fix"
	@echo "make mot17-public      Track all MOT17 training clips with provided detections"
	@echo "make mot17-trackeval   Score the full MOT17 public-detection run"

.venv/bin/python:
	$(PYTHON) -m venv .venv

setup: .venv/bin/python
	$(PY) -m ensurepip --upgrade
	$(PY) -m pip install --timeout 300 torch torchvision --index-url https://download.pytorch.org/whl/cu126
	$(PY) -m pip install --timeout 300 -r requirements.txt

install: setup

test:
	$(PY) -m pytest

smoke:  ## no data or GPU needed
	$(PY) -m multicam_tracker run --config configs/synthetic.yaml

# ---- Milestone 1: single-camera tracking on MOT17 --------------------------------------
mot17-demo:
	$(PY) -m multicam_tracker run --config configs/mot17/demo.yaml

mot17-demo-score:
	$(PY) -m multicam_tracker trackeval --config configs/mot17/demo.yaml

mot17-yolo-demo:
	$(PY) -m multicam_tracker run --config configs/mot17/yolo_demo.yaml

mot17-yolo-demo-score:
	$(PY) -m multicam_tracker trackeval --config configs/mot17/yolo_demo.yaml

mot17-occlusion-demo:
	$(PY) -m multicam_tracker run --config configs/mot17/occlusion_demo.yaml

mot17-public:
	$(PY) -m multicam_tracker run --config configs/mot17/public_frcnn.yaml

mot17-yolo:
	$(PY) -m multicam_tracker run --config configs/mot17/yolo_coco.yaml

mot17-yolo-data:
	$(PY) -m multicam_tracker prepare-yolo --dataset mot17 --root data/MOT17 --out data/yolo_mot17

mot17-yolo-train: mot17-yolo-data
	$(PY) -m multicam_tracker train-detector --data data/yolo_mot17/data.yaml --weights yolo11s.pt --imgsz 1280 --batch 4 \
		--epochs 50 --project runs/detector --name mot17

mot17-yolo-val:
	$(PY) -m multicam_tracker run --config configs/mot17/yolo_finetuned_val.yaml

mot17-trackeval:
	$(PY) -m multicam_tracker trackeval --config configs/mot17/public_frcnn.yaml

# ---- Milestones 2 & 4: WILDTRACK ---------------------------------------------------------
wildtrack-verify:
	$(PY) -m multicam_tracker verify-calibration --dataset wildtrack --root data/Wildtrack

wildtrack:
	$(PY) -m multicam_tracker run --config configs/wildtrack/base.yaml

wildtrack-ablate:
	$(PY) -m multicam_tracker ablate --configs configs/wildtrack/ablations/0[1-5]_*.yaml --out runs/wildtrack/ablations.md

wildtrack-demo:
	$(PY) -m multicam_tracker run --config configs/wildtrack/demo.yaml

# ---- Milestone 3: re-ID ------------------------------------------------------------------
reid-market:
	$(PY) -m multicam_tracker train-reid --config configs/reid/market1501_r50.yaml

reid-wildtrack-crops:
	$(PY) -m multicam_tracker build-reid-crops --dataset wildtrack --root data/Wildtrack --out data/wildtrack_reid

reid-wildtrack: reid-wildtrack-crops
	$(PY) -m multicam_tracker train-reid --config configs/reid/wildtrack_finetune.yaml

# ---- Milestone 5: deployment -------------------------------------------------------------
export:
	$(PY) -m multicam_tracker export-onnx --checkpoint runs/reid/market_r50/best.pt --out runs/reid/market_r50/reid.onnx

bench:
	$(PY) -m multicam_tracker benchmark --config configs/wildtrack/reid.yaml --frames 100
