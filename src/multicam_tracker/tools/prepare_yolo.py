"""Convert tracking datasets into Ultralytics YOLO detection format for fine-tuning.

Output: <out>/images/{train,val}/*, <out>/labels/{train,val}/*.txt, <out>/data.yaml. Images are
symlinked by default. Label lines are `0 cx cy w h`, normalised to [0, 1].
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from multicam_tracker.data.mot17 import load_mot17
from multicam_tracker.data.multiview import load_multiview

# MOT17 classes that are people: pedestrian, person on vehicle, static person. Distractors and
# reflections are left unlabelled.
MOT17_PERSON_CLASSES = (1, 2, 7)


def _place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _label_lines(boxes: np.ndarray, image_size) -> list[str]:
    H, W = image_size
    lines = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        x1, x2 = np.clip([x1, x2], 0, W)
        y1, y2 = np.clip([y1, y2], 0, H)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def _write(out: Path, split: str, stem: str, image: Path, lines, copy: bool) -> None:
    _place(image, out / "images" / split / f"{stem}{image.suffix}", copy)
    label = out / "labels" / split / f"{stem}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("\n".join(lines) + ("\n" if lines else ""))


def _data_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": {0: "person"}},
        sort_keys=False))


def prepare_mot17(root, out, detector: str = "FRCNN", copy: bool = False) -> dict:
    """ByteTrack half split: first half of every training sequence -> train, second half -> val."""
    out = Path(out)
    counts = {"train": 0, "val": 0}
    for split, frames in (("train", "first_half"), ("val", "second_half")):
        for seq in load_mot17(root, "train", detector, split=frames):
            stem = seq.name.rsplit("-", 1)[0]
            img_dir = Path(root) / "train" / seq.name / "img1"
            for f in seq.frames:
                gt = seq.gt(f)
                keep = np.isin(gt.classes, MOT17_PERSON_CLASSES) if gt.classes is not None else np.ones(len(gt.ids), bool)
                image = next(img_dir.glob(f"{f:06d}.*"))
                _write(out, split, f"{stem}_{f:06d}", image, _label_lines(gt.boxes[keep], seq.image_size), copy)
                counts[split] += 1
    _data_yaml(out)
    return counts


def prepare_multiview(root, kind: str, out, copy: bool = False) -> dict:
    """MVDet protocol: first 90% of frames -> train, last 10% -> val, every camera view."""
    out = Path(out)
    counts = {"train": 0, "val": 0}
    for split, part in (("train", "train"), ("val", "test")):
        seq = load_multiview(root, kind, split=part)
        for c in sorted(seq.cameras):
            img_dir = Path(root) / "Image_subsets" / f"C{c + 1}"
            images = {int(p.stem): p for p in img_dir.iterdir() if p.stem.isdigit()}
            for f in seq.frames:
                if f not in images:
                    continue
                lines = _label_lines(seq.gt_view(c, f).boxes, seq.image_size)
                _write(out, split, f"{kind}_C{c + 1}_{f:08d}", images[f], lines, copy)
                counts[split] += 1
    _data_yaml(out)
    return counts
