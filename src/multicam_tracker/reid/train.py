"""Re-ID training loop (Bag of Tricks recipe). Resumable, so it survives Kaggle session limits."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multicam_tracker.data.market1501 import load_market_style
from multicam_tracker.device import resolve_device
from multicam_tracker.reid.evaluate import evaluate_model
from multicam_tracker.reid.losses import batch_hard_triplet_loss
from multicam_tracker.reid.model import ReIDNet
from multicam_tracker.reid.sampler import PKSampler
from multicam_tracker.reid.transforms import test_transform, train_transform


@dataclass
class ReIDTrainConfig:
    data_root: str = "data/Market-1501-v15.09.15"
    train_dir: str = "bounding_box_train"
    query_dir: str = "query"
    gallery_dir: str = "bounding_box_test"
    backbone: str = "resnet50"
    pretrained: bool = True
    last_stride: int = 1
    input_size: list[int] = field(default_factory=lambda: [256, 128])
    p: int = 16
    k: int = 4
    epochs: int = 120
    lr: float = 3.5e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 10
    milestones: list[int] = field(default_factory=lambda: [40, 70])
    gamma: float = 0.1
    margin: float = 0.3
    label_smoothing: float = 0.1
    random_erasing: float = 0.5
    pad: int = 10
    amp: bool = True
    num_workers: int = 4
    eval_every: int = 10
    eval_batch_size: int = 128
    device: str = "auto"
    seed: int = 0
    output_dir: str = "runs/reid/market_r50"
    resume: bool = True
    init_from: str = ""  # checkpoint whose backbone initialises this run (fine-tuning)


class ReIDImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return self.transform(Image.open(s.path).convert("RGB")), s.label, s.camid


def lr_factor(epoch: int, cfg: ReIDTrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return 0.1 + 0.9 * epoch / cfg.warmup_epochs
    return cfg.gamma ** sum(epoch >= m for m in cfg.milestones)


def _save(path: Path, model, optimizer, scaler, epoch, meta, history):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "meta": meta, "history": history}, tmp)
    tmp.replace(path)


def build_model(cfg: ReIDTrainConfig, num_classes: int) -> ReIDNet:
    model = ReIDNet(cfg.backbone, num_classes, cfg.pretrained, cfg.last_stride)
    if cfg.init_from:
        state = torch.load(cfg.init_from, map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    return model


def train(cfg: ReIDTrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_amp = cfg.amp and device.startswith("cuda")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    data = load_market_style(cfg.data_root, cfg.train_dir, cfg.query_dir, cfg.gallery_dir)
    train_set = ReIDImageDataset(data["train"], train_transform(cfg.input_size, cfg.pad, cfg.random_erasing))
    sampler = PKSampler([s.label for s in data["train"]], cfg.p, cfg.k, cfg.seed)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=cfg.num_workers,
                        pin_memory=device.startswith("cuda"), persistent_workers=cfg.num_workers > 0)

    model = build_model(cfg, data["num_classes"]).to(device)
    meta = {**model.config, "input_size": list(cfg.input_size), "feat_dim": model.feat_dim}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_factor(e, cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history = {"loss": [], "eval": []}
    start, best_map = 0, -1.0

    last = out / "last.pt"
    if cfg.resume and last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start, history = ckpt["epoch"] + 1, ckpt["history"]
        best_map = max([e["mAP"] for e in history["eval"]], default=-1.0)
        sampler.epoch = start
        for _ in range(start):
            scheduler.step()
        log(f"resumed from {last} at epoch {start}")

    for epoch in range(start, cfg.epochs):
        model.train()
        t0, losses, correct, seen = time.time(), [], 0, 0
        for images, labels, _ in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                logits, feats = model(images)
                loss = ce(logits, labels) + batch_hard_triplet_loss(feats.float(), labels, cfg.margin)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        scheduler.step()
        history["loss"].append(float(np.mean(losses)))
        log(f"epoch {epoch + 1}/{cfg.epochs}  loss {history['loss'][-1]:.4f}  train-acc {correct / max(1, seen):.3f}"
            f"  lr {optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")

        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
            metrics = evaluate_model(model, data, test_transform(cfg.input_size), device, cfg.eval_batch_size)
            history["eval"].append({"epoch": epoch + 1, **metrics})
            log(f"  eval  rank-1 {metrics['rank1']:.4f}  mAP {metrics['mAP']:.4f}")
            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                _save(out / "best.pt", model, optimizer, scaler, epoch, meta, history)
        _save(last, model, optimizer, scaler, epoch, meta, history)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    return history
