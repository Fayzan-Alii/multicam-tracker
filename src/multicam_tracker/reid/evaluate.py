"""Market-1501 evaluation: CMC and mAP, excluding gallery images of the query's own identity
taken by the query's own camera (they make retrieval trivially easy)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def evaluate_rank(q_feat, q_pids, q_cams, g_feat, g_pids, g_cams, max_rank: int = 50):
    q = np.asarray(q_feat, dtype=np.float64)
    g = np.asarray(g_feat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    order = np.argsort(-(q @ g.T), axis=1, kind="stable")
    all_cmc, all_ap = [], []
    for i in range(len(q)):
        idx = order[i]
        keep = ~((g_pids[idx] == q_pids[i]) & (g_cams[idx] == q_cams[i]))
        hits = (g_pids[idx][keep] == q_pids[i]).astype(np.float64)
        if not hits.any():
            continue
        cmc = np.minimum(hits.cumsum(), 1.0)[:max_rank]
        all_cmc.append(np.pad(cmc, (0, max_rank - len(cmc)), mode="edge"))
        precision = hits.cumsum() / np.arange(1, len(hits) + 1)
        all_ap.append(float((precision * hits).sum() / hits.sum()))
    if not all_cmc:
        raise ValueError("no query has a valid gallery match")
    return np.mean(all_cmc, axis=0), float(np.mean(all_ap))


@torch.inference_mode()
def extract_features(model, samples, transform, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    feats = []
    for start in range(0, len(samples), batch_size):
        batch = torch.stack([transform(Image.open(s.path).convert("RGB"))
                             for s in samples[start:start + batch_size]]).to(device)
        feats.append(model(batch).float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 1))


def evaluate_model(model, data: dict, transform, device: str, batch_size: int = 128) -> dict:
    q, g = data["query"], data["gallery"]
    qf = extract_features(model, q, transform, device, batch_size)
    gf = extract_features(model, g, transform, device, batch_size)
    cmc, mAP = evaluate_rank(qf, np.array([s.pid for s in q]), np.array([s.camid for s in q]),
                             gf, np.array([s.pid for s in g]), np.array([s.camid for s in g]))
    return {"rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]), "mAP": mAP}
