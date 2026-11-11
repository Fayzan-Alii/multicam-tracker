"""Track lifecycle: TENTATIVE -> CONFIRMED <-> LOST -> DELETED."""

from __future__ import annotations

from enum import Enum

import numpy as np

from multicam_tracker.tracking.costs import normalize_rows


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


class Track:
    def __init__(self, track_id, mean, cov, box, score, embedding, timestamp, n_init, confirmed):
        self.track_id = track_id
        self.mean, self.cov = mean, cov
        self.box = np.asarray(box, dtype=np.float64)
        self.score = float(score)
        self.embedding = None if embedding is None else normalize_rows(embedding)
        self.world_xy = None
        self.world_cov = None
        self.hits = 1
        self.n_init = n_init
        self.last_update = timestamp
        self.last_predict = timestamp
        self.updated_now = True
        self.state = TrackState.CONFIRMED if confirmed or n_init <= 1 else TrackState.TENTATIVE

    @classmethod
    def create(cls, track_id, motion, dets, i, timestamp, n_init, confirmed=False) -> "Track":
        mean, cov = motion.initiate_from(dets, i)
        emb = None if dets.embeddings is None else dets.embeddings[i]
        track = cls(track_id, mean, cov, dets.boxes[i], dets.scores[i], emb, timestamp, n_init, confirmed)
        track._store_measurement(dets, i)
        return track

    def _store_measurement(self, dets, i) -> None:
        if dets.world_xy is not None:
            self.world_xy = dets.world_xy[i]
            self.world_cov = dets.world_cov[i]

    def predict(self, motion, timestamp: float) -> None:
        dt = timestamp - self.last_predict
        if dt > 0:
            self.mean, self.cov = motion.predict(self.mean, self.cov, dt)
            self.last_predict = timestamp
        self.updated_now = False

    def update(self, motion, dets, i, timestamp, ema: float, update_embedding: bool) -> None:
        self.mean, self.cov = motion.update_from(self.mean, self.cov, dets, i)
        self.box = dets.boxes[i].copy()
        self.score = float(dets.scores[i])
        self._store_measurement(dets, i)
        if update_embedding and dets.embeddings is not None:
            new = normalize_rows(dets.embeddings[i])
            self.embedding = new if self.embedding is None else normalize_rows(ema * self.embedding + (1 - ema) * new)
        self.hits += 1
        self.last_update = timestamp
        self.updated_now = True
        if self.state is TrackState.LOST or (self.state is TrackState.TENTATIVE and self.hits >= self.n_init):
            self.state = TrackState.CONFIRMED

    def mark_missed(self, timestamp: float, max_lost_s: float) -> None:
        if self.state is TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif timestamp - self.last_update > max_lost_s:
            self.state = TrackState.DELETED
        else:
            self.state = TrackState.LOST
