"""Single-camera multi-object tracker with ByteTrack's two-stage association.

Per frame:
  1. predict every track to the current timestamp;
  2. match high-confidence detections to confirmed and lost tracks (motion + optional appearance);
  3. match low-confidence detections to still-unmatched confirmed tracks on motion alone, which
     rescues partially occluded people that a single threshold would drop;
  4. match remaining high-confidence detections to tentative tracks;
  5. start tentative tracks from leftover detections above new_track_thresh.
"""

from __future__ import annotations

import numpy as np

from multicam_tracker.config import MotionConfig, TrackerConfig
from multicam_tracker.tracking.assignment import linear_assignment
from multicam_tracker.tracking.costs import cosine_distance, fuse_score
from multicam_tracker.tracking.kalman import GroundMotion, ImageMotion
from multicam_tracker.tracking.track import Track, TrackState
from multicam_tracker.types import Detections, TrackOutput


def make_motion(cfg: MotionConfig, fps: float):
    return GroundMotion(cfg) if cfg.space == "ground" else ImageMotion(cfg, fps)


class SingleCameraTracker:
    def __init__(self, cfg: TrackerConfig, motion, camera_id: int = 0):
        self.cfg = cfg
        self.motion = motion
        self.camera_id = camera_id
        self.tracks: list[Track] = []
        self._next_id = 1
        self._frames_seen = 0

    def _cost(self, tracks: list[Track], dets: Detections, use_appearance: bool,
              use_score: bool = True) -> np.ndarray:
        cfg = self.cfg
        cost = self.motion.motion_cost([t.mean for t in tracks], [t.cov for t in tracks], dets, cfg.gate_chi2)
        if use_score and cfg.fuse_score and isinstance(self.motion, ImageMotion):
            cost = fuse_score(cost, dets.scores)
        w = cfg.appearance_weight
        if use_appearance and w > 0 and dets.embeddings is not None and len(tracks) and len(dets):
            has_emb = np.array([t.embedding is not None for t in tracks])
            if has_emb.any():
                app = np.ones((len(tracks), len(dets)))
                app[has_emb] = cosine_distance(np.stack([t.embedding for t in tracks if t.embedding is not None]),
                                               dets.embeddings)
                combined = (1 - w) * cost + w * app
                # Appearance may re-rank plausible pairs but never licenses a physically impossible one.
                combined[cost >= 1.0 - 1e-9] = np.inf
                combined[has_emb[:, None] & (app > cfg.appearance_gate)] = np.inf
                cost = np.where(has_emb[:, None], combined, cost)
        return cost

    def _associate(self, tracks, dets, det_idx, threshold, use_appearance, timestamp, update_embedding,
                   use_score=True):
        """Match tracks to dets[det_idx]; update the matched tracks. Returns unmatched (tracks, det_idx)."""
        if not tracks or len(det_idx) == 0:
            return list(tracks), det_idx
        sub = dets[det_idx]
        matches, u_tracks, u_dets = linear_assignment(
            self._cost(tracks, sub, use_appearance, use_score), threshold, self.cfg.assignment
        )
        for ti, di in matches:
            tracks[ti].update(self.motion, sub, di, timestamp, self.cfg.embedding_ema, update_embedding)
        return [tracks[i] for i in u_tracks], det_idx[u_dets]

    def update(self, dets: Detections, timestamp: float) -> list[TrackOutput]:
        cfg = self.cfg
        first_frame = self._frames_seen == 0
        self._frames_seen += 1

        if isinstance(self.motion, GroundMotion) and len(dets):
            dets = dets[dets.ground_valid] if dets.ground_valid is not None else dets
        high = np.nonzero(dets.scores >= cfg.high_thresh)[0]
        low = np.nonzero((dets.scores >= cfg.low_thresh) & (dets.scores < cfg.high_thresh))[0]

        for track in self.tracks:
            track.predict(self.motion, timestamp)
        pool = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state is TrackState.TENTATIVE]

        rest, high_left = self._associate(pool, dets, high, cfg.match_thresh, True, timestamp, True)
        active_rest = [t for t in rest if t.state is TrackState.CONFIRMED]
        if cfg.second_stage:
            # Low-confidence boxes rescue existing tracks by motion alone. Score fusion
            # would reject even perfectly aligned boxes when their confidence is low.
            active_rest, _ = self._associate(
                active_rest, dets, low, cfg.second_match_thresh, False, timestamp, False, use_score=False
            )
        for track in active_rest + [t for t in rest if t.state is TrackState.LOST]:
            track.mark_missed(timestamp, cfg.max_lost_s)

        tentative_rest, high_left = self._associate(
            tentative, dets, high_left, cfg.unconfirmed_match_thresh, True, timestamp, True
        )
        for track in tentative_rest:
            track.mark_missed(timestamp, cfg.max_lost_s)

        for i in high_left:
            if dets.scores[i] >= cfg.new_track_thresh:
                self.tracks.append(
                    Track.create(self._next_id, self.motion, dets, i, timestamp, cfg.n_init, confirmed=first_frame)
                )
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.state is not TrackState.DELETED]
        return [self._output(t) for t in sorted(self.tracks, key=lambda t: t.track_id)
                if t.state is TrackState.CONFIRMED and t.updated_now]

    def _output(self, track: Track) -> TrackOutput:
        return TrackOutput(
            track_id=track.track_id,
            box=track.box.copy(),
            score=track.score,
            camera_id=self.camera_id,
            world_xy=None if track.world_xy is None else track.world_xy.copy(),
            world_cov=None if track.world_cov is None else track.world_cov.copy(),
            embedding=None if track.embedding is None else track.embedding.copy(),
        )
