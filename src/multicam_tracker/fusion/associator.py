"""Cross-camera fusion: turn per-camera local tracks into global identities on the ground plane.

Fusion consumes *tracks*, not raw detections. A local track already carries temporal identity
and a smoothed appearance, so once a (camera, local id) pair is linked to a global identity the
link is simply kept while it stays geometrically plausible. Per frame:

  1. sticky links: keep each existing local->global link that passes a loose gate;
  2. per camera, match unlinked local tracks to global tracks with no observation from that
     camera yet (so a global identity never takes two observations from one camera);
  3. cluster the leftovers across cameras (same-camera pairs can never join) into new identities;
  4. fuse each identity's observations in information form and apply one Kalman update;
  5. merge duplicate identities that agree for `merge_frames` frames without sharing a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from multicam_tracker.config import FusionConfig
from multicam_tracker.tracking.assignment import linear_assignment
from multicam_tracker.tracking.costs import cosine_distance, normalize_rows
from multicam_tracker.tracking.kalman import GroundMotion, mahalanobis_sq
from multicam_tracker.types import GlobalTrackOutput, TrackOutput


def fuse_measurements(zs, Rs) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance (information form) fusion of independent measurements."""
    info = np.zeros((2, 2))
    vec = np.zeros(2)
    for z, R in zip(zs, Rs):
        Ri = np.linalg.inv(R)
        info += Ri
        vec += Ri @ z
    R = np.linalg.inv(info)
    return R @ vec, R


@dataclass
class _GlobalTrack:
    gid: int
    mean: np.ndarray
    cov: np.ndarray
    embedding: np.ndarray | None
    last_update: float
    last_predict: float
    assigned: list[int] = field(default_factory=list)  # observation indices this frame


class CrossCameraFusion:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.motion = GroundMotion(cfg.motion)
        self.tracks: dict[int, _GlobalTrack] = {}
        self.links: dict[tuple[int, int], int] = {}
        self._close: dict[tuple[int, int], int] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ costs
    def _appearance(self, a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if self.cfg.appearance_weight <= 0 or a is None or b is None:
            return None
        return float(cosine_distance(a[None], b[None])[0, 0])

    def _combine(self, d2: float, app: float | None) -> float:
        cfg = self.cfg
        if d2 > cfg.gate_chi2:
            return np.inf
        motion = d2 / cfg.gate_chi2
        if app is None:
            return motion
        if app > cfg.appearance_gate:
            return np.inf
        return (1 - cfg.appearance_weight) * motion + cfg.appearance_weight * app

    def _track_cost(self, track: _GlobalTrack, o: TrackOutput) -> float:
        d2 = self.motion.gating_distance(track.mean, track.cov, o.world_xy[None], o.world_cov[None])[0]
        return self._combine(d2, self._appearance(track.embedding, o.embedding))

    def _pair_cost(self, a: TrackOutput, b: TrackOutput) -> float:
        if a.camera_id == b.camera_id:
            return np.inf
        S = self.motion.measurement_noise(a.world_cov) + self.motion.measurement_noise(b.world_cov)
        d2 = float(mahalanobis_sq((a.world_xy - b.world_xy)[None], S[None])[0])
        return self._combine(d2, self._appearance(a.embedding, b.embedding))

    # ----------------------------------------------------------------- stages
    def _assign(self, track: _GlobalTrack, i: int, o: TrackOutput, obs_gid: list) -> None:
        track.assigned.append(i)
        obs_gid[i] = track.gid
        self.links[(o.camera_id, o.track_id)] = track.gid

    def _sticky(self, obs, obs_gid) -> None:
        best: dict[tuple[int, int], tuple[float, int]] = {}  # (gid, camera) -> (d2, obs index)
        for i, o in enumerate(obs):
            key = (o.camera_id, o.track_id)
            gid = self.links.get(key)
            if gid is None:
                continue
            track = self.tracks.get(gid)
            d2 = np.inf if track is None else self.motion.gating_distance(
                track.mean, track.cov, o.world_xy[None], o.world_cov[None])[0]
            if d2 > self.cfg.link_gate_chi2:
                del self.links[key]
                continue
            slot = (gid, o.camera_id)
            if slot in best and best[slot][0] <= d2:  # two local tracks of one camera claim one identity
                del self.links[key]
                continue
            if slot in best:
                loser = obs[best[slot][1]]
                self.links.pop((loser.camera_id, loser.track_id), None)
            best[slot] = (d2, i)
        for (gid, _), (_, i) in best.items():
            self._assign(self.tracks[gid], i, obs[i], obs_gid)

    def _per_camera(self, obs, obs_gid) -> None:
        cams = sorted({o.camera_id for i, o in enumerate(obs) if obs_gid[i] is None})
        for cam in cams:
            cand = [i for i, o in enumerate(obs) if obs_gid[i] is None and o.camera_id == cam]
            tracks = [t for t in self.tracks.values() if all(obs[j].camera_id != cam for j in t.assigned)]
            if not cand or not tracks:
                continue
            cost = np.array([[self._track_cost(t, obs[i]) for i in cand] for t in tracks])
            matches, _, _ = linear_assignment(cost, self.cfg.match_thresh, self.cfg.assignment)
            for ti, ci in matches:
                self._assign(tracks[ti], cand[ci], obs[cand[ci]], obs_gid)

    def _cluster(self, obs, idx: list[int]) -> list[list[int]]:
        """Average-linkage agglomeration with a same-camera cannot-link constraint."""
        clusters = [[i] for i in idx]
        pair = {(a, b): self._pair_cost(obs[a], obs[b]) for a in idx for b in idx if a < b}

        def cost(a, b):
            return pair[(a, b)] if a < b else pair[(b, a)]

        while True:
            best, best_pair = self.cfg.cluster_thresh, None
            for x in range(len(clusters)):
                for y in range(x + 1, len(clusters)):
                    costs = [cost(a, b) for a in clusters[x] for b in clusters[y]]
                    if not np.all(np.isfinite(costs)):
                        continue
                    avg = float(np.mean(costs))
                    if avg < best:
                        best, best_pair = avg, (x, y)
            if best_pair is None:
                return clusters
            x, y = best_pair
            clusters[x] += clusters.pop(y)

    def _fused(self, obs, indices) -> tuple[np.ndarray, np.ndarray]:
        return fuse_measurements([obs[i].world_xy for i in indices],
                                 [self.motion.measurement_noise(obs[i].world_cov) for i in indices])

    def _mean_embedding(self, obs, indices) -> np.ndarray | None:
        embs = [obs[i].embedding for i in indices if obs[i].embedding is not None]
        return normalize_rows(np.mean(embs, axis=0)) if embs else None

    def _merge_duplicates(self, obs) -> None:
        cfg = self.cfg
        active = sorted(g for g, t in self.tracks.items() if t.assigned)
        seen = set()
        for x, gx in enumerate(active):
            for gy in active[x + 1:]:
                a, b = self.tracks.get(gx), self.tracks.get(gy)
                if a is None or b is None:
                    continue
                key = (gx, gy)
                seen.add(key)
                cams_a = {obs[i].camera_id for i in a.assigned}
                cams_b = {obs[i].camera_id for i in b.assigned}
                d2 = float(mahalanobis_sq((a.mean[:2] - b.mean[:2])[None], (a.cov[:2, :2] + b.cov[:2, :2])[None])[0])
                app = self._appearance(a.embedding, b.embedding)
                close = not (cams_a & cams_b) and d2 <= cfg.merge_chi2 and (app is None or app <= cfg.merge_appearance)
                if not close:
                    self._close.pop(key, None)
                    continue
                self._close[key] = self._close.get(key, 0) + 1
                if self._close[key] >= cfg.merge_frames:
                    z, R = self._fused(obs, b.assigned)
                    a.mean, a.cov = self.motion.update(a.mean, a.cov, z, R)
                    a.assigned += b.assigned
                    for link, gid in list(self.links.items()):
                        if gid == gy:
                            self.links[link] = gx
                    del self.tracks[gy]
        self._close = {k: v for k, v in self._close.items() if k in seen and k[1] in self.tracks}

    # ----------------------------------------------------------------- update
    def update(self, observations: list[TrackOutput], timestamp: float) -> list[GlobalTrackOutput]:
        cfg = self.cfg
        obs = [o for o in observations if o.world_xy is not None and np.all(np.isfinite(o.world_xy))]
        for t in self.tracks.values():
            dt = timestamp - t.last_predict
            if dt > 0:
                t.mean, t.cov = self.motion.predict(t.mean, t.cov, dt)
                t.last_predict = timestamp
            t.assigned = []
        existing = set(self.tracks)
        obs_gid: list[int | None] = [None] * len(obs)

        self._sticky(obs, obs_gid)
        self._per_camera(obs, obs_gid)
        for cluster in self._cluster(obs, [i for i in range(len(obs)) if obs_gid[i] is None]):
            z, R = self._fused(obs, cluster)
            mean, cov = self.motion.initiate(z, R)
            track = _GlobalTrack(self._next_id, mean, cov, self._mean_embedding(obs, cluster), timestamp, timestamp)
            self._next_id += 1
            self.tracks[track.gid] = track
            for i in cluster:
                self._assign(track, i, obs[i], obs_gid)

        for gid in existing:
            t = self.tracks[gid]
            if not t.assigned:
                continue
            z, R = self._fused(obs, t.assigned)
            t.mean, t.cov = self.motion.update(t.mean, t.cov, z, R)
            t.last_update = timestamp
            emb = self._mean_embedding(obs, t.assigned)
            if emb is not None:
                t.embedding = emb if t.embedding is None else normalize_rows(
                    cfg.embedding_ema * t.embedding + (1 - cfg.embedding_ema) * emb)

        self._merge_duplicates(obs)

        for gid in [g for g, t in self.tracks.items() if not t.assigned and timestamp - t.last_update > cfg.max_lost_s]:
            del self.tracks[gid]
        self.links = {k: g for k, g in self.links.items() if g in self.tracks}

        out = []
        for gid in sorted(self.tracks):
            t = self.tracks[gid]
            if not t.assigned:
                continue
            members = sorted(t.assigned, key=lambda i: obs[i].camera_id)
            out.append(GlobalTrackOutput(
                global_id=gid, world_xy=t.mean[:2].copy(), world_cov=t.cov[:2, :2].copy(),
                cameras=[obs[i].camera_id for i in members],
                local={obs[i].camera_id: obs[i].track_id for i in members},
                boxes={obs[i].camera_id: obs[i].box for i in members},
            ))
        return out
