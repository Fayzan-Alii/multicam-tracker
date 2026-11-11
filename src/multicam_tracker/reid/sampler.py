from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Batch sampler: P identities per batch, K images each (with replacement when an identity
    has fewer than K). Batch-hard triplet mining needs every batch to contain positives."""

    def __init__(self, labels, p: int, k: int, seed: int = 0):
        self.p, self.k, self.seed = p, k, seed
        self.by_label = defaultdict(list)
        for i, lab in enumerate(labels):
            self.by_label[int(lab)].append(i)
        self.labels = sorted(self.by_label)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.labels) // self.p

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        order = rng.permutation(self.labels)
        for b in range(len(self)):
            batch = []
            for lab in order[b * self.p:(b + 1) * self.p]:
                pool = self.by_label[int(lab)]
                batch.extend(int(i) for i in rng.choice(pool, size=self.k, replace=len(pool) < self.k))
            yield batch
