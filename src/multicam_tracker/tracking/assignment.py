"""Gated linear assignment.

Thresholding uses the "extended cost" construction from lapjv (as in ByteTrack): every row and
column may instead pair with a dummy at cost threshold/2, so a real match is only made when it
beats leaving both sides unmatched. Solving first and filtering afterwards can pick a
high-cost pair that blocks a valid cheaper one.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_native(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian algorithm with row/column potentials, O(n^2 m). Minimises total cost and
    assigns min(n, m) pairs. Mirrors scipy's linear_sum_assignment for finite costs."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = m, n
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j]: 1-based row matched to column j (0 = free)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (reduced < minv[1:])
            minv[1:][better] = reduced[better]
            way[1:][better] = j0
            candidates = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            used_cols = np.nonzero(used)[0]
            u[p[used_cols]] += delta
            v[used_cols] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:  # augment along the alternating path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def linear_assignment(
    cost: np.ndarray, threshold: float, backend: str = "scipy"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (matches (K, 2), unmatched_rows, unmatched_cols). Pairs with cost > threshold
    (including inf) are never matched."""
    if backend not in ("scipy", "native"):
        raise ValueError(f"unknown assignment backend {backend!r}; use 'scipy' or 'native'")
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=int), np.arange(n), np.arange(m)

    allowed = np.isfinite(cost) & (cost <= threshold)
    forbidden = threshold + 1.0 + (np.abs(cost[allowed]).max() if allowed.any() else 0.0)
    ext = np.full((n + m, n + m), threshold / 2.0)
    ext[:n, :m] = np.where(allowed, cost, forbidden * (n + m))
    ext[n:, m:] = 0.0

    solve = linear_sum_assignment if backend == "scipy" else solve_native
    rows, cols = solve(ext)
    keep = (rows < n) & (cols < m)
    rows, cols = rows[keep], cols[keep]
    ok = allowed[rows, cols]
    matches = np.column_stack([rows[ok], cols[ok]]).astype(int)
    unmatched_rows = np.setdiff1d(np.arange(n), matches[:, 0])
    unmatched_cols = np.setdiff1d(np.arange(m), matches[:, 1])
    return matches, unmatched_rows, unmatched_cols
