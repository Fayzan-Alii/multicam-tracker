import numpy as np
import pytest

from multicam_tracker.tracking.assignment import linear_assignment, solve_native


def as_set(matches):
    return {tuple(m) for m in matches.tolist()}


@pytest.mark.parametrize("backend", ["scipy", "native"])
def test_finds_unique_optimum(backend):
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    matches, ur, uc = linear_assignment(cost, threshold=10.0, backend=backend)
    assert as_set(matches) == {(0, 1), (1, 0), (2, 2)}
    assert ur.size == 0 and uc.size == 0


@pytest.mark.parametrize("backend", ["scipy", "native"])
def test_matches_above_threshold_are_released(backend):
    cost = np.array([[0.1, 0.9], [0.95, 0.99]])
    matches, ur, uc = linear_assignment(cost, threshold=0.5, backend=backend)
    assert as_set(matches) == {(0, 0)}
    assert ur.tolist() == [1] and uc.tolist() == [1]


@pytest.mark.parametrize("backend", ["scipy", "native"])
def test_infinite_cost_forbids_a_pair(backend):
    matches, _, _ = linear_assignment(np.array([[np.inf, 1.0], [1.0, np.inf]]), 5.0, backend)
    assert as_set(matches) == {(0, 1), (1, 0)}
    matches, ur, uc = linear_assignment(np.array([[np.inf, np.inf]]), 5.0, backend)
    assert matches.shape == (0, 2) and ur.tolist() == [0] and uc.tolist() == [0, 1]


@pytest.mark.parametrize("backend", ["scipy", "native"])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_problem(backend, shape):
    matches, ur, uc = linear_assignment(np.zeros(shape), 1.0, backend)
    assert matches.shape == (0, 2)
    assert ur.tolist() == list(range(shape[0])) and uc.tolist() == list(range(shape[1]))


@pytest.mark.parametrize("shape", [(5, 5), (4, 9), (9, 4), (1, 6), (12, 12)])
def test_native_solver_matches_scipy_optimum(shape):
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(sum(shape))
    for _ in range(20):
        cost = rng.uniform(0, 10, size=shape)
        rows, cols = solve_native(cost)
        ref_rows, ref_cols = linear_sum_assignment(cost)
        assert len(rows) == min(shape)
        assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
        np.testing.assert_allclose(cost[rows, cols].sum(), cost[ref_rows, ref_cols].sum())


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        linear_assignment(np.zeros((1, 1)), 1.0, backend="greedy")
