"""GPU smoke: every built module solves to finite output, deterministically;
per-knot weight arrays reproduce the scalar path bit-exactly."""
import numpy as np
import pytest

import gato
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}


def _inputs(plant, N, B):
    q0 = np.asarray(START[plant], dtype=np.float32)
    nq = q0.size
    x0 = np.concatenate([q0, np.zeros(nq, dtype=np.float32)])
    rng = np.random.default_rng(1234)
    X = np.tile(x0, (B, 1)) + rng.normal(0, 0.01, (B, 2 * nq)).astype(np.float32)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.35, 0.25, 0.5
    return X, goals


def _combos():
    return sorted(k for k in gato.available() if k[0] in START)


@pytest.mark.parametrize("plant,N", _combos())
@pytest.mark.parametrize("B", [1, 8])
def test_solve_finite_and_deterministic(make_solver, plant, N, B):
    X, goals = _inputs(plant, N, B)
    res_a = make_solver(plant, N, batch_size=B).solve(X, goals)
    res_b = make_solver(plant, N, batch_size=B).solve(X, goals)
    assert np.isfinite(res_a.xu).all()
    assert (res_a.stats.sqp_iters >= 1).all()
    assert res_a.xu.shape == (B, N * (res_a.nx + res_a.nu) - res_a.nu)
    # same inputs, fresh solver -> bit-identical trajectories
    np.testing.assert_array_equal(res_a.xu, res_b.xu)
    np.testing.assert_array_equal(res_a.stats.sqp_iters, res_b.stats.sqp_iters)


def test_per_knot_weights_match_scalar_path(make_solver, smallest_module):
    """(N,3) rows filled with the scalar weights (terminal ee = N_cost) must be
    bit-identical to the scalar path — the wave-C parity gate as a test."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)
    sp = dict(q_cost=2.0, qd_cost=1e-4, u_cost=1e-6, N_cost=50.0)

    s_scalar = make_solver(plant, N, batch_size=B, **sp)
    ref = s_scalar.solve(X, goals)

    s_knot = make_solver(plant, N, batch_size=B, **sp)
    w = np.tile([sp["q_cost"], sp["qd_cost"], sp["u_cost"]], (N, 1)).astype(np.float32)
    w[N - 1, 0] = sp["N_cost"]
    s_knot.set_cost_weights_per_knot(w)
    got = s_knot.solve(X, goals)

    np.testing.assert_array_equal(ref.xu, got.xu)

    # a mid-horizon ee-weight spike must actually change the solution
    s_via = make_solver(plant, N, batch_size=B, **sp)
    w2 = w.copy()
    w2[N // 2, 0] = 500.0
    s_via.set_cost_weights_per_knot(w2)
    via = s_via.solve(X, goals)
    assert np.abs(via.xu - ref.xu).max() > 1e-3


def test_arbitrary_batch_size(make_solver, smallest_module):
    plant, N = smallest_module
    X, goals = _inputs(plant, N, 3)  # not a power of two
    res = make_solver(plant, N, batch_size=3).solve(X, goals)
    assert res.xu.shape[0] == 3 and np.isfinite(res.xu).all()
