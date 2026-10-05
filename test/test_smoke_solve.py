"""GPU smoke: every built module solves to finite output, deterministically;
per-knot weight arrays reproduce the scalar path bit-exactly."""
import numpy as np
import pytest

import gato
from conftest import arm_problem  # noqa: E402
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}


def _inputs(plant, N, B):
    return arm_problem(plant, N, B, jitter=0.01)   # the reach problem with the suite's state jitter


def _combos():
    # Collection must not depend on which binaries happen to exist on the host.
    from pathlib import Path
    rows = (Path(__file__).with_name('receipt_modules.txt')).read_text().splitlines()
    return sorted((parts[0], int(parts[1])) for line in rows
                  if line.strip() and not line.lstrip().startswith('#')
                  for parts in [line.split()]
                  if parts[0] in START and (len(parts) == 2 or parts[2] == 'default'))


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


def test_terminal_merit_ignores_stale_shared_memory(urdfs):
    """The terminal knot has no control, but the merit's cost value read its u slot with zero
    weight; left unwritten, NaN/Inf from an earlier kernel made 0 * NaN = NaN and froze that
    solve's line search. A tiny-rho sweep leaves such values behind; every later solve in the
    process must still start from a finite merit (reproduced the bug every time before the fix)."""
    if ("iiwa14", 64) not in gato.available():
        pytest.fail("iiwa14 N=64 module missing from the receipt build")
    params = gato.SolverParams(max_sqp_iters=100, max_pcg_iters=200, pcg_tol=1e-3, solve_ratio=1.0,
                               mu=1.0, q_cost=10.0, qd_cost=0.1, u_cost=1e-6, N_cost=100.0,
                               q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0, rho=1e-3, adapt_rho=True)
    goal = np.tile(np.array([0.078, 0.344, 0.562, 0, 0, 0], np.float32), 64)
    B = 8
    sweep = np.power(10, -8 + np.arange(1, B + 1) / (B + 1) * 9).astype(np.float32)
    for rho_batch in (sweep, sweep, None):
        solver = gato.BSQP(model_path=str(urdfs["iiwa14"]), batch_size=B, N=64, dt=0.05, params=params,
                           plant_type="iiwa14", rho_batch=rho_batch)
        res = solver.solve(np.zeros((B, solver.nx), np.float32), np.tile(goal, (B, 1)))
        assert np.isfinite(res.stats.initial_merit).all(), res.stats.initial_merit
        assert np.isfinite(res.stats.min_merit).all()
