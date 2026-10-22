"""GPU smoke: every built module solves to finite output, deterministically;
per-knot weight arrays reproduce the scalar path bit-exactly."""
import numpy as np
import pytest

import gato
from conftest import arm_problem  # noqa: E402
from conftest import ARM_START as START  # noqa: E402  (every fixed-base arm in the receipt profile)

pytestmark = pytest.mark.gpu


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


def test_solve_budget_cutoff(make_solver, smallest_module):
    """solve(budget_s=...): the iteration-level cutoff on the raw solver. A budget
    no iteration fits runs exactly one iteration on every batch entry, bit-identical
    to a max_sqp_iters=1 solver, and reports it; None reports no budget stop."""
    plant, N = smallest_module
    x, goals = arm_problem(plant, N, B=3, jitter=0.05)
    s = make_solver(plant, N, batch_size=3)
    r = s.solve(x, goals, budget_s=1e-9)
    assert r.stats.budget_hit and (r.stats.sqp_iters == 1).all()
    one = make_solver(plant, N, batch_size=3, max_sqp_iters=1).solve(x, goals)
    assert not one.stats.budget_hit
    np.testing.assert_array_equal(r.xu, one.xu)
    full = make_solver(plant, N, batch_size=3).solve(x, goals)
    assert not full.stats.budget_hit and int(full.stats.sqp_iters.max()) >= 2
    with pytest.raises(ValueError, match="budget_s"):
        s.solve(x, goals, budget_s=-1.0)
    # a one-iteration solver under a generous budget: the iteration-cost history
    # has one slot and the solve ends on max_sqp_iters, not on the budget
    r1 = make_solver(plant, N, batch_size=3, max_sqp_iters=1).solve(x, goals, budget_s=1.0)
    assert not r1.stats.budget_hit and (r1.stats.sqp_iters == 1).all()
    np.testing.assert_array_equal(r1.xu, one.xu)


def test_merit_fused_and_unfused_paths_bit_identical(smallest_module, tmp_path):
    """GATO_MERIT_FUSED=0 (one candidate per block) and =1 (all candidates in one
    block per knot) give bit-identical solves; the default picks by SM count, so
    both paths are forced here in their own processes (the override is read once)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    plant, N = smallest_module
    here = Path(__file__).resolve().parent
    script = (
        "import sys, numpy as np, gato\n"
        f"sys.path.insert(0, {str(here)!r})\n"
        "from conftest import arm_problem, TEST_PARAMS, URDFS\n"
        f"X, goals = arm_problem({plant!r}, {N}, 4, jitter=0.01)\n"
        f"s = gato.BSQP(model_path=str(URDFS[{plant!r}]), batch_size=4, N={N}, dt=0.01, params=TEST_PARAMS, plant_type={plant!r})\n"
        "r = s.solve(X, goals)\n"
        "np.savez(sys.argv[1], xu=np.asarray(r.xu), iters=np.asarray(r.stats.sqp_iters))\n")
    out = {}
    for fused in ("0", "1"):
        env = dict(os.environ, GATO_MERIT_FUSED=fused)
        path = tmp_path / f"fused{fused}.npz"
        subprocess.run([sys.executable, "-c", script, str(path)], check=True, env=env, cwd=here)
        out[fused] = np.load(path)
    assert np.isfinite(out["0"]["xu"]).all() and (out["0"]["iters"] >= 1).all()
    np.testing.assert_array_equal(out["1"]["xu"], out["0"]["xu"])
    np.testing.assert_array_equal(out["1"]["iters"], out["0"]["iters"])


def test_stage_times_and_summary(make_solver, smallest_module):
    """SolverStats.stage_times_us is (iterations, 5) device microseconds per stage,
    summing to no more than the wall solve time; summary() names every stage and
    the merit launch; with stats off the array is empty and the plan is unchanged."""
    from gato.interface import STAGE_NAMES
    plant, N = smallest_module
    X, goals = _inputs(plant, N, 2)
    s = make_solver(plant, N, batch_size=2)
    res = s.solve(X, goals)
    st = res.stats
    iters = int(st.sqp_iters.max())
    assert st.stage_times_us.shape == (iters, 5) and np.isfinite(st.stage_times_us).all()
    assert (st.stage_times_us >= 0).all() and st.initial_merit_time_us >= 0
    assert st.stage_times_us[:, :3].min() > 0           # every started iteration reached the linear system
    assert st.stage_times_us.sum() + st.initial_merit_time_us <= res.solve_time_us + 100.0
    text = res.summary()
    assert all(name in text for name in STAGE_NAMES) and "merit launch" in text and "initial merit" in text
    assert isinstance(st.merit_fused, bool)

    # a fresh solver (the trust-region state carries across solves of one solver)
    quiet = make_solver(plant, N, batch_size=2)
    quiet.set_collect_stats(False)
    off = quiet.solve(X, goals)
    np.testing.assert_array_equal(off.xu, res.xu)          # collection never touches the plan
    assert off.stats.stage_times_us.shape == (0, 5) and "not collected" in off.summary()
