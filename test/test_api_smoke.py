"""Smoke coverage for public API names no other gate touched (audit 2026-09-19
§9): per-batch setters, cost/linsys/ADMM switches, resets, SolveResult
accessors, the hypothesis/estimator layer, MPCPolicy, and MPC_GATO.run_mpc_fig8.
Each: shapes / finiteness / determinism plus one semantic assertion — cheap,
GPU, non-slow."""
import numpy as np
import pytest

import gato
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS, FIG8_DEFAULT_PARAMS
from gato.common import figure8

pytestmark = pytest.mark.gpu

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}


def _problem(plant, N, B):
    q0 = np.asarray(START[plant], dtype=np.float32)
    x = np.tile(np.concatenate([q0, np.zeros_like(q0)]), (B, 1))
    ref = np.zeros((B, 6 * N), dtype=np.float32)
    ref[:, 0::6], ref[:, 1::6], ref[:, 2::6] = 0.35, 0.25, 0.5
    return x, ref


@pytest.fixture
def solver(make_solver, smallest_module):
    plant, N = smallest_module
    return make_solver(plant, N, batch_size=3)


def test_solve_result_accessors(solver, smallest_module):
    plant, N = smallest_module
    x, ref = _problem(plant, N, 3)
    r = solver.solve(x, ref)
    assert r.batch_size == 3 and r.n_fc == 0
    assert r.xu_b(1).shape == (solver.xu_size,)
    assert np.array_equal(r.u0(2), r.control_at(0, 2))
    assert r.control_at(10 ** 6, 0).shape == (solver.n_actuated,)   # clamped to the last control knot
    assert r.fc_at(0).shape == (0,) and r.fc_traj(0).shape == (N - 1, 0)
    assert r.diverged.shape == (3,) and not r.diverged.any()
    lam = solver.get_lambda()
    assert lam.shape == (3, N + 2, solver.nx) and np.isfinite(lam).all()


def test_per_batch_hyperparameter_setters(solver, smallest_module):
    plant, N = smallest_module
    x, ref = _problem(plant, N, 3)
    base = solver.solve(x, ref)
    s = solver
    s.solver.set_rho_penalty_batch(np.full(3, 1e-3, np.float32), True)
    s.set_drho_batch(np.array([1.0, 1.0, 1.0], np.float32))
    s.solver.set_mu_batch(np.array([1.0, 1.0, 1.0], np.float32))
    s.solver.set_pcg_tol_batch(np.array([1e-4, 1e-4, 1e-4], np.float32))
    s.reset()                                                 # stateful solver: reset before comparing
    same = s.solve(x, ref)
    np.testing.assert_array_equal(base.xu, same.xu)          # the defaults, re-set explicitly -> bitwise
    s.solver.set_mu_batch(np.array([1.0, 50.0, 1.0], np.float32))   # only row 1 changes
    s.reset()
    diff = s.solve(x, ref)
    np.testing.assert_array_equal(diff.xu[0], base.xu[0])
    np.testing.assert_array_equal(diff.xu[2], base.xu[2])
    assert not np.array_equal(diff.xu[1], base.xu[1])


def test_cost_weights_and_reset(solver, smallest_module):
    plant, N = smallest_module
    x, ref = _problem(plant, N, 3)
    base = solver.solve(x, ref)
    solver.set_cost_weights(q_cost=20.0)
    assert solver.params.q_cost == 20.0                       # params stays truthful
    heavy = solver.solve(x, ref)
    assert not np.array_equal(heavy.xu, base.xu)
    solver.set_cost_weights(q_cost=2.0)
    # the solver is STATEFUL across solves (adapted trust-region rho): reset()
    # restores the construction state -> bitwise the first solve again
    solver.reset()
    np.testing.assert_array_equal(solver.solve(x, ref).xu, base.xu)
    solver.set_cost_weights_per_knot(None)                    # clearing an unset table is a no-op
    solver.reset()
    np.testing.assert_array_equal(solver.solve(x, ref).xu, base.xu)


def test_linsys_and_admm_switches(solver, smallest_module):
    plant, N = smallest_module
    x, ref = _problem(plant, N, 3)
    solver.set_linsys("bdsv")
    assert solver.linsys == solver.params.linsys == "bdsv"
    r_b = solver.solve(x, ref)
    assert r_b.stats.linsys == "bdsv" and np.isfinite(r_b.xu).all()
    solver.set_linsys("bdsv_first")
    assert solver.solve(x, ref).stats.linsys == "bdsv_first"
    solver.set_linsys("pcg")
    solver.enable_limit_admm(rho=0.01, iters=5)
    solver.set_admm_linsys("bdsv_factor")
    solver.set_admm_merit(True)
    solver.set_admm_rho_adaptation(True)
    r = solver.solve(x, ref)
    assert r.stats.admm_r_prim.shape == (3,) and np.isfinite(r.xu).all()
    scale = solver.get_admm_rho_scale()
    assert scale.shape == (3,) and (scale > 0).all()
    solver.set_admm_linsys("pcg")
    solver.reset_rho(); solver.reset_dual()
    assert np.isfinite(solver.solve(x, ref).xu).all()
    solver.set_collect_stats(False)
    r2 = solver.solve(x, ref)
    assert r2.stats.pcg_iters.size == 0 and np.isfinite(r2.xu).all()   # stats off: no per-iteration arrays
    solver.set_collect_stats(True)


def test_hypotheses_estimators_and_policy(make_solver, smallest_module, urdfs):
    pin = pytest.importorskip("pinocchio")
    from gato import ForceEstimator, CEMForceEstimator, ForceHypothesisBatch, MPCController, MPCPolicy, TrajectoryReference
    plant, N = smallest_module
    B = 4
    model = pin.buildModelFromUrdf(str(urdfs[plant]))
    for est in (ForceEstimator(batch_size=B, seed=0), CEMForceEstimator(batch_size=B, seed=0)):
        s = make_solver(plant, N, batch_size=B)
        hyp = ForceHypothesisBatch(est, model, ee_frame="EE")
        assert hyp.batch_size == B
        ctrl = MPCController(s, hypotheses=hyp, linsys="pcg")
        traj = figure8(0.01, **FIG8_DEFAULT_PARAMS)[: 6 * (N + 20)]
        pol = MPCPolicy(ctrl, TrajectoryReference(traj, 0.01, N), dt_step=0.01)
        x0 = np.concatenate([np.asarray(START[plant], np.float32), np.zeros(s.nq, np.float32)])
        pol.reset(x0)
        u = pol(x0)
        assert u.shape == (s.n_actuated,) and np.isfinite(u).all()
        assert pol.last.best_id in range(B) and isinstance(pol.last.hypo_stats, dict)
        # deterministic under a fresh policy with the same seed
        s2 = make_solver(plant, N, batch_size=B)
        est2 = type(est)(batch_size=B, seed=0)
        pol2 = MPCPolicy(MPCController(s2, hypotheses=ForceHypothesisBatch(est2, model, ee_frame="EE"), linsys="pcg"),
                         TrajectoryReference(traj, 0.01, N), dt_step=0.01)
        pol2.reset(x0)
        np.testing.assert_array_equal(u, pol2(x0))


def test_mpc_gato_run_mpc_fig8_fixed_pacing_deterministic(smallest_module, urdfs):
    pin = pytest.importorskip("pinocchio")
    from gato.mpc_gato import MPC_GATO
    plant, N = smallest_module
    urdf = str(urdfs[plant])
    traj = figure8(0.01, **FIG8_DEFAULT_PARAMS)
    x0 = np.concatenate([np.asarray(START[plant]), np.zeros(len(START[plant]))])
    outs = []
    for _ in range(2):
        mpc = MPC_GATO(pin.buildModelFromUrdf(urdf), urdf, N=N, dt=0.01, batch_size=1, plant_type=plant, linsys="pcg")
        stats = mpc.run_mpc_fig8(x0, traj, sim_dt=0.001, sim_time=0.05, pace_by_solve_time=False)
        outs.append(stats)
    def same(a, b):
        if isinstance(a, dict):
            return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple, np.ndarray)):
            return len(a) == len(b) and all(same(u, v) for u, v in zip(a, b))
        if isinstance(a, float) and k_is_time(a):
            return True
        return a == b
    def k_is_time(v):   # wall-clock fields differ run to run; everything else must be bitwise
        return False
    a, b = outs
    assert type(a) is type(b)
    if isinstance(a, tuple):   # (trajectory, stats)
        assert same(a[0], b[0])
        a, b = a[1], b[1]
    keys = [k for k in a if "time" not in k.lower()]
    assert keys, list(a)
    for k in keys:
        assert same(a[k], b[k]), k
