"""ADMM inner-loop residual early exit (BSQP.set_admm_exit, W2a 2026-10-15) and
the controller's per-step ADMM inner-linsys policy. Mechanism tests: counts,
bit-parity and feasibility — never timing."""
import numpy as np
import pytest

import gato
from conftest import TEST_PARAMS, URDFS, arm_problem
from gato.controller import MPCController
from gato.policy import GoalReference

pytestmark = pytest.mark.gpu

RHO, ITERS = 10.0, 10          # the box test's binding configuration (test_rowgroups.py)
# Measured on this problem (iiwa14 N8 reach, B=4, 2026-10-15): at eps_rel 1e-2 a warm repeat
# exits after 1-1.6 inner iterations per SQP iteration on both arms with zero box violation;
# at 1e-3 the inner loop runs its budget (the fixed point is reached only to ~1e-2 relative).
EPS_ABS, EPS_REL = 0.0, 1e-2


def _inputs(plant, N, B):
    return arm_problem(plant, N, B, jitter=0.01)


def _admm(make_solver, plant, N, B, *, linsys="pcg", exit_on=True, check_every=1, max_pcg_iters=None):
    if max_pcg_iters is None:
        s = make_solver(plant, N, batch_size=B)
    else:
        s = gato.BSQP(model_path=str(URDFS[plant]), batch_size=B, N=N, dt=0.01, plant_type=plant,
                      params=TEST_PARAMS.replace(max_pcg_iters=max_pcg_iters))
    s.enable_limit_admm(rho=RHO, iters=ITERS)
    s.set_admm_linsys(linsys)
    if exit_on:
        s.set_admm_exit(EPS_ABS, EPS_REL, check_every)
    return s


def _warm_pair(s, X, goals):
    """A cold solve then a warm repeat (dual + trajectory warm start)."""
    first = s.solve(X, goals)
    second = s.solve(X, goals, first.xu)
    return first, second


def test_exit_off_is_the_fixed_budget_bitwise(make_solver, smallest_module):
    plant, N = smallest_module
    X, goals = _inputs(plant, N, 3)
    a = _admm(make_solver, plant, N, 3, exit_on=False)
    b = _admm(make_solver, plant, N, 3, exit_on=False)
    b.set_admm_exit(0.0, 0.0, 1)                       # explicit zeros = off
    ra, rb = a.solve(X, goals), b.solve(X, goals)
    np.testing.assert_array_equal(ra.xu, rb.xu)
    # the counts read the budget exactly: iters per SQP iteration, every entry
    assert ra.stats.admm_iters is not None
    np.testing.assert_array_equal(ra.stats.admm_iters, ITERS * ra.stats.sqp_iters)
    assert ra.stats.admm_linsys_iters.shape == (3,) and (ra.stats.admm_linsys_iters >= 0).all()
    with pytest.raises(ValueError):
        a.set_admm_exit(-1e-3, 0.0, 1)


def test_exit_stops_early_and_keeps_feasibility(make_solver, smallest_module):
    """With the residual exit the warm repeat runs fewer inner iterations than
    the budget, the residuals at exit satisfy the test on the reported scale,
    and the approximately-hard feasibility claim of the fixed budget holds."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)
    fixed = _admm(make_solver, plant, N, B, exit_on=False)
    _, rf = _warm_pair(fixed, X, goals)
    s = _admm(make_solver, plant, N, B)
    r1, r2 = _warm_pair(s, X, goals)
    for r in (r1, r2):
        assert np.isfinite(r.xu).all()
        assert (r.stats.admm_iters <= ITERS * r.stats.sqp_iters).all()
        assert (r.stats.admm_iters >= r.stats.sqp_iters).all()      # at least one inner iteration per SQP iteration
    # the warm repeat exits early on every entry (the loop starts near its fixed point)
    assert (r2.stats.admm_iters < ITERS * r2.stats.sqp_iters).all(), r2.stats.admm_iters
    # exit residuals satisfy the abs+rel test on the auxiliary scale (the per-group
    # test the kernel runs implies this whole-solve form)
    z = np.abs(s.get_admm_state()["z"]).reshape(B, -1).max(axis=1)
    assert (r2.stats.admm_r_prim <= EPS_ABS + EPS_REL * z + 1e-7).all()
    assert (r2.stats.admm_r_dual <= EPS_ABS + EPS_REL * RHO * z + 1e-7).all()
    # feasibility: the same claim the fixed budget makes (test_admm_enforces_boxes)
    assert r2.stats.row_max_violation.max() < 1e-4
    assert r2.stats.row_max_violation.max() <= 10 * max(rf.stats.row_max_violation.max(), 1e-6)


def test_exit_is_deterministic_and_batch_independent(make_solver, smallest_module):
    """Entries exit at their own iteration: the batched solve equals each entry
    solved alone (bitwise), and the run is bit-deterministic."""
    plant, N = smallest_module
    B = 3
    X, goals = _inputs(plant, N, B)

    def run(B_, Xb, gb):
        s = _admm(make_solver, plant, N, B_)
        return _warm_pair(s, Xb, gb)[1]

    r = run(B, X, goals)
    r_again = run(B, X, goals)
    np.testing.assert_array_equal(r.xu, r_again.xu)
    np.testing.assert_array_equal(r.stats.admm_iters, r_again.stats.admm_iters)
    for b in range(B):
        rb = run(1, X[b:b + 1], goals[b:b + 1])
        np.testing.assert_array_equal(r.xu[b], rb.xu[0])
        assert r.stats.admm_iters[b] == rb.stats.admm_iters[0]


def test_host_check_period_does_not_change_the_iterates(make_solver, smallest_module):
    """check_every only decides when the host stops launching empty inner
    iterations; the on-device skip mask fixes the iterates."""
    plant, N = smallest_module
    X, goals = _inputs(plant, N, 2)
    res = []
    for ce in (0, 1, 3):
        s = _admm(make_solver, plant, N, 2, check_every=ce)
        res.append(_warm_pair(s, X, goals)[1])
    for r in res[1:]:
        np.testing.assert_array_equal(res[0].xu, r.xu)
        np.testing.assert_array_equal(res[0].stats.admm_iters, r.stats.admm_iters)


def test_exit_requires_a_converged_inner_solve(make_solver, smallest_module):
    """The inexact-ADMM guard: with the PCG cap so low that no inner solve
    converges, the pcg arm never exits (the budget runs), while the factor
    arm (exact x-update) does."""
    plant, N = smallest_module
    X, goals = _inputs(plant, N, 2)
    capped = _admm(make_solver, plant, N, 2, max_pcg_iters=1)
    _, r = _warm_pair(capped, X, goals)
    np.testing.assert_array_equal(r.stats.admm_iters, ITERS * r.stats.sqp_iters)
    direct = _admm(make_solver, plant, N, 2, linsys="bdsv_factor", max_pcg_iters=1)
    _, rd = _warm_pair(direct, X, goals)
    assert (rd.stats.admm_iters < ITERS * rd.stats.sqp_iters).all()
    np.testing.assert_array_equal(rd.stats.admm_linsys_iters, rd.stats.admm_iters)   # one back-solve per inner iteration


def test_pcg_and_factor_arms_agree_at_exit(make_solver, smallest_module):
    """Inexact-ADMM check (plan H2.4b): the warm PCG arm's exit must not be
    read on a stale residual — its returned feasibility matches the exact
    factor arm's to within an order of magnitude and both meet the claim."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)
    rp = _warm_pair(_admm(make_solver, plant, N, B, linsys="pcg"), X, goals)[1]
    rf = _warm_pair(_admm(make_solver, plant, N, B, linsys="bdsv_factor"), X, goals)[1]
    vp, vf = rp.stats.row_max_violation.max(), rf.stats.row_max_violation.max()
    assert vp < 1e-4 and vf < 1e-4, (vp, vf)
    assert vp <= 10 * max(vf, 1e-6), (vp, vf)
    assert np.isfinite(rp.xu).all() and np.isfinite(rf.xu).all()


def test_controller_admm_linsys_auto_switches_on_warmness(make_solver, smallest_module):
    """admm_linsys='auto': a cold step (pred_err above the threshold) runs the
    factor arm, a warm step the PCG arm; StepResult reports which."""
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    solver.enable_limit_admm(rho=RHO, iters=ITERS, eps_abs=EPS_ABS, eps_rel=EPS_REL)
    ctrl = MPCController(solver, linsys="pcg", admm_linsys="auto", admm_threshold=0.1)
    assert ctrl.admm_threshold == 0.1
    x0 = np.zeros(solver.nx, dtype=np.float32)
    ref = GoalReference([0.35, 0.25, 0.5], N).window(0.0)
    ctrl.reset(x0)
    ctrl.warmup(x0, ref)
    r = ctrl.step(x0, ref)
    assert r.admm_linsys == "pcg" and solver.admm_linsys == "pcg"
    x_kick = x0.copy()
    x_kick[: solver.nq] += 0.5                       # pred_err ~1.2 > 0.1
    r = ctrl.step(x_kick, ref)
    assert r.pred_err > 0.1
    assert r.admm_linsys == "bdsv_factor" and solver.admm_linsys == "bdsv_factor"
    assert r.solve.stats.admm_iters is not None
    with pytest.raises(ValueError):
        MPCController(make_solver(plant, N), admm_linsys="direct")


def test_controller_admm_linsys_pinned_and_off(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver, admm_linsys="bdsv_factor")
    assert ctrl.admm_threshold is None and solver.admm_linsys == "bdsv_factor"
    # the default policy is auto with the SQP threshold as the fallback
    ctrl2 = MPCController(make_solver(plant, N), linsys="auto", bdsv_threshold=0.3)
    assert ctrl2.admm_linsys == "auto" and ctrl2.admm_threshold == 0.3
    # no ADMM group: the policy is inert and StepResult says so
    x0 = np.zeros(solver.nx, dtype=np.float32)
    ref = GoalReference([0.35, 0.25, 0.5], N).window(0.0)
    ctrl2.reset(x0)
    r = ctrl2.step(x0, ref)
    assert r.admm_linsys == "" and r.solve.stats.admm_iters is None
