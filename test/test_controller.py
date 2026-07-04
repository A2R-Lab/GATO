"""MPCController semantics: shift math, B=1 selection, StepResult wiring."""
import numpy as np
import pytest

from gato import MPCController
from gato.policy import GoalReference

pytestmark = pytest.mark.gpu


@pytest.fixture
def setup(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver)
    x0 = np.zeros(solver.nx, dtype=np.float32)
    ref = GoalReference([0.35, 0.25, 0.5], N).window(0.0)
    return ctrl, solver, x0, ref


def test_step_result_wiring(setup):
    ctrl, solver, x0, ref = setup
    ctrl.reset(x0)
    ctrl.warmup(x0, ref)
    r = ctrl.step(x0, ref)
    assert r.best_id == 0            # B == 1: nothing to select
    assert r.hypo_stats is None
    nx, nu = solver.nx, solver.nu
    np.testing.assert_array_equal(r.u, r.xu_best[nx: nx + nu])
    assert r.u.shape == (nu,)
    assert np.isfinite(r.xu_best).all()


def test_shift_warm_start_math(setup):
    ctrl, solver, x0, ref = setup
    ctrl.reset(x0)
    ctrl.warmup(x0, ref)
    r = ctrl.step(x0, ref)
    stride = solver.nx + solver.nu
    expected = np.concatenate([r.xu_best[stride:], r.xu_best[-stride:]])
    np.testing.assert_array_equal(ctrl._XU[0], expected)


def test_hold_warm_start(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver, warm_start="hold")
    x0 = np.zeros(solver.nx, dtype=np.float32)
    ref = GoalReference([0.35, 0.25, 0.5], N).window(0.0)
    ctrl.reset(x0)
    r = ctrl.step(x0, ref)
    np.testing.assert_array_equal(ctrl._XU[0], r.xu_best)


def test_warm_start_mode_validated(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    with pytest.raises(ValueError):
        MPCController(solver, warm_start="roll")
