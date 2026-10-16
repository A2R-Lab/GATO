"""MPCController(limits=...): the controller installs the URDF limit rows by
default on fixed-base arms (ADMM projection with the measured defaults), keeps
a caller-configured solver as configured, and reports the mechanism and the
winning plan's violation per row group in StepResult."""
import numpy as np
import pytest

from gato import MPCController
from gato.linsys_autotune import ADMM_FACTOR_MIN_N
from gato.policy import GoalReference
from gato.rowkinds import KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U

pytestmark = pytest.mark.gpu


def _tick(ctrl, solver):
    x0 = np.zeros(solver.nx, dtype=np.float32)
    ref = GoalReference([0.35, 0.25, 0.5], solver.N).window(0.0)
    ctrl.reset(x0)
    ctrl.warmup(x0, ref)
    return ctrl.step(x0, ref)


def test_default_installs_admm_limit_rows(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    assert solver.get_row_groups() == [] and solver.row_mech is None   # the raw solver stays explicit
    ctrl = MPCController(solver, linsys="pcg")
    assert ctrl.limits == "admm" and solver.row_mech == "admm" and solver.admm_active
    kinds = [g["kind"] for g in solver.get_row_groups()]
    assert kinds == [KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U]
    # the measured defaults: cap 10 with the residual exit at eps_rel 1e-2
    assert solver.admm_eps_rel == pytest.approx(1e-2) and solver.admm_eps_abs == 0.0
    # the inner linear solver follows the horizon unless tuned or pinned
    assert ctrl.admm_linsys == ("bdsv_factor" if N >= ADMM_FACTOR_MIN_N else "pcg")
    r = _tick(ctrl, solver)
    assert r.limits == "admm" and r.admm_linsys == ctrl.admm_linsys
    assert r.row_viol.shape == (3,) and np.isfinite(r.row_viol).all()
    assert r.solve.stats.admm_iters is not None and 1 <= int(r.solve.stats.admm_iters.max()) <= 10 * solver.params.max_sqp_iters


def test_off_installs_nothing(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver, linsys="pcg", limits="off")
    assert ctrl.limits == "" and solver.get_row_groups() == []
    r = _tick(ctrl, solver)
    assert r.limits == "" and r.row_viol is None and r.solve.stats.row_max_violation is None


def test_caller_configuration_stands(make_solver, smallest_module):
    """A solver with row groups already installed is left alone under the default."""
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    solver.enable_limit_barrier()
    ctrl = MPCController(solver, linsys="pcg")
    assert ctrl.limits == "barrier" and solver.row_mech == "barrier" and not solver.admm_active
    # an explicit mechanism re-installs
    ctrl2 = MPCController(solver, linsys="pcg", limits="telemetry")
    assert ctrl2.limits == "telemetry" and solver.row_mech == "telemetry"
    # and "off" keeps what is there, reporting it
    ctrl3 = MPCController(solver, linsys="pcg", limits="off")
    assert ctrl3.limits == "telemetry"


def test_explicit_mechanisms_and_overrides(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver, linsys="pcg", limits="admm", limits_kw=dict(eps_rel=None, iters=4))
    assert ctrl.limits == "admm" and solver.admm_eps_rel == 0.0     # fixed budget, no exit
    r = _tick(ctrl, solver)
    assert int(r.solve.stats.admm_iters.max()) == 4 * int(r.iterations)
    for mech in ("al", "telemetry", "barrier"):
        ctrl = MPCController(make_solver(plant, N, batch_size=1), linsys="pcg", limits=mech)
        assert ctrl.limits == mech and ctrl.solver.row_mech == mech
        r = _tick(ctrl, ctrl.solver)
        assert r.limits == mech and r.row_viol.shape == (3,)
    with pytest.raises(ValueError, match="limits must be"):
        MPCController(make_solver(plant, N, batch_size=1), limits="hard")


def test_batch_reports_the_winner(make_solver, smallest_module):
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=3)
    ctrl = MPCController(solver, linsys="pcg")
    r = _tick(ctrl, solver)
    viol = np.asarray(r.solve.stats.row_max_violation, dtype=np.float32).reshape(3, 3)
    np.testing.assert_array_equal(r.row_viol, viol[:, r.best_id])
