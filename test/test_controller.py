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


def test_linsys_defaults_fixed_base(make_solver, smallest_module):
    """Wired defaults (08-12): fixed-base controller = auto @ tau 0.1; the raw
    solver default stays pcg; explicit args always win."""
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    assert solver.linsys == "pcg"       # raw solver default unmoved
    ctrl = MPCController(solver)
    assert ctrl.linsys == "auto"
    assert ctrl.bdsv_threshold == pytest.approx(0.1)
    assert solver.linsys == "pcg"       # auto defers the path choice to step time

    pinned = MPCController(make_solver(plant, N, batch_size=1), linsys="bdsv")
    assert pinned.linsys == "bdsv" and pinned.solver.linsys == "bdsv"

    tuned = MPCController(make_solver(plant, N, batch_size=1),
                          linsys="auto", bdsv_threshold=0.25)
    assert tuned.bdsv_threshold == pytest.approx(0.25)


def test_linsys_defaults_floating():
    """Floating-base wired default = bdsv, solver AND controller."""
    import importlib.util
    if importlib.util.find_spec("gato.bsqpN16_go2") is None:
        pytest.skip("bsqpN16_go2 module not built")
    import gato
    from pathlib import Path
    urdf = Path(gato.__file__).resolve().parents[2] / "external" / "GRiD" \
        / "config" / "robot_assets" / "go2.urdf"
    solver = gato.BSQP(model_path=str(urdf), batch_size=1, N=16, dt=0.01,
                       plant_type="go2")
    assert solver.floating_base
    assert solver.linsys == "bdsv"
    ctrl = MPCController(solver)
    assert ctrl.linsys == "bdsv"


def test_linsys_tuned_entry_pickup(make_solver, smallest_module, tmp_path, monkeypatch):
    """MPCController(task_tag=...) resolves a tuned table entry (via
    $GATO_LINSYS_TUNING); an explicit linsys arg still wins."""
    from gato.linsys_autotune import save_tuning
    plant, N = smallest_module
    p = tmp_path / "tuning.json"
    save_tuning(plant, N, "unittest", {"policy": "bdsv_first", "tau": None}, path=p)
    monkeypatch.setenv("GATO_LINSYS_TUNING", str(p))
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver, task_tag="unittest")
    assert ctrl.linsys == "bdsv_first"
    assert solver.linsys == "bdsv_first"
    pinned = MPCController(make_solver(plant, N, batch_size=1),
                           linsys="pcg", task_tag="unittest")
    assert pinned.linsys == "pcg"


def test_linsys_auto_switches_on_pred_err(make_solver, smallest_module):
    """auto: warm step -> pcg; a kicked state past tau -> bdsv_first."""
    plant, N = smallest_module
    solver = make_solver(plant, N, batch_size=1)
    ctrl = MPCController(solver)      # wired default: auto @ 0.1
    x0 = np.zeros(solver.nx, dtype=np.float32)
    ref = GoalReference([0.35, 0.25, 0.5], N).window(0.0)
    ctrl.reset(x0)
    ctrl.warmup(x0, ref)
    r = ctrl.step(x0, ref)            # x == prediction: warm
    assert r.pred_err <= ctrl.bdsv_threshold
    assert solver.linsys == "pcg"
    x_kick = x0.copy()
    x_kick[: solver.nq] += 0.5
    r = ctrl.step(x_kick, ref)        # far off the shifted prediction: cold
    assert r.pred_err > ctrl.bdsv_threshold
    assert solver.linsys == "bdsv_first"


def test_linsys_auto_default_deterministic(make_solver, smallest_module):
    """Run-twice bitwise determinism of a short closed loop under the wired
    auto default (pred_err is deterministic, so the pcg/bdsv_first schedule
    and every trajectory must be bit-identical)."""
    plant, N = smallest_module

    def run():
        solver = make_solver(plant, N, batch_size=1)
        ctrl = MPCController(solver)
        x = np.zeros(solver.nx, dtype=np.float32)
        ref = GoalReference([0.35, 0.25, 0.5], N).window(0.0)
        ctrl.reset(x)
        ctrl.warmup(x, ref)
        out = []
        for k in range(6):
            if k == 3:                # one kick to exercise the cold branch
                x = x.copy()
                x[: solver.nq] += 0.3
            r = ctrl.step(x, ref)
            out.append(r.xu_best.copy())
            x = np.asarray(solver.sim_forward(x, r.u, solver.dt),
                           dtype=np.float32).reshape(solver.nx)
        return np.stack(out)

    np.testing.assert_array_equal(run(), run())
