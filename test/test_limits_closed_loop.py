"""Limits enforced in CLOSED LOOP: the ADMM box mechanism keeps the SIMULATED
plant (pinocchio integrates the applied torques; the planner does not grade
itself) inside a velocity cap that the figure-8 reference cannot satisfy, while
the telemetry-only run exceeds it — the closed-loop counterpart of the
solver-level box tests in test_rowgroups.py (examples/09_mpc_with_limits.py)."""
import numpy as np
import pytest

import gato
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS
from gato.rowkinds import KIND_BOX_QD

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

START = {"indy7": INDY7_START_CONFIGS["ready"]}
QD_CAP = 1.0   # rad/s
SIM_TIME, DT = 1.5, 0.01
MIN_N = 32     # measured: N <= 16 (<= 160 ms lookahead) cannot brake the figure-8 inside the cap


def _run(plant, N, urdf, enforce, eps_rel=None):
    pin = pytest.importorskip("pinocchio")
    from gato.mpc_gato import MPC_GATO
    mpc = MPC_GATO(pin.buildModelFromUrdf(urdf), urdf, N=N, dt=DT, batch_size=1, plant_type=plant,
                   params=dict(max_sqp_iters=5, rho=1e-3), linsys="pcg",
                   admm_linsys="auto" if eps_rel else "pcg")
    s = mpc.solver
    (s.enable_limit_admm(rho=0.01, iters=10, eps_rel=eps_rel) if enforce else s.enable_limit_telemetry())
    groups = s.get_row_groups()
    g = next(i for i, grp in enumerate(groups) if grp["kind"] == KIND_BOX_QD)
    n = groups[g]["n_rows"]
    s.set_row_group_bounds(g, -QD_CAP * np.ones(n), QD_CAP * np.ones(n))
    x0 = np.concatenate([np.asarray(START[plant]), np.zeros(len(START[plant]))])
    _, stats = mpc.run_mpc_fig8(x0, figure8(DT, **FIG8_DEFAULT_PARAMS), sim_dt=0.001,
                                sim_time=SIM_TIME, pace_by_solve_time=False)
    qd = np.abs(np.asarray(stats["joint_velocities"]))
    viol = np.asarray(stats["row_max_violation"])
    assert viol.shape == (len(qd), len(groups)) and np.isfinite(viol).all()
    if enforce:
        _run.inner = np.asarray(stats["admm_iters"], dtype=float)   # inner iterations per tick
    return qd, viol[:, g]


def test_admm_velocity_box_holds_in_closed_loop(urdfs):
    # the figure-8 defaults are the Indy7's task; the receipt profile builds indy7 N32/46/64/128
    combos = [k for k in gato.available() if k[0] == "indy7" and k[1] >= MIN_N]
    if not combos:
        pytest.fail(f"a receipt-profile indy7 module with N >= {MIN_N} must be built")
    plant, N = min(combos, key=lambda k: k[1])
    urdf = str(urdfs[plant])
    qd_free, _ = _run(plant, N, urdf, enforce=False)
    assert qd_free.max() > 1.5 * QD_CAP, "the reference must stress the cap for this test to mean anything"
    qd, plan_viol = _run(plant, N, urdf, enforce=True)
    # Measured (indy7, N32/N64, rho 0.01, 10 ADMM iterations per SQP step, 5 steps per tick):
    # the planner's residual box violation is 0.035-0.039 rad/s and the simulated plant peaks
    # at 1.03-1.04 rad/s against the 1.0 cap, versus 1.6-2.0 rad/s unenforced.
    assert plan_viol.max() < 0.05 * QD_CAP, plan_viol.max()
    assert qd.max() < 1.1 * QD_CAP, qd.max()


def test_admm_residual_exit_holds_in_closed_loop(urdfs):
    """The residual early exit (set_admm_exit, eps_rel 1e-2) with the warm-startedness
    inner-linsys policy keeps the same closed-loop enforcement as the fixed budget
    while running a fraction of the inner iterations (measured 2026-10-15, Indy7
    N32: 1.9 of 10 per SQP iteration on average; plan violation 2.4e-2 vs 3.5e-2,
    simulated peak 1.027 vs 1.034 rad/s)."""
    combos = [k for k in gato.available() if k[0] == "indy7" and k[1] >= MIN_N]
    if not combos:
        pytest.fail(f"a receipt-profile indy7 module with N >= {MIN_N} must be built")
    plant, N = min(combos, key=lambda k: k[1])
    urdf = str(urdfs[plant])
    qd, plan_viol = _run(plant, N, urdf, enforce=True, eps_rel=1e-2)
    inner = _run.inner
    assert plan_viol.max() < 0.05 * QD_CAP, plan_viol.max()
    assert qd.max() < 1.1 * QD_CAP, qd.max()
    # 5 SQP iterations x 10 = the budget per tick; the exit must actually fire
    assert inner.mean() < 0.5 * 50, inner.mean()
    assert inner.min() >= 5 and inner.max() <= 50
