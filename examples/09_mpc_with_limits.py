"""Intro example 9: joint limits enforced in closed loop.

The paper's figure-8 run (03_mpc_loop.py) plans with soft limit costs only; the
arm is free to exceed the URDF velocity limits when the reference demands it.
This example binds the limit row groups (URDF position / velocity / torque
boxes) to the ADMM-projection mechanism, tightens the velocity box to a cap the
figure-8 cannot satisfy at full speed, and runs the same MPC loop twice:

  * telemetry only  — the boxes are reported, not enforced (the paper setting);
  * ADMM            — the planner keeps every knot inside the boxes.

It then reports the SIMULATED joint velocities against the cap (the plant is
integrated by pinocchio from the applied torques, so this is not the planner
grading itself), the planner's own max violation per row group, and the
tracking error the cap costs. Mechanisms, defaults and provenance:
docs/constraints.md. Fixed simulation pacing: a correctness demo, not timing.

Needs the bsqpN64_indy7 module and a python with pinocchio; runs from any cwd:
    python examples/09_mpc_with_limits.py
"""
from pathlib import Path

import numpy as np
import pinocchio as pin

import gato
from gato import MPC_GATO
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS
from gato.rowkinds import KIND_BOX_QD, KIND_NAMES

REPO = Path(__file__).resolve().parents[1]
URDF = str(REPO / gato.robot_info("indy7")["urdf"])
N, DT, SIM_TIME = 64, 0.01, 3.0
QD_CAP = 1.0          # rad/s, every joint — well below what the figure-8 asks for
# rho=1e-3: the ADMM inner loop factors the Schur system directly and needs a
# regularized system; max_sqp_iters as in 06_constraints.py
PARAMS = dict(max_sqp_iters=5, rho=1e-3)


def run(enforce):
    model = pin.buildModelFromUrdf(URDF)
    mpc = MPC_GATO(model, model_path=URDF, N=N, dt=DT, batch_size=1, plant_type="indy7",
                   params=PARAMS, linsys="pcg")
    solver = mpc.solver
    if enforce:
        solver.enable_limit_admm(rho=0.01, iters=10)
    else:
        solver.enable_limit_telemetry()
    # the three canonical limit groups are installed in BOX_Q, BOX_QD, BOX_U order
    groups = solver.get_row_groups()
    qd_group = next(i for i, g in enumerate(groups) if g["kind"] == KIND_BOX_QD)
    n = groups[qd_group]["n_rows"]
    solver.set_row_group_bounds(qd_group, -QD_CAP * np.ones(n), QD_CAP * np.ones(n))

    x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(model.nv)))
    fig8 = figure8(DT, **FIG8_DEFAULT_PARAMS)
    _, stats = mpc.run_mpc_fig8(x0, fig8, sim_dt=0.001, sim_time=SIM_TIME, pace_by_solve_time=False)
    qd = np.abs(np.asarray(stats["joint_velocities"]))
    viol = np.asarray(stats["row_max_violation"])          # (steps, n_groups)
    return dict(steps=len(qd), qd_max=qd.max(), qd_over=float((qd > QD_CAP).any(axis=1).mean()),
                plan_viol=viol.max(axis=0), groups=groups,
                err=np.asarray(stats["goal_distances_knot0"]).mean())


for label, enforce in (("telemetry only", False), ("ADMM boxes", True)):
    r = run(enforce)
    print(f"\n{label} (Indy7, N={N}, {SIM_TIME} s, velocity box ±{QD_CAP} rad/s, {r['steps']} steps):")
    print(f"  simulated max |qd| : {r['qd_max']:.2f} rad/s   steps over the cap: {100 * r['qd_over']:.0f}%")
    print(f"  mean tracking err  : {r['err']:.4f} m")
    print("  planner max violation per row group:")
    for g, v in zip(r["groups"], r["plan_viol"]):
        print(f"    {KIND_NAMES[g['kind']]:<8} {v:.2e}")
