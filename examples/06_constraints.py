"""Intro example 6: constraints — joint/torque limits, linear control rows, collision.

The constraint layer is a set of ROW GROUPS bound to a mechanism:
  * enable_limit_admm()   — the URDF position/velocity/torque boxes under the
                            ADMM-projection mechanism ("approximately hard").
  * add_lin_u_rows(C, ...) — extra rows g = C @ u + d on the control (here: a
                            tighter torque cap on the first two joints), inheriting
                            the active mechanism.
  * set_collision_environment + enable_collision — clearance rows for the
                            codegen'd collision spheres against runtime obstacles.
Every solve reports the TRUE violation of the returned trajectory per row group
in stats.row_max_violation (telemetry is always on once a group exists).

Rulings, measured defaults and provenance: docs/constraints.md.
Needs the bsqpN64_indy7 module; runs from any cwd:
    python examples/06_constraints.py
"""
from pathlib import Path

import numpy as np

import gato
from gato import BSQP, SolverParams
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / gato.robot_info("indy7")["urdf"]
N, DT = 64, 0.01

# rho=1e-3: the ADMM inner loop factors the Schur system directly (f32 bdsv), which
# needs a regularized system (rho > 0)
solver = BSQP(URDF, batch_size=1, N=N, dt=DT,
              params=SolverParams(max_sqp_iters=5, rho=1e-3), plant_type="indy7")

# 1. URDF limit boxes (q / qd / tau) under ADMM. Call this BEFORE appending rows.
solver.enable_limit_admm(rho=0.01, iters=10)

# 2. Two extra LIN_U rows: |tau_0|, |tau_1| <= 30 N m (C picks the joints, lo/hi the box)
C = np.zeros((2, solver.nu), dtype=np.float32)
C[0, 0] = 1.0
C[1, 1] = 1.0
solver.add_lin_u_rows(C, lo=[-30.0, -30.0], hi=[30.0, 30.0])

# 3. A spherical obstacle near the figure-8 (world frame: x, y, z, radius), then the
#    clearance group: every codegen'd collision sphere stays >= margin away from it.
solver.set_collision_environment(spheres=[(0.0, 0.55, 0.95, 0.10)])
solver.enable_collision(margin=0.02)

x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(solver.nv))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)
res = solver.solve(x0[None], ref[None])

groups = solver.get_row_groups()
viol = np.asarray(res.stats.row_max_violation).reshape(len(groups), -1)
print(f"constrained solve (Indy7, N={N}): {res.solve_time_us / 1000:.3f} ms, "
      f"sqp_iters={res.stats.sqp_iters[0]}, final merit {res.stats.final_merit[0]:.4f}")
print(f"ADMM residuals: primal {res.stats.admm_r_prim[0]:.2e}  dual {res.stats.admm_r_dual[0]:.2e}")
print("max violation of the returned trajectory per row group:")
for g, v in zip(groups, viol[:, 0]):
    print(f"  {str(g.get('kind', g)):<12} mech={str(g.get('mech', '?')):<10} max_viol={v:.2e}")
