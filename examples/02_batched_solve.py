"""Intro example 2: a BATCHED GATO solve (the headline feature).

Solves M=8 figure-8-tracking problems simultaneously in one GPU call, each batch
member using a different damping parameter rho (log-spaced) — the batched
hyperparameter idea from Case Study 1. Prints the per-instance final merit and
which member converged best. This is GATO's core differentiator: tens-to-hundreds
of solves in one block-parallel launch.

Needs the bsqpN64_indy7 module built; runs from any cwd:
    python examples/02_batched_solve.py
"""
from pathlib import Path

import numpy as np

import gato
from gato import BSQP, SolverParams
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / gato.robot_info("indy7")["urdf"]
N, DT, M = 64, 0.01, 8
# one rho per batch member, log-spaced 1e-4..1e1 (the line search adapts each one per iter)
rho_batch = np.power(10, np.linspace(-4, 1, M)).astype(np.float32)
solver = BSQP(URDF, batch_size=M, N=N, dt=DT, params=SolverParams(max_sqp_iters=10),
              rho_batch=rho_batch, plant_type="indy7")

x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(solver.nv))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)

# all M members share the same problem here; only rho differs across the batch
res = solver.solve(np.tile(x0, (M, 1)), np.tile(ref, (M, 1)))

merits = res.stats.final_merit
best = int(np.argmin(merits))
print(f"GATO batched solve (Indy7, N={N}, M={M}) in one launch:")
print(f"  GPU solve time : {res.solve_time_us / 1000.0:.3f} ms for all {M} solves")
for i in range(M):
    mark = "  <- best" if i == best else ""
    print(f"  member {i}: rho={rho_batch[i]:.1e}  final_merit={merits[i]:.4f}{mark}")
