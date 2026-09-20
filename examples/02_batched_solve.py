"""Intro example 2: a BATCHED GATO solve (the headline feature).

Solves M=8 figure-8-tracking problems simultaneously in one GPU call, each batch
member using a different damping parameter rho (log-spaced) — the batched
hyperparameter idea from Case Study 1. Prints the per-instance final merit and
which member converged best. This is GATO's core differentiator: tens-to-hundreds
of solves in one block-parallel launch.

Run from the repo root (needs the bsqpN64_indy7 module built):
    python examples/02_batched_solve.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from gato.interface import BSQP
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS, SolverParams

URDF = os.path.join(os.path.dirname(__file__), "indy7_description", "indy7.urdf")
N, DT, M = 64, 0.01, 8
# one rho per batch member, log-spaced 1e-4..1e1
rho_batch = np.power(10, np.linspace(-4, 1, M)).astype(np.float32)
solver = BSQP(model_path=URDF, batch_size=M, N=N, dt=DT, params=SolverParams(max_sqp_iters=10, adapt_rho=True), rho_batch=rho_batch, plant_type="indy7")

nx, nu = solver.nx, solver.nu
x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(nx - len(INDY7_START_CONFIGS["ready"])))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)

# all M members share the same problem here; only rho differs across the batch
x0_B = np.tile(x0, (M, 1))
ref_B = np.tile(ref, (M, 1))
res = solver.solve(x0_B, ref_B)

merits = res.stats.final_merit
best = int(np.argmin(merits))
print(f"GATO batched solve (Indy7, N={N}, M={M}) in one launch:")
print(f"  GPU solve time : {res.solve_time_us / 1000.0:.3f} ms for all {M} solves")
for i in range(M):
    mark = "  <- best" if i == best else ""
    print(f"  member {i}: rho={rho_batch[i]:.1e}  final_merit={merits[i]:.4f}{mark}")
