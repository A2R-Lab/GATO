"""Intro example 8: the linear-system path (pcg / bdsv / bdsv_first) and warm starts.

Every SQP iteration solves the block-tridiagonal Schur system S lambda = gamma.
GATO offers three paths, switchable per solve at zero cost (set_linsys):
  * "pcg"        — iterative, warm-start friendly (the default fixed-base path),
  * "bdsv"       — direct block-Cholesky: exact, iteration-count free, flat cost,
  * "bdsv_first" — direct on SQP iteration 0 (the cold linearization), pcg after.
An MPC loop can pick per step from the warm-startedness signal
(MPCController(linsys="auto"): pred_err > bdsv_threshold -> bdsv_first), and
tools/autotune_linsys.py fits that threshold for a (plant, N, task) workload
and persists it to the tuning table MPCController(task_tag=...) reads.

This script solves ONE problem under each path, cold and warm-started, and
prints the per-iteration pcg iteration counts and the device solve time.
The times are illustrative only: timing needs a QUIET box (no other GPU work
— the benchmark harnesses refuse otherwise, examples/benchmarks/_bench.py).

Needs the bsqpN64_indy7 module; runs from any cwd:
    python examples/08_linsys_and_autotune.py
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

# rho > 0: the f32 direct solve needs a regularized Schur system
solver = BSQP(URDF, batch_size=1, N=N, dt=DT,
              params=SolverParams(max_sqp_iters=5, rho=1e-3), plant_type="indy7")
x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(solver.nv))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)

print(f"{'linsys':>11} {'start':>5} {'solve ms':>9} {'merit':>10}   pcg iters per SQP iter "
      "(bdsv reports 1 = one direct solve)")
for mode in ("pcg", "bdsv", "bdsv_first"):
    solver.set_linsys(mode)
    solver.reset()                                   # duals + adapted rho back to the defaults
    cold = solver.solve(x0[None], ref[None])         # xu_warm=None: hold-at-x0 seed
    warm = solver.solve(x0[None], ref[None], xu_warm=cold.xu)   # re-solve from the solution
    for tag, r in (("cold", cold), ("warm", warm)):
        print(f"{r.stats.linsys:>11} {tag:>5} {r.solve_time_us / 1000:9.3f} "
              f"{r.stats.final_merit[0]:10.4f}   {r.stats.pcg_iters[:, 0].tolist()}")

print("\nnext: tools/autotune_linsys.py --plant indy7 --N 64 --task-tag fig8   (quiet box)")
print("      then MPCController(solver, task_tag='fig8') picks the tuned policy")
