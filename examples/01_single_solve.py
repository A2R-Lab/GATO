"""Intro example 1: a single GATO trajectory-optimization solve.

Constructs one BSQP solver (batch_size=1) for the Indy7 and solves a single
figure-8-tracking problem, then prints the solver stats. This is the smallest
possible use of the core solver object.

Needs the bsqpN64_indy7 module built (see README); runs from any cwd:
    python examples/01_single_solve.py
"""
from pathlib import Path

import numpy as np

import gato
from gato import BSQP
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / gato.robot_info("indy7")["urdf"]   # the registry holds repo-relative URDF paths
N, DT = 64, 0.01

# build the solver (one problem instance; the defaults are gato.SolverParams())
solver = BSQP(URDF, batch_size=1, N=N, dt=DT, plant_type="indy7")

# initial state [q; qd] + a figure-8 EE reference over the horizon
x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(solver.nv))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)

# batched arrays even for batch_size=1: shape (1, ...)
res = solver.solve(x0.reshape(1, -1), ref.reshape(1, -1))   # cold start (hold at x0)
res = solver.solve(x0.reshape(1, -1), ref.reshape(1, -1), xu_warm=res.xu)   # warm-started re-solve

print(f"GATO single solve (Indy7, N={N}):")
print(f"  GPU solve time : {res.solve_time_us / 1000.0:.3f} ms")
print(f"  SQP iters      : {res.stats.sqp_iters}")
print(f"  final merit    : {res.stats.final_merit}")
print(f"  first control u0: {res.u0()}")
