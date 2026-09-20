"""Intro example 1: a single GATO trajectory-optimization solve.

Constructs one BSQP solver (batch_size=1) for the Indy7 and solves a single
figure-8-tracking QP, then prints the solver stats. This is the smallest possible
use of the core solver object.

Run from the repo root (needs the bsqpN64_indy7 module built — see README):
    python examples/01_single_solve.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from gato.interface import BSQP
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

URDF = os.path.join(os.path.dirname(__file__), "indy7_description", "indy7.urdf")
N, DT = 64, 0.01

# build the solver (one problem instance)
solver = BSQP(model_path=URDF, batch_size=1, N=N, dt=DT, plant_type="indy7")

nx, nu = solver.nx, solver.nu

# initial state + a figure-8 EE reference over the horizon
x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(nx - len(INDY7_START_CONFIGS["ready"])))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)

# batched arrays even for batch_size=1: shape (1, ...)
x0_B = x0.reshape(1, -1)
ref_B = ref.reshape(1, -1)
res = solver.solve(x0_B, ref_B)   # cold start (hold at x0); pass res.xu to warm-start the next call

print(f"GATO single solve (Indy7, N={N}):")
print(f"  GPU solve time : {res.solve_time_us / 1000.0:.3f} ms")
print(f"  SQP iters      : {res.stats.sqp_iters}")
print(f"  final merit    : {res.stats.final_merit}")
print(f"  first control u0: {res.u0()}")
