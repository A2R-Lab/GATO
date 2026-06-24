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
from bsqp.interface import BSQP
from bsqp.common import figure8
from bsqp.config import DEFAULT_SOLVER_PARAMS, FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

URDF = os.path.join(os.path.dirname(__file__), "indy7_description", "indy7.urdf")
N, DT = 64, 0.01

# build the solver (one problem instance)
sp = DEFAULT_SOLVER_PARAMS
solver = BSQP(model_path=URDF, batch_size=1, N=N, dt=DT,
              max_sqp_iters=sp["max_sqp_iters"], kkt_tol=sp["kkt_tol"],
              max_pcg_iters=sp["max_pcg_iters"], pcg_tol=sp["pcg_tol"],
              solve_ratio=sp["solve_ratio"], mu=sp["mu"],
              q_cost=sp["q_cost"], qd_cost=sp["qd_cost"], u_cost=sp["u_cost"],
              N_cost=sp["N_cost"], q_lim_cost=sp["q_lim_cost"],
              vel_lim_cost=sp["vel_lim_cost"], ctrl_lim_cost=sp["ctrl_lim_cost"],
              rho=sp["rho"], plant_type="indy7")

nx, nu = solver.nx, solver.nu

# initial state + a figure-8 EE reference over the horizon
x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(nx - len(INDY7_START_CONFIGS["ready"])))).astype(np.float32)
ref = figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32)

# batched arrays even for batch_size=1: shape (1, ...)
x0_B = x0.reshape(1, -1)
ref_B = ref.reshape(1, -1)
XU_B = np.zeros((1, N * (nx + nu) - nu), dtype=np.float32)
XU_B[:, :nx] = x0_B

XU_B, solve_us = solver.solve(x0_B, ref_B, XU_B)
stats = solver.get_stats()

print(f"GATO single solve (Indy7, N={N}):")
print(f"  GPU solve time : {solve_us / 1000.0:.3f} ms")
print(f"  SQP iters      : {stats.get('sqp_iters')}")
print(f"  final merit    : {stats.get('final_merit')}")
print(f"  first control u0: {XU_B[0, nx:nx + nu]}")
