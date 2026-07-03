"""Intro example 3: a closed-loop MPC tracking loop.

Runs the MPC_GATO controller tracking a figure-8 end-effector trajectory on the
Indy7 for a few seconds, then prints the average tracking error and per-step solve
time. This is the high-level wrapper used by the paper's Fig-3/Fig-5 experiments
(see examples/paper-figures/).

Run from the repo root (needs the bsqpN64_indy7 module built):
    python examples/03_mpc_loop.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pinocchio as pin
from gato.mpc_gato import MPC_GATO
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

URDF = os.path.join(os.path.dirname(__file__), "indy7_description", "indy7.urdf")
# M=1 here for a clean, deterministic tracking demo. (Batched solves are shown in
# 02_batched_solve.py; batched MPC with online force estimation is in
# examples/paper-figures/reproduce_fig5_disturbance.py, where the batch hypothesizes
# an actual disturbance — running the estimator without a disturbance just adds noise.)
N, DT, M = 64, 0.01, 1

model, _, _ = pin.buildModelsFromUrdf(URDF, os.path.dirname(URDF) + "/")
mpc = MPC_GATO(model, model_path=URDF, N=N, dt=DT, batch_size=M, plant_type="indy7",
               track_full_stats=False)

fig8 = figure8(DT, **FIG8_DEFAULT_PARAMS)
x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(model.nv)))

# pace_by_solve_time=True is the real-time MPC default (sim advances by the measured
# solve time, so a faster solver re-plans more often) — matches the paper benchmark.
_, stats = mpc.run_mpc_fig8(x0, fig8, sim_dt=0.001, sim_time=3.0, pace_by_solve_time=True)

err = np.asarray(stats["goal_distances_knot0"])
solve_ms = np.asarray(stats["solve_times"])
print(f"GATO MPC figure-8 (Indy7, N={N}, M={M}, 3 s):")
print(f"  control steps    : {len(err)}")
print(f"  mean tracking err: {err.mean():.4f} m")
print(f"  mean solve time  : {solve_ms.mean():.3f} ms/step")
