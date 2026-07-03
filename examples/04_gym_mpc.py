"""Intro example 4: MPC as a gymnasium policy, with a hypothesis batch.

Tracks a figure-8 with an UNMODELED constant -20 N force pulling down on the
end-effector. The environment (gato.envs.ArmTrackEnv) owns the simulation and
applies the disturbance; the solver does NOT know about it.

Two policies run the same task:
  * B=1  — plain MPC (believes f_ext = 0)
  * B=16 — hypothesis-batch MPC: each batch entry solves under a different
           guessed wrench, reality picks the winner each tick, and a force
           estimator refines the guesses (the paper's Fig-5/CS3 mechanism).

This demo exercises the full MPCPolicy / ArmTrackEnv / HypothesisBatch API.
NOTE: how much the batch actually helps is an open tuning question — the
force-estimator convergence (hypothesis frame conventions + radius schedule)
is tracked R&D (see docs/open-tasks); with the current estimator settings the
two policies track comparably in this configuration.

Run from the repo root (needs the bsqpN64_indy7 module built):
    python examples/04_gym_mpc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from gato import BSQP, MPCController, MPCPolicy, ForceEstimator, ForceHypothesisBatch
from gato.envs import ArmTrackEnv
from gato.policy import TrajectoryReference
from gato.common import figure8, _require_pin
from gato.config import DEFAULT_SOLVER_PARAMS, FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

URDF = os.path.join(os.path.dirname(__file__), "indy7_description", "indy7.urdf")
N, DT = 64, 0.01
F_EXT = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0])  # unmodeled EE force (world frame)
SIM_S = 3.0


def make_env(reference):
    x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(6)))
    return ArmTrackEnv(URDF, ctrl_dt=DT, sim_dt=1e-3, f_ext_world=F_EXT,
                       reference=reference, x0=x0, horizon_s=SIM_S)


def rollout(batch_size, with_hypotheses):
    sp = DEFAULT_SOLVER_PARAMS
    solver = BSQP(model_path=URDF, batch_size=batch_size, N=N, dt=DT, plant_type="indy7",
                  max_sqp_iters=sp["max_sqp_iters"], kkt_tol=sp["kkt_tol"],
                  max_pcg_iters=sp["max_pcg_iters"], pcg_tol=sp["pcg_tol"],
                  solve_ratio=sp["solve_ratio"], mu=sp["mu"],
                  q_cost=sp["q_cost"], qd_cost=sp["qd_cost"], u_cost=sp["u_cost"],
                  N_cost=sp["N_cost"], q_lim_cost=sp["q_lim_cost"],
                  vel_lim_cost=sp["vel_lim_cost"], ctrl_lim_cost=sp["ctrl_lim_cost"],
                  rho=sp["rho"])

    hypotheses = None
    if with_hypotheses:
        pin = _require_pin()
        model = pin.buildModelFromUrdf(URDF)
        estimator = ForceEstimator(batch_size=batch_size, initial_radius=5.0,
                                   min_radius=2.0, max_radius=40.0,
                                   smoothing_factor=0.5, seed=0, alpha=0.6, beta=0.5)
        hypotheses = ForceHypothesisBatch(estimator, model, ee_frame="EE")

    reference = TrajectoryReference(figure8(DT, **FIG8_DEFAULT_PARAMS), DT, N)
    env = make_env(reference)
    policy = MPCPolicy(MPCController(solver, hypotheses=hypotheses),
                       reference, dt_step=DT)

    obs, info = env.reset()
    policy.reset(obs)
    errs = []
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        errs.append(info["tracking_err"])
        if terminated or truncated:
            break
    assert not terminated, "state went non-finite"
    return float(np.mean(errs)), float(np.max(errs)), len(errs)


if __name__ == "__main__":
    print(f"Figure-8 tracking with an unmodeled {abs(F_EXT[2]):.0f} N EE force "
          f"(Indy7, N={N}, {SIM_S:.0f}s):")
    e1, m1, n1 = rollout(1, with_hypotheses=False)
    print(f"  B= 1 (no hypotheses)   : mean err {e1*100:6.2f} cm   max {m1*100:6.2f} cm   ({n1} steps)")
    e16, m16, n16 = rollout(16, with_hypotheses=True)
    print(f"  B=16 (force hypotheses): mean err {e16*100:6.2f} cm   max {m16*100:6.2f} cm   ({n16} steps)")
    print(f"  mean-error ratio B1/B16: {e1/e16:4.2f}x")
