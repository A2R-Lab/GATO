"""Intro example 7: a floating-base robot (Unitree go2) in closed loop.

Floating-base modules (gato.build(..., floating_base=True); the vendored go2 is
N16-only) store the state as [p; quat xyzw; joints; v], and the EE target frame
is the trunk imu — the "EE" cost pins the base pose. The recipe (from the w36
linsys sweep / the floating gates):
  * standing keyframe from gato.config.GO2_START_CONFIGS as the posture anchor
    (set_q_nom + set_q_pos_cost — the tracking cost alone leaves the legs free),
  * imu goal = the imu position at the standing pose (solver.ee_pos),
  * MPCController with the wired floating defaults (linsys "bdsv": the direct
    solve wins at every batch size on the 36-state Schur blocks) plus a
    reseed_threshold — a stale warm-start tail is a merit local minimum here.
The world is MuJoCo (gato.worlds.MuJoCoWorld, floating=True, a ground box under
the feet) when mujoco is installed, else the solver's own device integrator.

NOTE: the solver has no contact model, and a contactless free-flyer is
WEIGHTLESS in its own frame, so on MuJoCo it settles into a crouch rather than
holding the standing height (the fc-on-feet wave adds contact forces); on the
device integrator (no ground) it simply falls. This example shows the plumbing.

Needs the bsqpN16_go2 module and a python with pinocchio (+ mujoco for the
contact world); runs from any cwd:
    python examples/07_go2_floating.py [--steps 100]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

import gato
from gato import BSQP, MPCController, SolverParams
from gato.config import GO2_START_CONFIGS

REPO = Path(__file__).resolve().parents[1]
URDF = str(REPO / gato.robot_info("go2")["urdf"])
N, DT, SIM_DT = 16, 0.01, 1e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100, help="MPC ticks of DT (10 ms)")
    args = p.parse_args()

    solver = BSQP(URDF, batch_size=1, N=N, dt=DT, plant_type="go2",
                  params=SolverParams(q_cost=1.0, qd_cost=1e-2, u_cost=1e-4, N_cost=5.0,
                                      q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0))
    nq, nv = solver.nq, solver.nv
    q0 = np.asarray(GO2_START_CONFIGS["standing"], dtype=np.float64)   # [p; quat xyzw; 12 joints]
    x0 = np.concatenate([q0, np.zeros(nv)]).astype(np.float32)

    # posture anchor + the imu goal pinned at the standing pose
    solver.set_q_nom(q0)
    solver.set_q_pos_cost(50.0)
    goal = np.zeros(6 * N, dtype=np.float32)
    goal.reshape(N, 6)[:, :3] = solver.ee_pos(q0)   # imu position (solver.ee_frame == "imu_joint")

    ctrl = MPCController(solver, reseed_threshold=0.5)   # linsys resolves to the floating default (bdsv)
    ctrl.reset(x0)
    print(f"go2 floating-base MPC: nq={nq} nv={nv} nu={solver.nu} N={N}  linsys={ctrl.linsys}")

    if importlib.util.find_spec("mujoco") is not None:
        from gato.worlds import MuJoCoWorld
        world = MuJoCoWorld(URDF, floating=True, timestep=SIM_DT,
                            plane={"z": 0.0, "pos_xy": (0.0, 0.0), "size_xy": (1.0, 1.0)})
        step = world.step
        print("world: MuJoCo (ground box under the feet)")
    else:
        # device integrator: (B, nx) x (B, nu) -> (B, nx); no ground, so the robot falls
        def step(q, dq, u, dt):
            x1 = solver.sim_forward(np.concatenate([q, dq])[None], u[None], dt)[0]
            return x1[:nq].astype(np.float64), x1[nq:].astype(np.float64)
        print("world: device integrator (mujoco not installed — no contact)")

    q, dq = q0.copy(), np.zeros(nv)
    q[2] = 0.36   # start slightly above the stance height: a small drop onto the feet
    for k in range(args.steps):
        r = ctrl.step(np.concatenate([q, dq]).astype(np.float32), goal)
        u = np.clip(np.asarray(r.u, np.float64), -23.7, 23.7)   # go2 effort limit
        for _ in range(int(round(DT / SIM_DT))):
            q, dq = step(q, dq, u, SIM_DT)
        if k % 20 == 0 or k == args.steps - 1:
            print(f"  t={k * DT:5.2f}s  base z={q[2]:.3f}  |quat|={np.linalg.norm(q[3:7]):.4f}  "
                  f"pred_err={r.pred_err:.3f}{'  reseeded' if r.reseeded else ''}  "
                  f"solve {r.solve.solve_time_us / 1000:.3f} ms")
    print(f"finite state: {np.isfinite(q).all() and np.isfinite(dq).all()}")


if __name__ == "__main__":
    main()
