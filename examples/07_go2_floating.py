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

Default mode (no contact model): a contactless free-flyer is WEIGHTLESS in
its own frame, so on MuJoCo it settles into a crouch rather than holding the
standing height; on the device integrator (no ground) it simply falls. This
mode shows the plumbing.

--fc mode (the fc-on-feet variant, bsqpN16_go2_fc): one world-aligned wrench
[n; f] per foot frame is appended to every control. The S1 standing recipe:
fc_ref = mg/4 up on every foot, moment rows pinned (add_fc_box on
fc_slots(i, "n"), AL mechanism), posture anchor + imu goal at the STANCE pose
(the keyframe's base z = 0.35 has the feet 8.5 cm in the air — the feet touch
at z ≈ 0.287, derived from FK below). Holds the stance height with each foot
carrying mg/4; the solver's knot-0 wrench (StepResult.fc) is its contact
EXPLANATION and agrees with the world's normal forces.

Needs the bsqpN16_go2[_fc] module and a python with pinocchio (+ mujoco for the
contact world); runs from any cwd:
    python examples/07_go2_floating.py [--steps 100] [--fc]
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
FOOT_RADIUS = 0.022   # go2.urdf *_foot collision sphere


def stance_pose(solver, q_keyframe):
    """Lower the keyframe until the foot spheres rest on the plane (z = 0)."""
    q = np.array(q_keyframe, dtype=np.float64)
    foot_z = np.mean([solver.ee_pos(q, frame=f)[2] for f in solver.contact_frames])
    q[2] -= foot_z - FOOT_RADIUS
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100, help="MPC ticks of DT (10 ms)")
    p.add_argument("--fc", action="store_true", help="use the fc-on-feet variant (contact forces in the model)")
    args = p.parse_args()

    solver = BSQP(URDF, batch_size=1, N=N, dt=DT, plant_type="go2", variant="fc" if args.fc else None,
                  params=SolverParams(q_cost=5.0 if args.fc else 1.0, qd_cost=1e-1 if args.fc else 1e-2,
                                      u_cost=1e-4, N_cost=25.0 if args.fc else 5.0,
                                      q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0))
    nq, nv = solver.nq, solver.nv
    q0 = np.asarray(GO2_START_CONFIGS["standing"], dtype=np.float64)   # [p; quat xyzw; 12 joints]
    if args.fc:
        q0 = stance_pose(solver, q0)   # feet ON the ground, not the in-the-air keyframe
    x0 = np.concatenate([q0, np.zeros(nv)]).astype(np.float32)

    # posture anchor + the imu goal pinned at the (stance) pose
    solver.set_q_nom(q0)
    solver.set_q_pos_cost(50.0)
    goal = np.zeros(6 * N, dtype=np.float32)
    goal.reshape(N, 6)[:, :3] = solver.ee_pos(q0)   # imu position (solver.ee_frame == "imu_joint")

    if args.fc:
        # the S1 standing setpoint: mg/4 up on every foot, moment rows pinned
        mg = sum(i.mass for i in solver.model.inertias) * 9.81
        ref = np.zeros(solver.n_fc, dtype=np.float32)
        for i, frame in enumerate(solver.contact_frames):
            ref[solver.fc_slots(i, "f")[2]] = mg / 4
            solver.add_fc_box(0.0, 0.0, slots=solver.fc_slots(i, "n"), mech="al")
        solver.set_fc_ref(ref)                # fc_cost keeps its fc-build default (1e-2)
        print(f"fc-on-feet: frames {solver.contact_frames}, fc_ref fz = mg/4 = {mg / 4:.1f} N")

    # reseed_threshold: a stale warm-start tail is a merit local minimum on the
    # contactless model; on the fc model the one-step pred_err is ~0.5 of
    # joint-velocity jitter from the soft contact, so the default (off) is used
    ctrl = MPCController(solver, reseed_threshold=None if args.fc else 0.5)   # linsys -> floating default (bdsv)
    ctrl.reset(x0)
    print(f"go2 floating-base MPC: nq={nq} nv={nv} nu={solver.nu} N={N}  linsys={ctrl.linsys}")

    world = None
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
    q[2] += 0.01   # start slightly above: a small drop onto the feet
    for k in range(args.steps):
        r = ctrl.step(np.concatenate([q, dq]).astype(np.float32), goal)
        u = np.clip(np.asarray(r.u, np.float64), -23.7, 23.7)   # go2 effort limit
        for _ in range(int(round(DT / SIM_DT))):
            q, dq = step(q, dq, u, SIM_DT)
        if k % 20 == 0 or k == args.steps - 1:
            fc_note = f"  fz(solver)={np.round(r.fc[5::6], 1)}" if args.fc else ""
            fn_note = f"  fn(world)={world.last_contact['fn']:.1f}" if args.fc and world is not None else ""
            print(f"  t={k * DT:5.2f}s  base z={q[2]:.3f}  |quat|={np.linalg.norm(q[3:7]):.4f}  "
                  f"pred_err={r.pred_err:.3f}{'  reseeded' if r.reseeded else ''}  "
                  f"solve {r.solve.solve_time_us / 1000:.3f} ms{fc_note}{fn_note}")
    print(f"finite state: {np.isfinite(q).all() and np.isfinite(dq).all()}  final base z={q[2]:.3f}")


if __name__ == "__main__":
    main()
