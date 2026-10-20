"""Intro example 14: GATO as the controller inside a MuJoCo loop.

MuJoCo is the plant, GATO the controller, and the loop is yours: the pattern
for dropping the solver into an existing MuJoCo simulation. The steps are

  1. build the MuJoCo model from the SAME URDF the solver's dynamics were
     generated from (MjSpec; the collision meshes ship with the description);
  2. check that MuJoCo and the solver agree on the robot with the dynamics
     fingerprint (gato.fingerprint.check on MuJoCo's forward dynamics, the
     consumer contract's first step: a mismatch here means the controller
     compensates the wrong gravity or inertia);
  3. construct MPCController as it comes (limit rows on under the ADMM default)
     with tracking weights, and per control period call step(x, ref,
     budget_s=...), write the torques into data.qfrc_applied (data.ctrl when
     the model has motor actuators) and mj_step the substeps.

The task is a figure-8 sized from the URDF velocity limits (gato.tasks, speed
ratio 0.9, the arm can follow). Reported: the fingerprint verdict, the mean and
peak end-effector error, the simulated peak joint velocity and torque against
the URDF limits, the SQP iterations run and the share of ticks the budget cut.
Fixed simulation pacing (the sim advances 10 ms per tick whatever the wall
clock); the per-tick wall time is a diagnostic, not a benchmark.

Needs the bsqpN64_iiwa14 module and a python with mujoco and pinocchio:
    python examples/14_mujoco_closed_loop.py [--periods 2] [--budget-ms 5.0]
    python examples/14_mujoco_closed_loop.py --correctness-only --periods 1

Correctness-only mode disables the wall-clock budget and duration reporting;
the configured SQP iteration count is independent of shared-box load.
"""
import argparse
import os
import re

import mujoco
import numpy as np
import pinocchio as pin

import gato
from gato import BSQP, MPCController, SolverParams, tasks
from gato import fingerprint as fp

PLANT, N, DT, SIM_DT = "iiwa14", 64, 0.01, 0.001
TRACKING = SolverParams(max_sqp_iters=3, q_cost=10.0, qd_cost=1e-3)   # the tracking knobs (example 13)


def mujoco_model(urdf):
    """MjModel of the vendored URDF: meshes resolved next to the URDF, no contacts
    (free-space task; worlds.MuJoCoWorld adds a table and contact masks)."""
    xml = open(urdf).read()
    if "package://" in xml:                       # visual meshes we do not ship
        xml = re.sub(r"<visual\b.*?</visual>", "", xml, flags=re.DOTALL)
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = os.path.dirname(os.path.abspath(urdf))
    model = spec.compile()
    model.opt.timestep = SIM_DT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--periods", type=int, default=2)
    ap.add_argument("--budget-ms", type=float, default=5.0, help="step budget per control period")
    ap.add_argument("--correctness-only", action="store_true",
                    help="disable wall-clock budget and all duration reporting")
    args = ap.parse_args()
    if args.periods < 1 or (not args.correctness_only and args.budget_ms <= 0):
        ap.error("periods and budget-ms must be positive")
    budget_s = None if args.correctness_only else args.budget_ms * 1e-3
    urdf = str(gato.robot_urdf(PLANT))
    model = mujoco_model(urdf)
    data = mujoco.MjData(model)
    nq, nv = model.nq, model.nv

    # 2. the same robot? MuJoCo's forward dynamics at the fingerprint probes
    def mj_qdd(q, qd, u):
        data.qpos[:], data.qvel[:], data.qfrc_applied[:] = q, qd, u
        mujoco.mj_forward(model, data)
        return data.qacc.copy()
    res = fp.check(mj_qdd, PLANT)
    print(fp.report(res).splitlines()[0])
    if not res["ok"]:
        raise SystemExit("MuJoCo and the solver disagree on the robot; stop here")

    # the task: a figure-8 the arm can follow, from the URDF velocity limits
    pin_model = pin.buildModelFromUrdf(urdf)
    sc = tasks.fig8_scenarios(pin_model, PLANT, 1, seed=3, ratio_range=(0.9, 0.9))[0]
    traj, sim_time = tasks.fig8_reference(sc, N, periods=args.periods)
    lim = tasks.urdf_limits(pin_model)
    ee_id = pin_model.getFrameId(tasks.ee_frame_of(PLANT))
    pin_data = pin_model.createData()

    # 3. the controller as it comes, with tracking weights
    solver = BSQP(urdf, batch_size=1, N=N, dt=DT, params=TRACKING, plant_type=PLANT)
    ctrl = MPCController(solver)
    print(f"controller: limits={ctrl.limits!r} linsys={ctrl.linsys} admm_linsys={ctrl.admm_linsys} "
          f"N={N} dt={DT} budget_s={budget_s}")
    data.qpos[:] = sc["q0"]
    data.qvel[:] = 0.0
    x = np.concatenate([data.qpos, data.qvel]).astype(np.float32)
    ctrl.reset(x)
    ctrl.warmup(x, traj[: 6 * N])

    substeps = int(round(DT / SIM_DT))
    ticks = int(sim_time / DT)
    err, qd_peak, u_peak, its, hits, wall = [], [], [], [], 0, []
    for k in range(ticks):
        ref = traj[6 * k: 6 * (k + N)]
        if ref.shape[0] < 6 * N:
            break
        x = np.concatenate([data.qpos, data.qvel]).astype(np.float32)
        r = ctrl.step(x, ref, budget_s=budget_s)
        u = np.asarray(r.u, dtype=np.float64)
        its.append(r.iterations)
        hits += r.budget_hit
        if not args.correctness_only:
            wall.append(r.wall_s)
        for _ in range(substeps):
            data.ctrl[:] = 0.0
            data.qfrc_applied[:] = u            # joint torques, zero-order hold over the control period
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"nonfinite simulated state at tick {k}")
        pin.forwardKinematics(pin_model, pin_data, data.qpos.copy())
        pin.updateFramePlacements(pin_model, pin_data)
        err.append(np.linalg.norm(pin_data.oMf[ee_id].translation - ref[:3]))
        qd_peak.append(np.max(np.abs(data.qvel) / lim["qd"]))
        u_peak.append(np.max(np.abs(u) / lim["u"]))
    err, wall = np.asarray(err), np.asarray(wall) * 1e3
    print(f"{len(err)} ticks, {args.periods} figure-8 periods ({sim_time:.2f} s simulated), speed ratio {sc['ratio_est']:.2f}")
    print(f"  end-effector error   mean {err.mean():.4f} m   max {err.max():.4f} m")
    print(f"  simulated peak |qd| / limit {max(qd_peak):.2f}   peak |u| / limit {max(u_peak):.2f}")
    print(f"  SQP iterations per tick {np.mean(its):.2f} (max {max(its)}); budget cut {100 * hits / len(its):.0f}% of ticks")
    if not args.correctness_only:
        print(f"  step wall time per tick: p50 {np.percentile(wall, 50):.2f} ms  max {wall.max():.2f} ms  (diagnostic, shared machine)")
    print("\nThe plant is MuJoCo's integrator from the applied torques; the planner never grades itself. The same loop "
          "takes a CUDA state\nand reference (example 10) when the simulator lives on the GPU.")


if __name__ == "__main__":
    main()
