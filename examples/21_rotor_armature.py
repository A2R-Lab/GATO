"""Intro example 21: reflected rotor inertia (armature) on the TIAGo Pro right arm.

A geared joint carries its rotor's inertia multiplied by the square of the gear
ratio. URDF has no field for it, and on a light arm it can exceed the link
inertia: on this arm it is 0.17 to 0.43 kg m^2 per joint against a wrist link
inertia of a few 1e-3 kg m^2. gato.register_robot(..., joint_armature=...) bakes
the per-joint values into the dynamics and their derivatives, added to the
diagonal of the joint-space inertia as Pinocchio's model.armature.

The model is the reduced right arm Felix Muehlenberend derived from PAL's TIAGo
Pro description, with the armature he measured on the hardware (issue #20);
examples/tiago_pro_description/README.md records the source and the licence.
The setting is his: friction is compensated by an external controller, so the
optimizer plans with a loss-free model (joint_dynamics stays off and the URDF's
<dynamics> tags are not read) and the planned torque is the feedforward.

The example

  1. reads the measured values by joint name and checks the names against the
     solver's velocity order before applying them,
  2. registers the arm twice, with and without the armature,
  3. checks one device step of the rotor module against Pinocchio,
  4. runs the same reach in closed loop on ONE simulated plant, the arm with
     its rotors, once planning with them and once without. Only the planning
     model changes. It prints the tracking error, the plan's dynamics defect,
     the excursions past the URDF limits, and how the planned inertial torque
     splits between rotor and links.

One simulated reach, with the planned torque applied unclipped: an
illustration of what the planning model changes, not a hardware validation and
not a feasibility claim (both plans exceed the effort limit, and the example
prints by how much). Fixed simulation pacing: no timing is reported.
Needs pinocchio; the first run compiles two modules into the per-host cache.

    python examples/21_rotor_armature.py
    python examples/21_rotor_armature.py --urdf arm.urdf --params values.csv --ee-frame EE

Your own arm: the CSV needs the columns `joint` and `armature_kg_m2`, one row
per joint; the URDF needs collision geometry (the collision rows are part of
every module, so register_robot refuses a model without any).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pinocchio as pin

import gato
from gato import BSQP, MPCController, SolverParams
from gato.common import rk4

HERE = Path(__file__).resolve().parent / "tiago_pro_description"

N, DT = 32, 0.03                 # 30 ms knots, the reach setting of issue #20
SIM_DT, SIM_S = 0.001, 3.0
Q_START = np.array([-1.0, -0.6, 0.0, -1.2, 0.5, 0.5, 0.0])
Q_GOAL = np.array([-1.6, -1.0, 0.4, -0.6, 1.0, 0.2, 0.0])


def measured_armature(model, params):
    """The CSV's armature column as a vector in the model's velocity order.
    Every joint must be named once; an unknown or missing name is an error."""
    with open(params, newline="") as f:
        pairs = [(r["joint"], float(r["armature_kg_m2"])) for r in csv.DictReader(f)]
    rows = dict(pairs)
    if len(rows) != len(pairs):
        twice = sorted({n for n, _ in pairs if [m for m, _ in pairs].count(n) > 1})
        raise ValueError(f"the CSV names {twice} more than once")
    names = [model.names[j] for j in range(1, model.njoints)]
    idx_v = [model.joints[j].idx_v for j in range(1, model.njoints)]
    if idx_v != list(range(model.nv)):
        raise ValueError("this example expects one velocity slot per joint, in joint order")
    if sorted(names) != sorted(rows):
        raise ValueError(f"the CSV names {sorted(rows)} are not the model's joints {sorted(names)}")
    return names, np.array([rows[n] for n in names])


def closed_loop(solver, plant, x0, goal):
    """MPC on `solver`'s model, simulated on `plant` (RK4, zero-order hold)."""
    data = plant.createData()
    ctrl = MPCController(solver)
    ctrl.reset(x0)
    q, dq = x0[:plant.nq].copy(), x0[plant.nq:].copy()
    err, defect, tau_rotor, tau_link = [], [], [], []
    over = np.zeros(3)               # worst excursion past the position, velocity and effort limits
    for _ in range(int(round(SIM_S / DT))):
        step = ctrl.step(np.r_[q, dq], goal)
        # the acceleration the plan asks for on its first interval, and the
        # inertial torque it implies on each side of the split
        xu = np.asarray(step.xu_best, dtype=np.float64)
        nx, nu = 2 * plant.nv, plant.nv
        qdd = (xu[nx + nu + plant.nq:2 * nx + nu] - xu[plant.nq:nx]) / DT
        link_inertia = pin.crba(solver.model, solver.data, q) - np.diag(solver.model.armature)
        link_inertia = np.triu(link_inertia) + np.triu(link_inertia, 1).T
        tau_rotor.append(np.abs(plant.armature * qdd))
        tau_link.append(np.abs(link_inertia @ qdd))
        defect.append(step.defect)
        over[2] = max(over[2], np.max(np.abs(step.u) - plant.effortLimit))
        for _ in range(int(round(DT / SIM_DT))):
            q, dq = rk4(plant, data, q, dq, step.u, SIM_DT)
            over[0] = max(over[0], np.max(q - plant.upperPositionLimit), np.max(plant.lowerPositionLimit - q))
            over[1] = max(over[1], np.max(np.abs(dq) - plant.velocityLimit))
        err.append(np.linalg.norm(solver.ee_pos(q) - goal[:3]))
    return np.array(err), np.array(defect), np.maximum(over, 0.0), np.max(tau_rotor, axis=0), np.max(tau_link, axis=0)


def run(urdf=HERE / "tiago_pro_right_arm.urdf", params=HERE / "tiago_pro_right_arm_parameters.csv",
        ee_frame="arm_right_grasp_joint"):
    """The whole example. Returns the printed numbers: the device-step
    difference and, per planning model, the error trace, the defect trace and
    the limit excursions."""
    URDF = str(urdf)
    names, armature = measured_armature(pin.buildModelFromUrdf(URDF), params)
    print("joint, armature [kg m^2] (velocity order):")
    for n, a in zip(names, armature):
        print(f"  {n:<18} {a:.2f}")

    stem = "".join(c if c.isalnum() else "_" for c in Path(URDF).stem)
    common = dict(N=[N], ee_frame=ee_frame)
    gato.register_robot(f"{stem}_rotor", URDF, joint_armature=armature, **common)
    gato.register_robot(f"{stem}_rigid", URDF, **common)
    params = SolverParams(max_sqp_iters=5, defect_telemetry=True)
    rotor = BSQP(URDF, batch_size=1, N=N, dt=DT, params=params, plant_type=f"{stem}_rotor")
    rigid = BSQP(URDF, batch_size=1, N=N, dt=DT, params=params, plant_type=f"{stem}_rigid")
    assert np.allclose(rotor.model.armature, armature, rtol=1e-6) and not rigid.model.armature.any()

    # one device step against Pinocchio's forward dynamics with model.armature
    plant = rotor.model          # carries the baked armature: the simulated arm for both planners
    nv = plant.nv
    if Path(URDF).parent == HERE:
        q_start, q_goal = Q_START, Q_GOAL
    else:                        # another arm: two postures inside its position limits
        lo, hi = plant.lowerPositionLimit, plant.upperPositionLimit
        q_start, q_goal = lo + 0.4 * (hi - lo), lo + 0.6 * (hi - lo)
    rng = np.random.default_rng(0)
    x = np.r_[q_start, rng.uniform(-.3, .3, nv)]
    u = rng.uniform(-.1, .1, nv) * plant.effortLimit
    got = np.asarray(rotor.sim_forward(x[None], u[None], DT))[0]
    qdd = pin.aba(plant, rotor.data, x[:nv], x[nv:], u)
    want = np.r_[x[:nv] + DT * x[nv:] + .5 * DT**2 * qdd, x[nv:] + DT * qdd]
    out = {"device_step_diff": float(np.abs(got - want).max())}
    print(f"\ndevice step vs Pinocchio with armature: max abs difference {out['device_step_diff']:.2e}")
    assert np.allclose(got, want, atol=2e-4, rtol=1e-4)

    # the same reach, twice, on the plant that has the rotors
    x0 = np.r_[q_start, np.zeros(nv)]
    goal = np.zeros((N, 6), dtype=np.float32)
    goal[:, :3] = rotor.ee_pos(q_goal)
    goal = goal.ravel()
    print(f"\nreach of {np.linalg.norm(rotor.ee_pos(q_start) - goal[:3]):.2f} m, plant with rotors, "
          f"{SIM_S:.0f} s, planned torque applied as is:")
    for label, solver in (("with armature", rotor), ("without", rigid)):
        err, defect, over, t_rotor, t_link = closed_loop(solver, plant, x0, goal)
        out[label] = dict(err=err, defect=defect, over=over, tau_rotor=t_rotor, tau_link=t_link)
        print(f"  plan {label:<14} final error {err[-1] * 1e3:7.1f} mm, "
              f"mean over the last second {err[-int(1 / DT):].mean() * 1e3:7.1f} mm")
        print(f"    plan dynamics defect: max {defect.max():.2e}; excursion past the limits: "
              f"position {over[0]:.3f} rad, velocity {over[1]:.3f} rad/s, effort {over[2]:.2f} N m")
        if solver is rotor:
            with np.printoptions(precision=2, suppress=True):
                print(f"    largest planned inertial torque per joint [N m]: rotor {t_rotor}")
                print(f"    {'':>48} links {t_link}")
    return out


def main():
    p = argparse.ArgumentParser(description="rotor armature on the TIAGo Pro right arm (or your own arm).")
    p.add_argument("--urdf", default=str(HERE / "tiago_pro_right_arm.urdf"))
    p.add_argument("--params", default=str(HERE / "tiago_pro_right_arm_parameters.csv"),
                   help="CSV with the columns joint and armature_kg_m2")
    p.add_argument("--ee-frame", default="arm_right_grasp_joint", help="URDF fixed joint of the EE target frame")
    args = p.parse_args()
    run(args.urdf, args.params, args.ee_frame)


if __name__ == "__main__":
    main()
