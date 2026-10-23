"""Intro example 19: a standing reference for the Go2 through the base-pose weights.

On a floating-base plant the tracking objective's posture anchor carries the
base pose: the first six entries of the per-slot posture weights
(``set_q_pos_cost`` with an array) weigh the base position and orientation in the
tangent space of the log-map state cost, and the first seven entries of
``set_q_nom`` are the stored base pose (position, quaternion xyzw) they pull
toward. With them the imu target, which fixes one point of the body, becomes
a full standing reference. The cost menu itself is fixed-base in grid-rbd
0.5.4; the floating composition is the next GRiD batch.

The model is the fc-on-feet variant (contact wrenches in the control, the
standing setpoint of example 07: mg/4 up on every foot, the moment rows
pinned), since a contactless free-flyer cannot hold any height. The solve is
open loop from a base lowered by 3 cm and rolled by 6 degrees; the table gives
the planned base height and roll at the last knot with and without the base
weights.

Needs the bsqpN16_go2_fc module and pinocchio; runs from any cwd:
    python examples/19_go2_standing.py
"""
import numpy as np
import pinocchio as pin

import gato
from gato.config import GO2_START_CONFIGS

N, DT = 16, 0.01
FOOT_RADIUS = 0.022   # go2.urdf *_foot collision sphere
params = gato.SolverParams(max_sqp_iters=10, q_cost=5.0, qd_cost=1e-1, u_cost=1e-4, N_cost=25.0,
                           q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
urdf = str(gato.robot_urdf("go2"))


def solver():
    return gato.BSQP(urdf, batch_size=1, N=N, dt=DT, plant_type="go2", variant="fc", params=params)


s = solver()
nq, nv, nx, nu = s.nq, s.nv, s.nx, s.nu
# the stance: the standing keyframe lowered until the foot spheres rest on z = 0
q_stand = np.asarray(GO2_START_CONFIGS["standing"], np.float64)
foot_z = np.mean([s.ee_pos(q_stand, frame=f)[2] for f in s.contact_frames])
q_stand[2] -= foot_z - FOOT_RADIUS
p_imu = s.ee_pos(q_stand)                                        # the imu target at the stance
goals = np.zeros((1, N * 6), np.float32)
goals[0, 0::6], goals[0, 1::6], goals[0, 2::6] = p_imu
mg = sum(i.mass for i in s.model.inertias) * 9.81

# the start: the stance lowered by 3 cm and rolled by 6 degrees
q0 = q_stand.copy()
q0[2] -= 0.03
q0[3:7] = pin.Quaternion(pin.exp3(np.array([np.deg2rad(6.0), 0.0, 0.0]))).coeffs()   # xyzw
x0 = np.concatenate([q0, np.zeros(nv)]).astype(np.float32)[None]


def standing_setpoint(s):
    ref = np.zeros(s.n_fc, np.float32)
    for i, frame in enumerate(s.contact_frames):
        ref[s.fc_slots(i, "f")[2]] = mg / 4
        s.add_fc_box(0.0, 0.0, slots=s.fc_slots(i, "n"), mech="al")
    s.set_fc_ref(ref)


def base_pose(xu, k):
    q = xu[k * (nx + nu):k * (nx + nu) + nq].astype(np.float64)
    rpy = pin.rpy.matrixToRpy(pin.Quaternion(q[3], q[4], q[5], q[6]).toRotationMatrix())
    return q[2], np.rad2deg(rpy[0])


print("go2 N16 (fc-on-feet), a standing reference from a lowered and rolled base")
print(f"  start: height {q0[2]:.3f} m, roll 6.0 deg; stance height {q_stand[2]:.3f} m, roll 0; fz = mg/4 = {mg / 4:.1f} N per foot")
print("  planned base height along the horizon and the vertical foot forces at the first knot; the base weight")
print("  competes with the wrench regularization toward mg/4 (fc_cost), so lifting the body costs foot force")
print(f"  {'base weights (pos / rot)':26s} {'iters':>5s} {'knot 4':>8s} {'knot 8':>8s} {'knot 12':>8s} {'knot 15':>8s} {'roll 15':>8s}   fz at knot 0 [N]")
for wp, wo in ((0.0, 0.0), (500.0, 200.0), (5000.0, 2000.0), (50000.0, 20000.0)):
    s = solver()
    standing_setpoint(s)
    s.set_q_nom(q_stand)
    s.set_q_pos_cost(np.concatenate([[wp] * 3, [wo] * 3, [0.0], np.full(nq - 7, 50.0)]))
    res = s.solve(x0, goals)
    xu = res.xu_b(0)
    heights = [base_pose(xu, k)[0] for k in (4, 8, 12, 15)]
    roll = base_pose(xu, N - 1)[1]
    fz = res.fc_at(0)[5::6]
    print(f"  {wp:8.0f} / {wo:<15.0f} {int(res.stats.sqp_iters[0]):5d} " + " ".join(f"{h:8.4f}" for h in heights)
          + f" {roll:+8.2f}   " + " ".join(f"{f:5.1f}" for f in fz))
print("  the imu target and the joint posture already level the body; the base weights raise it toward the stance height")
