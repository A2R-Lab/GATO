"""Intro example 19: a standing reference for the Go2 through the base-pose weights.

On a floating-base plant the tracking objective's posture anchor carries the
base pose: the first six entries of the per-slot posture weights
(``set_q_pos_cost`` with an array) weigh the base position and orientation in the
tangent space of the log-map state cost, and the first seven entries of
``set_q_nom`` are the stored base pose (position, quaternion xyzw) they pull
toward. With them the imu target, which fixes one point of the body, becomes
a full standing reference; the cost menu's state term on the Go2 expresses the
same reference (docs/cost_menu.md), the anchor is the lighter path.

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

# The same reference as a cost-menu state term (grid-rbd 0.5.5 composes the menu on a
# floating base): the target is the stored stance state, the weights are tangent-indexed
# (base position, base orientation, joints, then the velocities). Quadratic activations
# reproduce the anchor's objective; the other activations are available on every row.
from gato.cost_menu import CostMenu, Term   # noqa: E402

if solver().cost_menu_available:
    s = solver()
    standing_setpoint(s)
    wp, wo = 5000.0, 2000.0
    s.set_cost_menu(CostMenu(
        state=Term(weight=np.concatenate([[wp] * 3, [wo] * 3, np.full(nq - 7, 50.0), np.full(nv, params.qd_cost)]),
                   target=np.concatenate([q_stand, np.zeros(nv)])),
        input=Term(weight=params.u_cost),
    ))
    res = s.solve(x0, goals)
    heights = [base_pose(res.xu_b(0), k)[0] for k in (4, 8, 12, 15)]
    print(f"  the menu's state term at {wp:.0f} / {wo:.0f} (imu term off): heights " + " ".join(f"{h:.4f}" for h in heights)
          + f", roll {base_pose(res.xu_b(0), N - 1)[1]:+.2f} deg")

# The centre-of-mass term (0.1.5; the vendored Go2 bakes GRiD's centroidal families):
# 0.5 w |c(q) - c_des|^2 on the world CoM, on top of whichever objective is on. A target
# shifted 3 cm toward the front-right foot at the stance pulls the planned CoM that way
# while the posture anchor and the foot-force setpoint hold the stance; the readout is
# the planned CoM at the last knot against the target.
if solver().com_cost_available:
    s = solver()
    standing_setpoint(s)
    s.set_q_nom(q_stand)
    s.set_q_pos_cost(np.concatenate([[5000.0] * 3, [2000.0] * 3, [0.0], np.full(nq - 7, 50.0)]))
    c_stand = s.com_position(q_stand)
    c_des = c_stand + np.array([0.03, -0.03, 0.0])
    print(f"  CoM at the stance {c_stand.round(4)}; target {c_des.round(4)} (3 cm toward the front-right foot)")
    for w in (0.0, 1e3, 1e4, 1e5):
        if w > 0:
            s.set_com_cost(c_des, w)
        res = s.solve(x0, goals)
        xu = res.xu_b(0)
        c_last = s.com_position(xu[(N - 1) * (nx + nu):(N - 1) * (nx + nu) + nq])
        print(f"  CoM weight {w:8.0f}: planned CoM at the last knot {c_last.round(4)}, "
              f"gap to the target {np.linalg.norm(c_last - c_des):.4f} m, height {base_pose(xu, N - 1)[0]:.4f} m")
    print("  the term trades posture and foot force for the shifted CoM; it is a cost, not a balance guarantee")
