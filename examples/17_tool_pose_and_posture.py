"""Intro example 17: a tool pose with a posture preference and a gravity-relative effort cost.

Three things the tracking objective has no slot for, in one menu on an arm:
an orientation target on the end effector (the log-map residual of the
rotation error), a linear feature that holds one joint, the elbow, near a
value while the rest of the arm is free, and an input cost measured against
the gravity torque at a reference configuration instead of zero, so that
holding the arm against gravity is not penalized as effort. The per-term
readout shows how the cost splits along the horizon.

Needs one built arm module and pinocchio (the ``[test]`` extra) for the frame
check; runs from any cwd:
    python examples/17_tool_pose_and_posture.py
"""
import numpy as np
import pinocchio as pin

import gato
from gato.cost_menu import CostMenu, LinearTerm, Term, TERMS
from gato.config import FR3_START_CONFIGS, GEN3_START_CONFIGS, IIWA14_START_CONFIGS, INDY7_START_CONFIGS, UR5E_START_CONFIGS

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["ready"],
         "fr3": FR3_START_CONFIGS["ready"], "gen3": GEN3_START_CONFIGS["ready"], "ur5e": UR5E_START_CONFIGS["ready"]}
plant, N = min((k for k in gato.available() if k[0] in START), key=lambda k: k[1])
DT = 0.01
solver = gato.BSQP(str(gato.robot_urdf(plant)), batch_size=1, N=N, dt=DT, plant_type=plant,
                   params=gato.SolverParams(max_sqp_iters=10, q_lim_cost=1e-3))
if not solver.cost_menu_available:
    raise SystemExit(f"{plant}: the module does not carry the cost menu")
nq, nv, nx, nu = solver.nq, solver.nv, solver.nx, solver.nu

q0 = np.asarray(START[plant], np.float64)
model, data = solver.model, solver.data
pin.framesForwardKinematics(model, data, q0)
M0 = data.oMf[solver.ee_frame_id].copy()
p_des = M0.translation + np.array([0.08, 0.0, -0.08])
R_des = pin.exp3(np.array([0.0, 0.0, np.deg2rad(30.0)])) @ M0.rotation   # the tool turned 30 degrees about the world z axis
elbow = nq // 2                                                          # the joint held by the linear feature
A = np.zeros((nx, nx)); A[elbow, elbow] = 1.0                            # the feature A x picks q[elbow]
Wx = np.zeros(nx); Wx[elbow] = 5.0
q_elbow_des = np.zeros(nx); q_elbow_des[elbow] = q0[elbow] + 0.3

menu = CostMenu(
    frame_pos=Term(weight=[5.0, 5.0, 5.0], target=p_des),
    orientation=Term(weight=[1.0, 1.0, 1.0], target=R_des),
    linear_x=LinearTerm(weight=Wx, A=A, target=q_elbow_des),
    state=Term(weight=[0.0] * nq + [1e-3] * nv),
    input=Term(weight=1e-5, target="gravity"),      # effort relative to g(q_ref), not to zero torque
    q_ref=q0,
)
solver.set_cost_menu(menu)

x = np.concatenate([q0, np.zeros(nv)]).astype(np.float32)[None]
goals = np.zeros((1, N * 6), np.float32)          # the menu does not read the reference argument
res = solver.solve(x, goals)
xu = res.xu_b(0)
q_last = xu[(N - 1) * (nx + nu):(N - 1) * (nx + nu) + nq].astype(np.float64)
pin.framesForwardKinematics(model, data, q_last)
M = data.oMf[solver.ee_frame_id]
err_p = np.linalg.norm(M.translation - p_des)
err_R = np.linalg.norm(pin.log3(R_des.T @ M.rotation))
print(f"{plant} N{N}: tool pose plus posture after {int(res.stats.sqp_iters[0])} iterations")
print(f"  start position error {np.linalg.norm(M0.translation - p_des):.3f} m, orientation error {np.linalg.norm(pin.log3(R_des.T @ M0.rotation)):.3f} rad")
print(f"  last-knot position error {err_p:.3f} m, orientation error {err_R:.3f} rad, elbow {q_last[elbow]:+.3f} (target {q_elbow_des[elbow]:+.3f}, start {q0[elbow]:+.3f})")
terms = solver.cost_menu_terms(res.xu)[0]
print("  cost per term, first knot vs last knot:")
for i, name in enumerate(TERMS):
    if terms[:, i].any():
        print(f"    {name:12s} {terms[0, i]:9.5f} {terms[N - 1, i]:9.5f}")
g = pin.computeGeneralizedGravity(model, data, q0)
print(f"  gravity torque at q_ref, norm {np.linalg.norm(g):.2f} Nm; first planned torque minus it, norm {np.linalg.norm(res.u0() - g):.2f} Nm")
