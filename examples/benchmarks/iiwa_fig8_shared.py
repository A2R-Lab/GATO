"""Single source of truth for the FAIR 3-way iiwa14 figure-8 (GATO / MPCGPU / BatchThneed).

All three solvers must track the IDENTICAL goal, measured at the IDENTICAL end-effector frame:
  - EE frame = the URDF **"EE" fixed joint** (+0.04 m beyond the L7 link frame). Since the
    2026-07-30 named-target regen (`fixed_target_name="EE"`), grid's end_effector_pose — GATO's
    and MPCGPU's grid.cuh COST — tracks EE, and so do the regenerated trajfiles; solver frame,
    goal frame, and metric frame all coincide (pre-regen these were L7 and old L7-frame
    data/baselines are NOT comparable).
  - fig8 (matches MPCGPU tools/gen_reference.cu exactly):
        center = EE(q0)              # so ee(0) == center  => zero initial error
        ee(t)  = [cx + A*sin(wt), cy, cz + 0.5*A*sin(2 wt)]   with wt = 2*pi/period * t*dt
    theta = 0 (vertical fig8 in the y=const plane); q0 = "readyC".
  - warm-start (all three): x_curr replicated + ZERO controls (infeasible on purpose; a gravity-comp
    feasible hold is a strict merit min that traps the SQP).

Load path: prefer the MPCGPU-generated goal file (examples/trajfiles/0_0_eepos.traj) so the goal is
BYTE-identical across repos; else synthesize from the formula (verified equal to the generator).
"""
import os
import numpy as np
import pinocchio as pin

# canonical iiwa14 URDF: the one GATO regen + MPCGPU both codegen from (md5 eeb7d4ff), NOT GRiD/robot_assets.
IIWA14_URDF = "/home/plancher/Desktop/GATO/examples/iiwa_description/iiwa14.urdf"
EE_FRAME = "EE"                                        # grid end_effector_pose == URDF "EE" fixed joint
Q0_READYC = np.array([0.0, 0.30, 0.0, -1.60, 0.0, 1.20, 0.0])   # bent start; EE ~ [0.5077, 0, 0.511]

# fig8 defaults — MUST equal tools/gen_reference.cu (A, period, dt)
FIG8_A = 0.15
FIG8_PERIOD = 6.0
DT = 0.01


def build_model(urdf=IIWA14_URDF):
    m = pin.buildModelFromUrdf(urdf)
    m.gravity.linear = np.array([0.0, 0.0, -9.81])     # match GATO/MPCGPU (-9.81)
    return m, m.createData()


def ee_pos(model, data, q, frame=EE_FRAME):
    """Position of `frame` at config q (grid-EE == URDF "EE" fixed joint)."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame)].translation.copy()


def fig8_center(model=None, data=None, q0=Q0_READYC):
    if model is None:
        model, data = build_model()
    return ee_pos(model, data, q0)


def figure8_goal(n_steps, A=FIG8_A, period=FIG8_PERIOD, dt=DT, center=None):
    """Flattened [x,y,z,0,0,0] per step (the layout GATO's run_mpc_fig8 + BatchThneed expect).
    Matches gen_reference.cu: ee(t) = center + [A sin(wt), 0, 0.5 A sin(2wt)], wt = 2pi/period * t*dt."""
    if center is None:
        center = fig8_center()
    omega = 2.0 * np.pi / period
    out = np.zeros(n_steps * 6)
    for t in range(n_steps):
        wt = omega * t * dt
        out[6 * t + 0] = center[0] + A * np.sin(wt)
        out[6 * t + 1] = center[1]
        out[6 * t + 2] = center[2] + 0.5 * A * np.sin(2.0 * wt)
    return out


def load_goal_file(prefix="/home/plancher/Desktop/MPCGPU/examples/trajfiles/0_0"):
    """Load MPCGPU's generated fig8 goal (BYTE-identical goal for all three). Returns flat 6-wide array
    or None if absent (caller then falls back to figure8_goal)."""
    path = prefix + "_eepos.traj"
    if not os.path.exists(path):
        return None
    rows = np.loadtxt(path, delimiter=",")
    return rows.reshape(-1).astype(float)


def ee_tracking_errors(model, data, joint_positions, goal_flat, dt=DT):
    """Per-control-step EE tracking error given the logged joint configs and the fig8 goal.
    Assumes fixed-dt pacing (goal index == step index). Returns np.array of |EE(q_i) - goal_i|."""
    errs = []
    n_goal = len(goal_flat) // 6
    for i, q in enumerate(joint_positions):
        if i >= n_goal:
            break
        p = ee_pos(model, data, np.asarray(q)[:model.nq])
        g = goal_flat[6 * i:6 * i + 3]
        errs.append(float(np.linalg.norm(p - g)))
    return np.array(errs)


if __name__ == "__main__":
    # off-GPU verification: EE center + fig8(0)==center + (if present) match vs the generated goal file
    m, d = build_model()
    c = fig8_center(m, d)
    print(f"EE(q0) center = {c.round(4)}   (gen_reference grid-FK prints [0.5077 0.0000 0.5110])")
    g = figure8_goal(4, center=c)
    print(f"fig8(0) = {g[0:3].round(4)}  (should == center)   fig8(1) = {g[6:9].round(4)}")
    gf = load_goal_file()
    if gf is not None:
        n = min(len(gf), len(figure8_goal(len(gf) // 6, center=c)))
        mine = figure8_goal(len(gf) // 6, center=c)
        diff = np.abs(gf[:n] - mine[:n]).max()
        print(f"generated goal file present ({len(gf)//6} steps): max|formula - file| = {diff:.2e} "
              f"(small => Python fig8 matches gen_reference)")
    else:
        print("no generated goal file yet (run gen_reference on GPU to produce the byte-identical goal)")
