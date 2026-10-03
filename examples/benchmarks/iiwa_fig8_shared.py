"""Single source of truth for the FAIR 3-way figure-8 (GATO / MPCGPU / the multi-threaded QDLDL-based CPU solver).

The harness is iiwa14-first (the 2026-07 FAIR config, byte-identical goal with MPCGPU) and
robot-parameterized: `robot("indy7")` gives the same problem on the paper's original Indy7
so Fig-3 can separate robot from timing boundary/hardware (paper: 18–21× on Indy7).

All three solvers must track the IDENTICAL goal, measured at the IDENTICAL end-effector frame:
  - EE frame = the URDF **"EE" fixed joint** (iiwa14: +0.04 m beyond the L7 link frame). Since the
    2026-07-30 named-target regen (`fixed_target_name="EE"`), grid's end_effector_pose — GATO's
    and MPCGPU's grid.cuh COST — tracks EE, and so do the regenerated trajfiles; solver frame,
    goal frame, and metric frame all coincide (pre-regen these were L7 and old L7-frame
    data/baselines are NOT comparable). The frame name comes from the plant registry.
  - fig8 (matches MPCGPU tools/gen_reference.cu exactly):
        center = EE(q0)              # so ee(0) == center  => zero initial error
        ee(t)  = [cx + A*sin(wt), cy, cz + 0.5*A*sin(2 wt)]   with wt = 2*pi/period * t*dt
    theta = 0 (vertical fig8 in the y=const plane). q0: iiwa14 = "readyC" (bent start);
    indy7 = gato.config INDY7_START_CONFIGS["ready"].
  - closed-loop tracking starts from x_curr replicated + ZERO controls.
    The historical GATO open-loop batch sweep instead uses x_curr at knot 0
    and a zero tail; its CLI records this separately from the hold seed.
    Do not substitute a gravity-compensated feasible hold (a strict merit
    minimum that can trap the SQP) or silently mix these initialization protocols.

Goal source: iiwa14 prefers the MPCGPU-generated goal file (<MPCGPU>/examples/trajfiles/0_0_eepos.traj,
MPCGPU = $MPCGPU_ROOT or <repo>/../MPCGPU) so the goal is BYTE-identical across repos; else it is
synthesized from the formula (verified equal to the generator). MPCGPU has no Indy7 trajfile, so
the Indy7 goal is ALWAYS synthesized — the Indy7 lane is a GATO-vs-CPU comparison only.
"""
import os
import numpy as np
import pinocchio as pin

from _bench import mpcgpu_root, pin_model, urdf_path

# canonical iiwa14 URDF: the one GATO regen + MPCGPU both codegen from (md5 eeb7d4ff), NOT GRiD/robot_assets.
IIWA14_URDF = urdf_path("iiwa14")
EE_FRAME = "EE"                                        # grid end_effector_pose == URDF "EE" fixed joint
Q0_READYC = np.array([0.0, 0.30, 0.0, -1.60, 0.0, 1.20, 0.0])   # bent start; EE ~ [0.5077, 0, 0.511]

# fig8 defaults — MUST equal tools/gen_reference.cu (A, period, dt)
FIG8_A = 0.15
FIG8_PERIOD = 6.0
DT = 0.01


class Robot:
    """Robot identity for the FAIR fig8 problem: `name`, `urdf`, `ee_frame`, `q0`, and whether
    MPCGPU's generated goal file applies (`shared_goal_file`; iiwa14 only)."""

    def __init__(self, name, urdf, ee_frame, q0, shared_goal_file):
        self.name, self.urdf, self.ee_frame = name, urdf, ee_frame
        self.q0 = np.asarray(q0, dtype=float)
        self.shared_goal_file = shared_goal_file

    def __repr__(self):
        return f"Robot({self.name!r}, ee_frame={self.ee_frame!r}, q0={self.q0.round(3).tolist()})"


ROBOTS = ("iiwa14", "indy7")


def robot(name="iiwa14"):
    """The FAIR fig8 identity of a supported robot (default = the iiwa14 reference lane)."""
    import gato
    if name not in ROBOTS:
        raise ValueError(f"unsupported fig8 robot {name!r} (choose from {ROBOTS})")
    ee_frame = gato.robot_info(name).get("ee_frame", EE_FRAME)
    if name == "iiwa14":
        return Robot(name, IIWA14_URDF, ee_frame, Q0_READYC, shared_goal_file=True)
    from gato.config import INDY7_START_CONFIGS
    return Robot(name, urdf_path(name), ee_frame, INDY7_START_CONFIGS["ready"], shared_goal_file=False)


def build_model(name="iiwa14"):
    m = pin_model(name)
    m.gravity.linear = np.array([0.0, 0.0, -9.81])     # match GATO/MPCGPU (-9.81)
    return m, m.createData()


def ee_pos(model, data, q, frame=EE_FRAME):
    """Position of `frame` at config q (grid-EE == URDF "EE" fixed joint)."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame)].translation.copy()


def fig8_center(model=None, data=None, q0=Q0_READYC, frame=EE_FRAME):
    if model is None:
        model, data = build_model()
    return ee_pos(model, data, q0, frame)


def figure8_goal(n_steps, A=FIG8_A, period=FIG8_PERIOD, dt=DT, center=None):
    """Flattened [x,y,z,0,0,0] per step (the layout GATO's run_mpc_fig8 + the CPU solver expect).
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


def load_goal_file(prefix=None):
    """Load MPCGPU's generated fig8 goal (BYTE-identical goal for all three). Returns flat 6-wide array
    or None if absent (caller then falls back to figure8_goal). Default prefix:
    <MPCGPU>/examples/trajfiles/0_0. This file is the iiwa14 goal; use goal_sequence() to pick
    the right source per robot."""
    if prefix is None:
        prefix = str(mpcgpu_root() / "examples" / "trajfiles" / "0_0")
    path = prefix + "_eepos.traj"
    if not os.path.exists(path):
        return None
    rows = np.loadtxt(path, delimiter=",")
    return rows.reshape(-1).astype(float)


def goal_sequence(rob, n_steps, center):
    """The fig8 goal for `rob` covering at least n_steps: MPCGPU's generated file when it applies
    (iiwa14) and is long enough, else the formula. Returns (flat goal, source) with source in
    {"mpcgpu-file", "formula"}. Same selection rule every lane used before parameterization."""
    goal = load_goal_file() if rob.shared_goal_file else None
    if goal is None or len(goal) // 6 < n_steps:
        return figure8_goal(n_steps, center=center), "formula"
    return goal, "mpcgpu-file"


def ee_tracking_errors(model, data, joint_positions, goal_flat, dt=DT, frame=EE_FRAME):
    """Per-control-step EE tracking error given the logged joint configs and the fig8 goal.
    Assumes fixed-dt pacing (goal index == step index). Returns np.array of |EE(q_i) - goal_i|."""
    errs = []
    n_goal = len(goal_flat) // 6
    for i, q in enumerate(joint_positions):
        if i >= n_goal:
            break
        p = ee_pos(model, data, np.asarray(q)[:model.nq], frame)
        g = goal_flat[6 * i:6 * i + 3]
        errs.append(float(np.linalg.norm(p - g)))
    return np.array(errs)


if __name__ == "__main__":
    # off-GPU verification: EE center + fig8(0)==center + (if present) match vs the generated goal file
    import sys
    rob = robot(sys.argv[1] if len(sys.argv) > 1 else "iiwa14")
    m, d = build_model(rob.name)
    c = fig8_center(m, d, rob.q0, rob.ee_frame)
    print(f"{rob}")
    print(f"EE(q0) center = {c.round(4)}" + ("   (gen_reference grid-FK prints [0.5077 0.0000 0.5110])"
                                              if rob.name == "iiwa14" else ""))
    g = figure8_goal(4, center=c)
    print(f"fig8(0) = {g[0:3].round(4)}  (should == center)   fig8(1) = {g[6:9].round(4)}")
    gf = load_goal_file() if rob.shared_goal_file else None
    if gf is not None:
        n = min(len(gf), len(figure8_goal(len(gf) // 6, center=c)))
        mine = figure8_goal(len(gf) // 6, center=c)
        diff = np.abs(gf[:n] - mine[:n]).max()
        print(f"generated goal file present ({len(gf)//6} steps): max|formula - file| = {diff:.2e} "
              f"(small => Python fig8 matches gen_reference)")
    elif rob.shared_goal_file:
        print("no generated goal file yet (run gen_reference on GPU to produce the byte-identical goal)")
    else:
        print(f"{rob.name}: no MPCGPU trajfile exists — goal is always synthesized from the formula")
