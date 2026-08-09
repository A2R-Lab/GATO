"""Contact wipe task (CL-3a's fair fight, 2026-08-09): shared task definition.

The task: the iiwa14 EE presses onto a table (MuJoCo world, table box 2 cm below
the 'ready' EE) and wipes back and forth along a surface line while the contact
normal force should sit at F_SET. Three arms share the identical world, wipe
path, pacing, and solver defaults — they differ ONLY in how contact enters the
OCP:

- ``pos``    position-only MPC. Force "control" = pressing the EE reference a
             calibrated depth below the surface (the stiffness-tuning approach).
- ``ucone``  pos + ``enable_u_cone``'s frozen-Jacobian friction cone: the
             contact force is faked as f = -pinv(J(q_press)^T) (tau - g(q_press)),
             FROZEN at the press configuration (the hack the docstring apologizes
             for). Same depth knob as ``pos``.
- ``fc``     forces-as-controls (GATO_CONTACT_FORCES build): fc slots carry the
             wrench, ``set_fc_ref`` carries the F_SET setpoint (no depth knob —
             the EE reference stays ON the surface), and the friction cone is an
             exact selection on the fc columns. R2/CL-2 cone machinery + the W2
             chain term, finally at a force where they matter.

Metrics are all CONTINUOUS (the W3 lesson): fn RMS error to F_SET, path RMS,
cone violation (world-truth ft vs mu*fn), contact-loss substeps, solve time.
Forces are sampled at SUBSTEP rate via MuJoCoWorld(record_contact=True).

Determinism: fixed pacing (pace_by_solve_time=False) + MuJoCo on a fixed binary
=> every cell is bit-reproducible; scenario variation is a deterministic grid
(wipe direction x stroke length), paired across arms by scenario id.

fc modules cannot co-load with default modules in one process (PyInit name
collision) — run the ``fc`` arm in its own process against the swapped .so
(see run_wipe_pool.sh).
"""
import os
import pickle
import subprocess
import time

import numpy as np

URDF = os.path.join(os.path.dirname(__file__), "..", "iiwa_description", "iiwa14.urdf")
URDF = os.path.abspath(URDF)

# ---- task constants (stamped into every pkl) ------------------------------
F_SET = 25.0          # N, mid of the 20-35 N spike-validated regime
MU = 0.6              # table sliding friction; the cone uses the SAME mu (honest)
TABLE_DROP = 0.02     # table top this far below the 'ready' EE
N_KNOTS = 16
DT = 0.03125
SIM_DT = 1e-3
T_APPROACH = 1.0      # descend to the press point
T_SETTLE = 1.0        # hold at the press point
T_STROKE_PAUSE = 0.3  # hold at each stroke end
V_WIPE = 0.05         # m/s along the surface
N_STROKES = 2         # out-and-back counts as 2 strokes
T_TAIL = 0.5
METRIC_SKIP = 0.2     # s after wipe start before force/path metrics count

# scenario grid: 12 directions x 2 stroke lengths = 24 paired scenarios
THETAS = np.deg2rad(np.arange(0, 360, 30))
STROKES = [0.06, 0.10]
SCENARIOS = [{"id": i, "theta": float(th), "stroke": float(L)}
             for i, (L, th) in enumerate((L, th) for L in STROKES for th in THETAS)]


_TIP_DROP_CACHE = {}


def tool_tip_drop(q0, ee):
    """EE-FRAME height above the tool's lowest contact point at q0 (~33 mm on
    the iiwa14 mesh). The EE frame origin is NOT the contact tip: placing the
    table relative to the frame starts the tool in collision, and a "surface"
    reference is physically unreachable — a hidden press bias for every arm
    (measured 2026-08-09: constant +13-15 N). Probed by bisecting the highest
    table top that does NOT touch at q0 (MuJoCo collision = the world's own
    contact geometry, so the probe is exact for the evaluation world)."""
    key = tuple(np.round(q0, 6))
    if key in _TIP_DROP_CACHE:
        return _TIP_DROP_CACHE[key]
    import mujoco
    from gato.worlds import MuJoCoWorld

    def touches(z):
        w = MuJoCoWorld(URDF, plane={"z": float(z), "pos_xy": (ee[0], ee[1]),
                                     "size_xy": (0.3, 0.3)})
        w.data.qpos[:] = q0
        w.data.qvel[:] = 0
        mujoco.mj_forward(w.model, w.data)
        return w.data.ncon > 0

    lo, hi = float(ee[2]) - 0.20, float(ee[2])  # lo clear, hi touching
    assert not touches(lo) and touches(hi)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if touches(mid):
            hi = mid
        else:
            lo = mid
    drop = float(ee[2]) - lo
    _TIP_DROP_CACHE[key] = drop
    return drop


def ready_press_geometry():
    """(model, q0, ee_frame, tip_drop, z_surface, press_center).

    The table top sits TABLE_DROP below the TOOL TIP at 'ready' (so the tool
    starts clear and descends into contact); press_center is on the surface.
    """
    import pinocchio as pin
    from gato.config import IIWA14_START_CONFIGS
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    q0 = IIWA14_START_CONFIGS["ready"].copy()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    ee = data.oMf[model.getFrameId("EE")].translation.copy()
    tip = tool_tip_drop(q0, ee)
    z_s = float(ee[2]) - tip - TABLE_DROP
    center = np.array([ee[0], ee[1], z_s])
    return model, q0, ee, tip, z_s, center


def plane_cfg(center, stroke):
    """Finite table box under the wipe segment (infinite-plane trap: see worlds.py)."""
    half = max(0.18, stroke + 0.08)
    return {"z": float(center[2]), "pos_xy": (float(center[0]), float(center[1])),
            "size_xy": (half, half), "friction": MU}


def wipe_reference(center, ee_start, tip_drop, theta, stroke, depth):
    """dt-sampled EE-FRAME reference [x,y,z,0,0,0] per knot: descend, settle,
    wipe, hold.

    depth: how far BELOW the surface the pressed TOOL TIP reference sits (0
    for the fc arm — its force comes from fc_ref, not penetration). The frame
    reference is tip_drop above the tip.
    """
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    z_ref = center[2] + tip_drop - depth  # frame z placing the TIP depth below surface
    press = np.array([center[0], center[1], z_ref])

    t_stroke = stroke / V_WIPE
    knots = []

    def hold(p, T):
        knots.extend([p] * int(round(T / DT)))

    def line(p0, p1, T):
        n = max(1, int(round(T / DT)))
        for i in range(n):
            knots.append(p0 + (p1 - p0) * (i + 1) / n)

    line(ee_start, press, T_APPROACH)
    hold(press, T_SETTLE)
    ends = [press + d * stroke, press]          # out, back
    p = press
    for k in range(N_STROKES):
        q = ends[k % 2]
        line(p, q, t_stroke)
        hold(q, T_STROKE_PAUSE)
        p = q
    hold(p, T_TAIL)
    t_wipe_start = T_APPROACH + T_SETTLE
    t_end = len(knots) * DT
    # pad generously past the horizon so run_mpc_fig8's window never starves
    hold(p, (8 * N_KNOTS) * DT)

    traj = np.zeros((len(knots), 6))
    traj[:, :3] = np.asarray(knots)
    return traj.ravel(), t_wipe_start, t_end


def frozen_pinv_cone_rows(model, q_press, n_u):
    """The ucone baseline map, frozen at q_press: rows [mu*f_n; f_t1; f_t2] of
    f = -pinv(J^T) (tau - g)  (quasi-static: tau = g - J^T f_ext => the external
    force ON the robot). Returns (C, d) for add_lin_u_rows(cone=True)."""
    import pinocchio as pin
    data = model.createData()
    J = pin.computeFrameJacobian(model, data, q_press, model.getFrameId("EE"),
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
    g0 = pin.computeGeneralizedGravity(model, data, q_press)
    pinvJT = np.linalg.pinv(J.T)                 # (3, 7): tau -> f
    S = np.array([[0.0, 0.0, MU],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    C = np.zeros((3, n_u))
    C[:, :7] = -S @ pinvJT
    d = S @ pinvJT @ g0
    # An SOC is invariant under uniform positive scaling, and the admm fold
    # lands rho * C^T C on the R block (interface docstring's rho-scale law):
    # pinv(J^T) has ||C|| ~ 1/sigma_min(J) ~ 6, so an unnormalized map at the
    # bound-default rho massively over-regularizes the torques (measured:
    # path RMS 53 mm vs 17 mm). Normalize the WHOLE map uniformly (the
    # docstring's own prescription for cone rows).
    scale = 1.0 / np.linalg.norm(C, 2)
    return C * scale, d * scale


def fc_cone_rows(n_u, n_act):
    """The exact cone on the fc FORCE slots ([n; f] layout: forces at +3..+5)."""
    C = np.zeros((3, n_u))
    C[0, n_act + 5] = MU     # mu * f_z
    C[1, n_act + 3] = 1.0    # f_x
    C[2, n_act + 4] = 1.0    # f_y
    return C


def fc_reference():
    """fc_ref for the press: pure +z reaction force ON the robot at F_SET."""
    r = np.zeros(6)
    r[5] = F_SET
    return r


def compute_metrics(stats, world, traj, t_wipe_start, t_end):
    """Continuous metrics from the tick stats + the substep contact trace."""
    ch = np.asarray(world.contact_history, dtype=float)  # (n_sub, 3): ncon, fn, ft
    n = len(ch)
    t_sub = (np.arange(n) + 1) * SIM_DT
    win = (t_sub >= t_wipe_start + METRIC_SKIP) & (t_sub <= t_end)
    fn, ft, ncon = ch[win, 1], ch[win, 2], ch[win, 0]

    ts = np.asarray(stats["timestamps"])
    ee = np.asarray(stats["ee_actual"])[:, :3]
    knot = np.minimum((ts / DT).astype(int), len(traj) // 6 - 1)
    ref = traj.reshape(-1, 6)[knot, :3]
    twin = (ts >= t_wipe_start + METRIC_SKIP) & (ts <= t_end)
    path_err = np.linalg.norm((ee - ref)[twin][:, :2], axis=1)  # surface-plane error

    cone_viol = np.maximum(0.0, ft - MU * fn)
    return {
        "fn_rms_err": float(np.sqrt(np.mean((fn - F_SET) ** 2))),
        "fn_mean": float(fn.mean()) if fn.size else 0.0,
        "ft_mean": float(ft.mean()) if ft.size else 0.0,
        "path_rms": float(np.sqrt(np.mean(path_err ** 2))),
        "cone_viol_mean": float(cone_viol.mean()),
        "cone_viol_max": float(cone_viol.max()) if cone_viol.size else 0.0,
        "contact_loss_frac": float((ncon == 0).mean()),
        "solve_ms_mean": float(np.mean(stats["solve_times"])),
        "n_sub_window": int(win.sum()),
    }


def protocol_stamp(arm, scenario, depth, world, solver_params, extra=None):
    import gato
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=os.path.dirname(URDF), capture_output=True,
                         text=True).stdout.strip()
    p = {
        "task": "contact_wipe_v1",
        "arm": arm,
        "scenario": scenario,
        "F_set": F_SET, "mu": MU, "table_drop": TABLE_DROP,
        "depth": depth,
        "N": N_KNOTS, "dt": DT, "sim_dt": SIM_DT,
        "v_wipe": V_WIPE, "n_strokes": N_STROKES,
        "world": world.params,
        "solver_params": dict(solver_params),
        "pace_by_solve_time": False,
        "gato_sha": sha,
        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        p.update(extra)
    return p


def save_cell(path, protocol, metrics, stats, world):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = {k: stats[k] for k in ("timestamps", "solve_times", "ee_actual",
                                  "joint_positions", "joint_velocities")
            if k in stats}
    with open(path, "wb") as f:
        pickle.dump({"protocol": protocol, "metrics": metrics,
                     "stats": keep,
                     "contact_history": np.asarray(world.contact_history)}, f)
