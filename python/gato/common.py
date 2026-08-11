"""
Common utilities for GATO trajectory optimization.
Shared functions used by both benchmark scripts and notebooks.
"""

import numpy as np


def _require_pin():
    """Import pinocchio on first use, with an actionable error.

    The base install is numpy-only; pinocchio is only needed by the model/EE
    metric layers (BSQP model loading, rk4, MPC_GATO) and ships in the
    [examples] extra.
    """
    try:
        import pinocchio as pin
    except ImportError as e:
        raise ImportError(
            "pinocchio is required for this feature (robot model / EE metrics / sim); "
            "install it with: pip install -e '.[examples]'"
        ) from e
    return pin


def figure8(dt, A_x=0.4, A_z=0.4, offset=[0.0, 0.5, 0.6], period=6, cycles=5, theta=np.pi/4):
    """
    Generate figure 8 trajectory for end-effector tracking.
    
    Args:
        dt: Time step
        A_x: Amplitude in X direction  
        A_z: Amplitude in Z direction
        offset: 3D offset for trajectory center
        period: Period of one figure-8 cycle
        cycles: Number of cycles to generate
        theta: Rotation angle around Z-axis
    
    Returns:
        Flattened array of trajectory points [x, y, z, 0, 0, 0] for each timestep
    """
    x_unrot = lambda t: offset[0] + A_x * np.sin(t)
    y_unrot = lambda t: offset[1]
    z_unrot = lambda t: offset[2] + A_z * np.sin(2*t)/2 + A_z/2
    
    # Rotation matrix around Z-axis
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    
    def get_rotated_coords(t):
        unrot = np.array([x_unrot(t), y_unrot(t), z_unrot(t)])
        rot = R @ unrot
        return rot[0], rot[1], rot[2]
    
    x = lambda t: get_rotated_coords(t)[0]
    y = lambda t: get_rotated_coords(t)[1]
    z = lambda t: get_rotated_coords(t)[2]
    
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    fig_8 = np.array([[x(t), y(t), z(t), 0.0, 0.0, 0.0] for t in timesteps]).reshape(-1)
    return np.tile(fig_8, int(cycles))


def rk4(model, data, q, dq, u, dt, fext=None):
    """
    RK4 integration for forward dynamics.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint positions
        dq: Joint velocities
        u: Control torques
        dt: Time step
        fext: External forces (optional)
    
    Returns:
        q_next: Joint positions at next timestep
        dq_next: Joint velocities at next timestep

    Floating base: ``u`` may be the ACTUATED torques only (len < model.nv);
    it is scattered into the full generalized-force vector with zeros on the
    leading base dofs (the mirror of the device ``loadStateAndFullControl``).
    """
    pin = _require_pin()
    u = np.asarray(u, dtype=np.float64)
    if u.shape[0] < model.nv:  # actuated-only torques: zero base wrench
        u = np.concatenate([np.zeros(model.nv - u.shape[0]), u])
    if fext is None:
        fext = pin.StdVec_Force()
        for _ in range(model.njoints):
            fext.append(pin.Force.Zero())
    
    # RK4 integration steps
    k1q = dq
    k1v = pin.aba(model, data, q, dq, u, fext)
    
    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = dq + k1v * dt/2
    k2v = pin.aba(model, data, q2, k2q, u, fext)
    
    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = dq + k2v * dt/2
    k3v = pin.aba(model, data, q3, k3q, u, fext)
    
    q4 = pin.integrate(model, q, k3q * dt)
    k4q = dq + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u, fext)
    
    dq_next = dq + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    avg_dq = (k1q + 2*k2q + 2*k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_dq * dt)
    
    return q_next, dq_next

def world_wrench_to_joint_local(model, data, q, w_world, frame_id):
    """World-axes wrench acting AT `frame_id`'s origin -> (joint_id, pin.Force) in that
    frame's parent-joint LOCAL frame about the joint origin — the pin.aba fext[j] slot.

    `w_world` = [force(3), torque(3)] in world axes. This is the single frame convention
    shared by the sim ground truth (mpc_gato/envs) and the solver hypothesis upload
    (hypotheses._world_to_gato); GRiD's GPU slot additionally swaps to Featherstone
    [angular; linear] order (verified vs pin.aba to 1e-8, fext_frame_probe 2026-07-07).
    """
    pin = _require_pin()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    f, tau = np.asarray(w_world[:3], float), np.asarray(w_world[3:], float)
    p = data.oMf[frame_id].translation
    F0 = pin.Force(f, tau + np.cross(p, f))          # about the world origin
    jid = model.frames[frame_id].parentJoint
    return jid, data.oMi[jid].actInv(F0)


def initialize_warm_start(x_start, N, nx, nu):
    """Initialize warm start trajectory."""
    XU = np.zeros(N*(nx+nu)-nu)
    for i in range(N):
        start_idx = i * (nx + nu)
        XU[start_idx:start_idx+nx] = x_start
    return XU


# ---- floating-base manifold helpers (pure numpy; no pinocchio/scipy) --------
#
# Numpy mirrors of the device manifold ops (gato/dynamics/manifold.cuh over
# glass::se3_difference), locked against pinocchio in test/test_manifold.py.
# Stored state = [p(3); quat xyzw(4); q_act; qd(nv)]; tangent = [v_lin; omega;
# joints] in the pin integrate/difference convention.

def _quat_to_rot(q):
    """3x3 rotation from a UNIT quaternion [x, y, z, w]."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _so3_left_jacobian(phi):
    """SO(3) left Jacobian (the SE(3) V matrix)."""
    th = np.linalg.norm(phi)
    S = np.array([[0, -phi[2], phi[1]],
                  [phi[2], 0, -phi[0]],
                  [-phi[1], phi[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + 0.5 * S + S @ S / 6.0
    return (np.eye(3) + (1 - np.cos(th)) / th**2 * S
            + (th - np.sin(th)) / th**3 * S @ S)


def state_difference(x_from, x_to, nq, nv):
    """Tangent x_to ⊟ x_from (2*nv,) for the stored state layout.

    Fixed base (nq == nv): plain subtraction. Floating base: SE(3) log on the
    base pose block ([v_lin; omega], pin convention), subtraction on the
    actuated joints and all velocities. Mirrors pin.difference (locked in
    test/test_manifold.py) without requiring pinocchio at runtime.
    """
    x_from = np.asarray(x_from, dtype=np.float64)
    x_to = np.asarray(x_to, dtype=np.float64)
    if nq == nv:
        return x_to - x_from
    qf, qt = x_from[3:7], x_to[3:7]
    qf = qf / np.linalg.norm(qf)
    qt = qt / np.linalg.norm(qt)
    # relative quaternion r = qf^-1 * qt (xyzw), then the SO(3) log
    xf, yf, zf, wf = -qf[0], -qf[1], -qf[2], qf[3]  # conjugate
    xt, yt, zt, wt = qt
    r = np.array([
        wf * xt + xf * wt + yf * zt - zf * yt,
        wf * yt - xf * zt + yf * wt + zf * xt,
        wf * zt + xf * yt - yf * xt + zf * wt,
        wf * wt - xf * xt - yf * yt - zf * zt,
    ])
    if r[3] < 0:
        r = -r
    sin_half = np.linalg.norm(r[:3])
    if sin_half < 1e-12:
        phi = 2.0 * r[:3]
    else:
        phi = (2.0 * np.arctan2(sin_half, r[3]) / sin_half) * r[:3]
    Rf = _quat_to_rot(qf)
    p_local = Rf.T @ (x_to[:3] - x_from[:3])
    rho = np.linalg.solve(_so3_left_jacobian(phi), p_local)
    return np.concatenate([rho, phi,
                           x_to[7:nq] - x_from[7:nq],
                           x_to[nq:] - x_from[nq:]])


def check_floating_state(x, nq, nv, what="x"):
    """Fail loud on a degenerate base quaternion (a zero quat tiled into a
    warm start is a silent NaN factory — the device manifold ops are
    undefined off the unit sphere). No-op on fixed base."""
    if nq == nv:
        return
    n = float(np.linalg.norm(np.asarray(x, dtype=np.float64)[3:7]))
    if abs(n - 1.0) > 1e-3:
        raise ValueError(
            f"{what}: base quaternion norm {n:.6f} != 1 (stored layout is "
            f"[p(3); quat xyzw(4); q_act; qd]) — not a valid floating state")
