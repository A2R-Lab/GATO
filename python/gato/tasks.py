"""Arm tasks sized to the URDF limits (October 2026, the joint-feasible MPC work).

A reference that asks for several times the joint velocity limit has no
feasible tracking solution: the limit rows then only decide how the plan
fails. The tasks here are scaled from the limits instead. ``feasible_fig8``
sets the figure-8 period so the joint speed the path demands is a chosen
fraction of the URDF velocity limit (``ratio`` below 1: the arm can follow;
above 1: the limits must bind and a feasible plan lags the reference), with a
differential-IK pass along the path as the estimate of the demanded joint
speed. ``fig8_scenarios`` and ``pickplace_scenarios`` draw seeded scenario
sets around a plant's workspace for the dev-set and holdout evaluations
(examples/benchmarks/limits_holdout.py); example 12 and the docs use the same
functions. Everything here is numpy plus pinocchio (imported inside the
functions); no solver, no GPU.
"""
import numpy as np

from .common import figure8
from .config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

# Start poses: the Indy7 'ready' pose of the paper; for the iiwa14 the bent
# start of the fair three-way figure-8 harness (examples/benchmarks/
# iiwa_fig8_shared.Q0_READYC, end effector near [0.51, 0, 0.51]), whose loop
# stays clear of the wrist singularity that the config 'ready' pose's
# neighbourhood runs into.
READY = {"indy7": np.asarray(INDY7_START_CONFIGS["ready"], dtype=float),
         "iiwa14": np.array([0.0, 0.30, 0.0, -1.60, 0.0, 1.20, 0.0])}
# Figure-8 centre of record per plant: the paper's Indy7 centre, and the
# iiwa14 harness centre (the end-effector position at its start pose).
FIG8_CENTER = {"indy7": np.asarray(FIG8_DEFAULT_PARAMS["offset"], dtype=float),
               "iiwa14": np.array([0.5077, 0.0, 0.511])}
# Figure-8 amplitude range (m) drawn by fig8_scenarios: the Indy7 around the
# paper's 0.4 m, the iiwa14 around the fair harness's smaller loop at its
# shorter reach.
FIG8_AMPLITUDE = {"indy7": (0.2, 0.4), "iiwa14": (0.10, 0.20)}
# Pick-and-place goal boxes (xyz lo, hi): the iiwa14 box is the paper's
# (PICKPLACE_DEFAULT_GOALS lie inside it), the Indy7 box sits around its
# figure-8 centre at comparable reach.
PICKPLACE_BOX = {"indy7": (np.array([-0.35, 0.30, 0.30]), np.array([0.35, 0.65, 0.90])),
                 "iiwa14": (np.array([0.0, -0.5, 0.2]), np.array([0.6, 0.5, 0.8]))}


def urdf_limits(model):
    """{'q_lo', 'q_hi', 'qd', 'u'} of a pinocchio model: the URDF <limit> tags
    (the physical limits the simulated plant is graded against; the solver's
    rows carry margin-shrunk copies, see get_row_groups())."""
    return dict(q_lo=np.asarray(model.lowerPositionLimit, dtype=float),
                q_hi=np.asarray(model.upperPositionLimit, dtype=float),
                qd=np.asarray(model.velocityLimit, dtype=float),
                u=np.asarray(model.effortLimit, dtype=float))


def _frame(model, ee_frame):
    fid = model.getFrameId(ee_frame)
    if fid >= model.nframes:
        raise ValueError(f"frame {ee_frame!r} not in the model")
    return fid


def ik_track(model, ee_frame, q0, path_xyz, *, first_iters=60, damping=2e-2, max_step=0.2):
    """Differential IK along an end-effector position path: damped least squares
    on the frame's position Jacobian, one update per path point (the first
    point gets ``first_iters`` updates so the path starts converged), each
    update clipped to ``max_step`` rad so an unreachable target leaves a large
    error instead of a diverging configuration. Returns (q, err): joint
    positions (T, nq) and the position error per point (T,)."""
    import pinocchio as pin
    data = model.createData()
    fid = _frame(model, ee_frame)
    path = np.asarray(path_xyz, dtype=float).reshape(-1, 3)
    q = np.array(q0, dtype=float).copy()
    qs, errs = np.zeros((len(path), model.nq)), np.zeros(len(path))

    def update(target):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        e = target - data.oMf[fid].translation
        J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(3), e)
        n = np.linalg.norm(dq)
        return e, dq * (max_step / n if n > max_step else 1.0)

    for k, target in enumerate(path):
        for _ in range(first_iters if k == 0 else 1):
            e, dq = update(target)
            q = pin.integrate(model, q, dq)
        e, _ = update(target)
        qs[k], errs[k] = q, np.linalg.norm(e)
    return qs, errs


Q_MARGIN = 0.15   # rad: IK solutions must sit this far inside the URDF position limits


def within_q_limits(model, qs, margin=Q_MARGIN):
    """True when every configuration in ``qs`` (T, nq) is at least ``margin``
    inside the URDF position limits (the limit rows themselves sit 0.1 rad
    inside; a task whose IK path runs along the position limit is not a
    velocity-limit task)."""
    lim = urdf_limits(model)
    qs = np.asarray(qs, dtype=float).reshape(-1, model.nq)
    return bool(np.all(qs >= lim["q_lo"] + margin) and np.all(qs <= lim["q_hi"] - margin))


def joint_speed_ratio(model, ee_frame, q0, path_xyz, dt):
    """Peak demanded joint speed along the IK path as a fraction of the URDF
    velocity limit: max over points and joints of |qd_j| / qd_lim_j, plus the
    IK position error max (m) and whether the path stays inside the position
    limits (reachability checks)."""
    qs, errs = ik_track(model, ee_frame, q0, path_xyz)
    qd = np.diff(qs, axis=0) / dt
    ratio = np.abs(qd) / urdf_limits(model)["qd"][None, :]
    return float(ratio.max()), float(errs.max()), within_q_limits(model, qs)


def fig8_path(params, dt):
    """(T, 3) positions of figure8(dt, **params)."""
    return np.asarray(figure8(dt, **params), dtype=float).reshape(-1, 6)[:, :3]


def feasible_fig8(model, ee_frame, q0, dt, *, ratio=0.8, cycles=1, **fig8_params):
    """Figure-8 parameters whose period makes the demanded peak joint speed
    ``ratio`` times the URDF velocity limit (differential-IK estimate from
    ``q0``; the speed of a fixed path scales with 1/period, so one rescale
    lands the target). ``fig8_params`` override FIG8_DEFAULT_PARAMS (A_x, A_z,
    offset, theta; the period is the output). Returns dict(params, ratio_est,
    ik_err_max, within_q_limits): the parameters, the estimate after
    rescaling, the IK reach error along the path (metres; large = the path is
    not reachable) and whether the IK path stays inside the position limits."""
    p = dict(FIG8_DEFAULT_PARAMS, cycles=1)
    p.update(fig8_params)
    r0, _, _ = joint_speed_ratio(model, ee_frame, q0, fig8_path(p, dt), dt)
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError("the figure-8 path demands no joint motion from q0; check the parameters")
    p["period"] = float(p["period"] * r0 / ratio)
    r1, err, inside = joint_speed_ratio(model, ee_frame, q0, fig8_path(p, dt), dt)
    p["cycles"] = cycles
    return dict(params=p, ratio_est=float(r1), ik_err_max=float(err), within_q_limits=inside)


def fig8_scenarios(model, plant, n, *, seed, dt=0.01, ratio_range=(0.6, 1.6), ee_frame="EE",
                   ik_err_max=0.005):
    """``n`` seeded figure-8 scenarios for ``plant``: amplitude in
    FIG8_AMPLITUDE[plant], centre within 0.1 m of the plant's centre of record, rotation 0 to 90
    degrees, and the period set by feasible_fig8 for a speed ratio drawn
    uniformly in ``ratio_range`` (below 1 the arm can follow, above 1 the
    velocity limits must bind). The episode starts ON the path: ``q0`` is the
    IK solution at the path's first point from the ready pose with 0.05 rad
    jitter (``q_ready``), so the run measures steady tracking rather than the
    reach from the rest pose; fig8_reference runs two periods.
    Scenarios whose path the IK cannot reach within ``ik_err_max`` and inside
    the position limits (Q_MARGIN), or whose rescaled estimate misses the
    drawn ratio by more than 15 % (the IK path jumped between
    configurations), are redrawn. Returns a list of dicts
    (id, plant, q0, params, ratio, ratio_est)."""
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > 50 * n:
            raise RuntimeError(f"fig8_scenarios: {len(out)} of {n} reachable scenarios after {tries} draws; widen ik_err_max or shrink the box")
        q_ready = READY[plant] + rng.normal(0.0, 0.05, model.nq)
        ratio = float(rng.uniform(*ratio_range))
        A = float(rng.uniform(*FIG8_AMPLITUDE[plant]))
        cand = dict(A_x=A, A_z=float(rng.uniform(0.5, 1.0)) * A,
                    offset=(FIG8_CENTER[plant] + rng.uniform(-0.1, 0.1, 3)).tolist(),
                    theta=float(rng.uniform(0.0, np.pi / 2)))
        f = feasible_fig8(model, ee_frame, q_ready, dt, ratio=ratio, **cand)
        if f["ik_err_max"] > ik_err_max or not f["within_q_limits"] or abs(f["ratio_est"] / ratio - 1.0) > 0.15:
            continue
        q0, _ = ik_track(model, ee_frame, q_ready, fig8_path(f["params"], dt)[:1], first_iters=200)
        out.append(dict(id=len(out), plant=plant, task="fig8", q0=q0[0].tolist(), q_ready=q_ready.tolist(),
                        params=f["params"], ratio=ratio, ratio_est=f["ratio_est"], dt=dt))
    return out


def pickplace_scenarios(model, plant, n, *, seed, n_goals=3, ramps=(0.0, 0.6, 1.2), ee_frame="EE",
                        ik_err_max=0.005):
    """``n`` seeded pick-and-place scenarios: ``n_goals`` goals drawn in the
    plant's goal box (each reachable by IK from the ready pose within
    ``ik_err_max`` and inside the position limits by Q_MARGIN), start pose = ready with 0.05 rad jitter, and a reference
    ramp (seconds of minimum-jerk travel between goals, run_mpc_goals'
    ``goal_ramp``) drawn from ``ramps``: 0 is the paper's step reference, which
    asks for maximal acceleration and makes the torque box bind. Returns a
    list of dicts (id, plant, q0, goals, ramp)."""
    rng = np.random.default_rng(seed)
    lo, hi = PICKPLACE_BOX[plant]
    out, tries = [], 0
    while len(out) < n:
        q0 = READY[plant] + rng.normal(0.0, 0.05, model.nq)
        goals = []
        while len(goals) < n_goals:
            tries += 1
            if tries > 50 * n * n_goals:
                raise RuntimeError(f"pickplace_scenarios: {tries} draws for {len(out)} scenarios; the goal box is mostly unreachable")
            g = rng.uniform(lo, hi)
            qg, err = ik_track(model, ee_frame, READY[plant], g[None, :], first_iters=200)
            if err.max() <= ik_err_max and within_q_limits(model, qg):
                goals.append(g.tolist())
        out.append(dict(id=len(out), plant=plant, task="pickplace", q0=q0.tolist(), goals=goals,
                        ramp=float(rng.choice(ramps))))
    return out


def fig8_reference(scenario, N, periods=2):
    """(flat trajectory, sim_time) for a figure-8 scenario: ``periods`` cycles
    to run, padded with extra cycles so the horizon window never runs off the
    end of the reference."""
    p = dict(scenario["params"])
    period, dt = float(p["period"]), float(scenario["dt"])
    p["cycles"] = periods + int(np.ceil(N * dt / period)) + 1
    return figure8(dt, **p), periods * period
