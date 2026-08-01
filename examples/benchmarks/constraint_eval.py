#!/usr/bin/env python
"""Constraint-mechanism evaluation matrix — the R1 harness (CL-1 item 4).

The standing harness behind the arc's evaluation rounds R1-R3
(docs/open-tasks/constraint_layer_locomotion_arc_plan_2026-07-10.md): every
mechanism-to-constraint binding is decided by experiment, and this driver is
where those experiments run.

R1 axes: {baseline | barrier_relaxed | admm | al} x {fig8 | reach | pickplace
| swing_heavy} x {indy7 | iiwa14}, plus EE-terminal-equality variants
(admm_ee, al_ee) on reach. Each cell is a FIXED-PACING closed-loop MPC episode
(one solve per dt, rk4 sim substeps) — bit-deterministic, so results are
GPU-contention-immune; only the recorded wall-clock times are load-sensitive
(never quote them from a shared box). swing_heavy emulates a heavy payload by
tightening the torque box to a fraction of the URDF limits via
set_row_group_bounds (the arc's user-ruled hard case).

Per step we record TRUE solution violation per row group (telemetry), the
EXECUTED torque's exceedance vs the active bounds, tracking error in the
solver EE frame, SQP iters, and merit. The report table compares mechanisms
per (plant, problem) against the arc gates: active-box violation <= 1e-5 and
fig8 tracking regression <= 2% vs the unconstrained baseline.

R2 additions: sweep-parameterized mechanism names (``<mech>~r<val>~i<val>`` —
r = the mechanism's primary strength knob: cone_rho for cone_* cells, mu for
rb, rho otherwise; i = ADMM inner iters), the ``press_mild`` problem (wide
cone + large normal-force headroom, so the "viol<=1e-5 at <=2% cost" gate is
tested where the cone barely binds, not only at the adversarial press cell),
and an ``--exact`` axis (SO-SQP exact-Hessian toggle; needs an
EXACT_HESSIAN_AVAILABLE module .so-swapped into python/gato — records/cells
are suffixed ``+ex`` so they never collide with default rows).

Usage:
  python examples/benchmarks/constraint_eval.py --run            # all cells
  python examples/benchmarks/constraint_eval.py --run --quick    # smoke subset
  python examples/benchmarks/constraint_eval.py --run --cells a,b,c  # explicit list
  python examples/benchmarks/constraint_eval.py --run --exact    # exact-Hessian axis
  python examples/benchmarks/constraint_eval.py --report         # tables from data
  python examples/benchmarks/constraint_eval.py --cell <name>    # one cell (child mode)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
DATA = ROOT / "examples" / "benchmarks" / "data" / "constraint_eval"

# ---- axes (extend here as bindings/problems land) -------------------------
N_KNOTS = 32
DT = 0.01
SIM_DT = 0.001
PLANTS = {
    "indy7": dict(urdf=ROOT / "examples" / "indy7_description" / "indy7.urdf",
                  start="ready"),
    "iiwa14": dict(urdf=ROOT / "examples" / "iiwa_description" / "iiwa14.urdf",
                   start="zeros"),
}
# mechanism -> (setup kwargs); EE variants bind the reach target as a terminal
# equality through whichever mechanism is active (the CL-1 waves' semantics).
# ADMM rho rides the COST-HESSIAN scale (R1 finding): rho >= 1 over-damps the
# u-block (natural scale u_cost=1e-6) -> controls freeze at the warm start and
# closed-loop MPC parks; rho ~ 0.005-0.02 tracks at-or-better than baseline
# with violation exactly 0. The wave-measured "rho-insensitivity 1..1000" was
# entirely inside the over-damped regime.
MECHANISMS = {
    "baseline": {},
    "rb": dict(mu=3e-3, delta=0.05),
    "admm": dict(rho=0.01, iters=10),
    "admm_m": dict(rho=0.01, iters=10, merit=True),     # + set_admm_merit (R1 ablation)
    "al": dict(rho=1.0),
    "admm_ee": dict(rho=0.01, iters=10, ee_rho=10.0),   # reach only
    "admm_m_ee": dict(rho=0.01, iters=10, ee_rho=10.0, merit=True),  # reach only
    "al_ee": dict(rho=1.0, ee_rho=1.0),                 # reach only
    # CL-2 cone cells ("press" problem): EE-force friction cone frozen at the
    # goal config (simulated grasp/contact), stacked ON the same-mechanism
    # limit boxes. The cone map is Frobenius-normalized (positively homogeneous
    # => cone-preserving), so cone_rho rides the SAME rho-scale-law pockets as
    # the boxes. cone_off = telemetry-only cone (the press baseline).
    "cone_off": dict(cone="soc", cone_mech="telemetry", cone_rho=0.0),
    "cone_soc_admm": dict(rho=0.01, iters=10, cone="soc", cone_mech="admm", cone_rho=0.01),
    "cone_soc_admm_m": dict(rho=0.01, iters=10, cone="soc", cone_mech="admm",
                            cone_rho=0.01, merit=True),  # R2 merit-under-cone ablation
    "cone_soc_al": dict(rho=1.0, cone="soc", cone_mech="al", cone_rho=1.0),
    "cone_pyr_admm": dict(rho=0.01, iters=10, cone="pyramid", cone_mech="admm", cone_rho=0.01),
    "cone_pyr_al": dict(rho=1.0, cone="pyramid", cone_mech="al", cone_rho=1.0),
    "cone_rb": dict(mu=3e-3, delta=0.05, cone="soc", cone_mech="barrier", cone_rho=3e-3),
    # CL-2 collision cells ("pillars" problem): per-sphere clearance rows
    # d_i(q_k) >= margin against two vertical capsule pillars at the fig8
    # lobe points (+ a floor plane), stacked ON the same-mechanism limit
    # boxes. cc_off = telemetry-only clearance (the pillars baseline —
    # tracks straight through the pillars and reports the penetration).
    # cc_rho = 1.0 BOUND by the 2b pillars round: clearance rows fold onto
    # the Q block (natural scale O(q_cost)), so unlike the cone's sharp
    # u-block pocket the rho pocket is WIDE AND FLAT — viol_max drops
    # monotonically 0.043->0.0027 (indy7) / 0.079->0.021 (iiwa14) over rho
    # 0.01->1.0 at FLAT tracking cost; 5.0 still stable and strictly better
    # (indy7 1.7mm / iiwa14 9.9mm) — the rho-scale law, Q-block edition.
    "cc_off": dict(cc_mech="telemetry", cc_rho=0.0),
    "cc_admm": dict(rho=0.01, iters=10, cc_mech="admm", cc_rho=1.0),
    "cc_al": dict(rho=1.0, cc_mech="al", cc_rho=1.0),
    "cc_rb": dict(mu=3e-3, delta=0.05, cc_mech="barrier", cc_rho=3e-3),
    # _lbdsv = ADMM inner-loop linsys A/B arms: identical cells to their base
    # except set_admm_linsys("bdsv_factor") — factor-once BDSV + factored
    # re-solves instead of the DEFAULT warm-started PCG (default flipped to pcg
    # 2026-08-01 on the quiet-box A/B: pcg 1.4-2.5x faster, identical tracking,
    # ~2-3x looser transients). Night-runner timing cells; solve_us is NOT
    # quotable from ordinary matrix runs.
    "admm_lbdsv": dict(rho=0.01, iters=10, admm_linsys="bdsv_factor"),
    "cone_soc_admm_lbdsv": dict(rho=0.01, iters=10, cone="soc", cone_mech="admm",
                                cone_rho=0.01, admm_linsys="bdsv_factor"),
    "cc_admm_lbdsv": dict(rho=0.01, iters=10, cc_mech="admm", cc_rho=1.0,
                          admm_linsys="bdsv_factor"),
    # P4.3 adaptive-rho recovery cells: OSQP-style per-solve rho scaling
    # (set_admm_rho_adaptation) evaluated where a MIS-SET rho is known to
    # hurt. _rho1 = cone at rho=1.0, 100x above the sharp 0.01 u-block pocket
    # (R2: parks the closed loop) — _rho1_ad asks whether adaptation recovers
    # it. cc_admm_ad starts at the bound 1.0 Q-block baseline where 5.0 is
    # known strictly better (2b: graze 0.021 -> 0.0099 iiwa14) — adaptation
    # should tighten toward it. cone_pyr_admm_eq = the pyramid-facet interval
    # cell with enable-time row equilibration (equilibrate=True; exact
    # reformulation — tests the TinyMPC-style normalization path end-to-end).
    "cone_soc_admm_rho1": dict(rho=0.01, iters=10, cone="soc", cone_mech="admm",
                               cone_rho=1.0),
    "cone_soc_admm_rho1_ad": dict(rho=0.01, iters=10, cone="soc", cone_mech="admm",
                                  cone_rho=1.0, rho_adapt=True),
    "cc_admm_ad": dict(rho=0.01, iters=10, cc_mech="admm", cc_rho=1.0,
                       rho_adapt=True),
    "cone_pyr_admm_eq": dict(rho=0.01, iters=10, cone="pyramid", cone_mech="admm",
                             cone_rho=0.01, equilibrate=True),
}
PROBLEMS = ["fig8", "reach", "pickplace", "swing_heavy"]
CONE_MECHS = [m for m in MECHANISMS if m.startswith("cone_")]
CONE_PROBLEMS = ["press", "press_mild"]
CC_MECHS = [m for m in MECHANISMS if m.startswith("cc_")]
EE_ONLY_PROBLEMS = {"admm_ee": ["reach"], "admm_m_ee": ["reach"], "al_ee": ["reach"],
                    **{m: CONE_PROBLEMS for m in CONE_MECHS},
                    **{m: ["pillars"] for m in CC_MECHS},
                    # A/B arms: one canonical problem each (they exist for the
                    # night runner's timing leg, not the evaluation matrix)
                    "admm_lbdsv": ["fig8"], "cone_soc_admm_lbdsv": ["press_mild"],
                    # adaptive-rho recovery cells: press_mild only (press is
                    # adversarial by design — it parks every hard mechanism
                    # and would measure the problem, not the adaptation)
                    "cone_soc_admm_rho1": ["press_mild"],
                    "cone_soc_admm_rho1_ad": ["press_mild"],
                    "cone_pyr_admm_eq": ["press_mild"]}
# per-problem (friction coefficient, N of normal-force headroom around the
# gravity-comp point): press is deliberately adversarial (tight cone, little
# headroom); press_mild barely binds — wide cone, big headroom AND a scaled-down
# reach (inertial torques dominate the implied static force during fast
# transients, so cone width alone cannot make the cell mild) — separating
# "mechanism costs nothing when feasible" from "mechanism fights the task".
# press_mild additionally SELF-CALIBRATES f_bias so the START pose's
# gravity-comp point is cone-feasible with >= PRESS_MILD_START_MARGIN margin
# (R2 finding: iiwa14's zeros start violated the goal-frozen cone by -6.7 —
# an infeasible-at-start cell parks EVERY hard mechanism by construction,
# measuring problem design, not the mechanism).
CONE_PARAMS = {"press": (0.5, 5.0), "press_mild": (0.9, 20.0)}
PRESS_MILD_START_MARGIN = 1.0
PRESS_MILD_GOAL_SCALE = 0.4
CONE_FACETS = 8
SWING_TORQUE_SCALE = 0.3     # heavy-payload emulation: torque box fraction
PICKPLACE_SEG_S = 1.2        # seconds per waypoint
PICKPLACE_GOALS = [          # _common.PICKPLACE_DEFAULT_GOALS (first 4)
    [0.5, -0.1865, 0.5], [0.5, 0.5, 0.2], [0.3, 0.3, 0.8], [0.6, -0.5, 0.2]]
SIM_S = {"fig8": 4.0, "reach": 2.0, "pickplace": 4.8, "swing_heavy": 3.0,
         "press": 2.0, "press_mild": 2.0, "pillars": 4.0}
# pillars: vertical capsules at PILLAR_LOBE_FRAC x the fig8 lobe amplitude
# (same rotation math as gato.common.figure8), so the straight fig8 grazes
# them by ~(sphere inflation + pillar radius - lobe overshoot); mechanisms
# must shave the lobe tips. Floor plane rides along as an INERT sanity row —
# it must clear every reachable sphere: near-base spheres sit at z ~0.06
# with r ~0.13, and a floor they permanently violate is an unsatisfiable
# row the base can't fix (the R2 infeasible-at-start class; measured: a
# z=0.05 floor pinned cc_viol at a constant 0.12 and drowned the pillars).
PILLAR_LOBE_FRAC = 0.9
PILLAR_RADIUS = 0.05
PILLAR_Z = (0.2, 1.3)
PILLAR_MARGIN = 0.02
FLOOR_Z = -0.25
REACH_DQ_GOAL = [1.2, 0.7, -0.7, 0.5, 0.4, 0.3, 0.3]  # reach/press goal offset


def cell_name(plant, mech, problem):
    return f"{plant}-{mech}-{problem}"


def resolve_mech(mech):
    """'cone_soc_admm~r0.02~i5' -> ('cone_soc_admm', params w/ overrides).
    r<float> = primary strength knob (cc_rho for collision mechs, cone_rho
    for cone mechs, mu for rb, rho otherwise); i<int> = ADMM inner iters."""
    base, *toks = mech.split("~")
    p = dict(MECHANISMS[base])
    for t in toks:
        if t[:1] == "r":
            key = ("cc_rho" if "cc_mech" in p else
                   ("cone_rho" if "cone" in p else ("mu" if "mu" in p else "rho")))
            p[key] = float(t[1:])
        elif t[:1] == "i":
            p["iters"] = int(t[1:])
        else:
            raise ValueError(f"bad sweep token {t!r} in {mech!r}")
    return base, p


def all_cells(quick=False):
    cells = []
    for plant in PLANTS:
        for mech in MECHANISMS:
            for prob in EE_ONLY_PROBLEMS.get(mech, PROBLEMS):
                cells.append(cell_name(plant, mech, prob))
    if quick:
        cells = [c for c in cells if c.startswith("indy7") and
                 c.endswith(("fig8", "reach"))]
    return cells


def provenance():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--ignore-submodules=untracked"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return dict(sha=sha, dirty=dirty, time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def gpu_load_note():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    util, mem = (int(v) for v in out.strip().split(","))
    quiet = util <= 5 and mem <= 2000
    if not quiet:
        print(f"[warn] GPU busy (util={util}%, mem={mem}MiB): results are "
              f"deterministic but recorded wall times are NOT quotable", file=sys.stderr)
    return quiet


# ---- problem definitions ---------------------------------------------------

def _dq_goal(s, problem):
    dq = np.array(REACH_DQ_GOAL)[:s.nq]
    return dq * PRESS_MILD_GOAL_SCALE if problem == "press_mild" else dq


def _start_x(s, plant):
    from gato.config import INDY7_START_CONFIGS
    if PLANTS[plant]["start"] == "ready":
        q0 = np.asarray(INDY7_START_CONFIGS["ready"], dtype=np.float64)
    else:
        q0 = np.zeros(s.nq)
    return np.concatenate([q0, np.zeros(s.nv)])


def build_problem(s, plant, problem):
    """Returns (x0, goal_of_step(step) -> per-knot 6N window, n_steps,
    ee_target or None, apply_bounds(s) or None)."""
    from gato.common import figure8
    from gato.config import FIG8_DEFAULT_PARAMS
    x0 = _start_x(s, plant)
    n_steps = int(SIM_S[problem] / DT)

    if problem in ("fig8", "pillars"):
        traj = figure8(DT, **FIG8_DEFAULT_PARAMS)

        def goal(step):
            off = 6 * step
            return traj[off:off + 6 * N_KNOTS]
        return x0, goal, n_steps, None, None

    if problem in ("reach", "swing_heavy") or problem.startswith("press"):
        # goal = solver-frame EE at a displaced (guaranteed-reachable) config
        # (press = reach with the EE-force cone frozen at the goal config)
        dq_goal = _dq_goal(s, problem)
        q_goal = x0[:s.nq] + dq_goal
        tgt = np.asarray(s.ee_pos(q_goal.astype(np.float32), frame="solver"),
                         dtype=np.float64)[:3]
        window = np.zeros(6 * N_KNOTS)
        window[0::6], window[1::6], window[2::6] = tgt

        def goal(step):
            return window

        apply_bounds = None
        if problem == "swing_heavy":
            def apply_bounds(sv):
                grp = sv.get_row_groups()[2]           # BOX_U
                lo = np.asarray(grp["lo"]) * SWING_TORQUE_SCALE
                hi = np.asarray(grp["hi"]) * SWING_TORQUE_SCALE
                sv.set_row_group_bounds(2, lo, hi)
        return x0, goal, n_steps, tgt, apply_bounds

    if problem == "pickplace":
        seg = int(PICKPLACE_SEG_S / DT)

        def goal(step):
            g = PICKPLACE_GOALS[min(step // seg, len(PICKPLACE_GOALS) - 1)]
            window = np.zeros(6 * N_KNOTS)
            window[0::6], window[1::6], window[2::6] = g
            return window
        return x0, goal, n_steps, None, None

    raise ValueError(problem)


def _ee_force_cone_map(s, q_ref, mu, f_bias):
    """(C, d) for the EE-force friction cone at the frozen config q_ref:
    implied static EE force f = pinv(J^T)(tau - g(q)) (world-aligned linear
    part, solver EE joint), cone rows [mu*(f_z + f_bias); f_x; f_y] — feasible
    iff the commanded force stays inside the mu-cone about +z with f_bias of
    normal headroom. Frobenius-normalized (cone-preserving: K is positively
    homogeneous), so cone_rho lives on the same scale as the box rho."""
    import pinocchio as pin
    q = np.asarray(q_ref, dtype=np.float64)
    jid = s.model.njoints - 1  # solver EE frame = last moving joint
    pin.computeJointJacobians(s.model, s.data, q)
    J = pin.getJointJacobian(s.model, s.data, jid,
                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    tau_g = pin.computeGeneralizedGravity(s.model, s.data, q)
    Cf = np.linalg.solve(J @ J.T, J)              # pinv(J^T): f = Cf @ (tau - g)
    S = np.array([[0.0, 0.0, mu], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    C = S @ Cf
    d = -C @ tau_g + np.array([mu * f_bias, 0.0, 0.0])
    scale = np.linalg.norm(C) / np.sqrt(C.shape[0])
    return C / scale, d / scale


def _press_mild_bias(s, q_goal, q_start, mu, f_bias):
    """Raise f_bias (never lower) until the START pose's gravity-comp point is
    cone-feasible with PRESS_MILD_START_MARGIN to spare. The margin is affine
    in f_bias (only row 0 of d moves), so one finite shift gives the slope."""
    import pinocchio as pin
    tau_g = pin.computeGeneralizedGravity(s.model, s.data,
                                          np.asarray(q_start, dtype=np.float64))

    def margin(fb):
        C, d = _ee_force_cone_map(s, q_goal, mu, fb)
        g = C @ tau_g + d
        return g[0] - np.linalg.norm(g[1:])

    m = margin(f_bias)
    if m >= PRESS_MILD_START_MARGIN:
        return f_bias
    slope = margin(f_bias + 1.0) - m
    return f_bias + (PRESS_MILD_START_MARGIN - m) / slope


def _pillars_env():
    """(capsules, planes) for the pillars problem: two vertical capsules at
    PILLAR_LOBE_FRAC x the fig8 lobe tips (t = +-pi/2 of the unrotated curve,
    rotated by theta about Z — the same math as gato.common.figure8) + the
    floor plane."""
    from gato.config import FIG8_DEFAULT_PARAMS as P
    th = P["theta"]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    caps = []
    for sx in (+1.0, -1.0):
        xy = R @ np.array([P["offset"][0] + sx * PILLAR_LOBE_FRAC * P["A_x"],
                           P["offset"][1]])
        caps.append((xy[0], xy[1], PILLAR_Z[0], xy[0], xy[1], PILLAR_Z[1],
                     PILLAR_RADIUS))
    return caps, [(0.0, 0.0, 1.0, FLOOR_Z)]


def enable_mechanism(s, mech, ee_target, q_goal=None, problem=None, q_start=None):
    base, p = resolve_mech(mech)
    if base in ("baseline", "cone_off", "cc_off"):
        s.enable_limit_telemetry()
    elif base in ("rb", "cone_rb", "cc_rb"):
        s.enable_limit_barrier(mu=p["mu"], delta=p["delta"])
    elif base.startswith("admm") or "_admm" in base:
        s.enable_limit_admm(rho=p["rho"], iters=p["iters"])
        if p.get("merit"):
            s.set_admm_merit(True)
    elif base in ("al", "al_ee") or base.endswith("_al"):
        s.enable_limit_al(rho=p["rho"])
    # mode-independent ADMM modifiers (no-ops outside ADMM mode)
    if "admm_linsys" in p:
        s.set_admm_linsys(p["admm_linsys"])
    if p.get("rho_adapt"):
        s.set_admm_rho_adaptation(True)
    if base.endswith("_ee"):
        assert ee_target is not None
        s.enable_ee_terminal_equality(ee_target.astype(np.float32), rho=p["ee_rho"])
    if "cone" in p:
        assert q_goal is not None
        mu_c, f_bias = CONE_PARAMS[problem]
        if problem == "press_mild" and q_start is not None:
            f_bias = _press_mild_bias(s, q_goal, q_start, mu_c, f_bias)
        C, d = _ee_force_cone_map(s, q_goal, mu_c, f_bias)
        s.enable_u_cone(C, d, mech=p["cone_mech"],
                        rho=(p["cone_rho"] or None), form=p["cone"],
                        facets=CONE_FACETS,
                        **(dict(admm_iters=p["iters"]) if "iters" in p else {}),
                        **(dict(delta=p["delta"]) if "delta" in p else {}),
                        **(dict(equilibrate=True) if p.get("equilibrate") else {}))
    if "cc_mech" in p:
        caps, planes = _pillars_env()
        s.set_collision_environment(capsules=caps, planes=planes)
        s.enable_collision(mech=p["cc_mech"], margin=PILLAR_MARGIN,
                           rho=(p["cc_rho"] or None),
                           **(dict(admm_iters=p["iters"]) if "iters" in p else {}),
                           **(dict(delta=p["delta"]) if "delta" in p else {}))


# ---- one cell: fixed-pacing closed-loop episode ----------------------------

def run_cell(name, exact=False, bdsv=False):
    import pinocchio  # noqa: F401  (rk4 needs the model)
    import gato
    from gato.common import rk4

    plant, mech, problem = name.split("-")
    base_mech, params = resolve_mech(mech)
    urdf = str(PLANTS[plant]["urdf"])
    # rho=1e-3: f32 bdsv (forced by AL mode, factored by ADMM's inner loop)
    # produces garbage steps on an UNREGULARIZED Schur system (R1 measured:
    # closed-loop cascade at the interface default rho=0.0). Also >= the
    # exact-Hessian f32 envelope rho >= 1e-4 (so_sqp_device RESULTS).
    # bdsv=True: force the bdsv linsys — the single-variable control arm for
    # +ex comparisons (exact mode force-switches to bdsv internally).
    s = gato.BSQP(model_path=urdf, batch_size=1, N=N_KNOTS, dt=DT,
                  plant_type=plant, rho=1e-3, exact_hessian=exact,
                  linsys=("bdsv" if bdsv else "pcg"))

    x0, goal_of, n_steps, ee_target, apply_bounds = build_problem(s, plant, problem)
    q_goal = (x0[:s.nq] + _dq_goal(s, problem)) \
        if problem.startswith("press") else None
    enable_mechanism(s, mech, ee_target, q_goal, problem, x0[:s.nq])
    if apply_bounds is not None:
        apply_bounds(s)
    groups = s.get_row_groups()
    u_lo = np.asarray(groups[2]["lo"], dtype=np.float64)
    u_hi = np.asarray(groups[2]["hi"], dtype=np.float64)

    nx, nu, nq = s.nx, s.nu, s.nq
    stride = nx + nu
    # hold warm start at x0 (controller.reset semantics)
    XU = np.zeros((1, N_KNOTS * stride - nu), dtype=np.float32)
    for k in range(N_KNOTS):
        XU[0, k * stride:k * stride + nx] = x0
    x = x0.copy()
    q, dq = x[:nq].copy(), x[nq:].copy()

    rec = dict(goal_dist=[], viol_max=[], u_exceed=[], iters=[], merit=[],
               solve_us=[], admm_r=[])
    substeps = int(round(DT / SIM_DT))
    for step in range(n_steps):
        window = goal_of(step).astype(np.float32)
        s.reset_rho()
        t0 = time.perf_counter()
        res = s.solve(x.astype(np.float32).reshape(1, -1),
                      window.reshape(1, -1), XU_B=XU)
        rec["solve_us"].append(1e6 * (time.perf_counter() - t0))

        xu = np.asarray(res.xu[0], dtype=np.float64)
        u0 = xu[nx:nx + nu]
        # shift warm start (controller "shift" semantics)
        XU[0, :-stride] = res.xu[0, stride:]
        XU[0, -stride:] = res.xu[0, -stride:]

        vm = res.stats.row_max_violation
        rec["viol_max"].append(np.asarray(vm, dtype=np.float64).reshape(-1).tolist()
                               if vm is not None else [])
        rec["u_exceed"].append(float(np.maximum(0.0, np.maximum(u0 - u_hi, u_lo - u0)).max()))
        rec["iters"].append(int(np.asarray(res.stats.sqp_iters).reshape(-1)[0]))
        rec["merit"].append(float(np.asarray(res.stats.final_merit).reshape(-1)[0]))
        rp = getattr(res.stats, "admm_r_prim", None)
        if rp is not None:
            rec["admm_r"].append(float(np.asarray(rp).reshape(-1)[0]))

        # play u0 for one dt (fixed pacing), then measure tracking there
        for _ in range(substeps):
            q, dq = rk4(s.model, s.data, q, dq, u0, SIM_DT)
        x = np.concatenate([q, dq])
        ee = np.asarray(s.ee_pos(q.astype(np.float32), frame="solver"),
                        dtype=np.float64)[:3]
        rec["goal_dist"].append(float(np.linalg.norm(ee - np.asarray(window[:3], dtype=np.float64))))

    gd = np.asarray(rec["goal_dist"])
    vmax = np.asarray([max(v) if v else np.nan for v in rec["viol_max"]])
    ue = np.asarray(rec["u_exceed"])
    tag = "+ex" if exact else ("+bdsv" if bdsv else "")
    out = dict(
        cell=name + tag, plant=plant, mechanism=mech + tag, problem=problem,
        params=params, exact=bool(exact), n_steps=n_steps,
        track_mean=float(gd.mean()), track_max=float(gd.max()),
        track_final=float(gd[-min(20, len(gd)):].mean()),
        viol_soln_max=float(np.nanmax(vmax)) if len(vmax) else None,
        viol_soln_mean=float(np.nanmean(vmax)) if len(vmax) else None,
        u_exceed_max=float(ue.max()),
        u_exceed_frac=float((ue > 1e-6).mean()),
        iters_mean=float(np.mean(rec["iters"])),
        solve_us_median=float(np.median(rec["solve_us"])),
        merit_final=rec["merit"][-1],
        admm_r_final=(rec["admm_r"][-1] if rec["admm_r"] else None),
        viol_groups_max=np.nanmax(np.asarray(
            [v for v in rec["viol_max"] if v], dtype=np.float64), axis=0).tolist()
            if any(rec["viol_max"]) else None,
    )
    # appended-group telemetry columns, found BY KIND (not by the old
    # hardcoded index 3, which broke once cells could append >1 group):
    # LIN_U (4) = the cone group (margin violation for SOC, facet violation
    # for pyramid); COLLISION (5) = the clearance group (margin - min d_i).
    vm_full = np.asarray([v for v in rec["viol_max"] if v], dtype=np.float64)
    if vm_full.ndim == 2:
        cone_gi = next((i for i, g in enumerate(groups) if g["kind"] == 4), None)
        cc_gi = next((i for i, g in enumerate(groups) if g["kind"] == 5), None)
        if cone_gi is not None and vm_full.shape[1] > cone_gi:
            out["cone_viol_max"] = float(vm_full[:, cone_gi].max())
            out["cone_viol_mean"] = float(vm_full[:, cone_gi].mean())
        if cc_gi is not None and vm_full.shape[1] > cc_gi:
            out["cc_viol_max"] = float(vm_full[:, cc_gi].max())
            out["cc_viol_mean"] = float(vm_full[:, cc_gi].mean())
    return out


# ---- orchestration + report ------------------------------------------------

def report(rows):
    latest = {}
    for r in rows:
        latest[r["cell"]] = r
    by_pp = {}
    for r in latest.values():
        by_pp.setdefault((r["plant"], r["problem"]), []).append(r)
    order = {m: i for i, m in enumerate(MECHANISMS)}

    def mech_key(m):  # sweep/exact variants sort next to their base mech
        return (order.get(m.split("+")[0].split("~")[0], 99), m)

    for (plant, prob), rs in sorted(by_pp.items()):
        rs.sort(key=lambda r: mech_key(r["mechanism"]))
        base = next((r for r in rs if r["mechanism"] in ("baseline", "cone_off")), None)
        print(f"\n### {plant} / {prob}  (steps={rs[0]['n_steps']})")
        has_cone = any(r.get("cone_viol_max") is not None for r in rs)
        cone_hdr = " cone viol max/mean |" if has_cone else ""
        print("| mech | track mean m | final m | soln viol max | u-exceed max (frac) | iters | reg vs base |" + cone_hdr)
        print("|---|---|---|---|---|---|---|" + ("---|" if has_cone else ""))
        for r in rs:
            reg = (f"{r['track_mean'] / base['track_mean'] - 1:+.1%}"
                   if base and base is not r and base["track_mean"] > 0 else "—")
            cone_col = ""
            if has_cone:
                cv = r.get("cone_viol_max")
                cone_col = (f" {cv:.2e} / {r['cone_viol_mean']:.2e} |"
                            if cv is not None else " — |")
            print(f"| {r['mechanism']} | {r['track_mean']:.4f} | {r['track_final']:.4f} "
                  f"| {r['viol_soln_max']:.2e} | {r['u_exceed_max']:.2e} ({r['u_exceed_frac']:.0%}) "
                  f"| {r['iters_mean']:.1f} | {reg} |" + cone_col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--cells", help="explicit comma-separated cell list (sweep driver)")
    ap.add_argument("--only", help="substring filter on cell names")
    ap.add_argument("--exact", action="store_true",
                    help="run cells with exact_hessian=True (+ex records)")
    ap.add_argument("--bdsv", action="store_true",
                    help="force the bdsv linsys (+bdsv records — the +ex control arm)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    results_path = DATA / "results.jsonl"

    if args.cell:
        print(json.dumps(run_cell(args.cell, exact=args.exact, bdsv=args.bdsv)))
        return

    if args.run:
        gpu_load_note()
        prov = provenance()
        cells = args.cells.split(",") if args.cells else all_cells(args.quick)
        if args.only:
            cells = [c for c in cells if args.only in c]
        child_extra = (["--exact"] if args.exact else []) + (["--bdsv"] if args.bdsv else [])
        with open(results_path, "a") as f:
            for cell in cells:
                r = subprocess.run([sys.executable, __file__, "--cell", cell]
                                   + child_extra, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[FAIL] {cell}\n{r.stderr[-2000:]}", file=sys.stderr)
                    continue
                rec = json.loads(r.stdout.strip().splitlines()[-1])
                rec["provenance"] = prov
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[ok] {rec['cell']}: track_mean={rec['track_mean']:.4f}m "
                      f"viol_max={rec['viol_soln_max']:.2e} "
                      f"u_exceed={rec['u_exceed_max']:.2e}")

    if args.report or args.run:
        if not results_path.exists():
            sys.exit("no results yet")
        rows = [json.loads(l) for l in open(results_path)]
        report(rows)


if __name__ == "__main__":
    main()
