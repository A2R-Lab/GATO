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

Usage:
  python examples/benchmarks/constraint_eval.py --run            # all cells
  python examples/benchmarks/constraint_eval.py --run --quick    # smoke subset
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
    "rb": dict(mu=1e-2, delta=0.1),
    "admm": dict(rho=0.01, iters=10),
    "admm_m": dict(rho=0.01, iters=10, merit=True),     # + set_admm_merit (R1 ablation)
    "al": dict(rho=100.0),
    "admm_ee": dict(rho=0.01, iters=10, ee_rho=10.0),   # reach only
    "admm_m_ee": dict(rho=0.01, iters=10, ee_rho=10.0, merit=True),  # reach only
    "al_ee": dict(rho=100.0, ee_rho=100.0),             # reach only
}
PROBLEMS = ["fig8", "reach", "pickplace", "swing_heavy"]
EE_ONLY_PROBLEMS = {"admm_ee": ["reach"], "admm_m_ee": ["reach"], "al_ee": ["reach"]}
SWING_TORQUE_SCALE = 0.3     # heavy-payload emulation: torque box fraction
PICKPLACE_SEG_S = 1.2        # seconds per waypoint
PICKPLACE_GOALS = [          # _common.PICKPLACE_DEFAULT_GOALS (first 4)
    [0.5, -0.1865, 0.5], [0.5, 0.5, 0.2], [0.3, 0.3, 0.8], [0.6, -0.5, 0.2]]
SIM_S = {"fig8": 4.0, "reach": 2.0, "pickplace": 4.8, "swing_heavy": 3.0}


def cell_name(plant, mech, problem):
    return f"{plant}-{mech}-{problem}"


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

    if problem == "fig8":
        traj = figure8(DT, **FIG8_DEFAULT_PARAMS)

        def goal(step):
            off = 6 * step
            return traj[off:off + 6 * N_KNOTS]
        return x0, goal, n_steps, None, None

    if problem in ("reach", "swing_heavy"):
        # goal = solver-frame EE at a displaced (guaranteed-reachable) config
        dq_goal = np.array([1.2, 0.7, -0.7, 0.5, 0.4, 0.3, 0.3])[:s.nq]
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


def enable_mechanism(s, mech, ee_target):
    p = MECHANISMS[mech]
    if mech == "baseline":
        s.enable_limit_telemetry()
    elif mech == "rb":
        s.enable_limit_barrier(mu=p["mu"], delta=p["delta"])
    elif mech.startswith("admm"):
        s.enable_limit_admm(rho=p["rho"], iters=p["iters"])
        if p.get("merit"):
            s.set_admm_merit(True)
    elif mech in ("al", "al_ee"):
        s.enable_limit_al(rho=p["rho"])
    if mech.endswith("_ee"):
        assert ee_target is not None
        s.enable_ee_terminal_equality(ee_target.astype(np.float32), rho=p["ee_rho"])


# ---- one cell: fixed-pacing closed-loop episode ----------------------------

def run_cell(name):
    import pinocchio  # noqa: F401  (rk4 needs the model)
    import gato
    from gato.common import rk4

    plant, mech, problem = name.split("-")
    urdf = str(PLANTS[plant]["urdf"])
    # rho=1e-3: f32 bdsv (forced by AL mode, factored by ADMM's inner loop)
    # produces garbage steps on an UNREGULARIZED Schur system (R1 measured:
    # closed-loop cascade at the interface default rho=0.0)
    s = gato.BSQP(model_path=urdf, batch_size=1, N=N_KNOTS, dt=DT,
                  plant_type=plant, rho=1e-3)

    x0, goal_of, n_steps, ee_target, apply_bounds = build_problem(s, plant, problem)
    enable_mechanism(s, mech, ee_target)
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
    out = dict(
        cell=name, plant=plant, mechanism=mech, problem=problem,
        params=MECHANISMS[mech], n_steps=n_steps,
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
    for (plant, prob), rs in sorted(by_pp.items()):
        rs.sort(key=lambda r: order.get(r["mechanism"], 99))
        base = next((r for r in rs if r["mechanism"] == "baseline"), None)
        print(f"\n### {plant} / {prob}  (steps={rs[0]['n_steps']})")
        print("| mech | track mean m | final m | soln viol max | u-exceed max (frac) | iters | reg vs base |")
        print("|---|---|---|---|---|---|---|")
        for r in rs:
            reg = (f"{r['track_mean'] / base['track_mean'] - 1:+.1%}"
                   if base and base is not r and base["track_mean"] > 0 else "—")
            print(f"| {r['mechanism']} | {r['track_mean']:.4f} | {r['track_final']:.4f} "
                  f"| {r['viol_soln_max']:.2e} | {r['u_exceed_max']:.2e} ({r['u_exceed_frac']:.0%}) "
                  f"| {r['iters_mean']:.1f} | {reg} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--only", help="substring filter on cell names")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    results_path = DATA / "results.jsonl"

    if args.cell:
        print(json.dumps(run_cell(args.cell)))
        return

    if args.run:
        gpu_load_note()
        prov = provenance()
        cells = all_cells(args.quick)
        if args.only:
            cells = [c for c in cells if args.only in c]
        with open(results_path, "a") as f:
            for cell in cells:
                r = subprocess.run([sys.executable, __file__, "--cell", cell],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[FAIL] {cell}\n{r.stderr[-2000:]}", file=sys.stderr)
                    continue
                rec = json.loads(r.stdout.strip().splitlines()[-1])
                rec["provenance"] = prov
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[ok] {cell}: track_mean={rec['track_mean']:.4f}m "
                      f"viol_max={rec['viol_soln_max']:.2e} "
                      f"u_exceed={rec['u_exceed_max']:.2e}")

    if args.report or args.run:
        if not results_path.exists():
            sys.exit("no results yet")
        rows = [json.loads(l) for l in open(results_path)]
        report(rows)


if __name__ == "__main__":
    main()
