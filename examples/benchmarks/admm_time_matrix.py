#!/usr/bin/env python
"""W2a timing legs (plan H2.4 + H1), QUIET BOX ONLY — the driver behind
run_timing_w2a.sh. Everything here is TIMING: the script refuses to run while
other compute processes hold the GPU (--allow-busy downgrades that to a
warning for plumbing smoke tests; numbers measured under contention are never
kept — rows are tagged `quotable: false`).

Legs (--leg, default all):

  matrix   the ADMM inner-loop time matrix on the closed-loop figure-8 with the
           velocity box tightened to ±1 rad/s (example 09's setting; rho 0.01,
           cap 10, residual exit at eps_rel 1e-2, warm-startedness threshold
           0.1), {plant} x {N} x {no kicks | q kicked every 25 ticks}:
             free          no row groups (the paper setting)          -- the unconstrained step
             fixed10-pcg   fixed 10 inner iterations, warm PCG        -- the Oct 12 default
             fixed10-fac   fixed 10, factor once + back-solves
             exit-pcg      residual exit, warm PCG
             exit-fac      residual exit, factor once + back-solves
             exit-auto     residual exit, inner arm chosen per tick from pred_err
             exit-fac-ce0  exit-fac with the host break off (device gating only)
             exit-fac-ce2  exit-fac with the host break every 2nd iteration
             exit3-fac     factor arm at eps_rel 1e-3
             default       the shipped controller default (per-box rho, exit, solver by horizon)
           Per cell: per-solve wall (ms) percentiles, inner iterations and inner
           linsys iterations per SQP iteration, max row violation, tracking.
           Fixed pacing (one solve per 10 ms control period, pinocchio plant).
  fit      tools/autotune_linsys.py --admm per (plant, N) in the matrix's cell
           configuration (same cap, rho, tolerance): fits the pred_err
           threshold where the PCG arm's cost crosses the factor arm's and
           PERSISTS admm_policy/admm_tau into python/gato/linsys_tuning.json
           (gitignored; MPCController(task_tag="fig8") resolves it).
  deadline H1's controller-call-boundary table: step() wall time per tick with
           no budget and with budgets of 100 / 80 / 60 % of the unbudgeted
           median (max_sqp_iters 8, like example 11), iterations per tick,
           budget hits, overruns.

Results: one JSON row per cell appended to --out (default
examples/benchmarks/data/admm_time_matrix/results.jsonl) with git provenance,
the GPU name and the quiet-box verdict; a table is printed at the end of each
leg. --quick shrinks everything (plumbing smoke; numbers are garbage).

  python examples/benchmarks/admm_time_matrix.py                 # all legs
  python examples/benchmarks/admm_time_matrix.py --leg matrix --plants indy7 --Ns 64
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _bench import REPO, git_provenance, gpu_info, pct, pct_line, require_quiet_gpu, run_kicked_arm

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "admm_time_matrix" / "results.jsonl"
QD_CAP, RHO, ITERS, EPS_REL, TAU = 1.0, 0.01, 10, 1e-2, 0.1
ADMM_CELLS = {
    "free": None,
    "fixed10-pcg": dict(linsys="pcg"),
    "fixed10-fac": dict(linsys="bdsv_factor"),
    "exit-pcg": dict(linsys="pcg", eps_rel=EPS_REL),
    "exit-fac": dict(linsys="bdsv_factor", eps_rel=EPS_REL),
    "exit-auto": dict(linsys="auto", eps_rel=EPS_REL, tau=TAU),
    # added October 16 after the first window: the host-sync cost of the exit
    # check (check_every 0 = device gating only, 2 = every other iteration) and
    # the tighter tolerance on the arm that can reach it
    "exit-fac-ce0": dict(linsys="bdsv_factor", eps_rel=EPS_REL, check_every=0),
    "exit-fac-ce2": dict(linsys="bdsv_factor", eps_rel=EPS_REL, check_every=2),
    "exit3-fac": dict(linsys="bdsv_factor", eps_rel=1e-3),
    # the shipped controller default (W2b: per-box rho 0.1/0.03/1e-5, cap 10, eps_rel 1e-2,
    # inner solver by horizon) on the same 1.0 rad/s velocity cap: the wall time of what ships
    "default": "default",
}


def _row(leg, quotable, **fields):
    prov = git_provenance()
    return dict(leg=leg, quotable=quotable, date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                sha=prov["short"] + ("+dirty" if prov["dirty"] else ""), gpu=gpu_info(), **fields)


def _append(out, row):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def leg_matrix(args, quotable, out):
    print(f"== matrix: plants {args.plants} Ns {args.Ns} sim {args.sim_time}s  quotable={quotable}", flush=True)
    rows = []
    for plant in args.plants:
        for N in args.Ns:
            for kick_every in (0, 25):
                for cell, cfg in ADMM_CELLS.items():
                    admm = cfg if cfg in (None, "default") else dict(rho=RHO, iters=ITERS, qd_cap=QD_CAP, **cfg)
                    tr = run_kicked_arm("pcg", TAU, plant=plant, N=N, sim_time=args.sim_time,
                                        kick_every=kick_every, admm=admm)
                    name = f"{plant}-N{N}-{'kick' if kick_every else 'quiet'}-{cell}"
                    extra = f"  track {tr['track_mean']:.4f}"
                    row = dict(plant=plant, N=N, kick_every=kick_every, cell=cell, steps=tr["steps"],
                               solve_ms=pct(tr["solve_ms"]), track_mean=tr["track_mean"],
                               track_max=tr["track_max"], config=admm)
                    if admm is not None:
                        it = np.asarray(tr["admm_iters"], float)
                        lin = np.asarray(tr["admm_linsys_iters"], float)
                        sqp = np.maximum(np.asarray(tr["sqp_iters"], float), 1.0)
                        row.update(inner_it_per_solve=float(it.mean()), inner_lin_per_solve=float(lin.mean()),
                                   inner_it_per_sqp_it=float((it / sqp).mean()), sqp_it_per_solve=float(sqp.mean()),
                                   row_viol_max=float(max(tr["row_viol"])),
                                   factor_frac=float(np.mean([a == "bdsv_factor" for a in tr["admm_linsys"]])))
                        extra += (f"  inner/SQP it {(it / sqp).mean():4.1f}  lin/solve {lin.mean():6.1f}"
                                  f"  viol {max(tr['row_viol']):.1e}")
                        if cell == "exit-auto":
                            extra += f"  factor {100 * row['factor_frac']:.0f}%"
                    print(pct_line(name, tr["solve_ms"], extra), flush=True)
                    _append(out, _row("matrix", quotable, **row))
                    rows.append(row)
    return rows


def leg_fit(args, quotable, out):
    print(f"== fit: tools/autotune_linsys.py --admm per (plant, N)  quotable={quotable}", flush=True)
    rc_all = 0
    for plant in args.plants:
        for N in args.Ns:
            cmd = [sys.executable, str(REPO / "tools" / "autotune_linsys.py"), "--plant", plant, "--N", str(N),
                   "--task-tag", "fig8", "--admm", "--linsys", "pcg", "--sim-time", str(args.sim_time),
                   "--admm-rho", str(RHO), "--admm-iters", str(ITERS), "--admm-eps-abs", "0",
                   "--admm-eps-rel", str(EPS_REL), "--admm-qd-cap", str(QD_CAP)]   # the matrix's cell config
            if not quotable:
                cmd += ["--allow-busy", "--dry-run"]
            print("$ " + " ".join(cmd), flush=True)
            res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            print(res.stdout, flush=True)
            if res.returncode != 0:
                print(res.stderr, file=sys.stderr, flush=True)
                rc_all = res.returncode
            verdict = [ln for ln in res.stdout.splitlines() if ln.startswith("fit:")]
            _append(out, _row("fit", quotable, plant=plant, N=N, rc=res.returncode,
                              verdict=verdict[-1] if verdict else None))
    return rc_all


def leg_deadline(args, quotable, out):
    """H1: step() wall per tick with and without a budget (example 11's loop on
    both arms' figure-8 tasks through MPC_GATO.run_mpc_fig8)."""
    from _bench import fig8_task, pin_model, urdf_path
    from gato.mpc_gato import MPC_GATO
    print(f"== deadline: plants {args.plants} N64  quotable={quotable}", flush=True)
    N, DT, ITERS_SQP = 64, 0.01, 8

    def run(plant, budget_s):
        mpc = MPC_GATO(pin_model(plant), model_path=urdf_path(plant), N=N, dt=DT, batch_size=1,
                       plant_type=plant, params=dict(max_sqp_iters=ITERS_SQP), linsys="pcg",
                       limits="off")   # H1's table is the unconstrained step
        walls, iters, hits = [], [], []
        orig = mpc.controller.step

        def step(x, g, **kw):
            r = orig(x, g, budget_s=budget_s, **kw)
            walls.append(r.wall_s); iters.append(r.iterations); hits.append(r.budget_hit)
            return r
        mpc.controller.step = step
        q0, fig8 = fig8_task(plant, N, DT)
        xs = np.hstack((q0, np.zeros(mpc.solver.nx - mpc.solver.nq)))
        _, stats = mpc.run_mpc_fig8(xs, fig8, sim_dt=0.001, sim_time=args.sim_time, pace_by_solve_time=False)
        w = 1e3 * np.asarray(walls)
        over = (w - 1e3 * budget_s) if budget_s else np.zeros_like(w)
        return dict(plant=plant, N=N, budget_ms=None if budget_s is None else 1e3 * budget_s,
                    steps=len(w), wall_ms=pct(w), iters_mean=float(np.mean(iters)),
                    iters_hist={str(k): int(v) for k, v in zip(*np.unique(iters, return_counts=True))},
                    budget_hit_frac=float(np.mean(hits)), over_frac=float((over > 0).mean()),
                    over_max_ms=float(max(over.max(), 0.0)),
                    track_mean=float(np.mean(stats["goal_distances"])))

    for plant in args.plants:
        base = run(plant, None)
        print(f"{plant} no budget: wall p50 {base['wall_ms']['p50']:.3f} p90 {base['wall_ms']['p90']:.3f}"
              f" max {base['wall_ms']['max']:.3f} ms  iters {base['iters_mean']:.2f}  ({base['steps']} ticks)"
              f"  track {base['track_mean']:.4f}", flush=True)
        _append(out, _row("deadline", quotable, **base))
        for frac in (1.0, 0.8, 0.6):
            b = frac * base["wall_ms"]["p50"] * 1e-3
            r = run(plant, b)
            print(f"{plant} budget {1e3 * b:.3f} ms ({100 * frac:.0f}%): wall p50 {r['wall_ms']['p50']:.3f} p90 {r['wall_ms']['p90']:.3f}"
                  f" max {r['wall_ms']['max']:.3f} ms  iters {r['iters_mean']:.2f}  hit {100 * r['budget_hit_frac']:.0f}%"
                  f"  over {100 * r['over_frac']:.0f}% by <= {r['over_max_ms']:.3f} ms  track {r['track_mean']:.4f}", flush=True)
            _append(out, _row("deadline", quotable, budget_frac=frac, **r))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leg", default="all", choices=["all", "matrix", "fit", "deadline"])
    ap.add_argument("--plants", default="indy7,iiwa14")
    ap.add_argument("--Ns", default="32,64")
    ap.add_argument("--sim-time", type=float, default=6.0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--quick", action="store_true", help="tiny subset, plumbing smoke; numbers are garbage")
    ap.add_argument("--allow-busy", action="store_true", help="run under contention (plumbing smoke only)")
    args = ap.parse_args()
    args.plants = args.plants.split(",")
    args.Ns = [int(n) for n in args.Ns.split(",")]
    if args.quick:
        args.plants, args.Ns, args.sim_time = args.plants[:1], args.Ns[:1], min(args.sim_time, 0.5)
    quotable = require_quiet_gpu(allow_busy=args.allow_busy) and not args.quick
    out = Path(args.out)
    print(f"results -> {out}  (quotable={quotable})", flush=True)
    rc = 0
    if args.leg in ("all", "matrix"):
        leg_matrix(args, quotable, out)
    if args.leg in ("all", "fit"):
        rc |= leg_fit(args, quotable, out)
    if args.leg in ("all", "deadline"):
        rc |= leg_deadline(args, quotable, out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
