#!/usr/bin/env python3
"""Merit launch timing: fused versus one block per candidate, and the block size.

Runs only in a user-assigned quiet window (--quotable); otherwise the rows are
marked not quotable. For every (plant, N, batch, block size, launch) a child
process (the launch override GATO_MERIT_FUSED is read once per process) solves
the reach problem warm-started from its own previous solution, max_sqp_iters
iterations per solve, and reports the per-stage device times from
SolverStats.stage_times_us: the merit column is the quantity of interest, the
whole solve the context. Block sizes other than the default use modules
registered as <plant>_mt<threads> (built with GATO_MERIT_THREADS=<threads>).
The MPS condition is set outside: start nvidia-cuda-mps-control and export
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE before running with --condition mps<pct>.

    python examples/benchmarks/merit_fusion_timing.py --out rows.jsonl --condition 5090 --quotable
    python examples/benchmarks/merit_fusion_timing.py --report rows.jsonl
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _bench import git_provenance

CHILD = r'''
import json, os, sys, time
import numpy as np
import gato
from gato.interface import STAGE_NAMES
plant, name, N, B, iters, warmup, solves = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
s = gato.BSQP(str(gato.robot_urdf(plant)), batch_size=B, N=N, dt=0.01, plant_type=name,
              params=gato.SolverParams(max_sqp_iters=iters, linsys="pcg"))
x = np.zeros((B, s.nx), np.float32)
goals = np.zeros((B, N * 6), np.float32)
goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
res = s.solve(x, goals)
for _ in range(warmup):
    res = s.solve(x, goals, xu_warm=res.xu)
stage = np.zeros((solves, 5)); wall = np.zeros(solves); init = np.zeros(solves); fused = None
for i in range(solves):
    res = s.solve(x, goals, xu_warm=res.xu)
    st = res.stats
    stage[i] = st.stage_times_us.sum(axis=0); wall[i] = res.solve_time_us; init[i] = st.initial_merit_time_us
    fused = bool(st.merit_fused)
    assert int(st.sqp_iters.max()) == iters
def q(v): return dict(p50=float(np.median(v)), p90=float(np.percentile(v, 90)), mean=float(v.mean()), max=float(v.max()))
print(json.dumps(dict(merit_fused=fused, solve_us=q(wall), initial_merit_us=q(init),
                      stages_us={n: q(stage[:, k]) for k, n in enumerate(STAGE_NAMES)}, n=solves)))
'''


def box_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception as exc:  # the row still records the failure
        return dict(error=f"{type(exc).__name__}: {exc}")
    return dict(gpu=out, other_processes=[l for l in apps.splitlines() if l.strip()],
                mps_percentage=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))


def collect(args):
    prov = git_provenance()
    if args.quotable and prov["dirty"]:
        raise SystemExit("a quotable run needs a clean committed checkout")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plants, Ns, Bs = args.plants.split(","), [int(n) for n in args.Ns.split(",")], [int(b) for b in args.Bs.split(",")]
    threads = args.threads.split(",")
    with out.open("a") as fh:
        for plant, N, B, th, fused in itertools.product(plants, Ns, Bs, threads, (0, 1)):
            name = plant if th == "default" else f"{plant}_mt{th}"
            env = dict(os.environ, GATO_MERIT_FUSED=str(fused))
            env.pop("GATO_MERIT_THREADS", None)
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, "-c", CHILD, plant, name, str(N), str(B), str(args.iters),
                                str(args.warmup), str(args.solves)], env=env, capture_output=True, text=True)
            row = dict(condition=args.condition, plant=plant, N=N, B=B, threads=th, fused_requested=fused,
                       iters=args.iters, warmup=args.warmup, solves=args.solves, sha=prov["short"],
                       dirty=prov["dirty"], date=prov["date"], quotable_times=bool(args.quotable),
                       box=box_state(), wall_s=round(time.perf_counter() - t0, 1))
            if r.returncode == 0:
                row.update(status="ok", result=json.loads(r.stdout.strip().splitlines()[-1]))
                m = row["result"]["stages_us"]["merit"]
                print(f"{args.condition} {plant} N{N} B{B:2d} threads {th:>7} fused {fused} -> merit p50 {m['p50']:8.1f} us "
                      f"solve p50 {row['result']['solve_us']['p50']:8.1f} us (used fused={row['result']['merit_fused']})", flush=True)
            else:
                row.update(status="error", error=r.stderr[-2000:])
                print(f"{args.condition} {plant} N{N} B{B} threads {th} fused {fused} -> ERROR", flush=True)
            fh.write(json.dumps(row) + "\n")
            fh.flush()


def report(paths):
    rows = [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)} rows ok, {len(rows) - len(ok)} errors; quotable: {sorted({r['quotable_times'] for r in rows})}; "
          f"source: {sorted({r['sha'] for r in rows})}")
    print("Merit device time per solve, median microseconds (p90 in brackets), and the whole solve's median; "
          "ratio = fused / one block per candidate.")
    for cond in sorted({r["condition"] for r in ok}):
        print(f"\n### condition {cond}\n")
        print("| plant | N | B | threads | merit unfused | merit fused | ratio | solve unfused | solve fused |")
        print("|---|---|---|---|---|---|---|---|---|")
        keys = sorted({(r["plant"], r["N"], r["B"], r["threads"]) for r in ok if r["condition"] == cond},
                      key=lambda k: (k[0], k[1], k[2], {"default": 0}.get(k[3], int(k[3]) if k[3].isdigit() else 9)))
        for plant, N, B, th in keys:
            g = {r["fused_requested"]: r["result"] for r in ok
                 if (r["condition"], r["plant"], r["N"], r["B"], r["threads"]) == (cond, plant, N, B, th)}
            if 0 not in g or 1 not in g:
                continue
            mu, mf = g[0]["stages_us"]["merit"], g[1]["stages_us"]["merit"]
            print(f"| {plant} | {N} | {B} | {th} | {mu['p50']:.1f} [{mu['p90']:.1f}] | {mf['p50']:.1f} [{mf['p90']:.1f}] "
                  f"| {mf['p50'] / max(mu['p50'], 1e-9):.2f} | {g[0]['solve_us']['p50']:.1f} | {g[1]['solve_us']['p50']:.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--condition", default="5090")
    ap.add_argument("--plants", default="indy7,iiwa14")
    ap.add_argument("--Ns", default="32,64")
    ap.add_argument("--Bs", default="1,8,16")
    ap.add_argument("--threads", default="default,64,128")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--solves", type=int, default=200)
    ap.add_argument("--quotable", action="store_true", help="the run is inside a user-assigned quiet window")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    if not args.out:
        ap.error("--out or --report required")
    collect(args)


if __name__ == "__main__":
    main()
