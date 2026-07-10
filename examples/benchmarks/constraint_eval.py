#!/usr/bin/env python
"""Constraint-mechanism evaluation matrix — harness SKELETON (CL-0.5).

The standing harness behind the arc's evaluation rounds R1-R3
(docs/open-tasks/constraint_layer_locomotion_arc_plan_2026-07-10.md): every
mechanism-to-constraint binding is decided by experiment, and this driver is
where those experiments run. Follows the bdsv_timing_session.py conventions:
GPU-idle preflight, per-cell child processes, provenance JSONL, one report.

CL-0 exercises the two bindings that exist (soft baseline off/on):
  mechanisms: none (baseline) | barrier_relaxed(mu, delta)
CL-1 adds admm_interval + al_phr on the same axes; CL-2 adds cones/collision;
the axes lists below are the single place to extend.

Usage:
  python examples/benchmarks/constraint_eval.py --run          # all cells
  python examples/benchmarks/constraint_eval.py --report       # tables from data
  python examples/benchmarks/constraint_eval.py --cell <name>  # one cell (child mode)
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
DATA = ROOT / "examples" / "benchmarks" / "data" / "constraint_eval"

# ---- axes (extend here as bindings/problems land) -------------------------
PLANT = "indy7"
URDF = ROOT / "examples" / "indy7_description" / "indy7.urdf"
KNOTS = 32
MECHANISMS = {
    "baseline": {},                                        # soft prior off — reference
    "barrier_relaxed": {"mu": 1e-2, "delta": 0.1},          # CL-0 soft prior
    # "admm_interval": {...},   # CL-1
    # "al_phr": {...},          # CL-1
}
PROBLEMS = ["fig8"]            # CL-1: + reach, pick-place, heavy-payload swing
BATCHES = [1, 16]
SOLVES = 20                    # fixed-input repeated solves per cell
WARMUP = 3


def cell_name(mech, problem, batch):
    return f"{mech}-{problem}-B{batch}"


def all_cells():
    return [cell_name(m, p, b) for m, p, b in itertools.product(MECHANISMS, PROBLEMS, BATCHES)]


def gpu_idle_preflight():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    util, mem = (int(v) for v in out.strip().split(","))
    if util > 5 or mem > 2000:
        sys.exit(f"GPU not idle (util={util}%, mem={mem}MiB) — no timing on a busy box")


def provenance():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--ignore-submodules=untracked"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return dict(sha=sha, dirty=dirty, time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run_cell(name):
    """Child-process body: one (mechanism, problem, batch) cell."""
    import numpy as np
    import gato
    from gato.certificate import kkt_residuals
    from gato.config import INDY7_START_CONFIGS

    mech, problem, bstr = name.rsplit("-", 2)
    B = int(bstr[1:])
    params = MECHANISMS[mech]

    q0 = np.asarray(INDY7_START_CONFIGS["ready"], dtype=np.float32)
    x0 = np.concatenate([q0, np.zeros_like(q0)])
    rng = np.random.default_rng(20260710)
    X = np.tile(x0, (B, 1)) + rng.normal(0, 0.01, (B, x0.size)).astype(np.float32)
    # fig8: static EE goal per knot (placeholder reference — CL-1 wires the real
    # fig8/pick-place/swing generators from examples/paper-figures/_common.py)
    goals = np.zeros((B, KNOTS * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.35, 0.25, 0.5

    s = gato.BSQP(model_path=str(URDF), batch_size=B, N=KNOTS, dt=0.01, plant_type=PLANT)
    if mech == "barrier_relaxed":
        s.enable_limit_barrier(**params)
    else:
        s.enable_limit_telemetry()  # violation reporting on every cell

    times, res = [], None
    for i in range(WARMUP + SOLVES):
        t0 = time.perf_counter()
        res = s.solve(X, goals)
        if i >= WARMUP:
            times.append(1e6 * (time.perf_counter() - t0))

    groups = s.get_row_groups()
    cert = [kkt_residuals(groups, res.xu[b].astype(np.float64), res.nx, res.nu)
            for b in range(B)]
    times = np.asarray(times)
    return dict(
        cell=name, mechanism=mech, problem=problem, batch=B, params=params,
        solve_us_median=float(np.median(times)), solve_us_iqr=float(np.subtract(*np.percentile(times, [75, 25]))),
        sqp_iters=res.stats.sqp_iters.tolist(),
        primal_max=max(c["primal"] for c in cert),
        n_active=[c["n_active"] for c in cert],
        final_merit=res.stats.final_merit.tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    results_path = DATA / "results.jsonl"

    if args.cell:
        print(json.dumps(run_cell(args.cell)))
        return

    if args.run:
        gpu_idle_preflight()
        prov = provenance()
        with open(results_path, "a") as f:
            for cell in all_cells():
                r = subprocess.run([sys.executable, __file__, "--cell", cell],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[FAIL] {cell}\n{r.stderr[-2000:]}", file=sys.stderr)
                    continue
                rec = json.loads(r.stdout.strip().splitlines()[-1])
                rec["provenance"] = prov
                f.write(json.dumps(rec) + "\n")
                print(f"[ok] {cell}: {rec['solve_us_median']:.0f}us median, "
                      f"primal_max={rec['primal_max']:.2e}")

    if args.report or args.run:
        if not results_path.exists():
            sys.exit("no results yet")
        rows = [json.loads(l) for l in open(results_path)]
        latest = {r["cell"]: r for r in rows}  # last write per cell wins
        print(f"\n| cell | median us | IQR | primal max | sqp iters |")
        print("|---|---|---|---|---|")
        for name in all_cells():
            r = latest.get(name)
            if r is None:
                continue
            print(f"| {r['cell']} | {r['solve_us_median']:.0f} | {r['solve_us_iqr']:.0f} "
                  f"| {r['primal_max']:.2e} | {r['sqp_iters']} |")


if __name__ == "__main__":
    main()
