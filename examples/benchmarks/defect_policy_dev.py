#!/usr/bin/env python3
"""Dynamics-defect policies on the development scenarios (Wave D), not a benchmark.

Reuses limits_holdout's tasks, controller configurations (MECHS) and episode
runner, and torque_clip_ab's actuator audit (pass-through or clipped to the URDF
effort limits at every 1 ms substep). For each policy, scenario and actuator
mode it records the per-tick dynamics defect, the SQP iterations run, the
substep torque and velocity peaks and the task outcome. Correctness only, one
worker; times are never quotable.

    python examples/benchmarks/defect_policy_dev.py --out dev-defect.jsonl \\
        --mechs admm-defect-tele,admm-sqp4,admm-sqp4-defect1e-3
    python examples/benchmarks/defect_policy_dev.py --report dev-defect.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _bench import git_provenance, pin_model
from limits_holdout import MECHS, run_episode, scenarios
from torque_clip_ab import ActuationAudit


def collect(args):
    source = git_provenance()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mechs = args.mechs.split(",")
    for m in mechs:
        if m not in MECHS:
            raise SystemExit(f"unknown mechanism {m!r}; see limits_holdout.MECHS")
    clips = {"both": (False, True), "0": (False,), "1": (True,)}[args.clip]
    wanted = {int(s) for s in args.scenarios.split(",")} if args.scenarios else None
    with out.open("a") as fh:
        for plant in args.plants.split(","):
            model = pin_model(plant)
            for task in args.tasks.split(","):
                for sc in scenarios(model, plant, task, args.n, args.seed):
                    if wanted is not None and sc["id"] not in wanted:
                        continue
                    for mech in mechs:
                        for clip in clips:
                            row = dict(set="dev", seed=args.seed, plant=plant, task=task, scenario=sc["id"],
                                       scenario_def={k: v for k, v in sc.items() if k not in ("id", "plant", "task")},
                                       mech=mech, clip=clip, N=args.N,
                                       sha=source["short"] + ("+dirty" if source["dirty"] else ""),
                                       date=source["date"], quotable_times=False)
                            try:
                                r = run_episode(model, plant, args.N, sc, mech, audit=ActuationAudit(model, clip))
                                for key in ("wall_s", "solve_ms"):
                                    r.pop(key, None)
                                row.update(status="ok", result=r)
                            except Exception as exc:  # keep the failure in the collection
                                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                            r = row.get("result", {})
                            d = r.get("defect") or {}
                            print(f"{plant} {task} #{sc['id']:3d} {mech:<22} {'clip' if clip else 'free'} {row['status']:5s} "
                                  f"reached {r.get('reached', '-')} iters {r.get('iters_mean', float('nan')):.2f} "
                                  f"defect p50 {d.get('p50', float('nan')):.2e} max {d.get('max', float('nan')):.2e} "
                                  f"qd {r.get('substeps', {}).get('qd_ratio', float('nan')):.3f} "
                                  f"u_applied {r.get('substeps', {}).get('applied_u_ratio', float('nan')):.3f}", flush=True)


def report(paths):
    rows = [json.loads(ln) for p in paths for ln in Path(p).read_text().splitlines() if ln.strip()]
    print("Correctness only. Per (plant, task, actuator) and policy: episodes, goals reached (mean of 3), "
          "time to all goals p50 (simulated s), SQP iterations per tick mean / max, defect p50 / p90 / max, "
          "substep qd/limit max, applied u/limit max, commanded u/limit max, episodes with clipping.")
    keys = sorted({(r["plant"], r["task"], r["clip"]) for r in rows}, key=str)
    for key in keys:
        sub = [r for r in rows if (r["plant"], r["task"], r["clip"]) == key and r["status"] == "ok"]
        errs = [r for r in rows if (r["plant"], r["task"], r["clip"]) == key and r["status"] != "ok"]
        print(f"\n### {key[0]} {key[1]} {'clipped' if key[2] else 'unclipped'}  ({len(errs)} errors)\n")
        print("| policy | n | reached | t_all p50 | iters mean / max | defect p50 / p90 / max | qd max | u applied max | u cmd max | clipped eps |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for mech in [m for m in MECHS if any(r["mech"] == m for r in sub)]:
            g = [r["result"] for r in sub if r["mech"] == mech]
            defs = [r["defect"] for r in g if r.get("defect")]
            t_all = [r["time_to_all"] for r in g if r.get("time_to_all") is not None]
            line = (f"| {mech} | {len(g)} | {np.mean([r.get('reached', 0) for r in g]):.2f} "
                    f"| {np.median(t_all):.2f} " if t_all else f"| {mech} | {len(g)} | {np.mean([r.get('reached', 0) for r in g]):.2f} | - ")
            line += f"| {np.mean([r['iters_mean'] for r in g]):.2f} / {max(r['iters_max'] for r in g)} "
            if defs:
                line += (f"| {np.median([d['p50'] for d in defs]):.2e} / {np.median([d['p90'] for d in defs]):.2e} "
                         f"/ {max(d['max'] for d in defs):.2e} ")
            else:
                line += "| - "
            ss = [r["substeps"] for r in g]
            line += (f"| {max(s['qd_ratio'] for s in ss):.3f} | {max(s['applied_u_ratio'] for s in ss):.3f} "
                     f"| {max(s['commanded_u_ratio'] for s in ss):.3f} | {sum(s['clipped_samples'] > 0 for s in ss)} |")
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--plants", default="fr3,gen3,ur5e")
    ap.add_argument("--tasks", default="pickplace")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--mechs", default="admm-defect-tele")
    ap.add_argument("--clip", default="both", choices=["both", "0", "1"])
    ap.add_argument("--scenarios", default=None, help="comma list of scenario ids (default: all n)")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    if not args.out:
        ap.error("--out or --report required")
    collect(args)


if __name__ == "__main__":
    main()
