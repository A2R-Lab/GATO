#!/usr/bin/env python3
"""Autotune the linsys policy for a (plant, N, task) workload.

Probes the workload once under pure pcg (per-solve pred_err + solve ms) and
once under pure bdsv (flat cost), fits the pred_err threshold where predicted
pcg cost crosses the bdsv cost (gato.linsys_autotune.fit_tau), decides
{pure pcg | pure bdsv | auto@tau} from the measured warm-startedness
distribution, optionally validates the winner, and persists the entry to the
tuning table that MPCController(task_tag=...) resolves at construction.

The probe/validate legs are TIMING — QUIET BOX ONLY (the script refuses to
run when other compute pids hold the GPU; --allow-busy overrides for
plumbing smoke tests, never for numbers you keep).

Tasks: the wired task is the disturbance-rich fig8 (same rig as the 08-12
CDF study, examples/benchmarks/_linsys_probe.py); --kick-every scales how
often the state is kicked (larger = warmer workload). Rerun go2 through this
under a real gait once fc-on-feet lands — the w36 "bdsv everywhere" verdict
came from a forced-iterations cold protocol.

Example:
    tools/autotune_linsys.py --plant iiwa14 --N 64 --task-tag fig8
    python -c "from gato.linsys_autotune import lookup; print(lookup('iiwa14', 64, 'fig8'))"
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "benchmarks"))


def pct(t):
    t = np.asarray(t, dtype=float)
    return {k: float(np.percentile(t, q)) for k, q in
            (("p50", 50), ("p90", 90), ("p99", 99))} | \
           {"max": float(t.max()), "mean": float(t.mean())}


def line(name, t, extra=""):
    s = pct(t)
    return (f"{name:<14} p50 {s['p50']:6.3f}  p90 {s['p90']:6.3f}  "
            f"p99 {s['p99']:6.3f}  max {s['max']:6.3f}  mean {s['mean']:6.3f} ms{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plant", default="indy7")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--task-tag", default="fig8")
    ap.add_argument("--sim-time", type=float, default=6.0)
    ap.add_argument("--kick-every", type=int, default=25,
                    help="kick period in MPC steps (larger = warmer workload)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the auto@tau validation arm")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; do not write the tuning table")
    ap.add_argument("--tuning-path", default=None,
                    help="override the table location (default: "
                         "python/gato/linsys_tuning.json or $GATO_LINSYS_TUNING)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run under GPU contention (plumbing smoke ONLY — "
                         "never persist numbers from a busy box)")
    args = ap.parse_args()

    from _linsys_probe import gpu_busy, run_arm
    from gato.linsys_autotune import decide_policy, fit_tau, save_tuning

    busy = gpu_busy()
    if busy and not args.allow_busy:
        sys.exit(f"REFUSING to time: GPU busy ({busy})")
    if busy:
        print(f"WARNING: probing under contention (pids: {busy}) — do NOT "
              "persist these numbers", flush=True)

    probe_kw = dict(plant=args.plant, N=args.N, sim_time=args.sim_time,
                    kick_every=args.kick_every, seed=args.seed)
    pcg = run_arm("pcg", **probe_kw)
    print(line("pcg", pcg["solve_ms"], f"  track {pcg['track_mean']:.4f}"))
    bdsv = run_arm("bdsv", **probe_kw)
    print(line("bdsv", bdsv["solve_ms"], f"  track {bdsv['track_mean']:.4f}"))
    bdsv_ms = float(np.median(bdsv["solve_ms"]))

    tau, diag = fit_tau(pcg["pred_err"], pcg["solve_ms"], bdsv_ms,
                        n_bins=args.n_bins)
    decision = decide_policy(pcg["pred_err"], tau)
    print(f"fit: bdsv_ms {bdsv_ms:.3f}  tau* {tau!r}  "
          f"cold_frac {decision['cold_frac']:.3f}  -> policy {decision['policy']}"
          + (f"@{decision['tau']:.3f}" if decision["policy"] == "auto" else ""))

    validation = None
    if decision["policy"] == "auto" and not args.no_validate:
        auto = run_arm("auto", decision["tau"], **probe_kw)
        cold = f"  cold {100 * np.mean(auto['cold']):.0f}%" if auto["cold"] else ""
        print(line(f"auto@{decision['tau']:.3f}", auto["solve_ms"],
                   f"  track {auto['track_mean']:.4f}{cold}"))
        validation = pct(auto["solve_ms"])

    if args.dry_run:
        print("dry run — tuning table not written")
        return
    if busy:
        sys.exit("REFUSING to persist numbers probed under contention "
                 "(rerun on a quiet box)")
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    entry = {
        "policy": decision["policy"],
        "tau": decision["tau"],
        "provenance": {
            "date": datetime.date.today().isoformat(),
            "sha": sha,
            "cold_frac": decision["cold_frac"],
            "probe": {"sim_time": args.sim_time, "kick_every": args.kick_every,
                      "seed": args.seed, "steps": pcg["steps"]},
            "pcg": pct(pcg["solve_ms"]),
            "bdsv_ms": bdsv_ms,
            "auto_validation": validation,
            "fit": diag,
        },
    }
    p = save_tuning(args.plant, args.N, args.task_tag, entry,
                    path=args.tuning_path)
    print(f"wrote {p}  key {args.plant}|N{args.N}|{args.task_tag}")
    print(json.dumps({k: entry[k] for k in ("policy", "tau")}))


if __name__ == "__main__":
    main()
