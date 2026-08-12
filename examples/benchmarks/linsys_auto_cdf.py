#!/usr/bin/env python3
"""Solve-time CDFs for the per-solve linsys policy (pcg / bdsv / bdsv_first /
auto@tau) under a disturbance-rich MPC workload.

Question (user, 2026-08-12): does warm-startedness-based per-solve selection
(linsys="auto": pred_err <= tau -> pcg, else bdsv_first) win the solve-time
CDF by capturing warm pcg's fast left edge AND bdsv's flat cold tail?
History: the 07-10 session (bdsv_timing_session.py --mpc) found auto matched
pcg but didn't beat p95 — but its protocol only ever kicked q by N(0,0.05)
and recorded p50/p95, so the cold tail was never populated or plotted.
Verdict (08-12, quiet box): YES — auto@0.08 matches the matched-step oracle
tail on indy7 and beats pure pcg at EVERY percentile incl. mean on iiwa14;
the wired controller default is auto@0.1 since 08-12.

Protocol: fig8 MPC, N=64 B=1, fixed pacing (deterministic), same seeded kick
schedule for every arm — see _linsys_probe.run_arm (shared with
tools/autotune_linsys.py). Per-step traces saved (solve ms, pred_err, pcg
iters, auto's cold picks), CDF plot + stats table emitted. A matched-step
min(pcg, bdsv) "oracle" curve shows the selection headroom. QUIET BOX ONLY.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _linsys_probe import URDFS, gpu_busy, run_arm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples" / "benchmarks" / "data" / "linsys_auto_cdf"
N, DT, SIM_TIME = 64, 0.01, 6.0
KICK_EVERY = 25
TAUS = [0.08, 0.17, 0.35]


def stats_line(name, t):
    t = np.asarray(t)
    return (f"{name:<14} p50 {np.percentile(t,50):6.3f}  p90 {np.percentile(t,90):6.3f}  "
            f"p99 {np.percentile(t,99):6.3f}  max {t.max():6.3f}  mean {t.mean():6.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="indy7", choices=list(URDFS))
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if other GPU compute pids exist (results noted noisy)")
    args = ap.parse_args()

    busy = gpu_busy()
    if busy and not args.allow_busy:
        sys.exit(f"REFUSING to time: GPU busy ({busy})")
    if busy:
        print(f"WARNING: timing under contention (pids: {busy}) — noisy numbers")
    OUT.mkdir(parents=True, exist_ok=True)

    arms = [("pcg", 0.0), ("bdsv", 0.0), ("bdsv_first", 0.0)] + \
           [("auto", t) for t in TAUS]
    rows = []
    for mode, tau in arms:
        r = run_arm(mode, tau, plant=args.plant, N=N, dt=DT, sim_time=SIM_TIME,
                    kick_every=KICK_EVERY)
        rows.append(r)
        tag = f"auto@{tau}" if mode == "auto" else mode
        extra = ""
        if r["cold"]:
            extra = f"  cold {100*np.mean(r['cold']):.0f}%"
        print(stats_line(tag, r["solve_ms"]) +
              f"  track {r['track_mean']:.4f}{extra}", flush=True)

    suffix = "" if args.plant == "indy7" else f"_{args.plant}"
    (OUT / f"cdf_traces{suffix}.json").write_text(json.dumps(rows))

    # matched-step oracle: min(pcg, bdsv) at each step index (same kick schedule;
    # trajectories diverge slightly after kicks, so this is an approximation)
    p = np.asarray(rows[0]["solve_ms"]); d = np.asarray(rows[1]["solve_ms"])
    n = min(len(p), len(d))
    oracle = np.minimum(p[:n], d[:n])
    print(stats_line("oracle(min)", oracle))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        t = np.sort(r["solve_ms"])
        tag = f"auto τ={r['tau']}" if r["mode"] == "auto" else r["mode"]
        ax.plot(t, np.arange(1, len(t) + 1) / len(t), label=tag,
                lw=2 if r["mode"] == "auto" else 1.4)
    ax.plot(np.sort(oracle), np.arange(1, n + 1) / n, "k--", lw=1,
            label="oracle min(pcg,bdsv)")
    ax.set_xscale("log")
    ax.set_xlabel("solve time [ms]"); ax.set_ylabel("CDF")
    ax.set_title(f"{args.plant} fig8 MPC N={N} B=1, mixed kicks every {KICK_EVERY} steps")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"linsys_auto_cdf{suffix}.png", dpi=140)
    print(f"wrote cdf_traces{suffix}.json + linsys_auto_cdf{suffix}.png")


if __name__ == "__main__":
    main()
