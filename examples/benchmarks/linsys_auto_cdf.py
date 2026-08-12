#!/usr/bin/env python3
"""Solve-time CDFs for the per-solve linsys policy (pcg / bdsv / bdsv_first /
auto@tau) under a disturbance-rich MPC workload.

Question (user, 2026-08-12): does warm-startedness-based per-solve selection
(linsys="auto": pred_err <= tau -> pcg, else bdsv_first) win the solve-time
CDF by capturing warm pcg's fast left edge AND bdsv's flat cold tail?
History: the 07-10 session (bdsv_timing_session.py --mpc) found auto matched
pcg but didn't beat p95 — but its protocol only ever kicked q by N(0,0.05)
and recorded p50/p95, so the cold tail was never populated or plotted.

Protocol here: indy7 fig8 MPC, N=64 B=1, fixed pacing (deterministic), same
seeded kick schedule for every arm — every 25 steps a kick cycling through
mild (q+=N(0,.03)) / medium (q+=N(0,.10)) / severe (q+=N(0,.20), qd+=N(0,.5)).
Per-step traces saved (solve ms, pred_err, pcg iters, auto's cold picks), CDF
plot + stats table emitted. A matched-step min(pcg, bdsv) "oracle" curve shows
the selection headroom. QUIET BOX ONLY.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "benchmarks"))
URDFS = {
    "indy7": str(ROOT / "examples" / "indy7_description" / "indy7.urdf"),
    # canonical iiwa14 (fig3-fair harness URDF, md5 eeb7d4ff — NOT GRiD/robot_assets)
    "iiwa14": str(ROOT / "examples" / "iiwa_description" / "iiwa14.urdf"),
}
OUT = ROOT / "examples" / "benchmarks" / "data" / "linsys_auto_cdf"
N, DT, SIM_TIME = 64, 0.01, 6.0
KICK_EVERY = 25
TAUS = [0.08, 0.17, 0.35]
PLANT = "indy7"


def start_and_ref():
    """(q0, fig8_flat) for PLANT — iiwa14 uses the shared fig3-fair harness."""
    from gato.common import figure8
    if PLANT == "indy7":
        from gato.config import INDY7_START_CONFIGS, FIG8_DEFAULT_PARAMS
        return INDY7_START_CONFIGS["ready"], figure8(DT, **FIG8_DEFAULT_PARAMS)
    import iiwa_fig8_shared as S
    n_steps = int(round(S.FIG8_PERIOD * 5 / DT))
    return S.Q0_READYC, S.figure8_goal(n_steps)


def run_arm(mode, tau=0.0):
    import pinocchio as pin
    from gato.mpc_gato import MPC_GATO
    from gato.controller import MPCController

    URDF = URDFS[PLANT]
    model = pin.buildModelFromUrdf(URDF)
    mpc = MPC_GATO(model, model_path=URDF, N=N, dt=DT, batch_size=1,
                   plant_type=PLANT)
    # always pass linsys explicitly: since 08-12 the controller DEFAULT is auto
    # (fixed-base), so an omitted arg would not give the pure-pcg arm
    kw = {"linsys": mode}
    if mode == "auto":
        kw["bdsv_threshold"] = tau
    mpc.controller = MPCController(mpc.solver, hypotheses=mpc.controller.hypotheses,
                                   warm_start="shift", reset_rho_each_step=True, **kw)

    pred_errs, iters, colds = [], [], []
    rng = np.random.default_rng(7)
    orig_step = mpc.controller.step
    nq, nx = mpc.solver.nq, mpc.solver.nx
    k = [0]

    def step(x, g, **skw):
        k[0] += 1
        if k[0] % KICK_EVERY == 0:
            x = x.copy()
            c = (k[0] // KICK_EVERY) % 3
            if c == 0:
                x[:nq] += rng.normal(0.0, 0.03, nq)
            elif c == 1:
                x[:nq] += rng.normal(0.0, 0.10, nq)
            else:
                x[:nq] += rng.normal(0.0, 0.20, nq)
                x[nq:nx] += rng.normal(0.0, 0.5, nx - nq)
        else:
            rng.normal(0.0, 1.0, 2 * nx - nq)   # keep the stream aligned across arms
        r = orig_step(x, g, **skw)
        pred_errs.append(r.pred_err)
        iters.append(int(np.asarray(r.solve.stats.pcg_iters).reshape(-1).max()))
        if mode == "auto":
            colds.append(bool(r.pred_err > tau or False))
        return r

    mpc.controller.step = step
    q0, fig8 = start_and_ref()
    xs = np.hstack((q0, np.zeros(nx - nq)))
    _, stats = mpc.run_mpc_fig8(xs, fig8, sim_dt=0.001, sim_time=SIM_TIME,
                                pace_by_solve_time=False)
    st = np.asarray(stats["solve_times"], dtype=float)
    gd = np.asarray(stats["goal_distances"], dtype=float)
    n = min(len(st), len(pred_errs))
    return {
        "mode": mode, "tau": tau, "steps": n,
        "solve_ms": st[:n].tolist(),
        "pred_err": pred_errs[:n],
        "pcg_iters": iters[:n],
        "cold": colds[:n] if mode == "auto" else None,
        "track_mean": float(gd.mean()), "track_max": float(gd.max()),
    }


def stats_line(name, t):
    t = np.asarray(t)
    return (f"{name:<14} p50 {np.percentile(t,50):6.3f}  p90 {np.percentile(t,90):6.3f}  "
            f"p99 {np.percentile(t,99):6.3f}  max {t.max():6.3f}  mean {t.mean():6.3f} ms")


def main():
    global PLANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="indy7", choices=list(URDFS))
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if other GPU compute pids exist (results noted noisy)")
    args = ap.parse_args()
    PLANT = args.plant

    import subprocess
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                           "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()
    if busy and not args.allow_busy:
        sys.exit(f"REFUSING to time: GPU busy ({busy})")
    if busy:
        print(f"WARNING: timing under contention (pids: {busy}) — noisy numbers")
    OUT.mkdir(parents=True, exist_ok=True)

    arms = [("pcg", 0.0), ("bdsv", 0.0), ("bdsv_first", 0.0)] + \
           [("auto", t) for t in TAUS]
    rows = []
    for mode, tau in arms:
        r = run_arm(mode, tau)
        rows.append(r)
        tag = f"auto@{tau}" if mode == "auto" else mode
        extra = ""
        if r["cold"]:
            extra = f"  cold {100*np.mean(r['cold']):.0f}%"
        print(stats_line(tag, r["solve_ms"]) +
              f"  track {r['track_mean']:.4f}{extra}", flush=True)

    suffix = "" if PLANT == "indy7" else f"_{PLANT}"
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
    ax.set_title(f"{PLANT} fig8 MPC N={N} B=1, mixed kicks every {KICK_EVERY} steps")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"linsys_auto_cdf{suffix}.png", dpi=140)
    print(f"wrote cdf_traces{suffix}.json + linsys_auto_cdf{suffix}.png")


if __name__ == "__main__":
    main()
