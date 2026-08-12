#!/usr/bin/env python3
"""W3.6: go2 (STATE_SIZE=36) pcg-vs-bdsv solve-time crossover over batch size.

QUIET BOX ONLY. The fixed-base crossover was measured at STATE_SIZE 12/14
(t512 pcg best, bdsv 2.35x on the factor path, warm B=1 pcg 2.5x — memory
`project_hybrid_pcg_bdsv_solver`); the floating Schur blocks are 36x36, which
moves both the factor cost (O(n^3) blocks) and the PCG spmv cost — this sweep
answers which linsys the go2 default should be, per batch regime.

Protocol: displaced actuated posture anchor (real SQP work every solve, the
liveness-gate config), identical deterministic inputs per (linsys, B);
kkt_tol=0 + a fresh hold-at-x warm start EVERY solve so each timed solve runs
the full max_sqp_iters of work (a re-used warm start converges in 1 iter and
measures nothing); 3 warm-up solves then NSOLVES timed; device solve_time_us
medians.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import gato  # noqa: E402

URDF = str(REPO / "external" / "GRiD" / "config" / "robot_assets" / "go2.urdf")
NQ, NV, NU, N = 19, 18, 12, 16


def standing_x():
    q = np.zeros(NQ, dtype=np.float32)
    q[2] = 0.35
    q[6] = 1.0
    q[7:] = np.tile([0.0, 0.9, -1.8], 4)
    return np.concatenate([q, np.zeros(NV, dtype=np.float32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--nsolves", type=int, default=50)
    ap.add_argument("--out", default=None, help="json output path")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    x = standing_x()
    q_nom = x[:NQ].copy()
    q_nom[7:] += 0.3

    rows = []
    for linsys in ("pcg", "bdsv"):
        for B in batches:
            s = gato.BSQP(model_path=URDF, batch_size=B, N=N, dt=0.01,
                          plant_type="go2", linsys=linsys, kkt_tol=0.0,
                          q_cost=1.0, qd_cost=1e-2, u_cost=1e-4, N_cost=5.0,
                          q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
            s.set_q_nom(q_nom)
            s.set_q_pos_cost(50.0)
            X = np.tile(x, (B, 1)).astype(np.float32)
            goals = np.zeros((B, N * 6), dtype=np.float32)
            from gato.common import initialize_warm_start
            XU0 = np.tile(initialize_warm_start(x, N, s.nx, s.nu).astype(np.float32),
                          (B, 1))
            for _ in range(3):
                s.solve(X, goals, XU0.copy())
            ts = []
            for _ in range(args.nsolves):
                r = s.solve(X, goals, XU0.copy())
                ts.append(float(np.asarray(r.stats.solve_time_us).ravel()[0]))
            med = float(np.median(ts))
            rows.append(dict(linsys=linsys, B=B, median_us=med,
                             p10_us=float(np.percentile(ts, 10)),
                             p90_us=float(np.percentile(ts, 90)),
                             sqp_iters=int(np.asarray(r.stats.sqp_iters).ravel()[0])))
            print(f"{linsys:5s} B={B:4d}  median {med:9.1f} us  "
                  f"[{rows[-1]['p10_us']:.1f}, {rows[-1]['p90_us']:.1f}]  "
                  f"sqp {rows[-1]['sqp_iters']}", flush=True)
            del s

    print("\n=== crossover table (bdsv/pcg median ratio) ===")
    for B in batches:
        p = next(r for r in rows if r["linsys"] == "pcg" and r["B"] == B)
        d = next(r for r in rows if r["linsys"] == "bdsv" and r["B"] == B)
        w = "bdsv" if d["median_us"] < p["median_us"] else "pcg"
        print(f"B={B:4d}  pcg {p['median_us']:9.1f}  bdsv {d['median_us']:9.1f}"
              f"  ratio {d['median_us']/p['median_us']:5.2f}  winner {w}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
