"""GATO batch-size x horizon timing sweep on the FAIR iiwa14 fig8 problem (iiwa_fig8_shared.py).

Times solver.solve() for B IDENTICAL problem replicas (open-loop warm-started MPC over the
fig8 goal sequence — same per-solve work as the tracking harness, no pinocchio sim in the
loop) so the batch axis measures pure batched-solve latency. At N=64, B=1 matches the
closed-loop harness config (SQP=1, PCG cap 200 / rel 1e-4, rho 0.01, FIG8 cost weights).

One (N, B) row per config; results append to a CSV consumed by
examples/paper-figures/reproduce_fig3_fair.py (fig3-left = the N=64 row set; the full
N x B grid is the fig3-right heatmap).

  PYTHONPATH=python /home/plancher/Desktop/GRiD/.venv/bin/python \
      examples/benchmarks/sweep_batch_iiwa_fig8.py [--N 64] [--batches 1,2,...,512] \
      [--solves 400] [--out examples/benchmarks/data/sweep_fig8_gato.csv]
"""
import os
import sys
import argparse
import importlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/plancher/Desktop/GATO/python")
sys.path.insert(0, "/home/plancher/Desktop/GATO/python/gato")
sys.path.insert(0, HERE)
import iiwa_fig8_shared as fig8mod

DT = fig8mod.DT
nx, nu = 14, 7
stride = nx + nu


def parse_args():
    p = argparse.ArgumentParser(description="GATO iiwa14 fig8 batched-solve timing sweep")
    p.add_argument("--N", type=int, default=64, help="knot points (module bsqpN{N}_iiwa14 must be built)")
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128",
                   help="comma list of batch sizes (256/512 are GATO-only extensions)")
    p.add_argument("--solves", type=int, default=400, help="solves per config (first 10 dropped)")
    p.add_argument("--out", default=os.path.join(HERE, "data", "sweep_fig8_gato.csv"),
                   help="CSV to append rows to ('' = print only)")
    return p.parse_args()


def main():
    args = parse_args()
    N = args.N
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    try:
        M = importlib.import_module(f"bsqpN{N}_iiwa14")
    except ImportError:
        sys.exit(f"ERROR: module bsqpN{N}_iiwa14 not built — cmake with -DKNOTS include {N}, -DPLANT iiwa14.")

    model, data = fig8mod.build_model()
    q0 = fig8mod.Q0_READYC
    center = fig8mod.fig8_center(model, data, q0)
    goal = fig8mod.load_goal_file()
    if goal is None or len(goal) // 6 < args.solves + N + 8:
        goal = fig8mod.figure8_goal(args.solves + N + 8, center=center)
    x0 = np.hstack((q0, np.zeros(7))).astype(np.float32)

    rows = []
    print(f"iiwa14 fig8 batch sweep: N={N} SQP=1 PCG<=200 rel 1e-4 rho 0.01, {args.solves} solves/config")
    print(f"{'B':>4} {'median_ms':>10} {'p90_ms':>8} {'per_traj_us':>12}")
    for B in batches:
        solver = M.BSQP_float(B, DT, 1, 1e-5, 200, 1e-4, 1.0, 10.0,
                              2.0, 1e-2, 2e-6, 50.0, 0.01, 0.0, 0.0, 1e-2)
        XU = np.zeros((B, N * stride - nu), dtype=np.float32)
        XU[:, :nx] = x0
        xcur = np.tile(x0, (B, 1))
        times = []
        for t in range(args.solves):
            ref = np.tile(goal[6 * t: 6 * (t + N)].astype(np.float32), (B, 1))
            XU[:, :nx] = xcur
            res = solver.solve(XU, DT, xcur.copy(), ref)
            times.append(float(res["sqp_time_us"]))
            XU = np.asarray(res["XU"], dtype=np.float32)
            xcur = XU[:, stride:stride + nx].copy()
            XU = np.concatenate([XU[:, stride:], XU[:, -stride:]], axis=1)  # one-stage shift + dup tail
        t = np.asarray(times[10:])  # drop warm-up solves
        med, p90 = np.median(t), np.percentile(t, 90)
        print(f"{B:>4} {med/1000:>10.4f} {p90/1000:>8.4f} {med/B:>12.1f}")
        rows.append((N, B, med / 1000, p90 / 1000, med / B, len(t)))
        del solver

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fresh = not os.path.exists(args.out)
        with open(args.out, "a") as f:
            if fresh:
                f.write("N,B,median_ms,p90_ms,per_traj_us,n_solves\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.1f},{r[5]}\n")
        print(f"[sweep] appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
