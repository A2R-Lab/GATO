"""GATO batch-size timing sweep on the FAIR iiwa14 fig8 problem (see iiwa_fig8_shared.py).

Times solver.solve() for B IDENTICAL problem replicas (open-loop warm-started MPC over the
fig8 goal sequence — same per-solve work as the tracking harness, no pinocchio sim in the
loop) so the batch axis measures pure batched-solve latency. B=1 matches the closed-loop
harness config (N=64, SQP=1, PCG cap 200 / rel 1e-4, rho 0.01, FIG8 cost weights).

  PYTHONPATH=python /home/plancher/Desktop/GRiD/.venv/bin/python \
      examples/benchmarks/sweep_batch_iiwa_fig8.py [B ...]   (default 1 2 4 8 16 32 64 128)
"""
import sys, os
import numpy as np
sys.path.insert(0, "/home/plancher/Desktop/GATO/python")
sys.path.insert(0, "/home/plancher/Desktop/GATO/python/gato")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iiwa_fig8_shared as fig8mod
import bsqpN64_iiwa14 as M

N, DT = 64, 0.01
nx, nu = 14, 7
stride = nx + nu
NSOLVES = 400
batches = [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8, 16, 32, 64, 128]

model, data = fig8mod.build_model()
q0 = fig8mod.Q0_READYC
center = fig8mod.fig8_center(model, data, q0)
goal = fig8mod.load_goal_file()
if goal is None or len(goal) // 6 < NSOLVES + N + 8:
    goal = fig8mod.figure8_goal(NSOLVES + N + 8, center=center)
x0 = np.hstack((q0, np.zeros(7))).astype(np.float32)

print(f"iiwa14 fig8 batch sweep: N={N} SQP=1 PCG<=200 rel 1e-4 rho 0.01, {NSOLVES} solves/config")
print(f"{'B':>4} {'median_ms':>10} {'p90_ms':>8} {'per_traj_us':>12}")
for B in batches:
    solver = M.BSQP_float(B, DT, 1, 1e-5, 200, 1e-4, 1.0, 10.0,
                          2.0, 1e-2, 2e-6, 50.0, 0.01, 0.0, 0.0, 1e-2)
    XU = np.zeros((B, N * stride - nu), dtype=np.float32)
    XU[:, :nx] = x0
    xcur = np.tile(x0, (B, 1))
    times = []
    for t in range(NSOLVES):
        ref = np.tile(goal[6 * t: 6 * (t + N)].astype(np.float32), (B, 1))
        XU[:, :nx] = xcur
        res = solver.solve(XU, DT, xcur.copy(), ref)
        times.append(float(res["sqp_time_us"]))
        XU = np.asarray(res["XU"], dtype=np.float32)
        xcur = XU[:, stride:stride + nx].copy()
        XU = np.concatenate([XU[:, stride:], XU[:, -stride:]], axis=1)  # one-stage shift + dup tail
    t = np.asarray(times[10:])  # drop warm-up solves
    med = np.median(t)
    print(f"{B:>4} {med/1000:>10.4f} {np.percentile(t, 90)/1000:>8.4f} {med/B:>12.1f}")
    del solver
