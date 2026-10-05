"""GATO batch-size x horizon timing sweep on the FAIR fig8 problem (iiwa_fig8_shared.py; iiwa14 default, --robot indy7).

Times solver.solve() for B IDENTICAL problem replicas (open-loop warm-started MPC over the
fig8 goal sequence — same per-solve work as the tracking harness, no pinocchio sim in the
loop) so the batch axis measures pure batched-solve latency. At N=64, B=1 matches the
closed-loop harness config (SQP=1, PCG cap 200 / rel 1e-4, rho 0.01, FIG8 cost weights).

The loop is the real-time-iteration MPC step: MPCController(warm_start="shift") solves from
the current state on the one-stage-shifted previous solution, and the solver's own predicted
x_1 becomes the next "measured" state (open loop; the controller's rho reset is OFF so the
adapted rho carries across solves exactly as the paper-era raw loop did).

One (N, B) row per config; results append to a CSV consumed by
examples/paper-figures/reproduce_fig3_fair.py (fig3-left = the N=64 row set; the full
N x B grid is the fig3-right heatmap). TIMING — quiet box only.

  python examples/benchmarks/sweep_batch_iiwa_fig8.py [--robot iiwa14] [--N 64] [--batches 1,2,...,512] \\
      [--solves 400] [--out examples/benchmarks/data/sweep_fig8_gato.csv]
The default CSV carries the robot for non-iiwa14 runs (sweep_fig8_gato_indy7.csv); the Indy7 goal is
always synthesized (MPCGPU has no Indy7 trajfile).
"""
import os
import sys
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

import iiwa_fig8_shared as fig8mod
import gato
from gato import BSQP, MPCController, SolverParams
from _bench import git_provenance

HERE = os.path.dirname(os.path.abspath(__file__))
DT = fig8mod.DT
# the 2026-07-07 benchmark config (== the closed-loop harness / SolverParams defaults),
# pinned to the paper-era pcg path (the controller default is "auto" since 08-12)
PARAMS = SolverParams(max_sqp_iters=1, max_pcg_iters=200, pcg_tol=1e-4, mu=10.0,
                      q_cost=2.0, qd_cost=1e-2, u_cost=2e-6, N_cost=50.0, q_lim_cost=0.01,
                      rho=1e-2, linsys="pcg")


def parse_args():
    p = argparse.ArgumentParser(description="GATO fig8 batched-solve timing sweep (FAIR harness)")
    p.add_argument("--robot", choices=fig8mod.ROBOTS, default="iiwa14",
                   help="plant (iiwa14 = the FAIR reference lane; indy7 = the paper's robot)")
    p.add_argument("--N", type=int, default=64, help="knot points (module bsqpN{N}_{robot} must be built)")
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128",
                   help="comma list of batch sizes (256/512 are GATO-only extensions)")
    p.add_argument("--solves", type=int, default=400, help="solves per config (first 10 dropped)")
    p.add_argument("--initial-guess", choices=("zero-tail", "hold"), default="zero-tail",
                   help="historical sweep uses zero-tail; hold reproduces the September checkpoint seed")
    p.add_argument("--goal-file", help="frozen flat 6-wide reference .npy (no pickle); otherwise load/generate fig8")
    p.add_argument("--check-only", action="store_true",
                   help="GPU correctness only: finite trajectories and controller/raw-loop parity; no timing output")
    p.add_argument("--record-call-boundary", action="store_true",
                   help="also save paired MPCController.step wall and internal solver samples; quiet window only")
    p.add_argument("--out", default=None,
                   help="CSV to append rows to ('' = print only; default data/sweep_fig8_gato[_<robot>].csv)")
    args = p.parse_args()
    if args.out is None:
        args.out = default_csv(args.robot)
    if args.record_call_boundary and not args.check_only:
        if os.environ.get('GATO_QUIET_WINDOW') != '1':
            p.error('call-boundary timing requires GATO_QUIET_WINDOW=1 in an assigned quiet window')
        if not args.out or Path(args.out + '.call-boundaries.json').exists():
            p.error('call-boundary timing requires a fresh --out path')
    return args


def default_csv(robot):
    suffix = "" if robot == "iiwa14" else f"_{robot}"
    return os.path.join(HERE, "data", f"sweep_fig8_gato{suffix}.csv")


def main():
    args = parse_args()
    if args.solves <= 10:
        sys.exit("--solves must exceed the ten warmup solves")
    N = args.N
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    rob = fig8mod.robot(args.robot)
    if (rob.name, N) not in gato.available():
        sys.exit(f"ERROR: module bsqpN{N}_{rob.name} not built — cmake with -DKNOTS include {N}, -DPLANT {rob.name}.")

    model, data = fig8mod.build_model(rob.name)
    q0 = rob.q0
    center = fig8mod.fig8_center(model, data, q0, rob.ee_frame)
    if args.goal_file:
        goal = np.load(args.goal_file, allow_pickle=False)
        if goal.ndim != 1 or goal.size < 6 * (args.solves + N):
            sys.exit("--goal-file must be a flat reference covering solves + N knots")
        goal_source = "file"
    else:
        goal, goal_source = fig8mod.goal_sequence(rob, args.solves + N + 8, center)
    goal = np.ascontiguousarray(goal, dtype=np.float32)
    if not np.isfinite(goal).all():
        sys.exit("reference must be finite")
    x0 = np.hstack((q0, np.zeros(model.nv))).astype(np.float32)

    rows, boundary_rows = [], []
    print(f"{rob.name} fig8 batch sweep: N={N} SQP=1 PCG<=200 rel 1e-4 rho 0.01, {args.solves} solves/config "
          f"(goal: {goal_source}, EE frame {rob.ee_frame}, center {center.round(4)})")
    print(f"initial_guess={args.initial_guess}; mode={'correctness' if args.check_only else 'timing'}")
    if not args.check_only:
        print(f"{'B':>4} {'median_ms':>10} {'p90_ms':>8} {'per_traj_us':>12}")
    for B in batches:
        solver = BSQP(rob.urdf, batch_size=B, N=N, dt=DT, params=PARAMS, plant_type=rob.name)
        nx, stride = solver.nx, solver.nx + solver.nu
        ctrl = MPCController(solver, warm_start="shift", linsys="pcg",
                             reset_rho_each_step=False)
        # The pre-migration raw sweep seeded ONLY knot zero. Holding every
        # state at x0 changes the optimization path even after ten warmups.
        seed = np.zeros(solver.xu_size, dtype=np.float32) if args.initial_guess == "zero-tail" else None
        ctrl.reset(x0, xu_warm=seed)
        raw_solver = None
        if args.check_only:
            raw_solver = BSQP(rob.urdf, batch_size=B, N=N, dt=DT, params=PARAMS, plant_type=rob.name)
            from gato.common import initialize_warm_start
            raw_seed = seed if seed is not None else initialize_warm_start(x0, N, solver.nx, solver.nu)
            raw_xu = np.tile(raw_seed, (B, 1)).astype(np.float32)
            raw_solver.reset_dual()
            raw_solver.reset_rho()
        xcur = x0.copy()
        times, call_times = [], []
        for t in range(args.solves):
            ref = goal[6 * t: 6 * (t + N)].astype(np.float32)
            if args.record_call_boundary and not args.check_only:
                start = time.perf_counter_ns()
                r = ctrl.step(xcur, ref)
                call_times.append((time.perf_counter_ns() - start) / 1000)
            else:
                r = ctrl.step(xcur, ref)
            if args.check_only:
                raw_xu[:, :nx] = xcur
                raw = raw_solver.solve(np.tile(xcur, (B, 1)), np.tile(ref, (B, 1)), raw_xu)
                if not np.isfinite(r.solve.xu).all() or r.solve.diverged.any():
                    raise AssertionError(f"nonfinite solve at B={B}, step={t}")
                np.testing.assert_array_equal(r.solve.xu, raw.xu)
                np.testing.assert_array_equal(r.solve.stats.pcg_iters, raw.stats.pcg_iters)
                raw_xu = np.concatenate([raw.xu[:, stride:], raw.xu[:, -stride:]], axis=1)
            else:
                times.append(float(r.solve.solve_time_us))
            xcur = r.xu_best[stride:stride + nx].copy()   # open loop: the solution's x_1 is the next state
        if args.check_only:
            print(f"B={B}: {args.solves} finite solves; bitwise raw/controller trajectory and PCG parity")
            del raw_solver, ctrl, solver
            continue
        t = np.asarray(times[10:])  # drop warm-up solves
        med, p90 = np.median(t), np.percentile(t, 90)
        print(f"{B:>4} {med/1000:>10.4f} {p90/1000:>8.4f} {med/B:>12.1f}")
        rows.append((N, B, med / 1000, p90 / 1000, med / B, len(t)))
        if call_times:
            wall = np.asarray(call_times[10:])
            if not np.isfinite(wall).all() or not np.isfinite(t).all() or np.any(wall <= 0) or np.any(t <= 0):
                raise ValueError('Invalid call-boundary samples')
            boundary_rows.append(dict(B=B,internal_solver_us=t.tolist(),controller_step_wall_us=wall.tolist()))
        del ctrl, solver

    if args.out and not args.check_only:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        goal_hash = hashlib.sha256(goal.tobytes()).hexdigest()
        goal_path = args.out + f".goal-{goal_hash}.npy"
        if not os.path.exists(goal_path):
            np.save(goal_path, goal, allow_pickle=False)
        with open(args.out + ".runs.jsonl", "a") as meta:
            meta.write(json.dumps({"source": git_provenance(), "robot": rob.name, "N": N, "batches": batches,
                                   "solves": args.solves, "initial_guess": args.initial_guess,
                                   "params": PARAMS.asdict(), "goal_sha256": goal_hash,
                                   "goal_file": os.path.abspath(goal_path),
                                   "goal_source": goal_source,
                                   "urdf_sha256": hashlib.sha256(Path(rob.urdf).read_bytes()).hexdigest(),
                                   "metric": "internal_solver_latency"}) + "\n")
        fresh = not os.path.exists(args.out)
        with open(args.out, "a") as f:
            if fresh:
                f.write("N,B,median_ms,p90_ms,per_traj_us,n_solves\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.1f},{r[5]}\n")
        print(f"[sweep] appended {len(rows)} rows -> {args.out}")
        if boundary_rows:
            with open(args.out + '.call-boundaries.json','x') as stream:
                json.dump(dict(source=git_provenance(),robot=rob.name,N=N,initial_guess=args.initial_guess,
                    goal_sha256=goal_hash,warmup_dropped=10,
                    boundary='MPCController.step wall time; excludes reference preparation, sensing, simulation and actuator I/O',
                    rows=boundary_rows),stream,indent=2)


if __name__ == "__main__":
    main()
