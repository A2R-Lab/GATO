"""Batched-CPU baseline for Fig-3 (the paper's *actual* CPU competitor): pysqpcpu.BatchThneed.

This is the multi-threaded C++ `BatchThneed` — it solves a batch of M
trajectory-optimization problems across `num_threads`
CPU cores in one call, so its per-step time scales SUB-LINEARLY with M (flat until the core
count, then linear), exactly like the paper's Fig-3 CPU line (~3 ms -> ~30 ms over M=1..128).
That is the fair comparison to GATO's batched GPU solve; single-solve x M overstates the CPU.

PREREQUISITE — build the module and set the runtime paths first:
    baselines/build_cpu_baseline.sh         # builds osqp + osqp-eigen + pysqpcpu (no ROS)
    source baselines/sqpcpu_env.sh          # LD_LIBRARY_PATH + PYTHONPATH for the .so + deps
Then (TIMING — run on a quiet box; other CPU load skews it):
    ../GRiD/.venv/bin/python baselines/run_batchthneed_fig8.py --batch-sizes 1,2,4,8,16,32,64,128

Output: baselines/batchthneed_fig8_results.pkl = list of {batch_size, batched_cpu_ms, ...}
which reproduce_fig3_scalability.py reads as the batched-CPU line.
"""
import sys, os, time, argparse, pickle
import numpy as np

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, G + '/python')
from bsqp.common import figure8, rk4
from bsqp.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS, DEFAULT_SOLVER_PARAMS

SP = DEFAULT_SOLVER_PARAMS  # GATO's OWN indy7 fig8 weights, so the CPU solves the identical problem


def _import_pysqpcpu():
    try:
        import pysqpcpu  # noqa: F401
        return pysqpcpu
    except ImportError as e:
        sys.exit(f"ERROR: cannot import pysqpcpu ({e}).\n"
                 "Build it and set paths first:\n"
                 "  baselines/build_cpu_baseline.sh && source baselines/sqpcpu_env.sh")


def run_one(pysqpcpu, urdf, model, N, dt, sim_time, sim_dt, fig8_traj, x_start,
            batch_size, num_threads, max_qp_iters):
    import pinocchio as pin
    data = model.createData()
    bt = pysqpcpu.BatchThneed(
        urdf_filename=urdf, eepos_frame_name="EE", batch_size=batch_size, N=N, dt=dt,
        max_qp_iters=max_qp_iters, num_threads=num_threads,
        Q_cost=SP['q_cost'], dQ_cost=SP['qd_cost'], R_cost=SP['u_cost'], QN_cost=SP['N_cost'])
    nq, nv, nx, nu = bt.nq, bt.nv, bt.nx, bt.nu
    q = x_start[:nq].copy(); dq = x_start[nq:nx].copy()
    f_ext = pin.StdVec_Force()
    for _ in range(model.njoints):
        f_ext.append(pin.Force.Zero())

    n_traj = len(fig8_traj) // 6
    solve_times, track_errs = [], []
    total = 0.0
    # one warm solve so the first timed step isn't allocation-dominated
    ee0 = fig8_traj[0:6 * N].reshape(N, 6)[:, :3].reshape(-1)
    bt.sqp(np.concatenate([q, dq]), ee0)
    while total < sim_time:
        off = int(total / dt)
        if off >= n_traj - 6 * N:
            break
        ee_g3 = fig8_traj[6 * off:6 * (off + N)].reshape(N, 6)[:, :3].reshape(-1)
        xcur = np.concatenate([q, dq])
        t0 = time.perf_counter()
        bt.sqp(xcur, ee_g3)                      # solves all `batch_size` problems across threads
        solve_times.append((time.perf_counter() - t0) * 1000.0)
        u = np.asarray(bt.get_results()[0])[nx:nx + nu]   # apply the (first) winning control
        for _ in range(int(round(dt / sim_dt))):
            q, dq = rk4(model, data, q, dq, u, sim_dt, f_ext)
            total += sim_dt
        track_errs.append(float(np.linalg.norm(bt.eepos(q)[:3] - ee_g3[:3])))

    st = np.array(solve_times)
    return {
        'N': N, 'batch_size': batch_size, 'num_threads': num_threads, 'success': True,
        'iterations': len(st),
        'batched_cpu_ms': float(np.median(st)),          # representative per-step batched-solve time
        'avg_cpu_time_ms': float(st.mean()), 'std_cpu_time_ms': float(st.std()),
        'avg_goal_distance': float(np.mean(track_errs)) if track_errs else None,
    }


def main():
    p = argparse.ArgumentParser(description="Batched-CPU (BatchThneed) fig8 baseline for Fig-3.")
    p.add_argument('--urdf', default=G + '/examples/indy7_description/indy7.urdf')
    p.add_argument('--N', type=int, default=64)
    p.add_argument('--batch-sizes', default='1,2,4,8,16,32,64,128')
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--sim-time', type=float, default=5.0)
    p.add_argument('--sim-dt', type=float, default=0.001)
    p.add_argument('--start-config', default='ready')
    p.add_argument('--max-qp-iters', type=int, default=1)
    p.add_argument('--num-threads', type=int, default=0, help="0 = use all cores (os.cpu_count)")
    p.add_argument('--out', default=G + '/baselines/batchthneed_fig8_results.pkl')
    args = p.parse_args()

    pysqpcpu = _import_pysqpcpu()
    import pinocchio as pin
    model = pin.buildModelFromUrdf(args.urdf)
    model.gravity.linear = np.array([0, 0, -9.81])
    fig8_traj = figure8(args.dt, **FIG8_DEFAULT_PARAMS)
    x_start = np.hstack((INDY7_START_CONFIGS[args.start_config], np.zeros(model.nv)))
    n_cores = args.num_threads or os.cpu_count()
    Ms = [int(x) for x in args.batch_sizes.split(',')]

    print("=" * 64)
    print(f"BatchThneed / batched-CPU baseline — Indy7 fig8, N={args.N}, threads={n_cores}")
    print("=" * 64)
    results = []
    for M in Ms:
        r = run_one(pysqpcpu, args.urdf, model, args.N, args.dt, args.sim_time, args.sim_dt,
                    fig8_traj, x_start, batch_size=M, num_threads=min(M, n_cores),
                    max_qp_iters=args.max_qp_iters)
        results.append(r)
        print(f"  M={M:4d} (thr={min(M, n_cores):2d}): {r['batched_cpu_ms']:8.3f} ms/batch-solve "
              f"| track {r['avg_goal_distance']:.4f} m | {r['iterations']} steps")
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nsaved {args.out}")
    print("BATCHTHNEED_FIG8_DONE")


if __name__ == "__main__":
    main()
