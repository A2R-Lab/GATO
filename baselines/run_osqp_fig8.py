"""OSQP/CPU baseline (the paper's CPU competitor) on the Indy7 figure-8 task.

Drives the pure-Python single-solve SQP solver from the sqpcpu submodule
(`sqpcpu/pinocchio_template.py::Thneed`, pinocchio + scipy.sparse + OSQP) through
the SAME closed-loop fig8 MPC structure as `examples/benchmark_fig8.py`, so the
solve-time / tracking numbers drop straight into the Fig-3 comparison. CPU, batch=1
by construction (OSQP is a single-problem QP solver). No C++/pybind build needed.

Run with a venv that has pinocchio + scipy + osqp (e.g. the GRiD .venv):
    python baselines/run_osqp_fig8.py --N 8,16,32,64 --sim-time 5
"""
import sys, os, time, argparse, pickle
import numpy as np

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, G + '/python')
sys.path.insert(0, G + '/baselines/sqpcpu')
from bsqp.common import figure8
from bsqp.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS, DEFAULT_SOLVER_PARAMS
from pinocchio_template import Thneed

# Use GATO's OWN indy7 fig8 cost weights so the CPU baseline solves the IDENTICAL weighted
# problem (fairness). With Thneed's old defaults (Q=100/dQ=0.01, ratio 10000 vs GATO's 200) the
# closed loop commanded violent torques and diverged (~0.9 m); matched weights track to ~0.056 m.
SP = DEFAULT_SOLVER_PARAMS  # q_cost/qd_cost/u_cost/N_cost/q_lim_cost/rho


def run_one(urdf, N, dt, sim_time, sim_dt, fig8_traj, x_start, max_qp_iters=1, sigma=None):
    if sigma is None:
        sigma = SP['rho']  # GATO's adaptive-rho init (0.01) as the OSQP primal Levenberg reg
    t = Thneed(urdf_filename=urdf, eepos_frame_name="EE", N=N, dt=dt,
               max_qp_iters=max_qp_iters, sigma=sigma,
               Q_cost=SP['q_cost'], dQ_cost=SP['qd_cost'], R_cost=SP['u_cost'],
               QN_cost=SP['N_cost'], Qlim_cost=SP['q_lim_cost'])
    nq, nv, nx, nu = t.nq, t.nv, t.nx, t.nu
    q = x_start[:nq].copy(); dq = x_start[nq:nx].copy()
    # warm start: stack current state across the horizon
    t.XU = np.zeros(N * (nx + nu) - nu)
    for i in range(N):
        t.XU[i * (nx + nu): i * (nx + nu) + nx] = np.concatenate([q, dq])

    solve_times, track_errs = [], []
    total_sim_time = 0.0
    n_traj = len(fig8_traj) // 6
    while total_sim_time < sim_time:
        eepos_offset = int(total_sim_time / dt)
        if eepos_offset >= n_traj - 6 * N:
            break
        # 6*N (pos+orient) GATO trace -> 3*N position-only goals for Thneed
        ee_g6 = fig8_traj[6 * eepos_offset: 6 * (eepos_offset + N)]
        ee_g3 = ee_g6.reshape(N, 6)[:, :3].reshape(-1)
        xcur = np.concatenate([q, dq])

        start = time.perf_counter()
        t.sqp(xcur, ee_g3)
        solve_times.append((time.perf_counter() - start) * 1000.0)  # ms

        # apply the first control over one MPC step (dt), sub-stepped at sim_dt
        u = t.XU[nx:nx + nu]
        for _ in range(int(round(dt / sim_dt))):
            q, dq = t.rk4(q, dq, u, sim_dt)
            total_sim_time += sim_dt
        track_errs.append(float(np.linalg.norm(t.eepos(q) - ee_g3[:3])))

    return {
        'N': N, 'batch_size': 1, 'success': True,
        'iterations': len(solve_times),
        'avg_cpu_time_ms': float(np.mean(solve_times)),
        'std_cpu_time_ms': float(np.std(solve_times)),
        'avg_goal_distance': float(np.mean(track_errs)),
        'max_goal_distance': float(np.max(track_errs)),
    }


def main():
    p = argparse.ArgumentParser(description="OSQP/CPU fig8 baseline (sqpcpu Thneed).")
    p.add_argument('--urdf', default=G + '/examples/indy7_description/indy7.urdf')
    p.add_argument('--N', default='8,16,32,64')
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--sim-time', type=float, default=5.0)
    p.add_argument('--sim-dt', type=float, default=0.001)
    p.add_argument('--start-config', default='ready')
    p.add_argument('--max-qp-iters', type=int, default=1,
                   help="SQP iters/step. Default 1 to match GATO's 1-iter real-time budget; with "
                        "GATO-matched cost weights the closed loop tracks at 1 iter (~0.056 m).")
    p.add_argument('--sigma', type=float, default=None,
                   help="OSQP primal Levenberg reg (rho*I on the Hessian). Default = GATO's rho "
                        "(DEFAULT_SOLVER_PARAMS['rho']=0.01); keeps the 1-iter KKT quasidefinite.")
    p.add_argument('--out', default=G + '/baselines/osqp_fig8_results.pkl')
    args = p.parse_args()

    import pinocchio as pin
    model = pin.buildModelFromUrdf(args.urdf)
    fig8_traj = figure8(args.dt, **FIG8_DEFAULT_PARAMS)
    x_start = np.hstack((INDY7_START_CONFIGS[args.start_config], np.zeros(model.nv)))
    Ns = [int(x) for x in args.N.split(',')]

    print("=" * 60)
    print("OSQP / CPU baseline (sqpcpu Thneed) — Indy7 figure-8")
    print("=" * 60)
    results = []
    for N in Ns:
        r = run_one(args.urdf, N, args.dt, args.sim_time, args.sim_dt, fig8_traj, x_start,
                    max_qp_iters=args.max_qp_iters, sigma=args.sigma)
        r['max_qp_iters'] = args.max_qp_iters
        r['sigma'] = args.sigma
        results.append(r)
        print(f"  N={N:3d}: {r['avg_cpu_time_ms']:8.3f} ± {r['std_cpu_time_ms']:6.3f} ms/solve "
              f"| tracking {r['avg_goal_distance']:.4f} m (max {r['max_goal_distance']:.4f}) "
              f"| {r['iterations']} steps")
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nsaved {args.out}")
    print("OSQP_FIG8_DONE")


if __name__ == '__main__':
    main()
