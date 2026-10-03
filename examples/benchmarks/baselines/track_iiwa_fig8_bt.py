"""Multi-threaded QDLDL-based CPU solver (pysqpcpu.BatchThneed: OSQP with QDLDL) fig8 tracking on the FAIR shared problem.
Same canonical fig8 as GATO/MPCGPU (center = grid-EE = URDF "EE" fixed joint at the robot's q0, A=0.15, T=6),
the robot's canonical URDF, EE frame from the plant registry (= grid end_effector_pose post-regen),
warm-start = zero controls, 1 QP iter. Tracking measured at EE from the logged joint configs (same
metric as GATO/MPCGPU). iiwa14 by default; `--robot indy7` runs the paper's robot (synthesized goal).

  baselines/build_cpu_baseline.sh            # once (default venv: the project .venv)
  source baselines/sqpcpu_env.sh             # LD_LIBRARY_PATH + PYTHONPATH for pysqpcpu
  python baselines/track_iiwa_fig8_bt.py [sim_time] [batch] [N] [out_csv] [--robot iiwa14]
"""
import argparse
import importlib.util
import os
import sys
import time
import numpy as np

from gato.common import rk4
from gato import SolverParams

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_bench():
    """../_bench.py by path (examples/benchmarks is not a package)."""
    spec = importlib.util.spec_from_file_location("_bench", os.path.join(os.path.dirname(HERE), "_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


fig8mod = _load_bench().import_sibling("iiwa_fig8_shared")
SP = SolverParams().asdict()  # GATO's own defaults, so the CPU baseline solves the same problem

_p = argparse.ArgumentParser(description="QDLDL-based CPU solver fig8 tracking (FAIR harness)")
_p.add_argument("sim_time", nargs="?", type=float, default=6.0, help="closed-loop seconds")
_p.add_argument("batch", nargs="?", type=int, default=1, help="B identical replicas (num_threads=B)")
_p.add_argument("N", nargs="?", type=int, default=64, help="horizon (the CPU solver takes N at construction)")
_p.add_argument("out_csv", nargs="?", default="", help="optional: append an (N,B,median) row")
_p.add_argument("--robot", choices=fig8mod.ROBOTS, default="iiwa14")
_a = _p.parse_args()
SIM_TIME, BATCH, N, OUT_CSV, ROBOT = _a.sim_time, _a.batch, _a.N, _a.out_csv, _a.robot
DT = fig8mod.DT


def _import_pysqpcpu():
    try:
        import pysqpcpu
        return pysqpcpu
    except ImportError as e:
        sys.exit(f"ERROR: cannot import pysqpcpu ({e}). Build + `source baselines/sqpcpu_env.sh` first.")


def main():
    import pinocchio as pin
    pysqpcpu = _import_pysqpcpu()
    rob = fig8mod.robot(ROBOT)
    model, data = fig8mod.build_model(rob.name)
    q0 = rob.q0.copy()
    center = fig8mod.fig8_center(model, data, q0, rob.ee_frame)

    n_needed = int(SIM_TIME / DT) + N + 8
    goal, goal_source = fig8mod.goal_sequence(rob, n_needed, center)
    n_goal = len(goal) // 6

    bt = pysqpcpu.BatchThneed(
        urdf_filename=rob.urdf, eepos_frame_name=rob.ee_frame,   # "EE" = grid-EE
        batch_size=BATCH, N=N, dt=DT, max_qp_iters=SP['max_sqp_iters'], num_threads=BATCH,
        Q_cost=SP['q_cost'], dQ_cost=SP['qd_cost'], R_cost=SP['u_cost'], QN_cost=SP['N_cost'])
    nq, nv, nx, nu = bt.nq, bt.nv, bt.nx, bt.nu
    print(f"{rob.name} QDLDL-based CPU fig8: center(EE)={center.round(4)} frame={rob.ee_frame} "
          f"A={fig8mod.FIG8_A} T={fig8mod.FIG8_PERIOD} N={N} nq={nq} goal_steps={n_goal} goal={goal_source}")

    q = q0.copy(); dq = np.zeros(nv)
    f_ext = pin.StdVec_Force()
    for _ in range(model.njoints):
        f_ext.append(pin.Force.Zero())

    ee0 = goal[0:6 * N].reshape(N, 6)[:, :3].reshape(-1)
    bt.sqp(np.concatenate([q, dq]), ee0)                   # one warm solve

    q_log, solve_ms = [], []
    total = 0.0
    while total < SIM_TIME:
        off = int(round(total / DT))
        if off >= n_goal - N:
            break
        ee_g3 = goal[6 * off:6 * (off + N)].reshape(N, 6)[:, :3].reshape(-1)
        t0 = time.perf_counter()
        bt.sqp(np.concatenate([q, dq]), ee_g3)
        solve_ms.append((time.perf_counter() - t0) * 1000.0)
        u = np.asarray(bt.get_results()[0])[nx:nx + nu]
        for _ in range(int(round(DT / 0.001))):
            q, dq = rk4(model, data, q, dq, u, 0.001, f_ext)
            total += 0.001
        q_log.append(q.copy())

    errs = fig8mod.ee_tracking_errors(model, data, q_log, goal, dt=DT, frame=rob.ee_frame)
    st = np.asarray(solve_ms, float)
    if len(errs):
        print(f"RESULT_BT steps={len(errs)} EE_mean={errs.mean():.6f} EE_max={errs.max():.6f} "
              f"EE_final={errs[-1]:.6f}  median_solve_ms={np.median(st):.4f}")
        print("trace:", " ".join(f"{errs[i]:.4f}" for i in range(0, len(errs), max(1, len(errs)//20))))
        if OUT_CSV:
            fresh = not os.path.exists(OUT_CSV)
            os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
            with open(OUT_CSV, "a") as f:
                if fresh:
                    f.write("N,B,median_ms,p90_ms,per_traj_us,n_solves,EE_mean\n")
                med, p90 = np.median(st), np.percentile(st, 90)
                f.write(f"{N},{BATCH},{med:.4f},{p90:.4f},{med*1000/BATCH:.1f},{len(st)},{errs.mean():.6f}\n")
            print(f"[bt] appended row -> {OUT_CSV}")
    else:
        print("no tracking samples")


if __name__ == "__main__":
    main()
