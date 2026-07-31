"""BatchThneed (CPU/OSQP baseline) iiwa14 fig8 tracking on the FAIR shared problem.
Same canonical fig8 as GATO/MPCGPU (center = grid-EE = URDF "EE" fixed joint at readyC, A=0.15, T=6),
iiwa14 URDF, EE frame = "EE" (= grid end_effector_pose post-regen), warm-start = zero controls, 1 QP
iter. Tracking measured at EE from the logged joint configs (same metric as GATO/MPCGPU).

  baselines/build_cpu_baseline.sh /home/plancher/Desktop/GRiD/.venv   # once
  source baselines/sqpcpu_env.sh                                       # LD_LIBRARY_PATH + PYTHONPATH
  PYTHONPATH=$PYTHONPATH:/home/plancher/Desktop/GATO/python \
    /home/plancher/Desktop/GRiD/.venv/bin/python baselines/track_iiwa_fig8_bt.py \
        [sim_time] [batch] [N] [out_csv]
"""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)                                   # iiwa_fig8_shared
sys.path.insert(0, os.path.dirname(BENCH) + "/../python")   # gato (…/GATO/python)
sys.path.insert(0, "/home/plancher/Desktop/GATO/python")
import iiwa_fig8_shared as fig8mod
from gato.common import rk4
from gato.config import DEFAULT_SOLVER_PARAMS as SP

SIM_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # B identical replicas (num_threads=B)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64      # BatchThneed takes N at construction
OUT_CSV = sys.argv[4] if len(sys.argv) > 4 else ""     # optional: append an (N,B,median) row
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
    model, data = fig8mod.build_model()
    q0 = fig8mod.Q0_READYC.copy()
    center = fig8mod.fig8_center(model, data, q0)

    n_needed = int(SIM_TIME / DT) + N + 8
    goal = fig8mod.load_goal_file()
    if goal is None or len(goal) // 6 < n_needed:
        goal = fig8mod.figure8_goal(n_needed, center=center)
    n_goal = len(goal) // 6

    bt = pysqpcpu.BatchThneed(
        urdf_filename=fig8mod.IIWA14_URDF, eepos_frame_name=fig8mod.EE_FRAME,   # "EE" = grid-EE
        batch_size=BATCH, N=N, dt=DT, max_qp_iters=SP['max_sqp_iters'], num_threads=BATCH,
        Q_cost=SP['q_cost'], dQ_cost=SP['qd_cost'], R_cost=SP['u_cost'], QN_cost=SP['N_cost'])
    nq, nv, nx, nu = bt.nq, bt.nv, bt.nx, bt.nu
    print(f"iiwa14 BatchThneed fig8: center(EE)={center.round(4)} frame={fig8mod.EE_FRAME} "
          f"A={fig8mod.FIG8_A} T={fig8mod.FIG8_PERIOD} N={N} nq={nq} goal_steps={n_goal}")

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

    errs = fig8mod.ee_tracking_errors(model, data, q_log, goal, dt=DT)
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
