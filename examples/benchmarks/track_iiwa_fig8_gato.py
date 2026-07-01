"""GATO iiwa14 fig8 tracking on the FAIR shared problem (see iiwa_fig8_shared.py).
Uses the canonical fig8 (center = grid-EE/L7 at readyC, A=0.15, T=6), fixed-dt pacing, and measures
tracking error at the L7 frame (grid-EE) so it is directly comparable to MPCGPU's validate_track and the
BatchThneed baseline. Run with the GRiD venv (pinocchio); needs the prebuilt bsqpN64_iiwa14 module.

  PYTHONPATH=/home/plancher/Desktop/GATO/python \
  /home/plancher/Desktop/GRiD/.venv/bin/python examples/benchmarks/track_iiwa_fig8_gato.py [sim_time]
"""
import sys, os
import numpy as np
sys.path.insert(0, "/home/plancher/Desktop/GATO/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iiwa_fig8_shared as fig8mod
from bsqp.mpc_controller import MPC_GATO

SIM_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
DT = fig8mod.DT
N = 64

model, data = fig8mod.build_model()
q0 = fig8mod.Q0_READYC
center = fig8mod.fig8_center(model, data, q0)

# prefer the byte-identical MPCGPU-generated goal; else synthesize (verified equal)
goal = fig8mod.load_goal_file()
n_needed = int(SIM_TIME / DT) + N + 8
if goal is None or len(goal) // 6 < n_needed:
    goal = fig8mod.figure8_goal(n_needed, center=center)
print(f"iiwa14 GATO fig8: center(L7)={center.round(4)} A={fig8mod.FIG8_A} T={fig8mod.FIG8_PERIOD} "
      f"N={N} sim_time={SIM_TIME}  goal_steps={len(goal)//6}")

x_start = np.hstack((q0, np.zeros(model.nv)))
mpc = MPC_GATO(model=model, N=N, dt=DT, batch_size=1, model_path=fig8mod.IIWA14_URDF,
               plant_type='iiwa14', constant_f_ext=None, track_full_stats=True)
# fixed-dt pacing (advance sim by dt each control step) => goal index == step index, matching MPCGPU's
# CONST_UPDATE_FREQ and the CPU baseline. This is also the fairest per-solve timing basis.
_, stats = mpc.run_mpc_fig8(x_start, goal, sim_dt=0.001, sim_time=SIM_TIME, pace_by_solve_time=False)

# measure at L7 from the logged joint configs (NOT solver.ee_pos, which reports the offset contact frame)
jp = stats.get('joint_positions', [])
errs = fig8mod.l7_tracking_errors(model, data, jp, goal, dt=DT)
st = np.asarray(stats['solve_times'], float)
sqp = np.asarray(stats.get('sqp_iters', []), float)
if len(errs):
    print(f"RESULT_GATO steps={len(errs)} L7_mean={errs.mean():.6f} L7_max={errs.max():.6f} "
          f"L7_final={errs[-1]:.6f}  median_solve_ms={np.median(st):.4f}  "
          f"sqp_iters~{np.mean(sqp) if len(sqp) else float('nan'):.2f}")
    print("trace:", " ".join(f"{errs[i]:.4f}" for i in range(0, len(errs), max(1, len(errs)//20))))
else:
    print("no tracking samples (joint_positions empty?)  keys:", list(stats.keys()))
