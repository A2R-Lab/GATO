"""GATO fig8 tracking on the FAIR shared problem (see iiwa_fig8_shared.py; iiwa14 default, --robot indy7).
Uses the canonical fig8 (center = grid-EE = URDF "EE" fixed joint at the robot's q0, A=0.15, T=6),
fixed-dt pacing, and measures tracking error at the EE frame so it is directly comparable to MPCGPU's
validate_track and the QDLDL-based CPU baseline. Needs the prebuilt bsqpN{N}_{robot} module and a
python with pinocchio (the project .venv).

  python examples/benchmarks/track_iiwa_fig8_gato.py [sim_time] [--robot iiwa14] [--N 64]
"""
import argparse
import numpy as np
import iiwa_fig8_shared as fig8mod
from gato.mpc_gato import MPC_GATO

_p = argparse.ArgumentParser(description="GATO fig8 closed-loop tracking (FAIR harness)")
_p.add_argument("sim_time", nargs="?", type=float, default=6.0)
_p.add_argument("--robot", choices=fig8mod.ROBOTS, default="iiwa14")
_p.add_argument("--N", type=int, default=64)
_a = _p.parse_args()
SIM_TIME, N = _a.sim_time, _a.N
DT = fig8mod.DT

rob = fig8mod.robot(_a.robot)
model, data = fig8mod.build_model(rob.name)
q0 = rob.q0
center = fig8mod.fig8_center(model, data, q0, rob.ee_frame)

# prefer the byte-identical MPCGPU-generated goal (iiwa14); else synthesize (verified equal)
n_needed = int(SIM_TIME / DT) + N + 8
goal, goal_source = fig8mod.goal_sequence(rob, n_needed, center)
print(f"{rob.name} GATO fig8: center(EE)={center.round(4)} A={fig8mod.FIG8_A} T={fig8mod.FIG8_PERIOD} "
      f"N={N} sim_time={SIM_TIME}  goal_steps={len(goal)//6} goal={goal_source}")

x_start = np.hstack((q0, np.zeros(model.nv)))
mpc = MPC_GATO(model=model, N=N, dt=DT, batch_size=1, model_path=rob.urdf,
               plant_type=rob.name, constant_f_ext=None, track_full_stats=True,
               # pin the paper-era pcg path (controller default is "auto" since
               # 08-12; committed baselines were measured under pcg)
               linsys="pcg")
# fixed-dt pacing (advance sim by dt each control step) => goal index == step index, matching MPCGPU's
# CONST_UPDATE_FREQ and the CPU baseline. This is also the fairest per-solve timing basis.
_, stats = mpc.run_mpc_fig8(x_start, goal, sim_dt=0.001, sim_time=SIM_TIME, pace_by_solve_time=False)

# measure at EE from the logged joint configs (uniform metric across all three solvers; post-regen
# the solver frame coincides with EE, so this equals solver.ee_pos up to the pin-vs-grid FK path)
jp = stats.get('joint_positions', [])
errs = fig8mod.ee_tracking_errors(model, data, jp, goal, dt=DT, frame=rob.ee_frame)
st = np.asarray(stats['solve_times'], float)
sqp = np.asarray(stats.get('sqp_iters', []), float)
if len(errs):
    print(f"RESULT_GATO steps={len(errs)} EE_mean={errs.mean():.6f} EE_max={errs.max():.6f} "
          f"EE_final={errs[-1]:.6f}  median_solve_ms={np.median(st):.4f}  "
          f"sqp_iters~{np.mean(sqp) if len(sqp) else float('nan'):.2f}")
    print("trace:", " ".join(f"{errs[i]:.4f}" for i in range(0, len(errs), max(1, len(errs)//20))))
else:
    print("no tracking samples (joint_positions empty?)  keys:", list(stats.keys()))
