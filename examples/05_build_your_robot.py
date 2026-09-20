"""Intro example 5: bring your own robot — gato.build on a URDF, then a smoke solve.

gato.build(urdf, name=..., N=[...], ee_frame=...) runs the GRiD codegen for the
robot (gato/dynamics/<name>/{grid.cuh, limits.cuh} + the registry entry) and
compiles the bsqpN{N}_<name> solver module(s) in one call. Fixed-base serial
chains with bounded <limit> tags; ee_frame must be a URDF FIXED joint (the EE
target frame the tracking cost and FK use). Codegen is skipped when the URDF /
GRiD pin / options are unchanged (a content key in the registry).

    python examples/05_build_your_robot.py path/to/robot.urdf --name myrobot --ee-frame EE
    python examples/05_build_your_robot.py            # rebuilds the vendored indy7 at N=16

NOTE: gato.build reconfigures the default build/ tree for exactly this
(plant, N) request — re-run ./tools/build.sh afterwards to restore the full
module matrix. Each module compile peaks ~6-7 GB of RAM; keep --jobs small.
"""
import argparse
from pathlib import Path

import numpy as np

import gato
from gato import BSQP, SolverParams

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="gato.build a URDF, then solve once.")
    p.add_argument("urdf", nargs="?", default=str(REPO / gato.robot_info("indy7")["urdf"]),
                   help="robot URDF (default: the vendored indy7)")
    p.add_argument("--name", default=None, help="plant name = module suffix (default: URDF stem)")
    p.add_argument("--N", type=int, default=16, help="horizon length to build")
    p.add_argument("--ee-frame", default="EE", help="URDF fixed joint of the EE target frame")
    p.add_argument("--jobs", type=int, default=2, help="parallel compile jobs (RAM-bound)")
    args = p.parse_args()

    name = args.name or Path(args.urdf).stem
    built = gato.build(args.urdf, name=name, N=[args.N], ee_frame=args.ee_frame, jobs=args.jobs)
    print(f"built modules: {built}")
    info = gato.robot_info(name)
    print(f"registry entry for {name!r}: nq={info['nq']} nv={info['nv']} ee_frame={info['ee_frame']}")

    # smoke solve: hold the rest posture, EE target = the EE position at rest
    solver = BSQP(args.urdf, batch_size=1, N=args.N, dt=0.01,
                  params=SolverParams(max_sqp_iters=5), plant_type=name)
    q0 = np.zeros(solver.nq, dtype=np.float32)
    x0 = np.concatenate([q0, np.zeros(solver.nv, dtype=np.float32)])
    goal = np.zeros(6 * args.N, dtype=np.float32)
    goal.reshape(args.N, 6)[:, :3] = solver.ee_pos(q0)   # pinocchio FK of the EE frame
    res = solver.solve(x0[None], goal[None])
    print(f"smoke solve ({name}, N={args.N}): {res.solve_time_us / 1000:.3f} ms, "
          f"sqp_iters={res.stats.sqp_iters[0]}, merit {res.stats.initial_merit[0]:.3g} -> "
          f"{res.stats.final_merit[0]:.3g}, finite={np.isfinite(res.xu).all()}")


if __name__ == "__main__":
    main()
