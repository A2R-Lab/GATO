"""Intro example 5: bring your own robot — register a URDF, then solve.

gato.register_robot(name, urdf, N=[...], ee_frame=...) runs the GRiD codegen
for the robot into GATO's per-host cache (seconds) and records it under `name`.
The first gato.BSQP(..., plant_type=name) for a horizon then compiles the
bsqpN{N}_<name> solver module once (a few minutes, logged when it starts) and
caches it; every later process loads it instantly. This works from a plain
`pip install gato` — no checkout, no submodule. Fixed-base serial chains with
bounded <limit> tags (floating_base=True for a quaternion free-flyer root);
ee_frame must be a URDF FIXED joint (the EE target frame the tracking cost and
FK use). Codegen is skipped when the URDF / grid-rbd release / options are
unchanged, and a changed toolchain or GPU architecture recompiles the module
rather than loading a stale one.

    python examples/05_build_your_robot.py path/to/robot.urdf --name myrobot --ee-frame EE
    python examples/05_build_your_robot.py            # the vendored indy7 under a new name, N=16
    python examples/05_build_your_robot.py --build-now    # compile at registration instead of first use

In a source checkout, gato.build(urdf, name=..., N=[...]) is the equivalent
that writes gato/dynamics/<name>/ and python/gato/ (what tools/build.sh and the
receipt profile use); the cache path above is the one a wheel user has.
"""
import argparse
from pathlib import Path

import numpy as np

import gato
from gato import BSQP, SolverParams


def main():
    p = argparse.ArgumentParser(description="register a URDF with GATO, then solve once.")
    p.add_argument("urdf", nargs="?", default=str(gato.robot_urdf("indy7")),
                   help="robot URDF (default: the vendored indy7)")
    p.add_argument("--name", default=None,
                   help="plant name = module suffix (default: '<urdf stem>_custom')")
    p.add_argument("--N", type=int, default=16, help="horizon length")
    p.add_argument("--ee-frame", default="EE", help="URDF fixed joint of the EE target frame")
    p.add_argument("--floating-base", action="store_true", help="quaternion free-flyer root")
    p.add_argument("--build-now", action="store_true", help="compile at registration (default: on first use)")
    p.add_argument("--jobs", type=int, default=2, help="parallel compile jobs (RAM-bound)")
    args = p.parse_args()

    name = args.name or f"{Path(args.urdf).stem}_custom"
    meta = gato.register_robot(name, args.urdf, N=[args.N], ee_frame=args.ee_frame,
                               floating_base=args.floating_base, build=args.build_now, jobs=args.jobs)
    print(f"registered {name!r}: nq={meta['nq']} nv={meta['nv']} ee_frame={meta['ee_frame']} "
          f"(cache: {gato._cache.cache_dir()})")

    # the first solver for (name, N) compiles the module unless --build-now did
    solver = BSQP(args.urdf, batch_size=1, N=args.N, dt=0.01,
                  params=SolverParams(max_sqp_iters=5), plant_type=name)
    print(f"module: {gato.available()[(name, args.N)]}")

    # smoke solve: hold the rest posture, EE target = the EE position at rest
    q0 = np.zeros(solver.nq, dtype=np.float32)
    if args.floating_base:
        q0[6] = 1.0                                       # unit quaternion (x, y, z, w)
    x0 = np.concatenate([q0, np.zeros(solver.nv, dtype=np.float32)])
    goal = np.zeros(6 * args.N, dtype=np.float32)
    goal.reshape(args.N, 6)[:, :3] = solver.ee_pos(q0)   # pinocchio FK of the EE frame
    res = solver.solve(x0[None], goal[None])
    print(f"smoke solve ({name}, N={args.N}): {res.solve_time_us / 1000:.3f} ms, "
          f"sqp_iters={res.stats.sqp_iters[0]}, merit {res.stats.initial_merit[0]:.3g} -> "
          f"{res.stats.final_merit[0]:.3g}, finite={np.isfinite(res.xu).all()}")


if __name__ == "__main__":
    main()
