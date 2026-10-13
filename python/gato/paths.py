"""Build settings for compiling your own C++/CUDA program against GATO's solver.

    python -m gato.paths --robot indy7 --N 32                 # readable
    python -m gato.paths --robot indy7 --N 32 --format cmake  # for CMake execute_process
    python -m gato.paths --robot indy7 --N 32 --format nvcc   # flags for a bare nvcc line

The solver is the header-only ``BSQP<T>`` class (gato/bsqp/bsqp.cuh); the
Python package wraps it. A translation unit that includes it needs: GATO's
header root, the robot's generated directory (grid.cuh + limits.cuh — from the
checkout's gato/dynamics, the wheel's bundled copy, or the per-host cache for
robots registered with gato.register_robot), the GLASS and collision headers
from the pinned grid-rbd wheel, and three defines selecting the horizon, the
plant and the plant adapter. Variants add one define each.
"""
from __future__ import annotations

import argparse
import json

from . import _cache, _grid

VARIANT_DEFINES = {"default": [], "fc": ["GATO_CONTACT_FORCES=1"], "eh": ["USE_EXACT_HESSIAN=1"]}
CUDA_FLAGS = ["-std=c++17", "-O3", "-DNDEBUG", "-use_fast_math"]


def settings(robot: str, N: int, variant: str = "default") -> dict:
    """Include directories, defines and flags for (robot, N, variant)."""
    if variant not in VARIANT_DEFINES:
        raise ValueError(f"variant must be one of {sorted(VARIANT_DEFINES)}, got {variant!r}")
    resolved = _cache.resolve(robot)
    if resolved is None:
        raise ValueError(f"unknown robot {robot!r}: not vendored and not registered with gato.register_robot")
    entry, plant_root = resolved
    plant_dir = plant_root / robot
    if not (plant_dir / "grid.cuh").exists():
        raise FileNotFoundError(f"{plant_dir}/grid.cuh is missing; (re)register the robot")
    root = _cache.native_root()
    _grid.require()
    return {
        "robot": robot, "N": int(N), "variant": variant,
        "include_dirs": [str(root / "gato"), str(plant_dir), str(_grid.glass_dir()), str(_grid.collision_dir())],
        "defines": [f"KNOT_POINTS={int(N)}", f"GATO_PLANT_NAME={robot}", 'GATO_PLANT_HEADER="dynamics/plant.cuh"',
                    *VARIANT_DEFINES[variant]],
        "cuda_flags": list(CUDA_FLAGS),
        "example": str(root / "examples" / "cpp"),
        "gato_root": str(root),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--variant", default="default", choices=sorted(VARIANT_DEFINES))
    ap.add_argument("--format", default="json", choices=["json", "cmake", "nvcc"])
    args = ap.parse_args(argv)
    s = settings(args.robot, args.N, args.variant)
    if args.format == "json":
        print(json.dumps(s, indent=2))
    elif args.format == "cmake":   # one line: include dirs | defines | flags, comma-separated within each part
        print("|".join([",".join(s["include_dirs"]), ",".join(s["defines"]), ",".join(s["cuda_flags"])]))
    else:
        print(" ".join([*s["cuda_flags"], *(f"-I{d}" for d in s["include_dirs"]),
                        *(f"-D{d}" for d in s["defines"])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
