"""gato.build: one-call robot codegen + solver-module build.

    import gato
    gato.build("path/to/robot.urdf", N=[32, 64])          # name defaults to the URDF stem
    gato.build("panda.urdf", name="panda", ee_frame="panda_hand")

Generates ``gato/dynamics/<name>/{grid.cuh, limits.cuh}`` from the URDF (via the
vendored GRiD code generator), registers the robot in ``python/gato/_registry.json``,
and compiles the ``bsqpN{N}_<name>`` pybind modules with CMake. Afterwards
``BSQP(model_path=urdf, N=..., plant_type=name)`` just works.

v1 targets a source checkout (the repo's CMakeLists + external/GRiD submodule are
required); a wheels-only JIT build rides the planned artifact cache.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REGISTRY_PATH = _PKG_DIR / "_registry.json"


def repo_root():
    """The GATO source checkout root (needed for codegen + CMake)."""
    root = _PKG_DIR.parents[1]
    if not (root / "CMakeLists.txt").exists() or not (root / "gato" / "dynamics").exists():
        raise RuntimeError(
            f"gato.build needs a GATO source checkout (CMakeLists.txt + gato/dynamics), "
            f"but {root} is not one. Clone the repo and `pip install -e .` from it."
        )
    return root


def load_registry():
    """Robot metadata registry: {name: {nq, nv, ee_frame, urdf}}."""
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _update_registry(name, meta):
    reg = load_registry()
    reg[name] = meta
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_urdf(urdf_path, ee_frame):
    """Parse with GRiD's URDFParser (same parse codegen uses); validate ee_frame."""
    root = repo_root()
    grid_root = root / "external" / "GRiD"
    if not (grid_root / "URDFParser").exists():
        raise RuntimeError(
            f"GRiD submodule not found at {grid_root}. Run: "
            f"git submodule update --init --recursive"
        )
    if str(grid_root) not in sys.path:
        sys.path.insert(0, str(grid_root))
    from URDFParser import URDFParser

    robot = URDFParser().parse(str(urdf_path), floating_base=False)  # fixed base
    if robot is None:
        raise ValueError(f"URDFParser could not parse {urdf_path}")

    fixed_names = [fj.name for fj in robot.fixed_joints]
    if ee_frame not in fixed_names:
        raise ValueError(
            f"ee_frame={ee_frame!r} is not a fixed joint of {Path(urdf_path).name} "
            f"(the EE target must be a fixed frame); available: {fixed_names}"
        )
    return robot


def _fmt(v):
    """Full-precision C++ float literal for a URDF limit value."""
    return repr(float(v))


def _write_limits(robot, out_path, name, urdf_name):
    """Emit the per-robot limits.cuh from the parsed URDF <limit> tags."""
    joints = robot.get_joints_ordered_by_id()
    rows = {"JOINT_LIMITS_DATA": [], "VEL_LIMITS_DATA": [], "CTRL_LIMITS_DATA": []}
    for j in joints:
        lo, hi = j.joint_limits
        vel, eff = j.get_velocity_limit(), j.get_effort_limit()
        for v, what in ((lo, "lower"), (hi, "upper"), (vel, "velocity"), (eff, "effort")):
            if v is None or not math.isfinite(float(v)):
                raise ValueError(
                    f"joint {j.name!r} has no finite {what} limit in {urdf_name} — "
                    f"GATO's barrier cost needs bounded <limit> tags on every joint "
                    f"(continuous/unbounded joints are not supported yet)"
                )
        rows["JOINT_LIMITS_DATA"].append((lo, hi, j.name))
        rows["VEL_LIMITS_DATA"].append((-vel, vel, j.name))
        rows["CTRL_LIMITS_DATA"].append((-eff, eff, j.name))

    nq = len(joints)
    out = [
        "#pragma once",
        f"// Joint/velocity/effort limit tables for {name}, from the URDF <limit> tags.",
        "// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit",
        "// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).",
        "",
        "namespace gato {",
        "namespace plant {",
    ]
    for table, entries in rows.items():
        out += ["", "        template<class T>",
                f"        __device__ constexpr T {table}[{nq}][2] = {{"]
        for i, (lo, hi, jname) in enumerate(entries):
            comma = "," if i < nq - 1 else ""
            out.append(f"            {{{_fmt(lo)} - JOINT_LIMIT_MARGIN<T>(), "
                       f"{_fmt(hi)} + JOINT_LIMIT_MARGIN<T>()}}{comma}  // {jname}")
        out.append("        };")
    out += ["", "}  // namespace plant", "}  // namespace gato", ""]
    Path(out_path).write_text("\n".join(out))


def codegen(urdf_path, name, ee_frame="EE", algorithm_list=None, out_dir=None,
            register=True):
    """Generate gato/dynamics/<name>/{grid.cuh, limits.cuh} + register the robot.

    Returns the registry metadata dict. This is the single codegen path — both
    gato.build() and tools/regen_grid.py go through it. algorithm_list=None uses
    GRiD's profile="all" (recommended; also emits the grid_plant cost surface).
    out_dir/register let tests generate elsewhere without touching the repo.
    """
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    root = repo_root()
    out_dir = Path(out_dir) if out_dir else root / "gato" / "dynamics" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = _parse_urdf(urdf_path, ee_frame)
    # Limits first: it validates every joint has bounded <limit> tags (cheap),
    # so an unsupported robot fails before the expensive grid.cuh generation.
    _write_limits(robot, out_dir / "limits.cuh", name, urdf_path.name)
    from GRiDCodeGenerator import GRiDCodeGenerator

    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
                            FILE_NAMESPACE="grid")
    kwargs = dict(
        include_homogenous_transforms=True,  # required for EE pose + gradient
        fixed_target_name=ee_frame,
        output_path=str(out_dir / "grid.cuh"),
    )
    if algorithm_list is None:
        kwargs["codegen_profile"] = "all"
    else:
        kwargs["algorithm_list"] = list(algorithm_list)
    gen.gen_all_code(**kwargs)

    try:
        urdf_rel = str(urdf_path.relative_to(root))
    except ValueError:
        urdf_rel = str(urdf_path)
    meta = {"nq": robot.get_num_pos(), "nv": robot.get_num_vel(),
            "ee_frame": ee_frame, "urdf": urdf_rel}
    if register:
        _update_registry(name, meta)
    return meta


def build(urdf_path, name=None, N=(32,), ee_frame="EE", arch=None, jobs=4,
          build_dir=None):
    """Codegen + compile the bsqpN{N}_<name> solver modules for a URDF.

    Args:
        urdf_path: robot URDF (fixed base).
        name: plant name (module suffix, dynamics dir); default = URDF stem.
        N: iterable of horizon lengths to build.
        ee_frame: fixed-joint name of the EE target frame (codegen + metrics).
        arch: CMAKE_CUDA_ARCHITECTURES override (default: native detection).
        jobs: parallel compile jobs (each TU pulls the large grid.cuh; keep small).
        build_dir: CMake build tree (default <repo>/build). NOTE: the tree is
            reconfigured for exactly this (plant, N) request.
    Returns: list of built (name, N) pairs.
    """
    name = name or Path(urdf_path).stem
    if not name.isidentifier():
        raise ValueError(f"plant name {name!r} must be a valid identifier "
                         f"(it becomes the module suffix); pass name= explicitly")
    Ns = [int(n) for n in ([N] if isinstance(N, int) else N)]

    root = repo_root()
    codegen(urdf_path, name, ee_frame=ee_frame)

    try:
        import pybind11
        pybind11_dir = pybind11.get_cmake_dir()
    except ImportError as e:
        raise RuntimeError(
            "pybind11 is required to build solver modules: pip install pybind11"
        ) from e

    build_dir = Path(build_dir) if build_dir else root / "build"
    cfg = ["cmake", "-S", str(root), "-B", str(build_dir),
           "-DCMAKE_BUILD_TYPE=Release", "-DGATO_BUILD_DEMO=OFF",
           f"-DPLANT={name}", f"-DKNOTS={';'.join(map(str, Ns))}",
           f"-DPython3_EXECUTABLE={sys.executable}",
           f"-Dpybind11_DIR={pybind11_dir}"]
    if arch:
        cfg.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    subprocess.run(cfg, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel", str(jobs)],
                   check=True)

    built = []
    for n in Ns:
        sos = list(_PKG_DIR.glob(f"bsqpN{n}_{name}.*.so")) + \
              [p for p in [_PKG_DIR / f"bsqpN{n}_{name}.so"] if p.exists()]
        if not sos:
            raise RuntimeError(f"build reported success but bsqpN{n}_{name}.so "
                               f"did not land in {_PKG_DIR}")
        built.append((name, n))
    return built
