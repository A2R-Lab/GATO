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


def _parse_urdf(urdf_path, ee_frame, floating_base=False):
    """Parse with GRiD's URDFParser (same parse codegen uses); validate ee_frame."""
    root = repo_root()
    grid_root = root / "external" / "GRiD"
    # GRiD packaging layout: grid_codegen at the GRiD root, URDFParser under
    # external/ (both importable from a raw checkout; no pip install needed).
    if not (grid_root / "external" / "URDFParser").exists():
        raise RuntimeError(
            f"GRiD submodule not found at {grid_root}. Run: "
            f"git submodule update --init --recursive"
        )
    for p in (str(grid_root), str(grid_root / "external")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from URDFParser import URDFParser

    robot = URDFParser().parse(str(urdf_path), floating_base=bool(floating_base))
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
    """Emit the per-robot limits.cuh from the parsed URDF <limit> tags.

    Floating base: the synthesized base free-joint carries no <limit> tags and
    its dofs are unbounded by construction — the tables cover the ACTUATED
    joints only (CL-3; fixed base is unchanged: every joint is actuated).
    ⚠ do not consume grid::init_joint_limits() instead: it is misaligned /
    uninitialized for floating robots (GRiD gen_init_joint_limits bug, ask
    filed 2026-08-09)."""
    joints = [j for j in robot.get_joints_ordered_by_id()
              if getattr(j, "jtype", None) != "floating"]
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


def _write_gato_abi(grid_cuh_path, out_path, name):
    """Emit gato_abi.cuh: caller-scratch arena offsets for the grid_plant step
    surfaces (FLOATING-BASE robots only).

    The plant_step_gradient_and_value / integrator_inner scratch pieces carry
    no emitted per-piece constants — the layout exists only inside the
    generated kernels' "GRID shared arena layout" recipes. Until GRiD emits
    carve helpers upstream (ASK 6, asks_from_gato_2026-08-09.md), parse the
    recipes of the two kernels GATO consumes and bake the element offsets.
    FAIL LOUD on drift: each parsed total must equal the TIER_SHARED count
    baked into the corresponding *_DYNAMIC_SHARED_MEM_BYTES sizer.

    The file's presence is also the preprocessor-level floating fact
    (__has_include in dynamics/grid_plant_step.cuh) — fixed-base robots must
    NOT get one (their TUs stay preprocessor-identical)."""
    import re
    text = Path(grid_cuh_path).read_text()

    def parse_arena(kernel_name):
        at = text.find(f"void {kernel_name}(")
        if at < 0:
            raise RuntimeError(f"{kernel_name} not found in {grid_cuh_path}")
        start = text.find("// GRID shared arena layout", at)
        end = text.find("extern __shared__", start)
        if start < 0 or end < 0:
            raise RuntimeError(f"arena recipe not found for {kernel_name}")
        pieces, int_count = [], None
        for line in text[start:end].splitlines()[1:]:
            if not line.strip():
                continue
            m = re.match(r"\s*//\s+(T|int|bytes)\s+(\w+)\[(\w+|.+?)\]\s*$", line)
            if not m:
                raise RuntimeError(f"unparsed arena line for {kernel_name}: {line!r}")
            kind, pname, count = m.groups()
            if kind == "T":
                pieces.append((pname, int(count)))
            elif kind == "int":
                int_count = int(count)
            # bytes (linalg helper region) handled symbolically on the C++ side
        return pieces, int_count

    def sizer_count(sizer_name):
        m = re.search(sizer_name + r"\(\)\s*\{\s*if constexpr \(TIER == TIER_SHARED\)\s+"
                      r"return grid_shared_arena_bytes<T>\((\d+),", text)
        if not m:
            raise RuntimeError(f"{sizer_name} TIER_SHARED count not found")
        return int(m.group(1))

    blocks = []
    for prefix, kernel, sizer in (
            ("SG", "integrator_with_gradient_kernel", "INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES"),
            ("ST", "integrator_kernel", "INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES")):
        pieces, int_count = parse_arena(kernel)
        total = sum(n for _, n in pieces)
        expect = sizer_count(sizer)
        if total != expect or int_count is None:
            raise RuntimeError(
                f"arena recipe drift for {kernel}: parsed T total {total} vs "
                f"{sizer} TIER_SHARED {expect} (ints: {int_count}) — the grid.cuh "
                f"emission changed; update _write_gato_abi / take GRiD's carve helpers")
        lines = [f"// {kernel} (TIER_SHARED / SCRATCH_IN_SMEM=true)"]
        off = 0
        for pname, n in pieces:
            lines.append(f"inline constexpr int {prefix}_{pname[2:].upper()} = {off};  // [{n}]")
            off += n
        lines.append(f"inline constexpr int {prefix}_T_COUNT = {off};")
        lines.append(f"inline constexpr int {prefix}_TOPOLOGY_INT_COUNT = {int_count};")
        blocks.append("\n".join(lines))

    # the fixed-target EE-pose inners are suffixed with the EE frame name
    # (arms: _EE; go2: _imu_joint) — plant.cuh's raw evaluators reach them
    # through these aliases (fixed base falls back to the historic _EE names)
    suffixes = set(re.findall(r"void end_effector_pose_inner_(\w+)\(", text))
    grad_suffixes = set(re.findall(r"void end_effector_pose_gradient_inner_(\w+)\(", text))
    if len(suffixes) != 1 or suffixes != grad_suffixes:
        raise RuntimeError(f"expected exactly one suffixed EE-pose inner pair, got "
                           f"{suffixes} / {grad_suffixes}")
    ee_suffix = suffixes.pop()

    out = "\n".join([
        "#pragma once",
        f"// GENERATED by gato.builder (_write_gato_abi) for {name} — DO NOT EDIT.",
        "// Caller-scratch arena element offsets for the grid_plant step surfaces,",
        "// parsed from the generated grid.cuh kernel arena recipes. Emitted for",
        "// FLOATING-BASE robots only; its presence is the preprocessor-level",
        "// floating fact (__has_include in dynamics/grid_plant_step.cuh).",
        "// The int topology region and the (constexpr-sized) linalg helper bytes",
        "// are carved AFTER the T region by the consumer.",
        "#define GATO_GRID_STEP_ABI 1",
        "",
        "// fixed-target EE-pose inner names (suffixed with the EE frame name)",
        f"#define GATO_GRID_EE_POSE_INNER end_effector_pose_inner_{ee_suffix}",
        f"#define GATO_GRID_EE_POSE_GRAD_INNER end_effector_pose_gradient_inner_{ee_suffix}",
        "",
        "namespace gato_abi {",
        "",
        blocks[0], "", blocks[1], "",
        "}  // namespace gato_abi",
        ""])
    Path(out_path).write_text(out)


def codegen(urdf_path, name, ee_frame="EE", algorithm_list=None, out_dir=None,
            register=True, collision_res=0.15, contact_frames=None,
            floating_base=False):
    """Generate gato/dynamics/<name>/{grid.cuh, limits.cuh} + register the robot.

    Returns the registry metadata dict. This is the single codegen path — both
    gato.build() and tools/regen_grid.py go through it. algorithm_list=None uses
    GRiD's profile="all" (recommended; also emits the grid_plant cost surface).
    out_dir/register let tests generate elsewhere without touching the repo.

    collision_res: sphere spacing in meters for the grid_collision namespace
    (CL-2). The URDF's <collision> geometry is spherized at this resolution
    (GRiD's spherizer; meshes need trimesh) and grid.cuh grows the
    grid_collision:: device ABI (config_free + the differentiable clearance
    family). REQUIRED for solver modules — the constraint layer
    (plant.cuh/rowgroups.cuh) compiles against grid_collision:: — so the
    default bakes it (0.15). None emits no collision code (headers-only /
    codegen-inspection use; bsqp modules will not compile against it).
    Coarser = fewer/fatter spheres; the sphere set IS the clearance
    row set, so keep it small (0.15 ⇒ ~30-45 spheres on the vendored arms).

    contact_frames: list of URDF fixed-joint names to bake as contact frames
    (CL-3 prep): grid.cuh grows the f_ext_body[_jacobian_*] contact-wrench map.
    None (default) bakes [ee_frame] — every GATO plant carries its EE contact
    map; pass [] to opt out.

    floating_base: parse/generate with a quaternion free-flyer root (CL-3;
    go2 nq=19, nv=18). The solver-side floating path is CL-3 W3.3+ — until it
    lands, floating headers are codegen/inspection-only (bsqp modules
    static_assert out).
    """
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    root = repo_root()
    out_dir = Path(out_dir) if out_dir else root / "gato" / "dynamics" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = _parse_urdf(urdf_path, ee_frame, floating_base=floating_base)
    # Limits first: it validates every joint has bounded <limit> tags (cheap),
    # so an unsupported robot fails before the expensive grid.cuh generation.
    _write_limits(robot, out_dir / "limits.cuh", name, urdf_path.name)
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

    # LAUNCH_CONFIG_ROBOT: the launch_configs dir is keyed by the URDF filename
    # stem (go2), NOT the URDF <robot name=...> (go2_description) — without the
    # explicit id the lookup silently misses and every algo falls back to the
    # conservative tier/threads default. FLOATING-ONLY for now: the vendored
    # iiwa14/indy7 grid.cuh were (accidentally) generated with that miss
    # ("KUKAiiwa14"/"indy" ≠ stem), so wiring the id for fixed base would
    # change their baked launch tables — a perf re-baseline, not a correctness
    # fix; queued for a timing window (SSOT cl3_floating_base_2026-08-09.md).
    gen_kwargs = dict(DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
    if floating_base:
        gen_kwargs["LAUNCH_CONFIG_ROBOT"] = Path(urdf_path).stem
    gen = GRiDCodeGenerator(robot, **gen_kwargs)
    kwargs = dict(
        include_homogenous_transforms=True,  # required for EE pose + gradient
        fixed_target_name=ee_frame,
        output_path=str(out_dir / "grid.cuh"),
    )
    if algorithm_list is None:
        kwargs["codegen_profile"] = "all"
    else:
        kwargs["algorithm_list"] = list(algorithm_list)
    n_spheres = None
    if collision_res is not None:
        from grid_codegen.algorithms._collision import (
            collision_spec_from_urdf, normalize_collision_tiers)
        # FAIL LOUD on silent coverage loss: GRiD's spherizer warns + SKIPs a
        # link whose collision mesh can't be resolved/loaded — but the sphere
        # set IS the clearance row set, so a skipped link silently vanishes
        # from collision checking. (A missing trimesh already raises; the
        # conservative bounding-box degrade stays a warning.)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings("error", message=r".*SKIPPING.*")
            spec = collision_spec_from_urdf(robot, str(urdf_path),
                                            resolution=float(collision_res))
        n_spheres = int(normalize_collision_tiers(spec)[-1]["n"])
        kwargs["collision_spec"] = spec
    if contact_frames is None:
        contact_frames = [ee_frame]
    if contact_frames:
        from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
        kwargs["contact_frames"] = contact_frames_from_urdf(robot, list(contact_frames))
    gen.gen_all_code(**kwargs)
    if floating_base:
        _write_gato_abi(out_dir / "grid.cuh", out_dir / "gato_abi.cuh", name)

    try:
        urdf_rel = str(urdf_path.relative_to(root))
    except ValueError:
        urdf_rel = str(urdf_path)
    meta = {"nq": robot.get_num_pos(), "nv": robot.get_num_vel(),
            "ee_frame": ee_frame, "urdf": urdf_rel,
            "floating_base": bool(floating_base)}
    if collision_res is not None:
        meta["collision"] = {"res": float(collision_res), "n_spheres": n_spheres}
    if contact_frames:
        meta["contact_frames"] = list(contact_frames)
    if register:
        _update_registry(name, meta)
    return meta


def build(urdf_path, name=None, N=(32,), ee_frame="EE", arch=None, jobs=4,
          build_dir=None, collision_res=0.15, contact_frames=None,
          floating_base=False):
    """Codegen + compile the bsqpN{N}_<name> solver modules for a URDF.

    Args:
        urdf_path: robot URDF (fixed-base serial chain, or floating_base=True
            for a quaternion free-flyer root — CL-3: the solver runs the
            grid_plant SE(3) step/linearization path, SI-EULER integrator).
        name: plant name (module suffix, dynamics dir); default = URDF stem.
        N: iterable of horizon lengths to build.
        ee_frame: fixed-joint name of the EE target frame (codegen + metrics).
        arch: CMAKE_CUDA_ARCHITECTURES override (default: native detection).
        jobs: parallel compile jobs (each TU pulls the large grid.cuh; keep small).
        build_dir: CMake build tree (default <repo>/build). NOTE: the tree is
            reconfigured for exactly this (plant, N) request.
        collision_res / contact_frames: see codegen() — collision sphere
            spacing (enables the grid_collision clearance rows) and CL-3
            contact-frame baking.
    Returns: list of built (name, N) pairs.
    """
    name = name or Path(urdf_path).stem
    if not name.isidentifier():
        raise ValueError(f"plant name {name!r} must be a valid identifier "
                         f"(it becomes the module suffix); pass name= explicitly")
    Ns = [int(n) for n in ([N] if isinstance(N, int) else N)]

    root = repo_root()
    codegen(urdf_path, name, ee_frame=ee_frame, collision_res=collision_res,
            contact_frames=contact_frames, floating_base=floating_base)

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
