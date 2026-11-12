"""Per-host cache of registered robots and compiled solver modules — the
`pip install gato` path.

    gato.register_robot("myarm", "arm.urdf", N=[32])      # codegen only (seconds)
    gato.BSQP("arm.urdf", N=32, plant_type="myarm", ...)   # first use compiles (minutes, once)

Layout (``GATO_CACHE_DIR`` or platformdirs' user cache, e.g. ~/.cache/gato):
    manifest.json                 name -> robot entry (urdf hash, options, codegen key, meta)
    robots/<codegen_key>/<name>/  grid.cuh, limits.cuh (one dir per URDF+options+generator)
    modules/<name>/<module_key>/  bsqpN{N}_{name}[_variant].so (a link into the store) + meta.json
    store/                        grid-rbd's consumer artifact store: artifacts, headers, logs

Modules compile through ``grid_rbd.build_artifact`` (since grid-rbd 0.7.0): GRiD
generates the robot's header, compiles GATO's bindings translation unit against
it with nvcc and links the extension; no CMake on this path. A module key hashes
everything that shapes the binary: the robot's codegen key (URDF bytes, options,
grid-rbd provenance), N and variant, the content identity of GATO's native
sources plus the dependency headers (tools/native_identity.py, the same identity
every module embeds), and the toolchain (CUDA arch, nvcc, host compiler, Python
ABI, pybind11). Unchanged inputs never recompile; any change is a miss, never a
stale binary; the store's own content key (sources, headers found by nvcc's
dependency pass, defines, flags, toolchain) dedupes the builds underneath. The
checkout's modules in python/gato/ (the receipt profile) take precedence over
the cache when both exist.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("gato")

_PKG_DIR = Path(__file__).resolve().parent
MANIFEST_SCHEMA = 1


# --- locations -------------------------------------------------------------

def cache_dir() -> Path:
    env = os.environ.get("GATO_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    import platformdirs

    return Path(platformdirs.user_cache_dir("gato"))


def validate_merit_threads(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"GATO_MERIT_THREADS must be an integer, got {value!r}") from None
    if n <= 0 or n % 32 or n > 1024:
        raise ValueError(f"GATO_MERIT_THREADS must be a multiple of 32 between 32 and 1024, got {n}")
    return n


def merit_threads() -> int | None:
    """GATO_MERIT_THREADS: the merit kernel's block size (issue #19); None = the
    configure-time default, 64 or 32 on the Jetson family (CMakeLists.txt).
    Compiled into the module and part of its cache key, so several sizes can
    live in one cache."""
    raw = os.environ.get("GATO_MERIT_THREADS", "").strip()
    return validate_merit_threads(raw) if raw else None


def native_root() -> Path:
    """The CMake source tree solver modules compile from: the wheel's bundled
    copy (gato/_native) or the source checkout around an editable install."""
    bundled = _PKG_DIR / "_native"
    if (bundled / "CMakeLists.txt").exists():
        return bundled
    root = _PKG_DIR.parents[1]
    if (root / "CMakeLists.txt").exists() and (root / "gato" / "dynamics").exists():
        return root
    raise RuntimeError("GATO's native sources are missing: neither a bundled gato/_native tree "
                       f"nor a source checkout around {_PKG_DIR}. Reinstall gato.")


def _identity_module():
    spec = importlib.util.spec_from_file_location("gato_native_identity", native_root() / "tools" / "native_identity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- identities -------------------------------------------------------------

def detect_arch() -> str | None:
    """CUDA architecture of the first GPU as CMake wants it ("120"), None if unknown."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        cap = out.strip().splitlines()[0].strip()
        major, minor = cap.split(".")
        return f"{int(major)}{int(minor)}"
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def toolchain_identity(arch: str | None) -> dict:
    """Everything outside the sources that shapes a binary (memoized per process)."""
    def tool(cmd, line):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
            return out.splitlines()[line]
        except Exception as e:  # a missing tool surfaces at CMake time with a real message
            return f"unavailable: {e.__class__.__name__}"

    import pybind11

    return {
        "arch": arch or "native",
        "nvcc": tool(["nvcc", "--version"], -1),
        "cxx": tool(["c++", "--version"], 0),
        "python_ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "pybind11": pybind11.__version__,
        "machine": platform.machine(),
    }


@functools.lru_cache(maxsize=None)
def _native_source_id(name: str, plant_root: str) -> str:
    return _identity_module().native_identity(native_root(), name, plant_root=Path(plant_root))


def module_key(robot: dict, N: int, variant: str, plant_root: Path, arch: str | None) -> tuple[str, dict]:
    """Content key of one solver module + the readable inputs it was derived from."""
    native_id = _native_source_id(robot["name"], str(plant_root))
    tc = toolchain_identity(arch)
    inputs = {"codegen_key": robot["codegen_key"], "N": int(N), "variant": variant,
              "native_source_id": native_id, "toolchain": tc, "schema": MANIFEST_SCHEMA}
    mt = merit_threads()
    if mt is not None:          # absent when unset, so existing cache keys are unchanged
        inputs["merit_threads"] = mt
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    return key, inputs


# --- manifest ----------------------------------------------------------------

@contextmanager
def _locked(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_manifest(cache: Path | None = None) -> dict:
    path = (cache or cache_dir()) / "manifest.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("schema") == MANIFEST_SCHEMA:
            return data
    return {"schema": MANIFEST_SCHEMA, "robots": {}}


def _save_manifest(cache: Path, data: dict):
    tmp = cache / "manifest.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(cache / "manifest.json")


def robot_entry(name: str, cache: Path | None = None) -> dict | None:
    return load_manifest(cache)["robots"].get(name)


def _vendored_entry(name: str) -> dict | None:
    """The robots GATO ships headers for (python/gato/_registry.json + gato/dynamics/<name>)
    are pre-registered: no register_robot call, keyed by their generated bytes."""
    from .builder import load_registry

    meta = load_registry().get(name)
    plant = native_root() / "gato" / "dynamics" / name
    if meta is None or not (plant / "grid.cuh").exists():
        return None
    h = hashlib.sha256()
    for f in ("grid.cuh", "limits.cuh"):
        h.update((plant / f).read_bytes())
    return {"name": name, "codegen_key": "vendored-" + h.hexdigest()[:16], "urdf": meta.get("urdf"),
            "N": [], "variant": "default", "meta": meta, "vendored": True}


def resolve(name: str, cache: Path | None = None) -> tuple[dict, Path] | None:
    """(manifest or vendored entry, directory holding <name>/grid.cuh) or None."""
    cache = cache or cache_dir()
    entry = robot_entry(name, cache)
    if entry is not None:
        return entry, cache / "robots" / entry["codegen_key"]
    entry = _vendored_entry(name)
    if entry is not None:
        return entry, native_root() / "gato" / "dynamics"
    return None


# --- registration -------------------------------------------------------------

def register_robot(name, urdf_path, *, N=(32,), ee_frame="EE", floating_base=False,
                   collision_res=0.15, contact_frames=None, variant="default",
                   build=False, cache_dir_override=None, arch=None, jobs=2, force=False,
                   joint_dynamics=False, friction_width=0.03, cost_frames=None, com=False,
                   joint_armature=None) -> dict:
    """Generate a robot's dynamics headers into the per-host cache and record it
    in the manifest. Compiles the solver modules now only with ``build=True``;
    otherwise the first ``gato.BSQP(..., plant_type=name)`` compiles (once).
    ``joint_dynamics=True`` bakes the URDF's ``<dynamics damping>`` and smooth
    friction (``friction_width`` in rad/s) into the dynamics and its
    derivatives (see gato.builder.codegen). ``joint_armature`` bakes a reflected
    rotor inertia per velocity slot (a scalar for every actuated slot, or a
    vector of length nv), as Pinocchio's ``model.armature``. ``cost_frames`` names extra joints
    baked as cost frames for the cost menu's frame terms (fixed base; the
    end-effector target is always frame 0)."""
    from .builder import _armature_option, _codegen_key, codegen

    if not str(name).isidentifier():
        raise ValueError(f"robot name {name!r} must be a valid identifier (it becomes the module suffix)")
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    cache = Path(cache_dir_override) if cache_dir_override else cache_dir()
    cost_frames = [str(f) for f in (cost_frames or [])]
    key = _codegen_key(urdf_path, ee_frame, collision_res, contact_frames, floating_base, joint_dynamics, friction_width,
                       cost_frames, com, joint_armature)
    dyn = cache / "robots" / key / name
    with _locked(cache):
        if force or not (dyn / "grid.cuh").exists() or not (dyn / "limits.cuh").exists():
            try:
                meta = codegen(urdf_path, name, ee_frame=ee_frame, out_dir=dyn, register=False,
                               collision_res=collision_res, contact_frames=contact_frames,
                               floating_base=floating_base, joint_dynamics=joint_dynamics,
                               friction_width=friction_width, cost_frames=cost_frames, com=com,
                               joint_armature=joint_armature)
            except ValueError as e:
                if "could not be resolved" in str(e):
                    raise ValueError(
                        f"{e}\nregister_robot needs the robot's collision meshes next to its URDF (the collision "
                        f"spheres are generated from them): keep the mesh folders with the URDF, or resolve "
                        f"package:// URIs through ROS_PACKAGE_PATH.") from e
                raise
            (dyn / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        else:
            meta = json.loads((dyn / "meta.json").read_text())
        meta = dict(meta, urdf=str(urdf_path))
        entry = {"name": name, "codegen_key": key, "urdf": str(urdf_path),
                 "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                 "options": {"ee_frame": ee_frame, "floating_base": bool(floating_base),
                             "collision_res": collision_res,
                             "contact_frames": list(contact_frames) if contact_frames else None,
                             "joint_dynamics": bool(joint_dynamics),
                             "friction_width": float(friction_width) if joint_dynamics else None,
                             "cost_frames": cost_frames or None, "com": bool(com),
                             "joint_armature": (lambda a: list(a) if isinstance(a, tuple) else a)(
                                 _armature_option(joint_armature))},
                 "N": sorted({int(n) for n in ([N] if isinstance(N, int) else N)}),
                 "variant": variant, "meta": meta, "registered": time.strftime("%Y-%m-%dT%H:%M:%S")}
        data = load_manifest(cache)
        data["robots"][name] = entry
        _save_manifest(cache, data)
    if build:
        for n in entry["N"]:
            ensure_module(name, n, variant, cache=cache, arch=arch, jobs=jobs)
    return meta


# --- modules -------------------------------------------------------------------

def _module_glob(out: Path, name: str, N: int, variant: str) -> list[Path]:
    suffix = "" if variant == "default" else f"_{variant}"
    return sorted(out.glob(f"bsqpN{N}_{name}{suffix}.*.so")) + sorted(out.glob(f"bsqpN{N}_{name}{suffix}.so"))


def find_module(name: str, N: int, variant: str = "default", cache: Path | None = None) -> Path | None:
    """The cached .so for (name, N, variant) if one was built from the CURRENT
    inputs (stale entries from older sources/toolchains are ignored)."""
    cache = cache or cache_dir()
    resolved = resolve(name, cache)
    if resolved is None:
        return None
    robot, plant_root = resolved
    if not (plant_root / name / "grid.cuh").exists():
        return None
    key, _ = module_key(robot, N, variant, plant_root, detect_arch())
    sos = _module_glob(cache / "modules" / name / key, name, N, variant)
    return sos[0] if sos else None


VARIANT_DEFINES = {"default": [], "fc": ["GATO_CONTACT_FORCES=1"], "eh": ["USE_EXACT_HESSIAN=1"], "rk4": ["GATO_INTEGRATOR=3"]}
EXTRA_FLAGS = ("-O3", "-use_fast_math", "-std=c++17", "-Xcompiler=-fvisibility=hidden")   # CMakeLists.txt Release + pybind11's visibility


def store_dir(cache: Path) -> Path:
    """grid-rbd's consumer store under GATO's cache (build_artifact's cache_dir)."""
    return cache / "store"


def _codegen_request(robot: dict) -> dict:
    """build_artifact's codegen keywords for a robot entry (register_robot's options, or the
    vendored registry meta): the options gato.builder.codegen bakes, so the store's header is
    the one the vendored path generates."""
    from .builder import algorithms_for

    opts = robot.get("options") or {}
    meta = robot.get("meta") or {}
    floating = bool(opts.get("floating_base", meta.get("floating_base", False)))
    ee = opts.get("ee_frame") or meta.get("ee_frame") or "EE"
    cost_frames = list(opts.get("cost_frames") or (meta.get("cost_frames") or [ee])[1:])
    contact = opts.get("contact_frames") or meta.get("contact_frames") or [ee]
    com = bool(opts.get("com", meta.get("com", False)))
    req = dict(floating_base=floating, ee_joint_names=[ee], contact_frames=list(contact), cost_frames=cost_frames,
               algorithm_list=list(algorithms_for(floating, cost_frames, com)), launch_config_profile="host",
               strict_collision=True)
    res = opts.get("collision_res", (meta.get("collision") or {}).get("res"))
    if res is not None:
        req.update(collision=True, collision_res=float(res))
    if opts.get("joint_dynamics", meta.get("joint_dynamics", False)):
        req.update(use_joint_dynamics=True, friction_model="tanh",
                   friction_width=float(opts.get("friction_width") or meta.get("friction_width")))
    if meta.get("armature"):    # the baked vector, so the store's header is the cached one
        req.update(armature=[float(v) for v in meta["armature"]])
    return req


def _robot_urdf(robot: dict) -> Path:
    """The URDF the store generates from: the registered path, or the vendored robot's copy
    (the checkout's examples/ tree, or the wheel's bundled description with its collision
    meshes)."""
    raw = robot.get("urdf")
    p = Path(raw) if raw else None
    if p is not None and not p.is_absolute():
        p = native_root() / p
    if p is None or not p.exists():
        raise RuntimeError(f"robot {robot['name']!r}: its URDF ({p}) is not available to compile a solver module; "
                           "register the robot from its URDF (gato.register_robot)")
    return p


def ensure_module(name: str, N: int, variant: str = "default", *, cache: Path | None = None,
                  arch: str | None = None, jobs: int = 2, quiet: bool = False) -> Path:
    """Return the cached module for (name, N, variant), compiling it once if missing through
    grid-rbd's artifact store (one translation unit; ``jobs`` is kept for callers and unused)."""
    cache = cache or cache_dir()
    resolved = resolve(name, cache)
    if resolved is None:
        raise ValueError(f"robot {name!r} is not registered in {cache}; call gato.register_robot first")
    robot, plant_root = resolved
    arch = arch or detect_arch()
    key, inputs = module_key(robot, N, variant, plant_root, arch)
    out = cache / "modules" / name / key
    existing = _module_glob(out, name, N, variant)
    if existing:
        return existing[0]
    suffix = "" if variant == "default" else f"_{variant}"
    modname = f"bsqpN{N}_{name}{suffix}"
    with _locked(cache):
        existing = _module_glob(out, name, N, variant)   # another process may have won the race
        if existing:
            return existing[0]
        if not quiet:
            log.warning("gato: compiling %s for sm_%s through grid-rbd's artifact store, a one-time build (typically "
                        "1 to 3 minutes on a desktop GPU, longer on Jetson); the result is cached in %s", modname, arch or "native", cache)
        import pybind11
        import grid_rbd

        root = native_root()
        urdf = _robot_urdf(robot)
        out.mkdir(parents=True, exist_ok=True)
        include = out / "include"             # limits.cuh alone: the store's generated grid.cuh must not be shadowed
        include.mkdir(exist_ok=True)
        shutil.copyfile(plant_root / name / "limits.cuh", include / "limits.cuh")
        defines = [f"KNOT_POINTS={int(N)}", f"GATO_PLANT_NAME={name}", f"GATO_MODULE_VARIANT={suffix}",
                   'GATO_PLANT_HEADER="dynamics/plant.cuh"', f'GATO_NATIVE_SOURCE_ID="{inputs["native_source_id"]}"', "NDEBUG",
                   *VARIANT_DEFINES[variant]]
        threads = inputs.get("merit_threads") or (32 if arch in ("72", "87") else None)   # CMakeLists.txt's Jetson default
        if threads:
            defines.append(f"GATO_MERIT_THREADS={int(threads)}")
        req = _codegen_request(robot)
        model = grid_rbd.describe_model(urdf, floating_base=req["floating_base"])
        t0 = time.time()
        try:
            art = grid_rbd.build_artifact(
                "gato", modname, urdf, sources=[root / "python" / "bindings.cu"], headers=[include / "limits.cuh"],
                include_dirs=[root / "gato", include, pybind11.get_include(), sysconfig.get_path("include")],
                defines=defines, extra_flags=list(EXTRA_FLAGS), artifact_name=modname + sysconfig.get_config_var("EXT_SUFFIX"),
                python_extension=True, expected_descriptor_id=model["descriptor_id"], cache_dir=store_dir(cache),
                cuda_arch=int(arch) if arch else None, **req)
        except Exception as e:
            raise RuntimeError(f"compiling {modname} failed: {e}") from e
        so = out / Path(art.path).name
        if so.is_symlink() or so.exists():
            so.unlink()
        so.symlink_to(art.path)
        (out / "meta.json").write_text(json.dumps({"inputs": inputs, "seconds": round(time.time() - t0, 1), "module": so.name,
                                                   "content_key": art.content_key, "store": str(art.store_dir),
                                                   "compiled": bool(art.compiled)}, indent=2, sort_keys=True) + "\n")
        _native_source_id.cache_clear()   # a later source edit in an editable checkout must re-key
        if not quiet:
            log.warning("gato: %s %s in %.0f s and cached", modname, "compiled" if art.compiled else "reused from the store",
                        time.time() - t0)
        return so


def load_module(name: str, N: int, variant: str = "default", *, cache: Path | None = None,
                build_if_missing: bool = True):
    """Import the cached module for (name, N, variant) as ``gato.<modname>``;
    None when the robot is not registered."""
    cache = cache or cache_dir()
    if resolve(name, cache) is None:
        return None
    so = find_module(name, N, variant, cache) if not build_if_missing else ensure_module(name, N, variant, cache=cache)
    if so is None:
        return None
    suffix = "" if variant == "default" else f"_{variant}"
    fullname = f"gato.bsqpN{N}_{name}{suffix}"
    if fullname in sys.modules:
        return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname, so)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def available_cached(cache: Path | None = None) -> dict:
    """{(plant, N, variant): filename} of cache modules built from the current inputs."""
    cache = cache or cache_dir()
    found = {}
    data = load_manifest(cache)
    arch = detect_arch()
    from .builder import load_registry

    robots = {name: (robot, cache / "robots" / robot["codegen_key"]) for name, robot in data["robots"].items()}
    for name in load_registry():
        if name not in robots:
            vend = _vendored_entry(name)
            if vend is not None:
                robots[name] = (vend, native_root() / "gato" / "dynamics")
    for name, (robot, plant_root) in robots.items():
        if not (plant_root / name / "grid.cuh").exists():
            continue
        built_ns = robot.get("N") or sorted({int(p.name.split("_")[0][5:]) for d in (cache / "modules" / name).glob("*")
                                             for p in d.glob("bsqpN*.so")} if (cache / "modules" / name).exists() else [])
        for n in built_ns:
            for variant in ("default", "fc", "eh", "rk4"):
                key, _ = module_key(robot, n, variant, plant_root, arch)
                sos = _module_glob(cache / "modules" / name / key, name, n, variant)
                if sos:
                    found[(name, int(n), variant)] = str(sos[0])
    return found
