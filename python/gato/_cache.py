"""Per-host cache of registered robots and compiled solver modules — the
`pip install gato` path.

    gato.register_robot("myarm", "arm.urdf", N=[32])      # codegen only (seconds)
    gato.BSQP("arm.urdf", N=32, plant_type="myarm", ...)   # first use compiles (minutes, once)

Layout (``GATO_CACHE_DIR`` or platformdirs' user cache, e.g. ~/.cache/gato):
    manifest.json                 name -> robot entry (urdf hash, options, codegen key, meta)
    robots/<codegen_key>/<name>/  grid.cuh, limits.cuh (one dir per URDF+options+generator)
    modules/<name>/<module_key>/  bsqpN{N}_{name}[_variant].so + meta.json
    build/<name>/<module_key>/    CMake tree while compiling (removed on success)

A module key hashes everything that shapes the binary: the robot's codegen key
(URDF bytes, options, grid-rbd provenance), N and variant, the content identity
of GATO's native sources plus the dependency headers (tools/native_identity.py,
the same identity every module embeds), and the toolchain (CUDA arch, nvcc,
host compiler, Python ABI, pybind11). Unchanged inputs never recompile; any
change is a miss, never a stale binary. The checkout's modules in python/gato/
(the receipt profile) take precedence over the cache when both exist.
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
        "cmake": tool(["cmake", "--version"], 0),
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
                   build=False, cache_dir_override=None, arch=None, jobs=2, force=False) -> dict:
    """Generate a robot's dynamics headers into the per-host cache and record it
    in the manifest. Compiles the solver modules now only with ``build=True``;
    otherwise the first ``gato.BSQP(..., plant_type=name)`` compiles (once)."""
    from .builder import _codegen_key, codegen

    if not str(name).isidentifier():
        raise ValueError(f"robot name {name!r} must be a valid identifier (it becomes the module suffix)")
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    cache = Path(cache_dir_override) if cache_dir_override else cache_dir()
    key = _codegen_key(urdf_path, ee_frame, collision_res, contact_frames, floating_base)
    dyn = cache / "robots" / key / name
    with _locked(cache):
        if force or not (dyn / "grid.cuh").exists() or not (dyn / "limits.cuh").exists():
            try:
                meta = codegen(urdf_path, name, ee_frame=ee_frame, out_dir=dyn, register=False,
                               collision_res=collision_res, contact_frames=contact_frames,
                               floating_base=floating_base)
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
                             "contact_frames": list(contact_frames) if contact_frames else None},
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


def ensure_module(name: str, N: int, variant: str = "default", *, cache: Path | None = None,
                  arch: str | None = None, jobs: int = 2, quiet: bool = False) -> Path:
    """Return the cached module for (name, N, variant), compiling it once if missing."""
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
            log.warning("gato: compiling %s for sm_%s — a one-time build (typically 2–6 minutes on a "
                        "desktop GPU, longer on Jetson); the result is cached in %s", modname, arch or "native", cache)
        import pybind11

        build = cache / "build" / name / key
        shutil.rmtree(build, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        cfg = ["cmake", "-S", str(native_root()), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               "-DGATO_BUILD_DEMO=OFF", "-DGATO_RECEIPT_PROFILE=OFF",
               f"-DMODULES={name}:{N}:{variant}", f"-DGATO_PLANT_ROOT={plant_root}",
               f"-DGATO_MODULE_OUTPUT_DIR={out}", f"-DPython3_EXECUTABLE={sys.executable}",
               f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"]
        if arch:
            cfg.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
        t0 = time.time()
        with open(out / "build.log", "w") as logf:
            try:
                subprocess.run(cfg, check=True, stdout=logf, stderr=subprocess.STDOUT)
                subprocess.run(["cmake", "--build", str(build), "--parallel", str(int(jobs))],
                               check=True, stdout=logf, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"compiling {modname} failed (exit {e.returncode}); see {out / 'build.log'}") from e
        built = _module_glob(out, name, N, variant)
        if not built:
            raise RuntimeError(f"build reported success but {modname}.so did not land in {out}")
        (out / "meta.json").write_text(json.dumps({"inputs": inputs, "seconds": round(time.time() - t0, 1),
                                                   "module": built[0].name}, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(build, ignore_errors=True)
        _native_source_id.cache_clear()   # a later source edit in an editable checkout must re-key
        if not quiet:
            log.warning("gato: %s compiled in %.0f s and cached", modname, time.time() - t0)
        return built[0]


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
            for variant in ("default", "fc", "eh"):
                key, _ = module_key(robot, n, variant, plant_root, arch)
                sos = _module_glob(cache / "modules" / name / key, name, n, variant)
                if sos:
                    found[(name, int(n), variant)] = str(sos[0])
    return found
