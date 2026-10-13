"""The pinned GRiD dependency: where the released `grid-rbd` wheel keeps the
code generator, the GLASS headers and the collision headers GATO compiles against.

GATO builds against exactly one released grid-rbd version (GRID_RBD_PIN) whose
bundled GLASS is the commit GATO's own kernels were validated with (GLASS_PIN).
The wheel's provenance manifest records both, so an installed package needs no
Git checkout; `require()` refuses anything else with the install command.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

GRID_RBD_PIN = "0.5.1"
GLASS_PIN = "9e57178ca146fa1cecae7e27c22260603b993aa6"
INSTALL_HINT = f'pip install "grid-rbd=={GRID_RBD_PIN}"  (or ./tools/install.sh)'


def installed_version() -> str | None:
    try:
        return version("grid-rbd")
    except PackageNotFoundError:
        return None


def provenance() -> dict:
    """The wheel's bundled provenance (peer commits + source hashes); {} for a checkout."""
    from grid_codegen import resources

    return resources.bundled_provenance() or {}


def glass_commit() -> str | None:
    return provenance().get("peers", {}).get("GLASS", {}).get("commit")


def require() -> dict:
    """Assert the installed grid-rbd is the pinned release; return its provenance."""
    got = installed_version()
    if got is None:
        raise RuntimeError(f"grid-rbd is not installed: {INSTALL_HINT}")
    if got != GRID_RBD_PIN:
        raise RuntimeError(f"grid-rbd {got} is installed but GATO pins {GRID_RBD_PIN}: {INSTALL_HINT}")
    prov = provenance()
    if not prov:
        raise RuntimeError("grid-rbd is an editable checkout without a provenance manifest; "
                           f"GATO builds against the released wheel: {INSTALL_HINT}")
    glass = prov.get("peers", {}).get("GLASS", {}).get("commit")
    if glass != GLASS_PIN:
        raise RuntimeError(f"grid-rbd {got} bundles GLASS {glass}, GATO's kernels are validated "
                           f"against {GLASS_PIN}; reinstall the pinned release: {INSTALL_HINT}")
    return prov


def glass_dir() -> Path:
    from grid_codegen import resources

    return Path(resources.resource_path("GLASS"))


def collision_dir() -> Path:
    from grid_codegen import resources

    return Path(resources.collision_include_dir())


def include_dirs() -> list[Path]:
    """Header roots every GATO native TU needs beyond the repo: GLASS (global
    `glass::`) and GRiD's collision primitives (`grid_collision_geometry.cuh`)."""
    require()
    return [glass_dir(), collision_dir()]


if __name__ == "__main__":  # CMake: semicolon list of include dirs
    print(";".join(str(p) for p in include_dirs()))
