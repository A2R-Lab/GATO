#!/usr/bin/env python3
"""Content identity embedded in each native module; no GPU or build side effects."""
import argparse
import hashlib
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


NATIVE_SUFFIXES = (".h", ".hpp", ".cuh", ".cu")


def _grid_dirs():
    """GLASS + collision header roots from the pinned grid-rbd wheel (gato._grid)."""
    sys.path.insert(0, str(ROOT / "python"))
    from gato import _grid

    return {"GLASS": _grid.glass_dir(), "grid_collision": _grid.collision_dir()}


def native_identity(root, plant, grid_dirs=None, plant_root=None):
    """Content hash of every native input of the (plant) modules: repo headers,
    the robot's generated headers (from ``plant_root``/<plant>, default the
    checkout's gato/dynamics), and the dependency headers (GLASS, GRiD
    collision). Dependency and robot headers are keyed by logical name, so the
    identity is independent of where the wheel or cache lives but changes with
    their contents."""
    root = Path(root)
    files = {root / p for p in ("CMakeLists.txt", "python/bindings.cu", "tools/native_identity.py")}
    for directory in ("gato", "gato/dynamics"):
        files.update(p for p in (root / directory).iterdir() if p.suffix in (".h", ".cuh", ".hpp"))
    for directory in ("gato/bsqp", "gato/utils"):
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"Missing native input directory: {folder}")
        files.update(p for p in folder.rglob("*") if p.suffix in NATIVE_SUFFIXES and p.is_file())
    entries = {p.relative_to(root).as_posix(): p for p in files}
    plant_dir = Path(plant_root) / plant if plant_root else root / "gato" / "dynamics" / plant
    if not plant_dir.is_dir():
        raise ValueError(f"Missing native input directory: {plant_dir}")
    for p in plant_dir.rglob("*"):
        if p.suffix in NATIVE_SUFFIXES and p.is_file():
            entries[f"gato/dynamics/{plant}/{p.relative_to(plant_dir).as_posix()}"] = p
    for label, folder in sorted((grid_dirs if grid_dirs is not None else _grid_dirs()).items()):
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Missing native input directory: {folder}")
        for p in folder.rglob("*"):
            if p.suffix in NATIVE_SUFFIXES and p.is_file():
                entries[f"{label}/{p.relative_to(folder).as_posix()}"] = p
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(entries[name].read_bytes()).digest())
    return digest.hexdigest()


def verify_receipt_modules(root=ROOT):
    sys.path.insert(0, str(root / "python"))
    identities = {}
    for line in (root / "test/receipt_modules.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        plant, knots, *variant = line.split()
        suffix = "" if not variant or variant[0] == "default" else "_" + variant[0]
        name = f"bsqpN{knots}_{plant}{suffix}"
        if plant not in identities:
            identities[plant] = native_identity(root, plant)
        module = importlib.import_module("gato." + name)
        if getattr(module, "NATIVE_SOURCE_ID", None) != identities[plant]:
            raise ValueError(f"{name}: stale/unidentified binary; reconfigure and rebuild the receipt profile")
    print("Receipt native module source identities verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant")
    parser.add_argument("--plant-root", help="directory holding <plant>/{grid.cuh,limits.cuh} (default: gato/dynamics)")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_receipt_modules()
    elif args.plant:
        print(native_identity(ROOT, args.plant, plant_root=args.plant_root))
    else:
        parser.error("choose --plant or --verify")
