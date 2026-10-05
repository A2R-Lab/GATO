#!/usr/bin/env python3
"""Content identity embedded in each native module; no GPU or build side effects."""
import argparse
import hashlib
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def native_identity(root, plant):
    root = Path(root)
    files = {root / p for p in ("CMakeLists.txt", "python/bindings.cu", "tools/native_identity.py")}
    for directory in ("gato", "gato/dynamics"):
        files.update(p for p in (root / directory).iterdir() if p.suffix in (".h", ".cuh", ".hpp"))
    for directory in ("gato/bsqp", "gato/utils", f"gato/dynamics/{plant}",
                      "external/GLASS", "external/GRiD/grid_codegen/collision"):
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"Missing native input directory: {folder}")
        files.update(p for p in folder.rglob("*") if p.suffix in (".h", ".hpp", ".cuh", ".cu") and p.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
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
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_receipt_modules()
    elif args.plant:
        print(native_identity(ROOT, args.plant))
    else:
        parser.error("choose --plant or --verify")
