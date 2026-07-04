"""Packaging/import hygiene: numpy-only import, registry schema, config sanity."""
import json
import subprocess
import sys
from pathlib import Path

import gato
from gato.config import STANDARD_BATCH_SIZES


def test_import_without_heavy_deps(repo_root):
    """`import gato` + the eager surface must work with pinocchio/torch/gymnasium
    unavailable (the base install is numpy-only)."""
    code = (
        "import sys\n"
        "for mod in ('pinocchio', 'torch', 'gymnasium'):\n"
        "    sys.modules[mod] = None\n"  # poisons import -> ImportError on use
        f"sys.path.insert(0, {str(repo_root / 'python')!r})\n"
        "import gato\n"
        "assert callable(gato.BSQP)\n"
        "assert isinstance(gato.available(), dict)\n"
        "assert isinstance(gato.robot_info('indy7'), dict)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_lazy_exports_resolve_to_modules():
    import gato as g
    # every lazy name must point at a module that defines it (catches renames)
    import importlib
    for name, mod in g._LAZY.items():
        spec = importlib.util.find_spec(mod, package="gato")
        assert spec is not None, f"lazy export {name} -> missing module {mod}"


def test_registry_schema(repo_root):
    reg_path = repo_root / "python" / "gato" / "_registry.json"
    assert reg_path.exists(), "vendored robots must be registered (_registry.json)"
    reg = json.loads(reg_path.read_text())
    for name in ("indy7", "iiwa14"):
        assert name in reg
        meta = reg[name]
        assert set(meta) >= {"nq", "nv", "ee_frame", "urdf"}
        assert meta["nq"] == meta["nv"] > 0
        assert (repo_root / meta["urdf"]).exists()


def test_config_batch_sizes():
    assert STANDARD_BATCH_SIZES and all(
        isinstance(b, int) and b >= 1 for b in STANDARD_BATCH_SIZES)


def test_available_shape():
    for (plant, N), fname in gato.available().items():
        assert isinstance(plant, str) and isinstance(N, int) and N >= 2
        assert fname.startswith(f"bsqpN{N}_{plant}.")
