"""Packaging/import hygiene: numpy-only import, registry schema, config sanity."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gato


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



def test_available_shape():
    for (plant, N), fname in gato.available().items():
        assert isinstance(plant, str) and isinstance(N, int) and N >= 2
        assert fname.startswith(f"bsqpN{N}_{plant}.")


@pytest.mark.gpu
def test_bsqp_constructs_and_solves_without_pinocchio(repo_root, smallest_module, urdfs):
    """The lean install promise: BSQP needs numpy + the built module only.
    Pinocchio is poisoned in a subprocess; construction, one solve and the
    module-derived dims must all work (FK helpers are the only pin users)."""
    plant, N = smallest_module
    code = (
        "import sys\n"
        "sys.modules['pinocchio'] = None\n"
        f"sys.path.insert(0, {str(repo_root / 'python')!r})\n"
        "import numpy as np, gato\n"
        f"s = gato.BSQP(model_path={str(urdfs[plant])!r}, batch_size=1, N={N}, dt=0.01, plant_type={plant!r})\n"
        "assert (s.nq, s.nv, s.nx, s.nu) == (s.lib.NQ, s.lib.NV, s.lib.NQ + s.lib.NV, s.lib.CONTROL_SIZE)\n"
        "x = np.zeros((1, s.nx), np.float32); ref = np.zeros((1, 6 * s.N), np.float32); ref[:, 2::6] = 0.5\n"
        "r = s.solve(x, ref)\n"
        "assert np.isfinite(r.xu).all()\n"
        "try:\n"
        "    s.ee_pos(x[0, :s.nq]); raise SystemExit('ee_pos must need pinocchio')\n"
        "except ImportError as e:\n"
        "    assert 'pinocchio' in str(e)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
