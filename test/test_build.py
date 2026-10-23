"""gato.build dogfood: codegen + compile + import + smoke solve on a temp name.

The test uses its own temporary CMake tree and leaves the receipt profile's
configuration intact.
"""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

import gato
from gato.builder import _REGISTRY_PATH

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

NAME = "testbot_indy7"


def _cleanup(repo_root):
    shutil.rmtree(repo_root / "gato" / "dynamics" / NAME, ignore_errors=True)
    for so in (repo_root / "python" / "gato").glob(f"bsqpN*_{NAME}*.so"):
        so.unlink()
    if _REGISTRY_PATH.exists():
        reg = json.loads(_REGISTRY_PATH.read_text())
        if reg.pop(NAME, None) is not None:
            _REGISTRY_PATH.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


def test_build_end_to_end(repo_root, urdfs, tmp_path):
    pytest.importorskip("pinocchio")
    try:
        # its OWN CMake tree: gato.build reconfigures the tree it is given for exactly
        # this (plant, N) request, and pointing it at the shared <repo>/build used to
        # leave that tree on the test plant — the next profile rebuild then failed
        # on a missing grid.cuh until someone re-ran cmake (2026-09-26, twice)
        built = gato.build(urdfs["indy7"], name=NAME, N=[8], jobs=2, build_dir=tmp_path / "build")
        assert built == [(NAME, 8)]
        assert (NAME, 8) in gato.available()
        meta = gato.robot_info(NAME)
        assert meta["nq"] == meta["nv"] == 6 and meta["ee_frame"] == "EE"

        s = gato.BSQP(model_path=str(urdfs["indy7"]), batch_size=2, N=8, dt=0.01,
                      plant_type=NAME)
        x = np.zeros((2, s.nx), dtype=np.float32)
        goals = np.zeros((2, 8 * 6), dtype=np.float32)
        goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
        res = s.solve(x, goals)
        assert np.isfinite(res.xu).all()
    finally:
        _cleanup(repo_root)


@pytest.mark.slow
def test_fresh_tree_targets_the_local_gpu(repo_root, tmp_path):
    """A fresh configure without an explicit architecture compiles for the local
    GPU (-arch=native), not the compiler default. The native default ran after
    project() had enabled CUDA until 2026-10-24, so fresh trees built sm_75
    modules that ran on the 5090 through JIT with other GLASS dispatch variants
    (PCG shifted by rounding, the direct solver returned a zero step)."""
    build = tmp_path / "cmk"
    pybind11_dir = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
    out = subprocess.run(["cmake", "-S", str(repo_root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                          "-DGATO_RECEIPT_PROFILE=OFF", "-DMODULES=iiwa14:8", f"-DPython3_EXECUTABLE={sys.executable}",
                          f"-Dpybind11_DIR={pybind11_dir}"], check=True, capture_output=True, text=True).stdout
    assert "Targeting CUDA Architectures: native" in out, out
    flags = (build / "CMakeFiles" / "bsqpN8_iiwa14.dir" / "flags.make").read_text()
    assert "-arch=native" in flags, flags
