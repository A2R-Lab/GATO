"""gato.build dogfood: codegen + compile + import + smoke solve on a temp name.

NOTE: gato.build reconfigures the repo's CMake build tree for its (plant, N)
request; re-run your usual `cmake -S . -B build -DPLANT=... -DKNOTS=...` after
this test if you drive builds by hand.
"""
import json
import shutil

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


def test_build_end_to_end(repo_root, urdfs):
    pytest.importorskip("pinocchio")
    try:
        built = gato.build(urdfs["indy7"], name=NAME, N=[8], jobs=4)
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
