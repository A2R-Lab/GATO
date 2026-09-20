"""Golden-value regression gate (plan Wave 0.4).

Every other determinism test is run-twice on a FRESH solver, which a
deterministic numeric regression passes. This gate pins the actual numbers:
for every module in test/receipt_modules.txt, a fixed problem is solved on
the pcg and bdsv linear-system paths (plus one constrained configuration per
arm) and xu / merits / iteration counts are compared BITWISE against
test/golden/<plant>_N<N>_<case>.npz.

Re-baseline (only with a documented numeric change — say why in the commit):
    GATO_GOLDEN_REBASELINE=1 pytest test/test_parity_golden.py
The goldens were first captured at GRiD bc9c4d7 / GLASS 78329b6 (2026-09-20)
so the Wave-1 pin bump is measured against them (plan D6).
"""
import os
from pathlib import Path

import numpy as np
import pytest

import gato
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
REBASELINE = os.environ.get("GATO_GOLDEN_REBASELINE") == "1"
ARM_START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}
GOAL_XYZ = (0.35, 0.25, 0.5)


def receipt_modules():
    out = []
    for line in (HERE / "receipt_modules.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            plant, n = line.split()
            out.append((plant, int(n)))
    return out


def _arm_problem(plant, N):
    q0 = np.asarray(ARM_START[plant], dtype=np.float32)
    x = np.concatenate([q0, np.zeros_like(q0)])[None, :]
    goals = np.zeros((1, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = GOAL_XYZ
    return x, goals


def _go2_problem(N):
    import test_floating_rowgroups as fr   # standing keyframe + imu goal at the current pose
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(fr.URDF), pin.JointModelFreeFlyer())
    x = fr._standing_x()
    return np.asarray(x, dtype=np.float32)[None, :], fr._goals_at(model, x, 1)


def _make(plant, N, urdfs):
    if plant == "go2":
        import test_floating_rowgroups as fr
        return fr._solver(1)
    return gato.BSQP(model_path=str(urdfs[plant]), batch_size=1, N=N, dt=0.01, plant_type=plant)


def _cases():
    cases = []
    for plant, N in receipt_modules():
        for linsys in ("pcg", "bdsv"):
            cases.append((plant, N, linsys))
        if plant != "go2" and N == 16:
            cases.append((plant, N, "admm_ee_pcg"))   # constraint layer: limit ADMM + EE terminal row
    return cases


@pytest.mark.parametrize("plant,N,case", _cases(), ids=lambda v: str(v))
def test_golden(plant, N, case, urdfs):
    if (plant, N) not in gato.available():
        pytest.fail(f"receipt module bsqpN{N}_{plant} is not built (test/receipt_modules.txt)")
    s = _make(plant, N, urdfs)
    x, goals = _go2_problem(N) if plant == "go2" else _arm_problem(plant, N)
    if case == "admm_ee_pcg":
        s.set_linsys("pcg")
        s.enable_limit_admm()
        s.enable_ee_terminal_equality(np.asarray(GOAL_XYZ, dtype=np.float32), rho=10.0)
    else:
        s.set_linsys(case)
    r = s.solve(x, goals)
    got = dict(xu=np.asarray(r.xu, np.float32),
               final_merit=np.asarray(r.stats.final_merit, np.float32),
               initial_merit=np.asarray(r.stats.initial_merit, np.float32),
               sqp_iters=np.asarray(r.stats.sqp_iters, np.int32),
               kkt_converged=np.asarray(r.stats.kkt_converged, np.int32))
    assert np.isfinite(got["xu"]).all()
    path = GOLDEN / f"{plant}_N{N}_{case}.npz"
    if REBASELINE or not path.exists():
        GOLDEN.mkdir(exist_ok=True)
        np.savez(path, **got)
        if not REBASELINE:
            pytest.fail(f"no golden for {path.name}; captured it — review and commit (or set GATO_GOLDEN_REBASELINE=1)")
        return
    want = np.load(path)
    for k, v in got.items():
        np.testing.assert_array_equal(v, want[k], err_msg=f"{path.name}[{k}] differs from golden — "
                                      "numeric regression, or a documented change needing GATO_GOLDEN_REBASELINE=1")
