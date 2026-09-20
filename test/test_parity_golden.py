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
so the Wave-1 pin bump is measured against them (plan D6) — bit-identical at
GRiD 3af782e / GLASS e83b086.
Re-baselined 2026-09-20 (Wave 2.A): BSQP.solve() became stateless and its cold
seed is hold-at-x (initialize_warm_start) instead of the old all-zeros
trajectory buffer; indy7 (non-zero start config) cases changed, iiwa14/go2
were bit-identical (their start states make the two seeds coincide).
"""
import os
from pathlib import Path

import numpy as np
import pytest

import gato
from conftest import TEST_PARAMS
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
REBASELINE = os.environ.get("GATO_GOLDEN_REBASELINE") == "1"
ARM_START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}
GOAL_XYZ = (0.35, 0.25, 0.5)


def receipt_modules():
    """[(plant, N, variant)] from test/receipt_modules.txt."""
    out = []
    for line in (HERE / "receipt_modules.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            out.append((parts[0], int(parts[1]), parts[2] if len(parts) > 2 else "default"))
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


def _make(plant, N, variant, urdfs):
    if plant == "go2":
        import test_floating_rowgroups as fr
        return fr._solver(1)
    params = TEST_PARAMS.replace(exact_hessian=True) if variant == "eh" else TEST_PARAMS
    return gato.BSQP(model_path=str(urdfs[plant]), batch_size=1, N=N, dt=0.01, params=params,
                     plant_type=plant, variant=variant)


def _cases():
    cases = []
    for plant, N, variant in receipt_modules():
        if variant == "eh":
            cases.append((plant, N, variant, "bdsv"))      # exact Hessian forces the direct path
            continue
        for linsys in ("pcg", "bdsv"):
            cases.append((plant, N, variant, linsys))
        if plant != "go2" and N == 16 and variant == "default":
            cases.append((plant, N, variant, "admm_ee_pcg"))   # constraint layer: limit ADMM + EE terminal row
    return cases


@pytest.mark.parametrize("plant,N,variant,case", _cases(), ids=lambda v: str(v))
def test_golden(plant, N, variant, case, urdfs):
    if (plant, N) not in gato.available(variant):
        pytest.fail(f"receipt module {gato.module_name(plant, N, variant)} is not built (test/receipt_modules.txt)")
    s = _make(plant, N, variant, urdfs)
    x, goals = _go2_problem(N) if plant == "go2" else _arm_problem(plant, N)
    if variant == "fc":
        s.set_fc_ref(np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 5.0], s.n_fc // 6).astype(np.float32))  # a 5 N press reference
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
    path = GOLDEN / (f"{plant}_N{N}_{case}.npz" if variant == "default" else f"{plant}_N{N}_{variant}_{case}.npz")
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
