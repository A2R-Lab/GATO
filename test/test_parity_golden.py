"""Golden-value regression gate (plan Wave 0.4).

Every other determinism test is run-twice on a FRESH solver, which a
deterministic numeric regression passes. This gate pins the actual numbers:
for every module in test/receipt_modules.txt, a fixed problem is solved on
the pcg and bdsv linear-system paths (plus constrained configurations: limit
ADMM + EE terminal row and collision AL/ADMM per arm at N16, masked CONTACT_POS
AL/ADMM on go2-fc) and xu / merits / iteration counts are compared BITWISE
against test/golden/<plant>_N<N>_<case>.npz.

Re-baseline (only with a documented numeric change — say why in the commit):
    GATO_GOLDEN_REBASELINE=1 pytest test/test_parity_golden.py
The goldens were first captured at GRiD bc9c4d7 / GLASS 78329b6 (2026-09-20)
so the Wave-1 pin bump is measured against them (plan D6) — bit-identical at
GRiD 3af782e / GLASS e83b086.
Re-baselined 2026-09-20 (Wave 2.A): BSQP.solve() became stateless and its cold
seed is hold-at-x (initialize_warm_start) instead of the old all-zeros
trajectory buffer; indy7 (non-zero start config) cases changed, iiwa14/go2
were bit-identical (their start states make the two seeds coincide).
Re-baselined 2026-09-26 (GRiD pin 3af782e -> 5904dbd): iiwa14 only. URDFParser
29d3d78 stopped rationalizing transform coefficients, so iiwa14's emitted
transforms gained +-1.6e-15 terms (six XImats constants plus the sin/cos-
dependent XImats / XmatsHom / dXmatsHom / d2XmatsHom update terms) that the
old parser snapped to exact 0; indy7/go2 emit no such terms and stayed
bit-identical. SENSITIVITY NOTE: these goldens are 10-iteration UNCONVERGED
solves — a 1e-7 nudge of one input coordinate moves xu by ~1% (iiwa14) and
~30% (indy7), so ANY numeric change shows up as an O(1) drift here. A drift
therefore says "something changed", never how much: attribute it with the
referee gates (test_dynamics_fingerprint / test_f_ext / test_exact_hessian /
test_rowgroups vs the pinocchio-pinned tables) plus a compute-sanitizer
memcheck of the cheapest module BEFORE re-baselining.
Re-baselined 2026-09-26 (evening): the two exact-Hessian cases only. A non-PD
direct factor now bumps the trust-region rho x100 (settings.h
NON_PD_RHO_FACTOR) instead of the ordinary x1.2 line-search adaptation; both
eh solves hit exactly one such factor mid-solve and end at a LOWER merit
(indy7 11.23 -> 10.39, iiwa14 21.61 -> 18.02, same 10 iterations). No other
golden solve hits a non-PD factor, so the rest stayed bit-identical.
Added 2026-10-01: the four collision_{al,admm} arm cases (captured with the
pre-refactor code) so the cooperative row-kind consolidation is gated bitwise
on all three cooperative kinds (EE_POS, COLLISION, CONTACT_POS).
"""
import os
from pathlib import Path

import numpy as np
import pytest

import gato
from conftest import TEST_PARAMS, arm_problem, ARM_GOAL

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
REBASELINE = os.environ.get("GATO_GOLDEN_REBASELINE") == "1"
GOAL_XYZ = ARM_GOAL


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
    return arm_problem(plant, N)          # ARM_START at rest, ARM_GOAL at every knot (the golden problem)


def _go2_problem(N):
    import test_floating_rowgroups as fr   # standing keyframe + imu goal at the current pose
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(fr.URDF), pin.JointModelFreeFlyer())
    x = fr._standing_x()
    return np.asarray(x, dtype=np.float32)[None, :], fr._goals_at(model, x, 1)


def _make(plant, N, variant, urdfs):
    if plant == "go2":
        from conftest import go2_solver
        return go2_solver(1, variant=variant)
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
            cases.extend((plant, N, variant, f"collision_{mech}") for mech in ("al", "admm"))   # cooperative COLLISION rows
        if plant == "go2" and variant == "fc":
            cases.extend((plant, N, variant, f"masked_contact_{mech}") for mech in ("al", "admm"))
    return cases


@pytest.mark.parametrize("plant,N,variant,case", _cases(), ids=lambda v: str(v))
def test_golden(plant, N, variant, case, urdfs):
    if (plant, N) not in gato.available(variant):
        pytest.fail(f"receipt module {gato.module_name(plant, N, variant)} is not built (test/receipt_modules.txt)")
    s = _make(plant, N, variant, urdfs)
    x, goals = _go2_problem(N) if plant == "go2" else _arm_problem(plant, N)
    if variant == "fc":
        s.set_fc_ref(np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 5.0], s.n_fc // 6).astype(np.float32))  # a 5 N press reference
    if case.startswith("masked_contact_"):
        # Exercise both knot and individual-row masks, changing targets, and the
        # cooperative CONTACT_POS Jacobian on the floating tangent chart.
        s.set_linsys("bdsv")
        p0 = s.contact_positions(x[0, :s.nq])
        targets = np.tile(p0.reshape(1, -1), (N, 1)).astype(np.float32)
        targets[N // 2:, 2] += 0.02
        mech = case.removeprefix("masked_contact_")
        gi = s.add_contact_pos_rows(targets=targets, mech=mech, rho=1000.0 if mech == "al" else 100.0)
        mask = np.ones((N, s.n_contact_rows), dtype=bool)
        mask[0] = False
        mask[1:N // 2, :3] = False
        s.set_row_group_mask(gi, mask)
    elif case.startswith("collision_"):
        # Cooperative COLLISION clearance rows (band-indexed state, uniform
        # one-sided bounds) against a sphere parked on the reach goal, so the
        # rows bind during the 10 iterations (telemetry: 0.17 m violation
        # unenforced -> 0.04 AL / 0.002 ADMM on indy7).
        s.set_linsys("bdsv")
        mech = case.removeprefix("collision_")
        s.set_collision_environment(spheres=[(*GOAL_XYZ, 0.08)])
        s.enable_collision(mech=mech, margin=0.02, rho=100.0 if mech == "al" else 1.0)
    elif case == "admm_ee_pcg":
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
