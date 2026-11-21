"""The refusals of docs/status.md's variant table: a feature a module does not
carry is refused with a message that names the build that carries it, never
silently ignored. Host tier: the build-time refusals. GPU tier: the shipped
receipt modules (indy7 N16 default/fc/eh, go2 N16 default)."""
import numpy as np
import pytest

import gato
from gato.cost_menu import CostMenu, Term
from conftest import ARM_GOAL, GO2_URDF, URDFS, go2_solver  # noqa: E402


def test_build_refuses_two_variants_and_the_floating_exact_hessian(tmp_path):
    with pytest.raises(ValueError, match="separate module variants"):
        gato.build(URDFS["indy7"], "refusal_arm", N=[8], contact_forces=True, exact_hessian=True, build_dir=tmp_path)
    with pytest.raises(ValueError, match="separate module variants"):
        gato.build(URDFS["indy7"], "refusal_arm", N=[8], exact_hessian=True, integrator="rk4", build_dir=tmp_path)
    with pytest.raises(ValueError, match="integrator must be None"):
        gato.build(URDFS["indy7"], "refusal_arm", N=[8], integrator="euler", build_dir=tmp_path)
    with pytest.raises(ValueError, match="fixed-base single-stage step only"):
        gato.build(GO2_URDF, "refusal_go2", N=[8], floating_base=True, exact_hessian=True, build_dir=tmp_path)


@pytest.mark.gpu
def test_module_refusals_name_the_build_that_carries_the_feature(make_solver):
    """On the default indy7 module: the fc API, the exact Hessian and the CoM term
    refuse; on the fc module the exact Hessian refuses; on the eh module the fc
    API refuses; on the Go2 the gravity target refuses and the arms refuse a second
    cost frame."""
    arm = make_solver("indy7", 16)
    with pytest.raises(RuntimeError, match="GATO_CONTACT_FORCES build"):
        arm.set_fc_ref(np.zeros(6, np.float32))
    with pytest.raises(RuntimeError, match="USE_EXACT_HESSIAN"):
        arm.set_exact_hessian(True)
    with pytest.raises(RuntimeError, match="centroidal families"):
        arm.set_com_cost([0.0, 0.0, 0.3])
    with pytest.raises(ValueError, match="second baked cost frame"):
        arm.set_cost_menu(CostMenu(frame_pos=Term(weight=1.0, target=ARM_GOAL), frame_pos_2=Term(weight=1.0, target=ARM_GOAL)))
    fc = make_solver("indy7", 16, variant="fc")
    with pytest.raises(RuntimeError, match="USE_EXACT_HESSIAN"):
        fc.set_exact_hessian(True)
    eh = make_solver("indy7", 16, variant="eh")
    with pytest.raises(RuntimeError, match="GATO_CONTACT_FORCES build"):
        eh.set_fc_ref(np.zeros(6, np.float32))
    assert eh.exact_hessian_available() and not arm.exact_hessian_available()
    go2 = go2_solver(1)
    with pytest.raises(ValueError, match="fixed-base plants only"):
        go2.set_cost_menu(CostMenu(input=Term(weight=1.0, target="gravity"), q_ref=np.zeros((go2.N, go2.nq), np.float32)))


@pytest.mark.gpu
def test_pcg_shared_memory_admission(make_solver, monkeypatch):
    """PCG keeps five vectors of (N + 2) * nx values in block shared memory. A
    shipped module reports the kernel's request against the device limit and
    fits. Where it would not fit (the report is replaced here; test/cuda/
    pcg_admission.cu runs the real refusal at N512), the defaults are the
    direct solves and every request for PCG is refused by name."""
    from gato.controller import MPCController
    arm = make_solver("indy7", 16)
    need, limit = arm.pcg_smem_fit()
    assert need == 4 * (5 * (16 + 2) * 12 + 32) + 20
    assert 48 * 1024 <= limit and arm.pcg_ok
    assert arm.linsys == "pcg" and arm.admm_linsys == "pcg"

    monkeypatch.setattr(gato.BSQP, "pcg_smem_fit", lambda self: (limit + 4, limit))
    assert not arm.pcg_ok
    for mode in ("pcg", "bdsv_first"):
        with pytest.raises(ValueError, match="linsys=.* needs PCG.*'bdsv'"):
            arm.set_linsys(mode)
    with pytest.raises(ValueError, match="admm_linsys='pcg' needs PCG.*'bdsv_factor'"):
        arm.set_admm_linsys("pcg")
    fresh = make_solver("indy7", 16)
    assert fresh.linsys == "bdsv" and fresh.admm_linsys == "bdsv_factor"
    ctrl = MPCController(fresh)
    assert ctrl.linsys == "bdsv" and ctrl.admm_linsys == "bdsv_factor"
    with pytest.raises(ValueError, match="needs PCG"):
        MPCController(fresh, linsys="auto")
    with pytest.raises(ValueError, match="needs PCG"):
        gato.BSQP(model_path=str(URDFS["indy7"]), batch_size=1, N=16, dt=0.01, plant_type="indy7",
                  params=gato.SolverParams(linsys="pcg"))
