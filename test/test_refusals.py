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
