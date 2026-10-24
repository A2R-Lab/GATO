"""The centre-of-mass term (0.1.5): 0.5 sum_r w_r (c_r(q) - c_des_r)^2 on top of
the objective in force, on modules whose header bakes the centroidal families
(gato.build(..., com=True); the vendored Go2). Gates: the KKT gradient and
Hessian increments against Pinocchio's centre of mass and its Jacobian
(Gauss-Newton), the merit increment against the term's value along a held
trajectory, per-solve targets, the refusals on a plant without the families,
and the term on top of the cost menu."""
import importlib.util

import numpy as np
import pytest

import gato
from conftest import GO2_DT, GO2_TEST_PARAMS, GO2_URDF, TEST_PARAMS, URDFS, go2_goals_at, go2_standing_x

pytestmark = pytest.mark.gpu


def _go2(B=1, variant=None):
    name = gato.module_name("go2", 16, variant or "default")
    if importlib.util.find_spec(f"gato.{name}") is None:
        pytest.fail(f"{name} is a receipt-profile module (test/receipt_modules.txt) and must be built")
    return gato.BSQP(model_path=str(GO2_URDF), batch_size=B, N=16, dt=GO2_DT, params=GO2_TEST_PARAMS,
                     plant_type="go2", variant=variant)


def _pin():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    return pin, model, model.createData()


def _held(s, x0, u=None):
    step = s.nx + s.nu
    xu = np.zeros((s.batch_size, s.xu_size), dtype=np.float64)
    for k in range(s.N):
        xu[:, k * step:k * step + s.nx] = x0
        if k < s.N - 1 and u is not None:
            xu[:, k * step + s.nx:(k + 1) * step] = u
    return xu


def test_refused_on_a_plant_without_the_families():
    plant, N = min((k for k in gato.available() if k[0] in URDFS), key=lambda k: k[1])
    s = gato.BSQP(model_path=str(URDFS[plant]), batch_size=1, N=N, dt=0.01, params=TEST_PARAMS, plant_type=plant)
    assert not s.com_cost_available
    with pytest.raises(RuntimeError, match="centroidal"):
        s.set_com_cost(np.zeros(3))
    s.clear_com_cost()   # a no-op without the term


def test_kkt_increment_matches_pinocchio_com_jacobian():
    """Gradient increment J_com^T W (c - c_des) on the tangent q slots and the
    Gauss-Newton Hessian increment J_com^T W J_com, at every knot, running and
    terminal, with a target that leaves a visible residual."""
    pin, model, data = _pin()
    s = _go2()
    s0 = _go2()
    nq, nv, N = model.nq, model.nv, s.N
    nx = 2 * nv   # the KKT blocks are tangent-sized on the floating base
    x0 = np.asarray(go2_standing_x(), dtype=np.float64)
    xu = _held(s, x0)
    goals = go2_goals_at(model, x0.astype(np.float32), 1)
    c0 = s.com_position(x0[:nq])
    target = c0 + np.array([0.05, -0.03, 0.04])
    w = np.array([2.0, 3.0, 5.0])
    s.set_com_cost(target, w)
    assert s.com_cost_on
    d1 = s.debug_setup_kkt(xu.astype(np.float32), x0[None].astype(np.float32), goals)
    d0 = s0.debug_setup_kkt(xu.astype(np.float32), x0[None].astype(np.float32), goals)
    Q1 = np.asarray(d1["Q"], np.float64).reshape(N, nx, nx).transpose(0, 2, 1)
    Q0 = np.asarray(d0["Q"], np.float64).reshape(N, nx, nx).transpose(0, 2, 1)
    q1 = np.asarray(d1["q"], np.float64).reshape(N, nx)
    q0 = np.asarray(d0["q"], np.float64).reshape(N, nx)
    # the control blocks over the N - 1 running slots (the terminal slot is never written)
    nu = s.nu
    assert np.array_equal(np.asarray(d1["R"]).ravel()[: (N - 1) * nu * nu], np.asarray(d0["R"]).ravel()[: (N - 1) * nu * nu])
    assert np.array_equal(np.asarray(d1["r"]).ravel()[: (N - 1) * nu], np.asarray(d0["r"]).ravel()[: (N - 1) * nu])
    J = pin.jacobianCenterOfMass(model, data, x0[:nq])        # 3 x nv, tangent columns
    r = c0 - target
    g_ref = J.T @ (w * r)
    H_ref = J.T @ (w[:, None] * J)
    for k in range(N):
        dg = q1[k] - q0[k]
        assert np.allclose(dg[:nv], g_ref, rtol=2e-4, atol=1e-5), (k, np.abs(dg[:nv] - g_ref).max())
        assert np.allclose(dg[nv:], 0.0, atol=1e-7)
        dH = Q1[k] - Q0[k]
        assert np.allclose(dH[:nv, :nv], H_ref, rtol=2e-4, atol=1e-5), (k, np.abs(dH[:nv, :nv] - H_ref).max())
        assert np.allclose(dH[nv:, :], 0.0, atol=1e-7) and np.allclose(dH[:, nv:], 0.0, atol=1e-7)


def test_merit_increment_is_the_term_value():
    """initial_merit of a held trajectory grows by sum_k 0.5 sum_r w_r (c_r - c_des_r)^2
    over the N knots when the term is on (the constraint part is unchanged)."""
    pin, model, data = _pin()
    s = _go2()
    s0 = _go2()
    nq = model.nq
    x0 = np.asarray(go2_standing_x(), dtype=np.float64)
    xu = _held(s, x0).astype(np.float32)
    goals = go2_goals_at(model, x0.astype(np.float32), 1)
    c0 = s.com_position(x0[:nq])
    target = c0 + np.array([0.02, 0.01, -0.03])
    w = np.array([4.0, 4.0, 1.0])
    s.set_com_cost(target, w)
    m1 = float(np.asarray(s.solve(x0[None].astype(np.float32), goals, xu_warm=xu).stats.initial_merit).ravel()[0])
    m0 = float(np.asarray(s0.solve(x0[None].astype(np.float32), goals, xu_warm=xu).stats.initial_merit).ravel()[0])
    expect = s.N * 0.5 * float(np.sum(w * (c0 - target) ** 2))
    assert expect > 1e-4
    assert abs((m1 - m0) - expect) < 1e-3 * (1.0 + expect), (m1 - m0, expect)


def test_per_solve_targets_and_clear():
    """(B, N, 3) targets: each solve's gradient uses its own target; clearing
    restores the untouched objective bit for bit."""
    pin, model, data = _pin()
    B = 2
    s = _go2(B)
    s0 = _go2(B)
    nq, nv, N = model.nq, model.nv, s.N
    nx = 2 * nv   # the KKT blocks are tangent-sized on the floating base
    x0 = np.asarray(go2_standing_x(), dtype=np.float64)
    xu = _held(s, x0)
    goals = go2_goals_at(model, x0.astype(np.float32), B)
    c0 = s.com_position(x0[:nq])
    targets = np.tile(c0, (B, N, 1))
    targets[0] += [0.03, 0.0, 0.0]
    targets[1] += [0.0, 0.0, -0.05]
    s.set_com_cost(targets, 2.0)
    X = np.tile(x0, (B, 1)).astype(np.float32)
    d1 = s.debug_setup_kkt(xu.astype(np.float32), X, goals)
    d0 = s0.debug_setup_kkt(xu.astype(np.float32), X, goals)
    q1 = np.asarray(d1["q"], np.float64).reshape(B, N, nx)
    q0 = np.asarray(d0["q"], np.float64).reshape(B, N, nx)
    J = pin.jacobianCenterOfMass(model, data, x0[:nq])
    for b in range(B):
        g_ref = J.T @ (2.0 * (c0 - targets[b, 0]))
        assert np.allclose((q1[b, 0] - q0[b, 0])[:nv], g_ref, rtol=2e-4, atol=1e-5)
    s.clear_com_cost()
    assert not s.com_cost_on
    d2 = s.debug_setup_kkt(xu.astype(np.float32), X, goals)
    assert np.array_equal(np.asarray(d2["q"]), np.asarray(d0["q"])) and np.array_equal(np.asarray(d2["Q"]), np.asarray(d0["Q"]))


def test_term_on_top_of_the_cost_menu():
    """With the floating-base cost menu on (the contact-force variant bakes it),
    the term adds the same increment to the menu's KKT blocks."""
    pin, model, data = _pin()
    s = _go2(variant="fc")
    s0 = _go2(variant="fc")
    if not s.cost_menu_available:
        pytest.fail("the go2 fc module is expected to carry the cost menu")
    from gato.cost_menu import CostMenu, Term
    nq, nv, N = model.nq, model.nv, s.N
    nx = 2 * nv   # the KKT blocks are tangent-sized on the floating base
    x0 = np.asarray(go2_standing_x(), dtype=np.float64)
    x_des = x0.copy()
    menu = CostMenu(state=Term(weight=np.full(2 * nv, 0.5), target=x_des))
    s.set_cost_menu(menu)
    s0.set_cost_menu(menu)
    xu = _held(s, x0)
    goals = go2_goals_at(model, x0.astype(np.float32), 1)
    c0 = s.com_position(x0[:nq])
    target = c0 + np.array([0.0, 0.04, 0.0])
    s.set_com_cost(target, 3.0)
    d1 = s.debug_setup_kkt(xu.astype(np.float32), x0[None].astype(np.float32), goals)
    d0 = s0.debug_setup_kkt(xu.astype(np.float32), x0[None].astype(np.float32), goals)
    q1 = np.asarray(d1["q"], np.float64).reshape(N, nx)
    q0 = np.asarray(d0["q"], np.float64).reshape(N, nx)
    J = pin.jacobianCenterOfMass(model, data, x0[:nq])
    g_ref = J.T @ (3.0 * (c0 - target))
    for k in range(N):
        assert np.allclose((q1[k] - q0[k])[:nv], g_ref, rtol=2e-4, atol=1e-5), k
