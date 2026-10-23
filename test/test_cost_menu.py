"""The cost menu (gato.cost_menu over grid_plant::tracking_cost_menu, grid-rbd 0.5.5).

Host tier: shape and validation rules of CostMenu.apply against a recording stub.
GPU tier: the menu with quadratic activations reproduces the tracking objective
(KKT blocks to rounding, the initial merit, the first SQP iteration); every term
matches RBDReference's menu oracle in the assembled KKT blocks; per-solve targets
keep the batch independent; the legacy cost setters are refused while a menu is
on; the gravity target is Pinocchio's generalized gravity; on the Go2 the menu is
in the tangent layout: its quadratic state term reproduces the floating tracking
objective, every term matches the oracle's tangent gradients and Hessians, and the
gravity target is refused."""
import types

import numpy as np
import pytest

import gato
from gato.cost_menu import ACTIVATIONS, CostMenu, LinearTerm, Term, TERMS
from conftest import ARM_GOAL, ARM_START, TEST_PARAMS, URDFS  # noqa: E402


# ---------------------------------------------------------------------------
# host tier: validation against a stub solver
# ---------------------------------------------------------------------------

class _Raw:
    def __init__(self):
        self.fields = {}
        self.activation = None
        self.on = None
        self.q_ref = None

    def set_cost_menu_on(self, on):
        self.on = bool(on)

    def set_cost_menu_field(self, name, values, per_solve):
        self.fields[name] = (np.array(values, dtype=np.float32), bool(per_solve))

    def set_cost_menu_activation(self, a):
        self.activation = np.array(a, dtype=np.int32)

    def set_cost_menu_gravity_target(self, q_ref):
        self.q_ref = np.array(q_ref, dtype=np.float32)


def _stub(N=4, B=2, nq=7, frames=1, available=True):
    nx, nu = 2 * nq, nq
    sizes = dict(x_des=nx, Q=nx, x_lower=nx, x_upper=nx, u_des=nu, R=nu, ub_lower=nu, ub_upper=nu, ee_des=3, W=3,
                 Ax=nx * nx, bx=nx, tx=nx, Wx=nx, lx_lower=nx, lx_upper=nx, Au=nu * nu, bu=nu, tu=nu, Wu=nu,
                 lu_lower=nu, lu_upper=nu, R_des=9, Wo=3, v_des=6, Wv=6, v_lower=6, v_upper=6, cf_des=3, Wcf=3,
                 cf_lower=3, cf_upper=3, eps=8)
    lib = types.SimpleNamespace(COST_MENU_AVAILABLE=available, COST_MENU_FIELDS=sizes, NUM_COST_FRAMES=frames)
    return types.SimpleNamespace(lib=lib, N=N, batch_size=B, nq=nq, nv=nq, nx=nx, nu=nu, solver=_Raw())


def test_menu_broadcast_shared_and_per_solve():
    s = _stub(N=4, B=2)
    nx = s.nx
    tgt = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    CostMenu(frame_pos=Term(weight=[1.0, 2.0, 3.0], target=tgt),
             state=Term(weight=np.ones((4, nx)), target=0.0, activation=["quadratic", "none", "pseudo_huber", "quadratic"], eps=0.1)
             ).apply(s)
    raw = s.solver
    assert raw.on is True
    W, ps = raw.fields["W"]
    assert ps is False and W.shape == (12,) and np.array_equal(W.reshape(4, 3)[2], [1.0, 2.0, 3.0])
    ee, ps = raw.fields["ee_des"]
    assert ps is True and np.array_equal(ee, tgt.reshape(-1))
    act = raw.activation.reshape(4, 8)
    assert list(act[:, TERMS.index("state")]) == [0, -1, 1, 0]
    assert list(act[:, TERMS.index("frame_pos")]) == [0, 0, 0, 0]
    assert (act[:, 2:] == -1).all()
    eps = raw.fields["eps"][0].reshape(4, 8)
    assert np.allclose(eps[:, TERMS.index("state")], 0.1)


def test_menu_linear_feature_is_column_major():
    s = _stub(N=2, B=1, nq=2)
    A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    CostMenu(linear_x=LinearTerm(weight=1.0, A=A, b=np.arange(4), target=np.ones(4))).apply(s)
    flat = s.solver.fields["Ax"][0].reshape(2, 16)[0]
    for m in range(4):
        for n in range(4):
            assert flat[m + 4 * n] == A[m, n]
    assert np.array_equal(s.solver.fields["bx"][0].reshape(2, 4)[1], np.arange(4))


@pytest.mark.parametrize("bad,match", [
    (dict(frame_pos=Term(weight=[1.0, 1.0], target=[0, 0, 0])), "expected shape"),
    (dict(frame_pos=Term(weight=[-1.0, 1.0, 1.0], target=[0, 0, 0])), "nonnegative"),
    (dict(frame_pos=Term(weight=1.0, target=[np.nan, 0, 0])), "finite"),
    (dict(state=Term(weight=1.0, activation="pseudo_huber", eps=0.0)), "eps must be positive"),
    (dict(state=Term(weight=1.0, activation="quadratic_barrier")), "need lower and upper"),
    (dict(state=Term(weight=1.0, activation="quadratic_barrier", lower=1.0, upper=-1.0)), "at most its upper"),
    (dict(frame_pos=Term(weight=1.0, activation="relaxed_log_barrier", eps=0.1, lower=0.0, upper=1.0)), "no bounds"),
    (dict(orientation=Term(weight=1.0, target=np.eye(3) * 2.0)), "rotation matrix"),
    (dict(orientation=Term(weight=1.0, target=None)), "rotation matrix"),
    (dict(linear_u=Term(weight=1.0)), "LinearTerm"),
    (dict(frame_pos_2=Term(weight=1.0, target=[0, 0, 0])), "second baked cost frame"),
    (dict(state=Term(weight=1.0, activation="huber")), "unknown activation"),
    (dict(state=Term(weight=1.0, activation=["quadratic"] * 3)), "one per knot"),
    (dict(input=Term(weight=1.0, target="gravity")), "q_ref"),
    (dict(input=Term(weight=1.0, target="zeros")), "'gravity'"),
    ({}, "empty cost menu"),
])
def test_menu_rejects_bad_terms(bad, match):
    with pytest.raises(ValueError, match=match):
        CostMenu(**bad).apply(_stub())


def test_menu_unavailable_module_is_an_error():
    with pytest.raises(ValueError, match="not available"):
        CostMenu(input=Term(weight=1.0)).apply(_stub(available=False))


def test_menu_gravity_target_needs_q_ref_shape():
    s = _stub(N=3, nq=7)
    CostMenu(input=Term(weight=1.0, target="gravity"), q_ref=np.zeros(7)).apply(s)
    assert s.solver.q_ref.shape == (21,) and "u_des" not in s.solver.fields
    with pytest.raises(ValueError, match="q_ref"):
        CostMenu(input=Term(weight=1.0, target="gravity"), q_ref=np.zeros((2, 7))).apply(s)


# ---------------------------------------------------------------------------
# GPU tier
# ---------------------------------------------------------------------------

def _equivalent_menu(s, p):
    """The menu that spells the tracking objective: EE weight q_cost (N_cost at the
    last knot) toward the reach goal, qd_cost on the velocities, u_cost on u."""
    W = np.tile([p.q_cost] * 3, (s.N, 1)).astype(np.float32)
    W[s.N - 1] = p.N_cost
    return CostMenu(frame_pos=Term(weight=W, target=np.array(ARM_GOAL)),
                    state=Term(weight=[0.0] * s.nq + [p.qd_cost] * s.nv),
                    input=Term(weight=p.u_cost))


def _trajectory(s, plant, seed=0, scale=0.05):
    """(x0, ref (N, 6), xu): the start state, the reach reference and a jittered hold."""
    q0 = np.asarray(ARM_START[plant], dtype=np.float32)
    x0 = np.concatenate([q0, np.zeros_like(q0)])
    rng = np.random.default_rng(seed)
    nx, nu, N = s.nx, s.nu, s.N
    xu = np.zeros(N * (nx + nu) - nu, dtype=np.float32)
    for k in range(N):
        xu[k * (nx + nu):k * (nx + nu) + nx] = x0 + rng.normal(0, scale, nx).astype(np.float32)
        if k < N - 1:
            xu[k * (nx + nu) + nx:(k + 1) * (nx + nu)] = rng.normal(0, 1.0, nu).astype(np.float32)
    ref = np.zeros((N, 6), dtype=np.float32)
    ref[:, :3] = ARM_GOAL
    return x0, ref, xu


def _kkt(s, x0, ref, xu):
    d = s.debug_setup_kkt(xu[None], x0[None], ref.ravel()[None])
    N, nx, nu = s.N, s.nx, s.nu
    return (np.asarray(d["Q"]).reshape(N, nx, nx), np.asarray(d["q"]).reshape(N, nx),
            np.asarray(d["R"]).reshape(N, nu, nu), np.asarray(d["r"]).reshape(N, nu))


def _knot(xu, k, nx, nu):
    x = xu[k * (nx + nu):k * (nx + nu) + nx]
    u = xu[k * (nx + nu) + nx:(k + 1) * (nx + nu)] if k * (nx + nu) + nx + nu <= xu.size else np.zeros(nu, np.float32)
    return x.astype(np.float64), u.astype(np.float64)


@pytest.mark.gpu
def test_menu_reproduces_tracking_objective(make_solver, smallest_module):
    """Quadratic activations with the default weights: KKT blocks equal to rounding
    (the EE Jacobian comes from frame_jacobian, a different arithmetic order than
    the pose gradient), the initial merit equal, the first SQP step equal to
    rounding. Later iterations may pick different line-search candidates on
    ties, which is the known ulp sensitivity, not a modeling difference."""
    plant, N = smallest_module
    p = TEST_PARAMS
    s0 = make_solver(plant, N, max_sqp_iters=1)
    assert s0.cost_menu_available
    x0, ref, xu = _trajectory(s0, plant)
    Q0, q0, R0, r0 = _kkt(s0, x0, ref, xu)
    s1 = make_solver(plant, N, max_sqp_iters=1)
    s1.set_cost_menu(_equivalent_menu(s1, p))
    assert s1.cost_menu is not None and s1.solver.cost_menu_on()
    Q1, q1, R1, r1 = _kkt(s1, x0, ref, xu)
    np.testing.assert_allclose(Q1, Q0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(q1, q0, rtol=1e-5, atol=1e-6)
    # the terminal R/r slots are never written by the KKT kernel (a discarded dummy)
    np.testing.assert_array_equal(R1[:-1], R0[:-1])
    np.testing.assert_array_equal(r1[:-1], r0[:-1])
    # one SQP step from the same warm start
    X = np.tile(x0, (1, 1))
    a = make_solver(plant, N, max_sqp_iters=1).solve(X, ref.ravel()[None])
    sm = make_solver(plant, N, max_sqp_iters=1)
    sm.set_cost_menu(_equivalent_menu(sm, p))
    b = sm.solve(X, ref.ravel()[None])
    np.testing.assert_allclose(b.stats.initial_merit, a.stats.initial_merit, rtol=1e-6)
    np.testing.assert_array_equal(b.stats.step_size, a.stats.step_size)
    np.testing.assert_allclose(b.xu, a.xu, rtol=1e-3, atol=1e-3)


def _oracle(plant):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(URDFS[plant])))


def _rotation_small(rng, max_angle):
    """exp of a random axis-angle with angle in [0, max_angle]."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    th = rng.uniform(0.0, max_angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


@pytest.mark.gpu
def test_menu_terms_match_rbdreference(make_solver, smallest_module):
    """Every term and activation against RBDReference's menu oracle in the assembled
    KKT blocks (GN Hessians), at a running knot, a knot with another activation
    table and the terminal knot (input terms off there). The joint-limit barriers
    are off so the blocks are the menu alone."""
    plant, N = smallest_module
    s = make_solver(plant, N, q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
    nq, nx, nu = s.nq, s.nx, s.nu
    ee = s.ee_frame
    rng = np.random.default_rng(7)
    rbd = _oracle(plant)
    x0, ref, xu = _trajectory(s, plant, seed=3, scale=0.2)
    # per-knot data: targets vary along the horizon, activations change at knot 5
    ee_des = np.tile(np.array(ARM_GOAL), (N, 1)) + rng.normal(0, 0.05, (N, 3))
    W = np.abs(rng.normal(1.0, 0.3, (N, 3)))
    x_des = rng.normal(0, 0.3, (N, nx)); Q = np.abs(rng.normal(0.5, 0.2, (N, nx)))
    x_act = ["pseudo_huber"] * 5 + ["quadratic_barrier"] * (N - 5)
    x_lo = x_des - 0.1; x_hi = x_des + 0.1
    u_des = rng.normal(0, 2.0, (N, nu)); Rw = np.abs(rng.normal(1e-2, 3e-3, (N, nu)))
    u_act = ["quadratic"] * 5 + ["relaxed_log_barrier"] * (N - 5)
    u_lo = np.full((N, nu), -5.0); u_hi = np.full((N, nu), 5.0)
    Ax = rng.normal(0, 1.0, (nx, nx)); bx = rng.normal(0, 0.1, nx); tx = rng.normal(0, 0.2, (N, nx)); Wx = np.abs(rng.normal(0.3, 0.1, nx))
    Au = rng.normal(0, 1.0, (N, nu, nu)); bu = rng.normal(0, 0.1, (N, nu)); tu = rng.normal(0, 0.2, nu); Wu = np.abs(rng.normal(0.3, 0.1, (N, nu)))
    # orientation targets within a radian of the start orientation: the log map is
    # ill-conditioned at a residual of pi (its branch boundary), where float32 loses digits
    R_start = rbd._frame_placement(np.asarray(ARM_START[plant], dtype=np.float64), ee)[0]
    R_des = np.stack([R_start @ _rotation_small(rng, 1.0) for _ in range(N)]); Wo = np.abs(rng.normal(0.7, 0.2, 3))
    v_des = rng.normal(0, 0.5, (N, 6)); Wv = np.abs(rng.normal(0.4, 0.1, 6))
    menu = CostMenu(
        frame_pos=Term(weight=W, target=ee_des, activation="pseudo_huber", eps=0.02),
        state=Term(weight=Q, target=x_des, activation=x_act, eps=0.05, lower=x_lo, upper=x_hi),
        input=Term(weight=Rw, target=u_des, activation=u_act, eps=0.5, lower=u_lo, upper=u_hi),
        linear_x=LinearTerm(weight=Wx, A=Ax, b=bx, target=tx),
        linear_u=LinearTerm(weight=Wu, A=Au, b=bu, target=tu, activation="pseudo_huber", eps=0.3),
        orientation=Term(weight=Wo, target=R_des),
        velocity=Term(weight=Wv, target=v_des, activation=["quadratic_barrier"] * 5 + ["smooth_hinge"] * (N - 5), eps=0.02,
                      lower=v_des - 0.05, upper=v_des + 0.05),
    )
    s.set_cost_menu(menu)
    Qk, qk, Rk, rk = _kkt(s, x0, ref, xu)
    values = s.cost_menu_terms(xu[None])[0]   # (N, 8) per-term values along the same trajectory
    assert values.shape == (N, len(TERMS)) and np.isfinite(values).all()
    assert np.all(values[:, TERMS.index("frame_pos_2")] == 0.0)   # an off term reads zero
    act = lambda name: ACTIVATIONS[name]
    for k in (0, 5, N - 1):
        x, u = _knot(xu, k, nx, nu)
        q, qd = x[:nq], x[nq:]
        terminal = (k == N - 1)
        gx = np.zeros(nx); Hx = np.zeros((nx, nx)); gu = np.zeros(nu); Hu = np.zeros((nu, nu))
        vals = {}
        vals["frame_pos"], g, H = rbd.frame_position_cost(q, ee, ee_des[k], W[k], act("pseudo_huber"), 0.02); gx += g; Hx += H
        vals["state"], g, H = rbd.state_cost(x, x_des[k], Q[k], act(x_act[k]), 0.05, x_lo[k], x_hi[k]); gx += g; Hx += H
        vals["linear_x"], g, H = rbd.linear_feature_cost(x, Ax, bx, tx[k], Wx, act("quadratic"), 0.0); gx += g; Hx += H
        R_cm = np.asarray(R_des[k]).T.reshape(-1)   # column-major, as the device buffer
        vals["orientation"], g, H = rbd.frame_orientation_cost(q, ee, R_cm, Wo, act("quadratic"), 0.0); gx += g; Hx += H
        v_kind = "quadratic_barrier" if k < 5 else "smooth_hinge"
        vals["velocity"], g, H = rbd.frame_velocity_cost(q, qd, ee, v_des[k], Wv, act(v_kind), 0.02, v_des[k] - 0.05, v_des[k] + 0.05); gx += g; Hx += H
        if not terminal:
            vals["input"], g, H = rbd.input_cost(u, u_des[k], Rw[k], act(u_act[k]), 0.5, u_lo[k], u_hi[k]); gu += g; Hu += H
            vals["linear_u"], g, H = rbd.linear_feature_cost(u, Au[k], bu[k], tu, Wu[k], act("pseudo_huber"), 0.3); gu += g; Hu += H
        else:
            vals["input"] = vals["linear_u"] = 0.0   # no control at the terminal knot
        for name, v in vals.items():
            np.testing.assert_allclose(values[k, TERMS.index(name)], v, rtol=2e-4, atol=1e-5, err_msg=f"value of {name} at knot {k}")
        scale_x = max(1.0, np.abs(gx).max()); scale_H = max(1.0, np.abs(Hx).max())
        np.testing.assert_allclose(qk[k], gx, rtol=2e-4, atol=2e-4 * scale_x, err_msg=f"q at knot {k}")
        np.testing.assert_allclose(Qk[k], Hx, rtol=2e-4, atol=2e-4 * scale_H, err_msg=f"Q at knot {k}")
        if not terminal:
            np.testing.assert_allclose(rk[k], gu, rtol=2e-4, atol=2e-4 * max(1.0, np.abs(gu).max()), err_msg=f"r at knot {k}")
            np.testing.assert_allclose(Rk[k], Hu, rtol=2e-4, atol=2e-4 * max(1.0, np.abs(Hu).max()), err_msg=f"R at knot {k}")


@pytest.mark.gpu
def test_menu_per_solve_targets_keep_the_batch_independent(make_solver, smallest_module):
    """Per-solve (B, N, 3) EE targets: each solve of the batch is bit-identical to the
    batch-one solve with that target shared."""
    plant, N = smallest_module
    p = TEST_PARAMS
    B = 3
    q0 = np.asarray(ARM_START[plant], dtype=np.float32)
    X = np.tile(np.concatenate([q0, np.zeros_like(q0)]), (B, 1))
    goals = np.zeros((B, N * 6), dtype=np.float32)
    targets = np.array(ARM_GOAL)[None, None, :] + np.array([[0.0, 0.0, 0.0], [0.05, -0.05, 0.0], [-0.05, 0.05, 0.05]])[:, None, :]
    targets = np.broadcast_to(targets, (B, N, 3)).astype(np.float32)
    sb = make_solver(plant, N, batch_size=B)
    menu = _equivalent_menu(sb, p)
    menu.frame_pos.target = targets
    sb.set_cost_menu(menu)
    got = sb.solve(X, goals)
    for b in range(B):
        s1 = make_solver(plant, N, batch_size=1)
        m1 = _equivalent_menu(s1, p)
        m1.frame_pos.target = targets[b]
        s1.set_cost_menu(m1)
        one = s1.solve(X[b:b + 1], goals[:1])
        np.testing.assert_array_equal(got.xu[b], one.xu[0])


@pytest.mark.gpu
def test_menu_refuses_legacy_setters_and_clears(make_solver, smallest_module):
    plant, N = smallest_module
    s = make_solver(plant, N)
    s.set_cost_menu(_equivalent_menu(s, TEST_PARAMS))
    for call in (lambda: s.solver.set_q_nom(np.zeros(s.nq, np.float32)),
                 lambda: s.set_cost_weights_per_knot(np.ones((N, 3), np.float32)),
                 lambda: s.solver.set_cost_weights(1, 1, 1, 1, 0, 0, 0),
                 lambda: s.solver.set_q_pos_cost(1.0)):
        with pytest.raises(RuntimeError, match="cost menu is on"):
            call()
    s.clear_cost_menu()
    assert s.cost_menu is None and not s.solver.cost_menu_on()
    s.solver.set_q_nom(np.zeros(s.nq, np.float32))   # accepted again
    s.solver.set_q_nom(np.zeros(0, np.float32))
    # back on the tracking objective: the default path, bit-identical to a fresh solver
    q0 = np.asarray(ARM_START[plant], dtype=np.float32)
    X = np.concatenate([q0, np.zeros_like(q0)])[None]
    goals = np.zeros((1, N * 6), dtype=np.float32); goals[0, 0::6], goals[0, 1::6], goals[0, 2::6] = ARM_GOAL
    s.solver.reset_rho()
    a = s.solve(X, goals)
    b = make_solver(plant, N).solve(X, goals)
    np.testing.assert_array_equal(a.xu, b.xu)


@pytest.mark.gpu
def test_menu_gravity_target_is_the_generalized_gravity(make_solver, smallest_module):
    """input.target = "gravity" with q_ref: at a hold trajectory with u = 0 the input
    gradient is R (u - g(q_ref)) = -g(q_ref), Pinocchio's generalized gravity
    (the sign check: GATO's gravity constant matches the physical, downward, one)."""
    import pinocchio as pin
    plant, N = smallest_module
    s = make_solver(plant, N, q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
    nq, nx, nu = s.nq, s.nx, s.nu
    rng = np.random.default_rng(11)
    q_ref = np.asarray(ARM_START[plant], dtype=np.float64)[None, :] + rng.normal(0, 0.3, (N, nq))
    s.set_cost_menu(CostMenu(input=Term(weight=1.0, target="gravity"), q_ref=q_ref))
    x0, ref, xu = _trajectory(s, plant, seed=1, scale=0.0)
    xu = xu.copy()
    for k in range(N - 1):
        xu[k * (nx + nu) + nx:(k + 1) * (nx + nu)] = 0.0
    _, _, _, rk = _kkt(s, x0, ref, xu)
    model = pin.buildModelFromUrdf(str(URDFS[plant]))
    data = model.createData()
    for k in (0, N // 2, N - 2):
        g = pin.computeGeneralizedGravity(model, data, q_ref[k])
        np.testing.assert_allclose(rk[k], -g, rtol=1e-4, atol=1e-4)


def _go2_setup(**kw):
    from conftest import go2_solver, go2_standing_x, go2_goals_at, require_module
    require_module("bsqpN16_go2")
    s = go2_solver(1, **kw)
    x = go2_standing_x().astype(np.float32)
    ref = go2_goals_at(s.model, x, 1)
    return s, x, ref


def _go2_trajectory(s, x, seed=5, scale=0.1):
    """A jittered hold on the standing state: the base quaternion stays unit (perturbed and renormalized)."""
    from gato.common import initialize_warm_start
    rng = np.random.default_rng(seed)
    xu = initialize_warm_start(x, s.N, s.nx, s.nu).astype(np.float32)
    nx, nu = s.nx, s.nu
    for k in range(s.N):
        xk = xu[k * (nx + nu):k * (nx + nu) + nx]
        xk += rng.normal(0, scale, nx).astype(np.float32)
        xk[3:7] /= np.linalg.norm(xk[3:7])
        if k < s.N - 1:
            xu[k * (nx + nu) + nx:(k + 1) * (nx + nu)] = rng.normal(0, 1.0, nu).astype(np.float32)
    return xu


def _go2_kkt(s, x, ref, xu):
    d = s.debug_setup_kkt(xu[None], x[None], ref)
    ns, nu, N = 2 * s.nv, s.nu, s.N
    return (np.asarray(d["Q"]).reshape(N, ns, ns), np.asarray(d["q"]).reshape(N, ns),
            np.asarray(d["R"]).reshape(N, nu, nu), np.asarray(d["r"]).reshape(N, nu))


@pytest.mark.gpu
@pytest.mark.parametrize("variant", [None, "fc"], ids=["default", "fc"])
def test_menu_floating_quadratic_state_reproduces_tracking_objective(variant):
    """Go2: the menu's quadratic state term with the tracking objective's tangent weights
    (joint posture off, velocities under qd_cost, zero base rows) and the input term
    reproduce the floating tracking objective's KKT blocks with the imu term off
    (q_cost = N_cost = 0), to rounding; on the contact-force variant the wrench rows
    (fc_cost around a reference) come out identical too."""
    from conftest import go2_standing_q, require_module
    if variant == "fc":
        require_module("bsqpN16_go2_fc")
    s, x, ref = _go2_setup(variant=variant, q_cost=0.0, N_cost=0.0)
    assert s.cost_menu_available and s.cost_menu_tangent
    if variant == "fc":
        fc_ref = np.linspace(-1.0, 1.0, s.n_fc).astype(np.float32)
        s.set_fc_ref(fc_ref)
    xu = _go2_trajectory(s, x)
    Q0, q0, R0, r0 = _go2_kkt(s, x, ref, xu)
    nv = s.nv
    x_des = np.concatenate([go2_standing_q(), np.zeros(nv)])
    s2, _, _ = _go2_setup(variant=variant, q_cost=0.0, N_cost=0.0)
    if variant == "fc":
        s2.set_fc_ref(fc_ref)
    s2.set_cost_menu(CostMenu(state=Term(weight=[0.0] * nv + [s2.params.qd_cost] * nv, target=x_des),
                              input=Term(weight=s2.params.u_cost)))
    Q1, q1, R1, r1 = _go2_kkt(s2, x, ref, xu)
    np.testing.assert_allclose(Q1, Q0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(q1, q0, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(R1[:-1], R0[:-1])
    np.testing.assert_array_equal(r1[:-1], r0[:-1])


@pytest.mark.gpu
def test_menu_floating_terms_match_rbdreference():
    """Go2: every term in the tangent layout against RBDReference (state_cost_tangent,
    the frame terms on the imu with tangent-sized derivatives, the input and the
    linear features over the actuated inputs and [q[7:]; qd]), at a running knot, a
    knot with other activations and the terminal knot; the limit barriers off."""
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    from conftest import GO2_URDF, go2_standing_q
    s, x, ref = _go2_setup(q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
    N, nq, nv, nu = s.N, s.nq, s.nv, s.nu
    ns, nlx = 2 * nv, 2 * nv - 6
    rng = np.random.default_rng(9)
    xu = _go2_trajectory(s, x, seed=4, scale=0.15)
    x_des = np.tile(np.concatenate([go2_standing_q(), np.zeros(nv)]), (N, 1)) + rng.normal(0, 0.1, (N, nq + nv))
    x_des[:, 3:7] /= np.linalg.norm(x_des[:, 3:7], axis=1, keepdims=True)
    Q = np.abs(rng.normal(0.5, 0.2, (N, ns)))
    x_act = ["pseudo_huber"] * 5 + ["smooth_hinge"] * (N - 5)
    x_lo = -0.5 + rng.normal(0, 0.05, (N, ns)); x_hi = x_lo + 0.6
    u_des = rng.normal(0, 2.0, (N, nu)); Rw = np.abs(rng.normal(1e-2, 3e-3, (N, nu)))
    Ax = rng.normal(0, 1.0, (nlx, nlx)); bx = rng.normal(0, 0.1, nlx); tx = rng.normal(0, 0.2, (N, nlx)); Wx = np.abs(rng.normal(0.3, 0.1, nlx))
    Au = rng.normal(0, 1.0, (nu, nu)); bu = rng.normal(0, 0.1, nu); tu = rng.normal(0, 0.2, nu); Wu = np.abs(rng.normal(0.3, 0.1, nu))
    rbd = RBDReference(URDFParser().parse(str(GO2_URDF), floating_base=True))
    ee = s.ee_frame
    q_stand = go2_standing_q()
    R_start, p_start, _ = rbd._frame_placement(q_stand, ee)
    R_des = np.stack([R_start @ _rotation_small(rng, 0.8) for _ in range(N)]); Wo = np.abs(rng.normal(0.7, 0.2, 3))
    p_des = p_start[None, :] + rng.normal(0, 0.05, (N, 3)); W = np.abs(rng.normal(1.0, 0.3, (N, 3)))
    v_des = rng.normal(0, 0.5, (N, 6)); Wv = np.abs(rng.normal(0.4, 0.1, 6))
    s.set_cost_menu(CostMenu(
        frame_pos=Term(weight=W, target=p_des, activation="pseudo_huber", eps=0.02),
        state=Term(weight=Q, target=x_des, activation=x_act, eps=0.05, lower=x_lo, upper=x_hi),
        input=Term(weight=Rw, target=u_des, activation="quadratic_barrier", lower=np.full((N, nu), -5.0), upper=np.full((N, nu), 5.0)),
        linear_x=LinearTerm(weight=Wx, A=Ax, b=bx, target=tx),
        linear_u=LinearTerm(weight=Wu, A=Au, b=bu, target=tu, activation="pseudo_huber", eps=0.3),
        orientation=Term(weight=Wo, target=R_des),
        velocity=Term(weight=Wv, target=v_des),
    ))
    Qk, qk, Rk, rk = _go2_kkt(s, x, ref, xu)
    values = s.cost_menu_terms(xu[None])[0]
    act = lambda name: ACTIVATIONS[name]
    for k in (0, 5, N - 1):
        xk, uk = _knot(xu, k, s.nx, nu)
        q, qd = xk[:nq], xk[nq:]
        terminal = (k == N - 1)
        gx = np.zeros(ns); Hx = np.zeros((ns, ns)); gu = np.zeros(nu); Hu = np.zeros((nu, nu)); vals = {}
        vals["frame_pos"], g, H = rbd.frame_position_cost(q, ee, p_des[k], W[k], act("pseudo_huber"), 0.02); gx += g; Hx += H
        vals["state"], g, H = rbd.state_cost_tangent(xk, x_des[k], Q[k], act(x_act[k]), 0.05, x_lo[k], x_hi[k]); gx += g; Hx += H
        vals["linear_x"], g, H = rbd.linear_feature_cost(np.concatenate([q[7:], qd]), Ax, bx, tx[k], Wx, act("quadratic"), 0.0)
        gx[6:] += g; Hx[6:, 6:] += H
        R_cm = np.asarray(R_des[k]).T.reshape(-1)
        vals["orientation"], g, H = rbd.frame_orientation_cost(q, ee, R_cm, Wo, act("quadratic"), 0.0); gx += g; Hx += H
        vals["velocity"], g, H = rbd.frame_velocity_cost(q, qd, ee, v_des[k], Wv, act("quadratic"), 0.0); gx += g; Hx += H
        if not terminal:
            vals["input"], g, H = rbd.input_cost(uk, u_des[k], Rw[k], act("quadratic_barrier"), 0.0, np.full(nu, -5.0), np.full(nu, 5.0)); gu += g; Hu += H
            vals["linear_u"], g, H = rbd.linear_feature_cost(uk, Au, bu, tu, Wu, act("pseudo_huber"), 0.3); gu += g; Hu += H
        else:
            vals["input"] = vals["linear_u"] = 0.0
        np.testing.assert_allclose(qk[k], gx, rtol=2e-4, atol=2e-4 * max(1.0, np.abs(gx).max()), err_msg=f"q at knot {k}")
        np.testing.assert_allclose(Qk[k], Hx, rtol=2e-4, atol=2e-4 * max(1.0, np.abs(Hx).max()), err_msg=f"Q at knot {k}")
        if not terminal:
            np.testing.assert_allclose(rk[k], gu, rtol=2e-4, atol=2e-4 * max(1.0, np.abs(gu).max()), err_msg=f"r at knot {k}")
            np.testing.assert_allclose(Rk[k], Hu, rtol=2e-4, atol=2e-4 * max(1.0, np.abs(Hu).max()), err_msg=f"R at knot {k}")
        for name, v in vals.items():
            np.testing.assert_allclose(values[k, TERMS.index(name)], v, rtol=2e-4, atol=1e-5, err_msg=f"value of {name} at knot {k}")


@pytest.mark.gpu
def test_menu_floating_refuses_gravity_target_and_bad_quaternion():
    from conftest import go2_standing_q
    s, x, ref = _go2_setup()
    with pytest.raises(ValueError, match="fixed-base plants only"):
        s.set_cost_menu(CostMenu(input=Term(weight=1.0, target="gravity"), q_ref=go2_standing_q()))
    with pytest.raises(ValueError, match="unit quaternion"):
        s.set_cost_menu(CostMenu(state=Term(weight=1.0, target=np.zeros(s.nx))))
    with pytest.raises(ValueError, match="stored state"):
        s.set_cost_menu(CostMenu(state=Term(weight=1.0)))
    assert s.cost_menu is None
