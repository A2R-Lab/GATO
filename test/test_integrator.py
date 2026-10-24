"""The rk4 module variant: the classical fourth-order step on the generated grid
integrator (0.1.5). The fixed-base default modules keep the hand-rolled
trapezoidal step (bit-identical, see the goldens); the rk4 variant routes the
step, its linearization and the merit's integrator error through the same
grid step twins the floating base uses.

Gates: the device step against a numpy RK4 on Pinocchio's ABA, the A and B
blocks of the KKT linearization against central differences of that same
step, a solve that converges on the arm reach problem, and the refusals of
the combinations the module cannot build."""
import importlib.util

import numpy as np
import pytest

import gato
from conftest import ARM_START, GO2_DT, GO2_TEST_PARAMS, GO2_URDF, TEST_PARAMS, URDFS, arm_problem, go2_goals_at, go2_standing_x

PLANT, N = "iiwa14", 16


def _rk4_module():
    name = gato.module_name(PLANT, N, "rk4")
    if importlib.util.find_spec(f"gato.{name}") is None:
        pytest.fail(f"{name} is a receipt-profile module (test/receipt_modules.txt) and must be built")


def _solver(batch_size=1, variant="rk4", **kw):
    return gato.BSQP(model_path=str(URDFS[PLANT]), batch_size=batch_size, N=N, dt=0.01,
                     params=TEST_PARAMS, plant_type=PLANT, variant=variant, **kw)


def test_module_names_and_variants():
    """Host: the rk4 token is a variant with its own module name."""
    assert gato.module_name(PLANT, N, "rk4") == f"bsqpN{N}_{PLANT}_rk4"
    assert gato.module_name(PLANT, N) == f"bsqpN{N}_{PLANT}"
    with pytest.raises(ValueError):
        gato.module_name(PLANT, N, "midpoint")
    from gato.interface import INTEGRATORS, VARIANTS
    assert "rk4" in VARIANTS and INTEGRATORS["rk4"] == "rk4" and INTEGRATORS["default"] is None


def test_build_refuses_the_combinations_the_module_cannot_compile(tmp_path):
    """Host: integrator= is its own variant; fc and eh keep the single-stage step."""
    from gato.builder import build
    urdf = URDFS[PLANT]
    with pytest.raises(ValueError, match="separate module variants"):
        build(urdf, name="x", integrator="rk4", exact_hessian=True, build_dir=tmp_path)
    with pytest.raises(ValueError, match="separate module variants"):
        build(urdf, name="x", integrator="rk4", contact_forces=True, build_dir=tmp_path)
    with pytest.raises(ValueError, match="integrator must be"):
        build(urdf, name="x", integrator="midpoint", build_dir=tmp_path)


@pytest.mark.gpu
def test_module_reports_its_step():
    _rk4_module()
    s = _solver()
    assert s.integrator == "rk4" and s.lib.INTEGRATOR == 3
    s0 = _solver(variant="default")
    assert s0.integrator == "trapezoidal" and getattr(s0.lib, "INTEGRATOR", 2) == 2


def _pin():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(URDFS[PLANT]))
    return pin, model, model.createData()


def _state(seed=3):
    rng = np.random.default_rng(seed)
    q = np.asarray(ARM_START[PLANT], dtype=np.float64) + rng.normal(0, 0.2, len(ARM_START[PLANT]))
    qd = rng.normal(0, 0.5, len(q))
    u = rng.normal(0, 5.0, len(q))
    return q, qd, u


@pytest.mark.gpu
def test_device_step_is_rk4_on_the_same_dynamics():
    """sim_forward on the rk4 module reproduces a numpy RK4 over Pinocchio's ABA
    to float rounding, and differs from the trapezoidal module's step by the
    integrator's own truncation term (so the variant is not a relabel)."""
    _rk4_module()
    from gato.common import rk4
    pin, model, data = _pin()
    q, qd, u = _state()
    dt = 0.01
    s = _solver()
    x = np.concatenate([q, qd]).astype(np.float32)
    x_dev = np.asarray(s.sim_forward(x[None], u.astype(np.float32)[None], dt)).ravel()
    q_ref, qd_ref = rk4(model, data, q, qd, u, dt)
    x_ref = np.concatenate([q_ref, qd_ref])
    assert np.allclose(x_dev, x_ref, rtol=1e-4, atol=2e-5), np.abs(x_dev - x_ref).max()
    x_trap = np.asarray(_solver(variant="default").sim_forward(x[None], u.astype(np.float32)[None], dt)).ravel()
    gap = np.abs(x_trap - x_ref).max()
    assert gap > 1e-4, "the trapezoidal module's step should differ from RK4 at dt = 0.01 with this velocity"


@pytest.mark.gpu
def test_linearization_matches_central_differences_of_the_step():
    """A = d x_{k+1} / d x_k and B = d x_{k+1} / d u_k from setup_kkt (debug_setup_kkt)
    against central differences of the device step itself (float64 reference
    from the same numpy RK4), at every running knot of a held trajectory."""
    _rk4_module()
    from gato.common import rk4
    pin, model, data = _pin()
    s = _solver()
    nx, nu, dt = s.nx, s.nu, s.dt
    q, qd, u = _state(seed=5)
    x0 = np.concatenate([q, qd])
    xu = np.zeros(s.xu_size, dtype=np.float64)
    step = nx + nu
    for k in range(N):
        xu[k * step:k * step + nx] = x0
        if k < N - 1:
            xu[k * step + nx:(k + 1) * step] = u
    _, goals = arm_problem(PLANT, N)
    d = s.debug_setup_kkt(xu[None].astype(np.float32), x0[None].astype(np.float32), goals)
    # the dump is column-major per knot (A nx x nx, B nx rows by nu columns) over
    # N slots; the terminal slot is never written by the running-knot chain
    A = np.asarray(d["A"], dtype=np.float64).reshape(N, nx, nx).transpose(0, 2, 1)[:N - 1]
    B = np.asarray(d["B"], dtype=np.float64).reshape(N, nu, nx).transpose(0, 2, 1)[:N - 1]
    nq = model.nq

    def f(x, uu):
        qn, qdn = rk4(model, data, x[:nq], x[nq:], uu, dt)
        return np.concatenate([qn, qdn])

    eps = 1e-5
    A_fd = np.zeros((nx, nx)); B_fd = np.zeros((nx, nu))
    for i in range(nx):
        e = np.zeros(nx); e[i] = eps
        A_fd[:, i] = (f(x0 + e, u) - f(x0 - e, u)) / (2 * eps)
    for j in range(nu):
        e = np.zeros(nu); e[j] = eps
        B_fd[:, j] = (f(x0, u + e) - f(x0, u - e)) / (2 * eps)
    assert np.allclose(A[0], A_fd, rtol=2e-3, atol=2e-3), np.abs(A[0] - A_fd).max()
    assert np.allclose(B[0], B_fd, rtol=2e-3, atol=2e-4), np.abs(B[0] - B_fd).max()
    # every running knot linearizes the same held state
    assert np.allclose(A[1:], A[0]) and np.allclose(B[1:], B[0])


@pytest.mark.gpu
def test_solve_converges_on_the_reach_problem():
    """The rk4 module solves the arm reach problem to a comparable final cost and
    the same end-effector error class as the trapezoidal module; two fresh
    solvers give bit-identical results (a solver instance carries its
    trust-region state across calls, so repeats on one instance differ)."""
    _rk4_module()
    x, goals = arm_problem(PLANT, N)
    s = _solver()
    r1 = s.solve(x, goals)
    r2 = _solver().solve(x, goals)
    assert np.array_equal(r1.xu, r2.xu)
    assert np.all(np.isfinite(r1.xu))
    s0 = _solver(variant="default")
    r0 = s0.solve(x, goals)
    err = lambda r, sol: np.linalg.norm(sol.ee_pos(r.xu[0, -(sol.nx):][:sol.nq]) - goals[0, :3])
    assert err(r1, s) < 2 * max(err(r0, s0), 1e-2) + 1e-3


@pytest.mark.gpu
def test_floating_base_rk4_step_and_linearization():
    """The go2 rk4 module: the device step reproduces Pinocchio's free-flyer RK4
    (stage velocities averaged, one retraction), and the tangent A and B blocks
    of the linearization match central differences of the device step itself
    through the manifold retraction and difference."""
    name = gato.module_name("go2", 16, "rk4")
    if importlib.util.find_spec(f"gato.{name}") is None:
        pytest.fail(f"{name} is a receipt-profile module (test/receipt_modules.txt) and must be built")
    import pinocchio as pin
    from gato.common import rk4, state_difference
    urdf = str(GO2_URDF)
    model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    data = model.createData()
    nq, nv = model.nq, model.nv
    s = gato.BSQP(model_path=urdf, batch_size=1, N=16, dt=GO2_DT, params=GO2_TEST_PARAMS,
                  plant_type="go2", variant="rk4")
    assert s.integrator == "rk4"
    x0 = np.asarray(go2_standing_x(), dtype=np.float64)
    rng = np.random.default_rng(1)
    x0[nq:] = rng.normal(0, 0.3, nv)
    u = rng.normal(0, 3.0, s.nu)
    dt = GO2_DT

    def f(x, uu):
        return np.asarray(s.sim_forward(x.astype(np.float32)[None], uu.astype(np.float32)[None], dt)).ravel().astype(np.float64)

    qr, qdr = rk4(model, data, x0[:nq], x0[nq:], u, dt)
    assert np.abs(state_difference(np.concatenate([qr, qdr]), f(x0, u), nq, nv)).max() < 1e-5

    N, ts, step = s.N, 2 * nv, s.nx + s.nu
    xu = np.zeros(s.xu_size)
    for k in range(N):
        xu[k * step:k * step + s.nx] = x0
        if k < N - 1:
            xu[k * step + s.nx:(k + 1) * step] = u
    d = s.debug_setup_kkt(xu[None].astype(np.float32), x0[None].astype(np.float32),
                          go2_goals_at(model, x0.astype(np.float32), 1))
    A = np.asarray(d["A"], dtype=np.float64).reshape(N, ts, ts).transpose(0, 2, 1)[0]
    B = np.asarray(d["B"], dtype=np.float64).reshape(N, s.nu, ts).transpose(0, 2, 1)[0]

    def retract(x, dx):
        return np.concatenate([pin.integrate(model, x[:nq], dx[:nv]), x[nq:] + dx[nv:]])

    eps = 1e-3
    A_fd = np.zeros((ts, ts)); B_fd = np.zeros((ts, s.nu))
    for i in range(ts):
        e = np.zeros(ts); e[i] = eps
        A_fd[:, i] = state_difference(f(retract(x0, -e), u), f(retract(x0, e), u), nq, nv) / (2 * eps)
    for j in range(s.nu):
        e = np.zeros(s.nu); e[j] = eps
        B_fd[:, j] = state_difference(f(x0, u - e), f(x0, u + e), nq, nv) / (2 * eps)
    assert np.abs(A - A_fd).max() < 5e-3, np.abs(A - A_fd).max()
    assert np.abs(B - B_fd).max() < 5e-3, np.abs(B - B_fd).max()
