"""CL-3 floating-base bring-up gates (go2, N=16).

Oracle = pinocchio with a free-flyer root. The module's stored state is
[p(3); quat xyzw(4); q_act(12); qd(18)] (pin layout); the tangent is
[v_lin; omega; joints] (pin integrate/difference convention — locked against
pinocchio in test_manifold.py). Dynamics step = SI-EULER (the floating
default): v' = v + dt·aba(q, v, tau),  q' = integrate(q, v'·dt).

The linearization gate follows the verify-analytic-vs-FD rule: the device
tangent A|B blocks are checked against central finite differences of the
pinocchio step map, differenced/retracted on the manifold.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pin = pytest.importorskip("pinocchio")
if importlib.util.find_spec("gato.bsqpN16_go2") is None:
    pytest.skip("bsqpN16_go2 module not built", allow_module_level=True)

import gato
import gato.bsqpN16_go2 as mod

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "external" / "GRiD" / "config" / "robot_assets" / "go2.urdf"

N = 16
DT = 0.01
NQ, NV = 19, 18
NX = NQ + NV          # stored state
TS = 2 * NV           # tangent state
NU = 12
XU_STRIDE = NX + NU


@pytest.fixture(scope="module")
def model():
    return pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())


def _standing_q():
    q = np.zeros(NQ)
    q[2] = 0.35          # base height
    q[6] = 1.0           # quat w (xyzw)
    q[7:] = np.tile([0.0, 0.9, -1.8], 4)   # hip/thigh/calf per leg
    return q


def _rand_state(rng, scale_q=0.05, scale_v=0.2):
    q = _standing_q()
    dq = rng.normal(0, scale_q, NV)
    model_ = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    q = pin.integrate(model_, q, dq)
    v = rng.normal(0, scale_v, NV)
    return np.concatenate([q, v])


def _si_euler(model, data, x, u, dt):
    q, v = x[:NQ], x[NQ:]
    tau = np.concatenate([np.zeros(6), u])
    a = pin.aba(model, data, q, v, tau)
    v2 = v + dt * a
    q2 = pin.integrate(model, q, v2 * dt)
    return np.concatenate([q2, v2])


def _diff(model, xa, xb):
    """xb ⊟ xa (tangent, 2*NV): pin difference on q, subtract on qd."""
    dq = pin.difference(model, xa[:NQ], xb[:NQ])
    return np.concatenate([dq, xb[NQ:] - xa[NQ:]])


def _retract(model, x, dx):
    """x ⊞ dx."""
    q = pin.integrate(model, x[:NQ], dx[:NV])
    return np.concatenate([q, x[NQ:] + dx[NV:]])


def _solver(B):
    return gato.BSQP(model_path=str(URDF), batch_size=B, N=N, dt=DT,
                     plant_type="go2", q_cost=1.0, qd_cost=1e-2, u_cost=1e-4,
                     N_cost=5.0, q_lim_cost=1e-3, vel_lim_cost=0.0,
                     ctrl_lim_cost=0.0)


def test_module_dims():
    assert (mod.NQ, mod.NV) == (NQ, NV)
    assert mod.FLOATING_BASE
    assert mod.CONTROL_SIZE == NU and mod.ACTUATED_SIZE == NU
    assert mod.NUM_BODIES == 13


def test_interface_shapes(model):
    s = _solver(2)
    assert s.floating_base and s.nq == NQ and s.nv == NV and s.nx == NX
    assert s.XU_B.shape == (2, N * XU_STRIDE - NU)


def test_sim_forward_vs_pinocchio(model):
    """One device SI-EULER step == the pinocchio oracle (f32 tolerance),
    quaternion stays unit-norm."""
    rng = np.random.default_rng(42)
    data = model.createData()
    s = _solver(2)
    for trial in range(5):
        x = _rand_state(rng)
        u = rng.normal(0, 2.0, NU)
        got = np.asarray(s.solver.sim_forward(x.astype(np.float32),
                                              u.astype(np.float32), DT))
        assert got.shape == (2, NX)
        np.testing.assert_allclose(got[0], got[1], rtol=0, atol=0)
        want = _si_euler(model, data, x, u, DT)
        assert abs(np.linalg.norm(got[0][3:7]) - 1.0) < 1e-5
        np.testing.assert_allclose(got[0], want, rtol=1e-3, atol=2e-3)


def _kkt_blocks(s, xu, x_s):
    ref = np.zeros(N * 6, dtype=np.float32)
    B = s.batch_size
    out = s.solver.debug_setup_kkt(np.tile(xu, (B, 1)).astype(np.float32), DT,
                                   np.tile(x_s, (B, 1)).astype(np.float32),
                                   np.tile(ref, (B, 1)).astype(np.float32))
    A = np.asarray(out["A"])[0].reshape(N, TS, TS)
    Bm = np.asarray(out["B"])[0].reshape(N, NU, TS)   # col-major per knot
    c = np.asarray(out["c"])[0].reshape(N, TS)
    return (np.transpose(A, (0, 2, 1)),      # -> (N, row, col) from col-major
            np.transpose(Bm, (0, 2, 1)), c)


def _rand_traj(rng, model):
    xu = np.zeros(N * XU_STRIDE - NU, dtype=np.float64)
    for k in range(N):
        x = _rand_state(rng, 0.03, 0.1)
        base = k * XU_STRIDE
        xu[base:base + NX] = x
        if k < N - 1:
            xu[base + NX:base + XU_STRIDE] = rng.normal(0, 1.0, NU)
    return xu


def test_linearization_vs_pin_fd(model):
    """Device tangent A|B + defect c against central-FD of the pinocchio step
    map at several knots."""
    rng = np.random.default_rng(7)
    data = model.createData()
    s = _solver(1)
    xu = _rand_traj(rng, model)
    x_s = xu[:NX].copy()
    A, B, c = _kkt_blocks(s, xu, x_s)

    eps = 1e-4
    for k in (0, 5, N - 2):
        base = k * XU_STRIDE
        x_k = xu[base:base + NX]
        u_k = xu[base + NX:base + XU_STRIDE]
        x_kp1 = xu[base + XU_STRIDE:base + XU_STRIDE + NX]
        pred = _si_euler(model, data, x_k, u_k, DT)

        # defect: c_{k+1} = x_{k+1}^traj ⊟ pred
        np.testing.assert_allclose(c[k + 1], _diff(model, pred, x_kp1),
                                   rtol=1e-3, atol=1e-3)

        A_fd = np.zeros((TS, TS))
        for j in range(TS):
            e = np.zeros(TS); e[j] = eps
            fp = _si_euler(model, data, _retract(model, x_k, e), u_k, DT)
            fm = _si_euler(model, data, _retract(model, x_k, -e), u_k, DT)
            A_fd[:, j] = _diff(model, fm, fp) / (2 * eps)
        np.testing.assert_allclose(A[k], A_fd, rtol=5e-3, atol=5e-3)

        B_fd = np.zeros((TS, NU))
        for j in range(NU):
            e = np.zeros(NU); e[j] = eps
            fp = _si_euler(model, data, x_k, u_k + e, DT)
            fm = _si_euler(model, data, x_k, u_k - e, DT)
            B_fd[:, j] = _diff(model, fm, fp) / (2 * eps)
        np.testing.assert_allclose(B[k], B_fd, rtol=5e-3, atol=5e-3)


def test_c0_is_manifold_gap(model):
    """c_0 = x_0 ⊟ x_s (stored-format x_s, tangent defect)."""
    rng = np.random.default_rng(3)
    s = _solver(1)
    xu = _rand_traj(rng, model)
    x_s = _rand_state(rng, 0.02, 0.05)
    _, _, c = _kkt_blocks(s, xu, x_s)
    np.testing.assert_allclose(c[0], _diff(model, x_s, xu[:NX]),
                               rtol=1e-4, atol=1e-4)


def test_solve_smoke_and_determinism(model):
    """solve() returns finite output and is bit-deterministic run-to-run."""
    rng = np.random.default_rng(11)
    B = 4
    q0 = _standing_q()
    X = np.zeros((B, NX), dtype=np.float32)
    for b in range(B):
        X[b] = _rand_state(rng, 0.01, 0.02)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.0, 0.0, q0[2]

    r1 = _solver(B).solve(X, goals)
    assert np.isfinite(r1.xu).all()
    r2 = _solver(B).solve(X, goals)
    np.testing.assert_array_equal(r1.xu, r2.xu)
