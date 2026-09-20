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
import importlib
from pathlib import Path

import numpy as np
import pinocchio as pin   # [test] extra — a missing dep is a broken env, never a skip
import pytest

import gato
from conftest import TEST_PARAMS

pytestmark = pytest.mark.gpu   # cpu-lane deselects; on a GPU box the module MUST exist


@pytest.fixture(scope="module")
def mod():
    """The go2 N16 module — part of the receipt profile (test/receipt_modules.txt),
    so its absence on a GPU box is a FAILURE, not a skip (no per-robot gating)."""
    try:
        return importlib.import_module("gato.bsqpN16_go2")
    except ImportError as e:
        pytest.fail(f"bsqpN16_go2 not built (./tools/build.sh --profile receipt): {e}")

from conftest import (GO2_URDF as URDF, GO2_N as N, GO2_DT as DT, GO2_NQ as NQ, GO2_NV as NV,  # noqa: E402
                      GO2_NU as NU, GO2_NX as NX, GO2_XU_STRIDE as XU_STRIDE, go2_standing_q as _standing_q,
                      go2_solver)

TS = 2 * NV           # tangent state


@pytest.fixture(scope="module")
def model(go2_model):
    return go2_model


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
    return go2_solver(B, q_lim_cost=1e-3)   # this file keeps the plant q barrier on


def test_module_dims(mod):
    assert (mod.NQ, mod.NV) == (NQ, NV)
    assert mod.FLOATING_BASE
    assert mod.CONTROL_SIZE == NU and mod.ACTUATED_SIZE == NU
    assert mod.NUM_BODIES == 13


def test_interface_shapes(model):
    s = _solver(2)
    assert s.floating_base and s.nq == NQ and s.nv == NV and s.nx == NX
    assert s.xu_size == N * XU_STRIDE - NU


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
    out = s.debug_setup_kkt(np.tile(xu, (B, 1)), np.tile(x_s, (B, 1)), np.tile(ref, (B, 1)))
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


# ---------------------------------------------------------------------------
# fc-on-feet (Wave F, 2026-09-20): the go2 "fc" module VARIANT (bsqpN16_go2_fc,
# receipt profile) carries one world-aligned wrench [n; f] per baked foot
# frame as the tail of every control (CONTROL_SIZE = 12 + 24). The device
# path composes the fc columns and the dfext/dq chain term from the grid
# step's full-force B block (grid_plant_step.cuh); the oracle here is
# pinocchio's aba with the same world wrenches re-expressed in the feet's
# parent-joint frames (gato.common.world_wrench_to_joint_local, the sim-side
# convention), so the gate is the verify-analytic-vs-FD rule against an
# INDEPENDENT dynamics engine, not the device's own composition.
# ---------------------------------------------------------------------------

from conftest import GO2_FEET as FEET, GO2_FC as NFC, go2_solver as _go2_solver, go2_standing_x as _standing_x  # noqa: E402
from gato.common import world_wrench_to_joint_local  # noqa: E402

CS = NU + NFC                 # fc-build control width
XU_STRIDE_FC = NX + CS


@pytest.fixture(scope="module")
def mod_fc():
    """The go2 N16 fc module — receipt profile (test/receipt_modules.txt), so its
    absence on a GPU box is a FAILURE, not a skip."""
    try:
        return importlib.import_module("gato.bsqpN16_go2_fc")
    except ImportError as e:
        pytest.fail(f"bsqpN16_go2_fc not built (./tools/build.sh --profile receipt): {e}")


def _fc_solver(B, **kw):
    return _go2_solver(B, variant="fc", q_lim_cost=1e-3, **kw)


def _feet_fext(model, data, q, fc):
    """fc (24: [n_w; f_w] per foot, world axes, about the foot frame origin) ->
    pin.aba's joint-local StdVec_Force."""
    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force.Zero())
    for i, name in enumerate(FEET):
        n_w, f_w = fc[6 * i:6 * i + 3], fc[6 * i + 3:6 * i + 6]
        jid, F = world_wrench_to_joint_local(model, data, q, np.concatenate([f_w, n_w]),
                                             model.getFrameId(name))
        fext[jid] = fext[jid] + F
    return fext


def _si_euler_fc(model, data, x, u_full, dt):
    q, v = x[:NQ], x[NQ:]
    tau = np.concatenate([np.zeros(6), u_full[:NU]])
    fext = _feet_fext(model, data, q, u_full[NU:])
    a = pin.aba(model, data, q, v, tau, fext)
    v2 = v + dt * a
    q2 = pin.integrate(model, q, v2 * dt)
    return np.concatenate([q2, v2])


def _rand_fc(rng, scale=20.0):
    fc = rng.normal(0, scale, NFC)
    fc[5::6] += 30.0    # a standing-like upward bias on the vertical force rows
    return fc


def _kkt_blocks_fc(s, xu, x_s):
    ref = np.zeros(N * 6, dtype=np.float32)
    out = s.debug_setup_kkt(xu[None, :], x_s[None, :], ref[None, :])
    A = np.asarray(out["A"])[0].reshape(N, TS, TS)
    Bm = np.asarray(out["B"])[0].reshape(N, CS, TS)   # col-major per knot
    c = np.asarray(out["c"])[0].reshape(N, TS)
    return np.transpose(A, (0, 2, 1)), np.transpose(Bm, (0, 2, 1)), c


def _rand_traj_fc(rng, model, fc_scale=20.0):
    xu = np.zeros(N * XU_STRIDE_FC - CS, dtype=np.float64)
    for k in range(N):
        base = k * XU_STRIDE_FC
        xu[base:base + NX] = _rand_state(rng, 0.03, 0.1)
        if k < N - 1:
            xu[base + NX:base + NX + NU] = rng.normal(0, 1.0, NU)
            xu[base + NX + NU:base + XU_STRIDE_FC] = _rand_fc(rng, fc_scale)
    return xu


def test_fc_module_dims(mod_fc):
    assert (mod_fc.NQ, mod_fc.NV) == (NQ, NV) and mod_fc.FLOATING_BASE
    assert mod_fc.ACTUATED_SIZE == NU and mod_fc.FC_SIZE == NFC and mod_fc.CONTROL_SIZE == CS
    assert mod_fc.NUM_CONTACT_FRAMES == len(FEET) == 4


def test_fc_interface_slots(model):
    s = _fc_solver(2)
    assert (s.n_actuated, s.n_fc, s.nu) == (NU, NFC, CS)
    assert s.contact_frames == list(FEET)
    assert s.xu_size == N * XU_STRIDE_FC - CS
    assert s.fc_slots(0) == list(range(6))
    assert s.fc_slots("FL_foot_joint") == list(range(6, 12))
    assert s.fc_slots("RL_foot_joint", "f") == [21, 22, 23]
    assert s.fc_slots(2, "n") == [12, 13, 14]
    with pytest.raises(ValueError):
        s.fc_slots("imu_joint")
    # ee_pos resolves any URDF frame: the foot frames are the points the fc wrenches act about
    q = _standing_q()
    for f in FEET:
        p = s.ee_pos(q, frame=f)
        assert p.shape == (3,) and 0.05 < p[2] < 0.12, (f, p)   # 8.5 cm up at the in-the-air keyframe
    with pytest.raises(ValueError, match="frame"):
        s.ee_pos(q, frame="not_a_frame")
    # the default module has no slots
    with pytest.raises(RuntimeError, match="fc"):
        _go2_solver(1).fc_slots(0)


def test_fc_sim_forward_vs_pinocchio(model):
    """One device SI-EULER step under foot wrenches == the pinocchio oracle
    with the same world wrenches at the four foot frames."""
    rng = np.random.default_rng(21)
    data = model.createData()
    s = _fc_solver(2)
    for _ in range(5):
        x = _rand_state(rng)
        u_full = np.concatenate([rng.normal(0, 2.0, NU), _rand_fc(rng)])
        got = np.asarray(s.sim_forward(x, u_full, DT))   # (nx,), (nu,) -> every batch entry
        np.testing.assert_allclose(got[0], got[1], rtol=0, atol=0)
        want = _si_euler_fc(model, data, x, u_full, DT)
        assert abs(np.linalg.norm(got[0][3:7]) - 1.0) < 1e-5
        np.testing.assert_allclose(got[0], want, rtol=1e-3, atol=2e-3)
        # the wrench MOVES the step (not a silently-dropped tail)
        want0 = _si_euler(model, data, x, u_full[:NU], DT)
        assert np.abs(want - want0).max() > 1e-4


def test_fc_zero_wrench_sim_is_bitwise_default(model):
    """fc = 0: the fc module's step is BITWISE the default module's (the
    mapped wrench is a zero array on both — the tail cannot perturb a
    wrench-free rollout)."""
    rng = np.random.default_rng(22)
    s_fc, s0 = _fc_solver(1), _go2_solver(1, q_lim_cost=1e-3)
    for _ in range(3):
        x = _rand_state(rng).astype(np.float32)
        u = rng.normal(0, 2.0, NU).astype(np.float32)
        a = np.asarray(s_fc.sim_forward(x, np.concatenate([u, np.zeros(NFC, np.float32)]), DT))
        b = np.asarray(s0.sim_forward(x, u, DT))
        np.testing.assert_array_equal(a, b)


def test_fc_linearization_vs_pin_fd(model):
    """Device tangent A|B under nonzero foot wrenches vs central FD of the
    pinocchio step map: the fc columns of B (B_full·(-dτ/dfext·dfext/dfc)) and
    the A-block's dfext/dq chain term (the world wrench rotates with the body)
    — both at a wrench large enough that the chain term is not cosmetic."""
    rng = np.random.default_rng(23)
    data = model.createData()
    s = _fc_solver(1)
    xu = _rand_traj_fc(rng, model, fc_scale=25.0)
    x_s = xu[:NX].copy()
    A, B, c = _kkt_blocks_fc(s, xu, x_s)

    eps = 1e-4
    for k in (0, 6, N - 2):
        base = k * XU_STRIDE_FC
        x_k = xu[base:base + NX]
        u_k = xu[base + NX:base + XU_STRIDE_FC]
        x_kp1 = xu[base + XU_STRIDE_FC:base + XU_STRIDE_FC + NX]
        pred = _si_euler_fc(model, data, x_k, u_k, DT)
        np.testing.assert_allclose(c[k + 1], _diff(model, pred, x_kp1), rtol=1e-3, atol=1e-3)

        A_fd = np.zeros((TS, TS))
        for j in range(TS):
            e = np.zeros(TS); e[j] = eps
            fp = _si_euler_fc(model, data, _retract(model, x_k, e), u_k, DT)
            fm = _si_euler_fc(model, data, _retract(model, x_k, -e), u_k, DT)
            A_fd[:, j] = _diff(model, fm, fp) / (2 * eps)
        np.testing.assert_allclose(A[k], A_fd, rtol=5e-3, atol=5e-3)

        B_fd = np.zeros((TS, CS))
        for j in range(CS):
            e = np.zeros(CS); e[j] = eps
            fp = _si_euler_fc(model, data, x_k, u_k + e, DT)
            fm = _si_euler_fc(model, data, x_k, u_k - e, DT)
            B_fd[:, j] = _diff(model, fm, fp) / (2 * eps)
        np.testing.assert_allclose(B[k], B_fd, rtol=5e-3, atol=5e-3)
        # the fc columns are live (a dropped composition would be all-zero)
        assert np.abs(B[k][:, NU:]).max() > 1e-3


def test_fc_chain_term_is_real(model):
    """The A-block dq columns at a nonzero wrench differ from the fixed-f_ext
    linearization by the chain term: re-linearizing with the SAME wrench held
    q-independent (uploaded as the P4.6 band, fc slots zeroed) must NOT match
    the fc-slot linearization — and the FD gate above says which one is right."""
    rng = np.random.default_rng(24)
    data = model.createData()
    s = _fc_solver(1)
    xu = _rand_traj_fc(rng, model, fc_scale=40.0)
    x_s = xu[:NX].copy()
    A_fc, _, _ = _kkt_blocks_fc(s, xu, x_s)
    # same wrenches as a fixed body-major band per knot, fc slots zeroed
    nb = s.n_bodies
    band = np.zeros((1, N, 6 * nb), dtype=np.float32)
    xu0 = xu.copy()
    for k in range(N - 1):
        base = k * XU_STRIDE_FC
        fext = _feet_fext(model, data, xu[base:base + NQ], xu[base + NX + NU:base + XU_STRIDE_FC])
        for jid in range(1, model.njoints):
            F = fext[jid]
            band[0, k, 6 * (jid - 1):6 * (jid - 1) + 6] = np.concatenate([F.angular, F.linear])
        xu0[base + NX + NU:base + XU_STRIDE_FC] = 0.0
    s.set_f_ext_B(band)
    A_band, _, _ = _kkt_blocks_fc(s, xu0, x_s)
    diff = np.abs(A_fc[:N - 1, :, :NV] - A_band[:N - 1, :, :NV]).max()
    assert diff > 1e-3, f"chain term missing: max|A_fc - A_band| = {diff}"
    # the velocity columns carry no chain term (dfext/dqd = 0)
    np.testing.assert_allclose(A_fc[:N - 1, :, NV:], A_band[:N - 1, :, NV:], rtol=1e-4, atol=1e-4)


def test_fc_zero_wrench_linearization_is_bitwise_default(model):
    """fc = 0 on the fc module: A, c and the actuated B columns are BITWISE
    the default module's (the chain term is linear in fc, x - 0 == x)."""
    rng = np.random.default_rng(25)
    s_fc, s0 = _fc_solver(1), _go2_solver(1, q_lim_cost=1e-3)
    xu_fc = _rand_traj_fc(rng, model, fc_scale=0.0)
    for k in range(N - 1):
        xu_fc[k * XU_STRIDE_FC + NX + NU:(k + 1) * XU_STRIDE_FC] = 0.0
    xu0 = np.zeros(N * XU_STRIDE - NU)
    for k in range(N):
        xu0[k * XU_STRIDE:k * XU_STRIDE + NX] = xu_fc[k * XU_STRIDE_FC:k * XU_STRIDE_FC + NX]
        if k < N - 1:
            xu0[k * XU_STRIDE + NX:(k + 1) * XU_STRIDE] = xu_fc[k * XU_STRIDE_FC + NX:k * XU_STRIDE_FC + NX + NU]
    x_s = xu_fc[:NX].copy()
    A1, B1, c1 = _kkt_blocks_fc(s_fc, xu_fc, x_s)
    A0, B0, c0 = _kkt_blocks(s0, xu0, x_s)
    # knots 0..N-2 carry a linearization (the last knot's A/B slot is never
    # written — it is whatever the fresh device buffer held); c is complete
    np.testing.assert_array_equal(A1[:N - 1], A0[:N - 1])
    np.testing.assert_array_equal(c1, c0)
    np.testing.assert_array_equal(B1[:N - 1, :, :NU], B0[:N - 1])
    assert np.abs(B1[:N - 1, :, NU:]).max() > 1e-3   # fc columns still live at fc = 0 (fc-independent)


def test_fc_solve_finite_deterministic(model):
    """fc-build solve with regularized fc slots: finite, run-twice bitwise, and
    fc_traj has the (N-1, 24) shape."""
    rng = np.random.default_rng(26)
    B = 2
    X = np.stack([_rand_state(rng, 0.01, 0.02) for _ in range(B)]).astype(np.float32)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 2::6] = _standing_q()[2]
    xus = []
    for _ in range(2):
        s = _fc_solver(B)
        s.set_fc_cost(1e-3)
        r = s.solve(X, goals)
        assert np.isfinite(r.xu).all()
        assert r.fc_traj(0).shape == (N - 1, NFC) and r.u0().shape == (NU,)
        xus.append(np.asarray(r.xu).copy())
    np.testing.assert_array_equal(xus[0], xus[1])


def _standing_problem(model, B=1):
    x = _standing_x().astype(np.float32)
    from conftest import go2_goals_at
    return np.tile(x, (B, 1)), go2_goals_at(model, x.astype(np.float64), B)


def test_fc_box_pin_zeroes_fc(model):
    """LIN_U box rows [0,0] on every fc slot (AL mech) pin the wrenches to ~0."""
    s = _fc_solver(1)
    s.set_fc_cost(1e-4)
    s.enable_limit_al()
    s.add_fc_box(0.0, 0.0, mech="al")
    X, goals = _standing_problem(model)
    r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    assert np.abs(r.fc_traj(0)).max() < 5e-2


def test_fc_ref_pulls_feet_toward_reference(model):
    """A dominant fc_cost tracks a per-foot reference: mg/4 up on every foot,
    moment rows pinned by add_fc_box on fc_slots(i, "n") — the S1 standing
    setpoint the closed-loop gate uses."""
    s = _fc_solver(1)
    mg = sum(i.mass for i in model.inertias) * 9.81
    ref = np.zeros(NFC, dtype=np.float32)
    for i in range(4):
        ref[s.fc_slots(i, "f")[2]] = mg / 4
        s.add_fc_box(0.0, 0.0, slots=s.fc_slots(i, "n"), mech="al")   # enforced (default mech is telemetry)
    s.set_fc_cost(1e2)
    s.set_fc_ref(ref)
    X, goals = _standing_problem(model)
    r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    fc = r.fc_traj(0)
    err = np.linalg.norm(fc - ref[None, :], axis=1)
    assert err.mean() < 0.3 * np.linalg.norm(ref), (err.mean(), np.linalg.norm(ref))
    assert np.abs(fc[:, [j for i in range(4) for j in s.fc_slots(i, "n")]]).max() < 5e-2


def test_fc_debug_contact_oracle_is_fixed_base_only(model):
    """The generalized-dims contact oracle is not defined on the manifold: it
    must refuse loudly on floating modules (it used to return wrong numbers)."""
    s = _go2_solver(1)
    with pytest.raises(RuntimeError, match="fixed-base"):
        s.solver.debug_contact_dynamics(np.zeros(NQ, np.float32), np.zeros(NQ, np.float32),
                                        np.zeros(NQ, np.float32), np.zeros(NFC, np.float32))
