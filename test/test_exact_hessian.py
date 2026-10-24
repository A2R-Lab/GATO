"""Exact-Hessian (SO-SQP) oracle gate: device stage-block projection vs numpy.

Compares the setup_kkt exact path (assemble blkdiag(Q,R) + lambda^T d2a
contraction + glass::psd_project) against an independent numpy computation:
GN blocks read from the DEVICE (exact-off debug_setup_kkt), d2a from central
finite differences of pinocchio's computeABADerivatives (f64), projection via
np.linalg.eigh with the same eps = 1e-6*(1+maxdiag) clip.

Runs on the "eh" module VARIANT (bsqpN{N}_{plant}_eh, part of the receipt
profile); skips only where no eh module is built.
"""
import numpy as np
import pytest

import gato

pytestmark = [pytest.mark.gpu]

pin = pytest.importorskip("pinocchio")


def _exact_combo():
    combos = sorted(gato.available("eh"))
    return min(combos, key=lambda k: k[1]) if combos else None


COMBO = _exact_combo()
needs_exact = pytest.mark.skipif(COMBO is None, reason="no eh variant module built (gato.build(..., exact_hessian=True))")


def _fd_d2a(model, data, q, v, tau, h=1e-5):
    """4 stacked tensors [i, j, k] = d2(qdd_i)/d(first_j)d(second_k), f64 central FD."""
    nv = model.nv

    def first_order(qq, vv):
        pin.computeABADerivatives(model, data, qq, vv, tau)
        return np.array(data.ddq_dq), np.array(data.ddq_dv), np.array(data.Minv)

    d2a_dqdq = np.zeros((nv, nv, nv))
    d2a_dvdq = np.zeros((nv, nv, nv))
    d2a_dvdv = np.zeros((nv, nv, nv))
    d2a_dtdq = np.zeros((nv, nv, nv))
    for k in range(nv):
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        dq_p, dv_p, mi_p = first_order(qp, v)
        dq_m, dv_m, mi_m = first_order(qm, v)
        d2a_dqdq[:, :, k] = (dq_p - dq_m) / (2 * h)
        d2a_dvdq[:, :, k] = (dv_p - dv_m) / (2 * h)
        d2a_dtdq[:, :, k] = (mi_p - mi_m) / (2 * h)
        vp, vm = v.copy(), v.copy()
        vp[k] += h
        vm[k] -= h
        _, dv_pv, _ = first_order(q, vp)
        _, dv_mv, _ = first_order(q, vm)
        d2a_dvdv[:, :, k] = (dv_pv - dv_mv) / (2 * h)
    return d2a_dqdq, d2a_dvdq, d2a_dvdv, d2a_dtdq


def _contraction(w, tensors, nv):
    """E (3nv x 3nv, symmetric): the lambda^T d2a term over z = [q; v; u]."""
    d2a_dqdq, d2a_dvdq, d2a_dvdv, d2a_dtdq = tensors
    E = np.zeros((3 * nv, 3 * nv))
    Eqq = np.einsum("i,ijk->jk", w, d2a_dqdq)
    Evv = np.einsum("i,ijk->jk", w, d2a_dvdv)
    Evq = np.einsum("i,ijk->jk", w, d2a_dvdq)   # rows = v, cols = q
    Euq = np.einsum("i,ijk->jk", w, d2a_dtdq)   # rows = u, cols = q
    E[:nv, :nv] = 0.5 * (Eqq + Eqq.T)
    E[nv:2 * nv, nv:2 * nv] = 0.5 * (Evv + Evv.T)
    E[nv:2 * nv, :nv] = Evq
    E[:nv, nv:2 * nv] = Evq.T
    E[2 * nv:, :nv] = Euq
    E[:nv, 2 * nv:] = Euq.T
    return E


def _project(P, eps_scale=1e-5):
    # 1e-5 relative floor (device setup_kkt.cuh): must clear the f32 Jacobi
    # reconstruction noise; ABS on the diag per the prototype's _eps_for
    eps = eps_scale * (1.0 + np.max(np.abs(np.diag(P))))
    W, V = np.linalg.eigh(P)
    return (V * np.maximum(W, eps)) @ V.T


def _mk(plant, N, B, exact):
    from conftest import TEST_PARAMS, URDFS
    s = gato.BSQP(model_path=str(URDFS[plant]), batch_size=B, N=N, dt=0.01, params=TEST_PARAMS.replace(max_sqp_iters=4),
                  plant_type=plant, variant="eh")
    if exact:
        s.set_exact_hessian(True)
    return s


def _problem(s, B, N, seed=7):
    rng = np.random.default_rng(seed)
    nq = s.nq
    q0 = rng.uniform(-0.6, 0.6, nq)
    x0 = np.concatenate([q0, rng.normal(0, 0.3, nq)]).astype(np.float32)
    X = np.tile(x0, (B, 1)) + rng.normal(0, 0.02, (B, 2 * nq)).astype(np.float32)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.45, 0.15, 0.5
    traj = N * (s.nx + s.nu) - s.nu
    XU0 = np.zeros((B, traj), dtype=np.float32)
    for k in range(N):
        off = k * (s.nx + s.nu)
        XU0[:, off:off + s.nx] = X + rng.normal(0, 0.01, (B, s.nx)).astype(np.float32)
        if k < N - 1:
            XU0[:, off + s.nx:off + s.nx + s.nu] = rng.normal(0, 1.0, (B, s.nu)).astype(np.float32)
    return X, goals, XU0


@needs_exact
def test_projection_matches_numpy_lambda_zero():
    """lambda = 0: device blocks == eigh-clip projection of the GN blocks (1b)."""
    plant, N = COMBO
    B = 2
    gn = _mk(plant, N, B, exact=False)
    ex = _mk(plant, N, B, exact=True)
    X, goals, XU0 = _problem(gn, B, N)
    nx, nu, nv = gn.nx, gn.nu, gn.nv

    assert np.count_nonzero(ex.get_lambda()) == 0  # fresh solver: lagged lambda = 0
    kkt_gn = gn.debug_setup_kkt(XU0.ravel(), X.ravel(), goals.ravel())
    kkt_ex = ex.debug_setup_kkt(XU0.ravel(), X.ravel(), goals.ravel())
    Qg = np.asarray(kkt_gn["Q"], dtype=np.float64).reshape(B, N, nx, nx)
    Rg = np.asarray(kkt_gn["R"], dtype=np.float64).reshape(B, N, nu, nu)
    Qe = np.asarray(kkt_ex["Q"]).reshape(B, N, nx, nx)
    Re = np.asarray(kkt_ex["R"]).reshape(B, N, nu, nu)

    for b in range(B):
        for k in range(N - 1):
            P = np.zeros((nx + nu, nx + nu))
            P[:nx, :nx] = Qg[b, k]
            P[nx:, nx:] = Rg[b, k]
            Pp = _project(P)
            scale = max(1.0, np.abs(P).max())
            np.testing.assert_allclose(Qe[b, k], Pp[:nx, :nx], rtol=0, atol=5e-5 * scale)
            np.testing.assert_allclose(Re[b, k], Pp[nx:, nx:], rtol=0, atol=5e-5 * scale)
        # terminal block: projected alone
        Pp = _project(Qg[b, N - 1])
        scale = max(1.0, np.abs(Qg[b, N - 1]).max())
        np.testing.assert_allclose(Qe[b, N - 1], Pp, rtol=0, atol=5e-5 * scale)


@needs_exact
def test_contraction_matches_pinocchio_fd():
    """lambda != 0: device blocks == projection of GN + lambda^T d2a (FD oracle, 1c)."""
    plant, N = COMBO
    B = 2
    gn = _mk(plant, N, B, exact=False)
    ex = _mk(plant, N, B, exact=True)
    X, goals, XU0 = _problem(gn, B, N)
    nx, nu, nv = gn.nx, gn.nu, gn.nv
    dt = ex.dt

    # populate lagged lambda with a real Schur solve, then rebuild the KKT at XU0
    ex.solver.solve(XU0.ravel(), dt, X.ravel(), goals.ravel())
    lam = np.asarray(ex.get_lambda(), dtype=np.float64).reshape(B, N + 2, nx)
    assert np.abs(lam).max() > 0
    kkt_gn = gn.debug_setup_kkt(XU0.ravel(), X.ravel(), goals.ravel())
    kkt_ex = ex.debug_setup_kkt(XU0.ravel(), X.ravel(), goals.ravel())
    Qg = np.asarray(kkt_gn["Q"], dtype=np.float64).reshape(B, N, nx, nx)
    Rg = np.asarray(kkt_gn["R"], dtype=np.float64).reshape(B, N, nu, nu)
    Qe = np.asarray(kkt_ex["Q"]).reshape(B, N, nx, nx)
    Re = np.asarray(kkt_ex["R"]).reshape(B, N, nu, nu)

    model, data = ex.model, ex.data
    checked = 0
    for b in range(B):
        for k in (0, N // 2, N - 2):
            off = k * (nx + nu)
            q = XU0[b, off:off + nv].astype(np.float64)
            v = XU0[b, off + nv:off + nx].astype(np.float64)
            tau = XU0[b, off + nx:off + nx + nu].astype(np.float64)
            tensors = _fd_d2a(model, data, q, v, tau)
            lam_kp1 = lam[b, k + 2]  # padded layout: constraint row j at slot j+1
            w = 0.5 * dt * dt * lam_kp1[:nv] + dt * lam_kp1[nv:]
            E = _contraction(w, tensors, nv)
            P = np.zeros((nx + nu, nx + nu))
            P[:nx, :nx] = Qg[b, k]
            P[nx:, nx:] = Rg[b, k]
            P += E
            Pp = _project(P)
            scale = max(1.0, np.abs(P).max())
            np.testing.assert_allclose(Qe[b, k], Pp[:nx, :nx], rtol=0, atol=2e-4 * scale)
            np.testing.assert_allclose(Re[b, k], Pp[nx:, nx:], rtol=0, atol=2e-4 * scale)
            checked += 1
    assert checked == 6
    # the contraction must actually matter somewhere (lambda != 0 changes blocks)
    kkt_ex0 = _mk(plant, N, B, exact=True).debug_setup_kkt(XU0, X, goals)
    Qe0 = np.asarray(kkt_ex0["Q"]).reshape(B, N, nx, nx)
    assert np.abs(Qe - Qe0).max() > 1e-6


def _kinematic_position_hessian(model, data, fid, q, h=1e-6):
    """H[r, i, j] = d^2 p_r / dq_i dq_j of the frame position by central
    differences of the LOCAL_WORLD_ALIGNED frame Jacobian's linear rows."""
    nv = model.nv
    H = np.zeros((3, nv, nv))
    for j in range(nv):
        qp, qm = q.copy(), q.copy()
        qp[j] += h
        qm[j] -= h
        Jp = pin.computeFrameJacobian(model, data, qp, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
        Jm = pin.computeFrameJacobian(model, data, qm, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
        H[:, :, j] = (Jp - Jm) / (2 * h)
    return H


@needs_exact
def test_cost_curvature_matches_pinocchio_kinematic_hessian():
    """set_cost_curvature on (exact_hessian off): every Q block of a held
    trajectory equals the projection of the Gauss-Newton block read from the
    device plus the residual-weighted kinematic Hessian of the end-effector
    position (central differences of pinocchio's frame Jacobian), running
    knots at q_cost and the terminal knot at N_cost; the R blocks and the
    gradients are untouched. The reference is held away from the arm so the
    residual, and with it the curvature term, is large."""
    plant, N = COMBO
    from conftest import URDFS
    s = _mk(plant, N, 1, exact=False)
    s0 = _mk(plant, N, 1, exact=False)
    s.set_cost_curvature(True)
    assert s.cost_curvature and not s.exact_hessian
    model = pin.buildModelFromUrdf(str(URDFS[plant]))
    data = model.createData()
    fid = s.ee_frame_id
    rng = np.random.default_rng(11)
    q = rng.uniform(-0.8, 0.8, s.nq)
    x0 = np.concatenate([q, rng.normal(0, 0.2, s.nq)])
    step = s.nx + s.nu
    xu = np.zeros(s.xu_size)
    for k in range(N):
        xu[k * step:k * step + s.nx] = x0
    goals = np.zeros((1, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.9, -0.6, 1.2
    d_gn = s0.debug_setup_kkt(xu[None].astype(np.float32), x0[None].astype(np.float32), goals)
    d_cc = s.debug_setup_kkt(xu[None].astype(np.float32), x0[None].astype(np.float32), goals)
    nx, nu, nq = s.nx, s.nu, s.nq
    Q_gn = np.asarray(d_gn["Q"], np.float64).reshape(N, nx, nx).transpose(0, 2, 1)
    Q_cc = np.asarray(d_cc["Q"], np.float64).reshape(N, nx, nx).transpose(0, 2, 1)
    R_gn = np.asarray(d_gn["R"], np.float64).reshape(N, nu, nu).transpose(0, 2, 1)
    R_cc = np.asarray(d_cc["R"], np.float64).reshape(N, nu, nu).transpose(0, 2, 1)
    assert np.array_equal(np.asarray(d_gn["q"]), np.asarray(d_cc["q"]))
    assert np.array_equal(np.asarray(d_gn["r"]), np.asarray(d_cc["r"]))
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, fid)
    r = np.array(data.oMf[fid].translation) - goals[0, :3].astype(np.float64)
    H = _kinematic_position_hessian(model, data, fid, q)
    curv = np.einsum("r,rij->ij", r, H)
    assert np.abs(curv).max() > 1e-2, "the curvature term should be visible at this residual"
    for k in range(N):
        w = s.params.N_cost if k == N - 1 else s.params.q_cost
        if k == N - 1:
            P = Q_gn[k].copy()
            P[:nq, :nq] += w * curv
            expect_Q = _project(P)
        else:
            # the device projects the whole stage block [[Q, 0], [0, R]] (its eps
            # floor reaches R too, as on the dynamics-curvature path) and keeps
            # the diagonal blocks
            P = np.zeros((nx + nu, nx + nu))
            P[:nx, :nx] = Q_gn[k]
            P[nx:, nx:] = R_gn[k]
            P[:nq, :nq] += w * curv
            Pp = _project(P)
            expect_Q, expect_R = Pp[:nx, :nx], Pp[nx:, nx:]
            scale = 1.0 + np.abs(expect_R).max()
            assert np.allclose(R_cc[k], expect_R, atol=5e-4 * scale), (k, np.abs(R_cc[k] - expect_R).max())
        scale = 1.0 + np.abs(expect_Q).max()
        assert np.allclose(Q_cc[k], expect_Q, atol=5e-4 * scale), (k, np.abs(Q_cc[k] - expect_Q).max())
        assert not np.allclose(Q_cc[k][:nq, :nq], Q_gn[k][:nq, :nq], atol=1e-6)


@needs_exact
def test_cost_curvature_refusals():
    """The switch needs the eh module, and excludes the cost menu."""
    plant, N = COMBO
    from conftest import URDFS, TEST_PARAMS
    s_default = gato.BSQP(model_path=str(URDFS[plant]), batch_size=1, N=N, dt=0.01, params=TEST_PARAMS,
                          plant_type=plant, variant="default")
    with pytest.raises(RuntimeError, match="eh variant"):
        s_default.set_cost_curvature(True)
    s = _mk(plant, N, 1, exact=False)
    s.set_cost_curvature(True)
    if s.cost_menu_available:
        from gato.cost_menu import CostMenu, Term
        with pytest.raises((RuntimeError, ValueError)):
            s.set_cost_menu(CostMenu(frame_pos=Term(weight=1.0, target=np.zeros(3))))
    s.set_cost_curvature(False)
    assert not s.cost_curvature
