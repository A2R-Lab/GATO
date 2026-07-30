"""Exact-Hessian (SO-SQP) oracle gate: device stage-block projection vs numpy.

Compares the setup_kkt exact path (assemble blkdiag(Q,R) + lambda^T d2a
contraction + glass::psd_project) against an independent numpy computation:
GN blocks read from the DEVICE (exact-off debug_setup_kkt), d2a from central
finite differences of pinocchio's computeABADerivatives (f64), projection via
np.linalg.eigh with the same eps = 1e-6*(1+maxdiag) clip.

Skips unless a built module was compiled with -DGATO_EXACT_HESSIAN=ON (canary
workflow: build_eh/ + .so swap; the default receipt suite skips these).
"""
import numpy as np
import pytest

import gato

pytestmark = [pytest.mark.gpu]

pin = pytest.importorskip("pinocchio")


def _exact_combo():
    import importlib
    for plant, N in sorted(gato.available()):
        try:
            mod = importlib.import_module(f"gato.bsqpN{N}_{plant}")
        except ImportError:
            continue
        if getattr(mod, "EXACT_HESSIAN_AVAILABLE", False):
            return plant, N
    return None


COMBO = _exact_combo()
needs_exact = pytest.mark.skipif(COMBO is None, reason="no module built with -DGATO_EXACT_HESSIAN=ON")


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
    from conftest import URDFS
    s = gato.BSQP(model_path=str(URDFS[plant]), batch_size=B, N=N, dt=0.01,
                  plant_type=plant, max_sqp_iters=4)
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

    assert np.count_nonzero(ex.solver.get_lambda()) == 0  # fresh solver: lagged lambda = 0
    kkt_gn = gn.solver.debug_setup_kkt(XU0.ravel(), gn.dt, X.ravel(), goals.ravel())
    kkt_ex = ex.solver.debug_setup_kkt(XU0.ravel(), ex.dt, X.ravel(), goals.ravel())
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
    lam = np.asarray(ex.solver.get_lambda(), dtype=np.float64).reshape(B, N + 2, nx)
    assert np.abs(lam).max() > 0
    kkt_gn = gn.solver.debug_setup_kkt(XU0.ravel(), dt, X.ravel(), goals.ravel())
    kkt_ex = ex.solver.debug_setup_kkt(XU0.ravel(), dt, X.ravel(), goals.ravel())
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
    kkt_ex0 = _mk(plant, N, B, exact=True).solver.debug_setup_kkt(XU0.ravel(), dt, X.ravel(), goals.ravel())
    Qe0 = np.asarray(kkt_ex0["Q"]).reshape(B, N, nx, nx)
    assert np.abs(Qe - Qe0).max() > 1e-6
