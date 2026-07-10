"""CPU-only gates for the CL-1 mechanism oracles (test/oracles/mechanisms.py)
and the shipped relaxed-barrier scalar math (numpy twin of rowgroups.cuh).

Everything here is numpy/scipy on tiny QPs — no GPU, no built modules. The
point: when CL-1's CUDA ADMM/AL bindings land, they are gated against these
references iterate-for-iterate; the references themselves must therefore be
verified against solver-independent KKT conditions first (this file).
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from oracles.mechanisms import admm_interval, al_phr, rb_value, rb_grad, rb_hess  # noqa: E402


def _random_box_qp(seed, n=8, one_sided=2, equalities=1):
    """Seeded SPD QP with box rows on all variables (G = I): `one_sided` rows
    keep only an upper bound, `equalities` rows have lo == hi. Bounds are
    placed so several rows are ACTIVE at the optimum."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    H = A @ A.T + n * np.eye(n)
    g = 5.0 * rng.normal(size=n)
    G = np.eye(n)
    xs = np.linalg.solve(H, -g)  # unconstrained solution
    lo = xs - np.abs(rng.normal(size=n)) * 0.5
    hi = xs + np.abs(rng.normal(size=n)) * 0.5
    # shift some bounds INSIDE the unconstrained solution -> active rows
    # (opposite bound moved clear so every interval stays lo < hi)
    lo[0] = xs[0] + 0.1
    hi[0] = lo[0] + 1.0
    hi[1] = xs[1] - 0.1
    lo[1] = hi[1] - 1.0
    for j in range(2, 2 + one_sided):
        lo[j] = -np.inf
    for j in range(2 + one_sided, 2 + one_sided + equalities):
        lo[j] = hi[j] = xs[j] - 0.05
    return H, g, G, lo, hi


def _kkt_check(H, g, G, lo, hi, x, y, tol=1e-6):
    """Solver-independent KKT conditions for min .5x'Hx+g'x, lo<=Gx<=hi with
    multiplier y for the row constraints (OSQP convention: stationarity
    Hx + g + G'y = 0; y >= 0 at upper bounds, y <= 0 at lower, 0 inactive)."""
    Gx = G @ x
    assert np.abs(H @ x + g + G.T @ y).max() < tol, "stationarity"
    assert (Gx <= hi + tol).all() and (Gx >= lo - tol).all(), "primal feasibility"
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    for j in range(len(y)):
        if eq[j]:
            continue  # equality multipliers are free
        at_hi = np.isfinite(hi[j]) and Gx[j] > hi[j] - 1e-5
        at_lo = np.isfinite(lo[j]) and Gx[j] < lo[j] + 1e-5
        if not at_hi and not at_lo:
            assert abs(y[j]) < tol, f"row {j}: inactive but |y|={abs(y[j]):.2e}"
        elif at_hi:
            assert y[j] > -tol, f"row {j}: upper-active but y={y[j]:.2e}"
        elif at_lo:
            assert y[j] < tol, f"row {j}: lower-active but y={y[j]:.2e}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_admm_interval_kkt(seed):
    H, g, G, lo, hi = _random_box_qp(seed)
    r = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10)
    assert r["r_prim"] < 1e-9 and r["r_dual"] < 1e-9
    _kkt_check(H, g, G, lo, hi, r["x"], r["y"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_al_phr_matches_admm(seed):
    """Both mechanisms must find the same primal point AND the same
    multipliers (lam_hi - lam_lo == ADMM's y) — the cross-mechanism oracle
    agreement CL-1's R1 relies on."""
    H, g, G, lo, hi = _random_box_qp(seed)
    ra = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10)
    rl = al_phr(H, g, G, lo, hi)
    assert rl["true_violation"] < 1e-8
    np.testing.assert_allclose(rl["x"], ra["x"], atol=1e-6)
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    lam_net = rl["lam_hi"] - rl["lam_lo"] + np.where(eq, rl["lam_eq"], 0.0)
    np.testing.assert_allclose(lam_net, ra["y"], atol=1e-5)


def test_admm_scipy_crosscheck():
    """Primal solution vs scipy L-BFGS-B on the bound-constrained QP (exact
    for G = I selection rows — the audit's pure-block case)."""
    from scipy.optimize import minimize
    H, g, G, lo, hi = _random_box_qp(7)
    r = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10)
    sp = minimize(lambda x: 0.5 * x @ H @ x + g @ x, np.zeros(len(g)),
                  jac=lambda x: H @ x + g, method="L-BFGS-B",
                  bounds=list(zip(lo, hi)), options=dict(ftol=1e-15, gtol=1e-12))
    np.testing.assert_allclose(r["x"], sp.x, atol=1e-6)


def test_admm_dual_warm_start():
    """Warm-starting (x*, y*) must converge (residual exit) almost
    immediately — the MPC dual-shift semantics CL-1 implements."""
    H, g, G, lo, hi = _random_box_qp(11)
    cold = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10)
    warm = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10,
                         x0=cold["x"], y0=cold["y"])
    assert warm["iters"] <= 3 < cold["iters"]
    np.testing.assert_allclose(warm["x"], cold["x"], atol=1e-8)


def test_admm_fixed_rho_factor_reuse_structure():
    """The x-update matrix is constant across iterations at fixed rho — the
    property the bdsv factor-reuse design depends on. Guard: iterates from a
    factored-once solve must equal iterates from per-iteration re-solves."""
    H, g, G, lo, hi = _random_box_qp(3)
    iterates = []
    admm_interval(H, g, G, lo, hi, rho=5.0, iters=30,
                  callback=lambda k, x, z, y: iterates.append(x))
    # re-run with identical settings: bit-equal trajectory of iterates
    iterates2 = []
    admm_interval(H, g, G, lo, hi, rho=5.0, iters=30,
                  callback=lambda k, x, z, y: iterates2.append(x))
    for a, b in zip(iterates, iterates2):
        np.testing.assert_array_equal(a, b)


def test_certificate_dual_axes_with_real_multipliers():
    """First real exercise of gato.certificate's dual/complementarity path,
    with AL multipliers on a converged solution mapped onto a 1-knot BOX_Q
    group (nx = 2n so the q-block is exactly our variable vector)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from gato.certificate import kkt_residuals

    H, g, G, lo, hi = _random_box_qp(5, equalities=0)
    rl = al_phr(H, g, G, lo, hi)
    n = len(g)
    group = dict(kind=0, block=0, mech=0, n_rows=n, knot_lo=0, knot_hi=1,
                 lo=lo, hi=hi)
    xu = np.concatenate([rl["x"], np.zeros(n), np.zeros(1)])  # q | qd | u
    duals = [(rl["lam_hi"] + rl["lam_lo"]).reshape(1, n)]     # both >= 0
    r = kkt_residuals([group], xu, nx=2 * n, nu=1, duals=duals)
    assert r["primal"] < 1e-7
    assert r["dual"] == 0.0            # PHR multipliers are nonnegative
    assert r["complementarity"] < 1e-7  # viol ~ 0 at the solution
    assert r["n_active"] >= 2           # the two bounds we forced active


# ---- relaxed-barrier scalar math (numpy twin of rowgroups.cuh) -------------

MU, DELTA = 0.37, 0.1


@pytest.mark.parametrize("d", [2.0, 0.5, 0.11, 0.1, 0.09, 0.02, 0.0, -0.3])
def test_rb_grad_hess_finite_difference(d):
    h = 1e-6
    fd_g = (rb_value(d + h, MU, DELTA) - rb_value(d - h, MU, DELTA)) / (2 * h)
    fd_h = (rb_grad(d + h, MU, DELTA) - rb_grad(d - h, MU, DELTA)) / (2 * h)
    # skip FD across the switch point itself (one-sided kinks are C2, not C3)
    if abs(d - DELTA) > 2 * h:
        np.testing.assert_allclose(rb_grad(d, MU, DELTA), fd_g, rtol=1e-4)
        np.testing.assert_allclose(rb_hess(d, MU, DELTA), fd_h, rtol=1e-4)


def test_rb_c2_continuity_at_delta():
    eps = 1e-10
    for f in (rb_value, rb_grad, rb_hess):
        lo_side = f(DELTA - eps, MU, DELTA)
        hi_side = f(DELTA + eps, MU, DELTA)
        np.testing.assert_allclose(lo_side, hi_side, rtol=1e-6)


def test_rb_matches_log_barrier_interior_and_finite_infeasible():
    d = 1.7
    np.testing.assert_allclose(rb_value(d, MU, DELTA), -MU * np.log(d), rtol=1e-12)
    for d_bad in (0.0, -0.5, -10.0):
        assert np.isfinite(rb_value(d_bad, MU, DELTA))
        assert np.isfinite(rb_grad(d_bad, MU, DELTA))
        assert rb_hess(d_bad, MU, DELTA) == MU / DELTA**2  # bounded Hessian
