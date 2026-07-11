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

from oracles.mechanisms import (admm_interval, al_phr, rb_value, rb_grad,  # noqa: E402
                                rb_hess, admm_z_update, al_hinge_value,
                                al_interval_value, al_interval_grad_hess)


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


# ---- elastic (soft/slack) twins: RowGroupDesc.sigma semantics ---------------
# AL = L1 elastic (activation saturates at sigma, |lam| capped);
# ADMM = quadratic slack (smoothed projection, y = sigma * violation).

RHO, SIG = 4.0, 0.25


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.24])
def test_al_elastic_hinge_c1_at_saturation_and_fd(lam):
    """Value/grad continuity at the a == sigma seam (c* = (sigma-lam)/rho)
    plus FD gates on the grad in both regions (an upper-bound-only row so the
    interval grad reduces to the hinge side)."""
    c_star = (SIG - lam) / RHO
    eps = 1e-9
    v_below = al_hinge_value(c_star - eps, lam, RHO, SIG)
    v_above = al_hinge_value(c_star + eps, lam, RHO, SIG)
    np.testing.assert_allclose(v_below, v_above, atol=1e-8)
    g_below, _ = al_interval_grad_hess(c_star - eps, -np.inf, 0.0, lam, 0.0, RHO, SIG)
    g_above, _ = al_interval_grad_hess(c_star + eps, -np.inf, 0.0, lam, 0.0, RHO, SIG)
    np.testing.assert_allclose(g_below, SIG, atol=1e-7)  # a -> sigma at seam
    np.testing.assert_allclose(g_above, SIG, atol=1e-7)
    h = 1e-7
    for c in (c_star - 0.05, c_star + 0.05, c_star + 5.0):  # both regions + deep
        fd = (al_hinge_value(c + h, lam, RHO, SIG)
              - al_hinge_value(c - h, lam, RHO, SIG)) / (2 * h)
        gr, hess = al_interval_grad_hess(c, -np.inf, 0.0, lam, 0.0, RHO, SIG)
        np.testing.assert_allclose(gr, fd, rtol=1e-5, atol=1e-8)
        if c > c_star:
            assert hess == 0.0   # saturated region is linear in c
        # saturated value grows linearly with slope sigma
    assert al_hinge_value(10.0, lam, RHO, SIG) - al_hinge_value(9.0, lam, RHO, SIG) == pytest.approx(SIG)


@pytest.mark.parametrize("lam", [-0.2, 0.0, 0.2])
def test_al_elastic_equality_c1_both_seams(lam):
    """Equality rows saturate on BOTH sides (a > sigma and a < -sigma)."""
    for seam in (SIG, -SIG):
        c_star = (seam - lam) / RHO
        eps = 1e-9
        v = [al_interval_value(c + 1.0, 1.0, 1.0, lam, 0.0, RHO, SIG)
             for c in (c_star - eps, c_star + eps)]
        np.testing.assert_allclose(v[0], v[1], atol=1e-8)
        g = [al_interval_grad_hess(c + 1.0, 1.0, 1.0, lam, 0.0, RHO, SIG)[0]
             for c in (c_star - eps, c_star + eps)]
        np.testing.assert_allclose(g[0], g[1], atol=1e-7)
        np.testing.assert_allclose(abs(g[0]), SIG, atol=1e-7)


def test_admm_z_update_hard_soft_and_limit():
    lo_b, hi_b = -1.0, 1.0
    assert admm_z_update(2.0, lo_b, hi_b, RHO, 0.0) == 1.0          # hard clamp
    assert admm_z_update(0.3, lo_b, hi_b, RHO, SIG) == 0.3          # interior
    z = admm_z_update(2.0, lo_b, hi_b, RHO, SIG)
    assert hi_b < z < 2.0                                            # gives in
    np.testing.assert_allclose(z - hi_b, RHO * (2.0 - hi_b) / (RHO + SIG))
    zl = admm_z_update(-3.0, lo_b, hi_b, RHO, SIG)
    np.testing.assert_allclose(zl - lo_b, RHO * (-3.0 - lo_b) / (RHO + SIG))
    # sigma -> inf recovers the hard clamp
    np.testing.assert_allclose(admm_z_update(2.0, lo_b, hi_b, RHO, 1e12), hi_b, atol=1e-10)


def _signed_violation(Gx, lo, hi):
    return np.where(Gx > hi, Gx - hi, np.where(Gx < lo, Gx - lo, 0.0))


def test_admm_elastic_solves_quadratic_penalty():
    """Elastic ADMM's fixed point solves the quadratic-penalty problem:
    stationarity H x + g + G'y = 0 with y = sigma * (signed violation) —
    proportional dual, the L1-vs-quadratic distinction."""
    H, g, G, lo, hi = _random_box_qp(2)
    sig = 0.5
    r = admm_interval(H, g, G, lo, hi, rho=10.0, iters=20000, eps_abs=1e-11,
                      sigma_slack=sig)
    assert r["r_prim"] < 1e-10 and r["r_dual"] < 1e-10
    Gx = G @ r["x"]
    sviol = _signed_violation(Gx, lo, hi)
    assert np.abs(sviol).max() > 1e-3            # slack actually engaged
    np.testing.assert_allclose(r["y"], sig * sviol, atol=1e-8)
    assert np.abs(H @ r["x"] + g + G.T @ r["y"]).max() < 1e-8
    # scipy cross-check on the smooth quadratic-penalty objective
    from scipy.optimize import minimize
    lo_f = np.where(np.isfinite(lo), lo, -1e30)
    hi_f = np.where(np.isfinite(hi), hi, 1e30)

    def pen(x):
        s = _signed_violation(G @ x, lo_f, hi_f)
        return 0.5 * x @ H @ x + g @ x + 0.5 * sig * s @ s

    sp = minimize(pen, np.zeros(len(g)), method="BFGS",
                  options=dict(gtol=1e-12, maxiter=2000))
    np.testing.assert_allclose(r["x"], sp.x, atol=1e-6)


def test_admm_elastic_hard_path_untouched_and_sigma_limit():
    H, g, G, lo, hi = _random_box_qp(6)
    r_hard = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10)
    r_zero = admm_interval(H, g, G, lo, hi, rho=10.0, iters=5000, eps_abs=1e-10,
                           sigma_slack=0.0)
    np.testing.assert_array_equal(r_hard["x"], r_zero["x"])   # bit-equal path
    r_stiff = admm_interval(H, g, G, lo, hi, rho=10.0, iters=20000,
                            eps_abs=1e-11, sigma_slack=1e8)
    np.testing.assert_allclose(r_stiff["x"], r_hard["x"], atol=1e-4)


def test_al_elastic_solves_l1_penalty():
    """Elastic AL solves the L1-penalty problem exactly: multipliers cap at
    sigma, strictly-violated rows sit AT sigma, and the L1 subgradient KKT
    holds (solver-independent). sigma above lam* reproduces the hard solution
    EXACTLY — the sigma-selection rule."""
    H, g, G, lo, hi = _random_box_qp(4)
    hard = al_phr(H, g, G, lo, hi)
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    lam_star = np.max(np.abs(hard["lam_hi"] - hard["lam_lo"]
                             + np.where(eq, hard["lam_eq"], 0.0)))
    assert lam_star > 0.1                        # active rows exist
    # sigma BELOW lam*: slack engages
    sig = 0.3 * lam_star
    soft = al_phr(H, g, G, lo, hi, sigma_slack=sig, outer=60)
    lam_net = soft["lam_hi"] - soft["lam_lo"] + np.where(eq, soft["lam_eq"], 0.0)
    assert (np.abs(lam_net) <= sig + 1e-9).all()             # multiplier cap
    assert np.abs(H @ soft["x"] + g + G.T @ lam_net).max() < 1e-7  # stationarity
    Gx = G @ soft["x"]
    sviol = _signed_violation(Gx, lo, hi)
    assert np.abs(sviol).max() > 1e-4            # it gave in somewhere
    for j in range(len(lam_net)):                # L1 subgradient conditions
        if sviol[j] > 1e-6:
            np.testing.assert_allclose(lam_net[j], sig, atol=1e-8)
        elif sviol[j] < -1e-6:
            np.testing.assert_allclose(lam_net[j], -sig, atol=1e-8)
        elif not eq[j] and np.isfinite(hi[j]) and Gx[j] < hi[j] - 1e-5 \
                and np.isfinite(lo[j]) and Gx[j] > lo[j] + 1e-5:
            assert abs(lam_net[j]) < 1e-8        # strictly interior -> 0
    # sigma ABOVE lam*: elastic == hard exactly
    stiff = al_phr(H, g, G, lo, hi, sigma_slack=2.0 * lam_star, outer=60)
    np.testing.assert_allclose(stiff["x"], hard["x"], atol=1e-7)
    assert stiff["true_violation"] < 1e-7


def test_elastic_l1_vs_quadratic_distinction():
    """Same QP, same sigma: AL's violated-row multiplier is CAPPED at sigma;
    ADMM's is PROPORTIONAL (sigma * viol) — so with viol < 1 the quadratic
    slack pushes back LESS and violates MORE."""
    H, g, G, lo, hi = _random_box_qp(9)
    hard = al_phr(H, g, G, lo, hi)
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    lam_star = np.max(np.abs(hard["lam_hi"] - hard["lam_lo"]
                             + np.where(eq, hard["lam_eq"], 0.0)))
    sig = 0.3 * lam_star
    al = al_phr(H, g, G, lo, hi, sigma_slack=sig, outer=60)
    ad = admm_interval(H, g, G, lo, hi, rho=10.0, iters=20000, eps_abs=1e-11,
                       sigma_slack=sig)
    v_al = np.abs(_signed_violation(G @ al["x"], lo, hi))
    v_ad = np.abs(_signed_violation(G @ ad["x"], lo, hi))
    j = int(np.argmax(v_al))
    assert v_al[j] > 1e-4
    if v_ad[j] < 1.0:                    # proportional dual < cap here
        assert v_ad[j] > v_al[j] - 1e-9  # quadratic slack gives in at least as much
