"""Pure-numpy reference implementations of the CL-1 constraint mechanisms.

These are the iterate-level oracles for the constraint-layer arc
(docs/open-tasks/constraint_layer_locomotion_arc_plan_2026-07-10.md): the CUDA
ADMM/AL bindings must reproduce these updates on matched QP data. The
vocabulary mirrors TrajoptMPCReference's handler seam; the structure mirrors
the GATO design decisions they exist to de-risk:

  * admm_interval: OSQP-style splitting with FIXED rho inside the loop, so the
    x-update matrix (H + sigma*I + rho*G^T G) is constant and factored ONCE —
    exactly the bdsv factor-reuse pattern (factorBDSVBatched + per-iteration
    solveBDSVFactoredBatched). rho adaptation happens BETWEEN calls (= between
    SQP iterations), never inside.
  * al_phr: PHR augmented Lagrangian with the PDDP-settled semantics — hinge
    rows for inequalities, always-active equality rows, TRUE-violation
    reporting, outer dual update lambda <- max(0, lambda + rho*g).

Both solvers take sigma_slack (numpy twin of RowGroupDesc.sigma, the slack
toggle): <= 0 keeps the exact hard path. Soft semantics differ BY DESIGN:
  * AL = L1 elastic (slack xi >= 0 weighted sigma*xi, minimized analytically):
    the activation a = lam + rho*c saturates at sigma, the outer update caps
    |lam| <= sigma, and the converged point solves the L1-penalty problem
    min 0.5 x'Hx + g'x + sigma*||viol||_1 (EXACTLY hard when sigma > lam*).
  * ADMM = quadratic slack (smoothed z-projection, slope rho/(rho+sigma) past
    the bound): dual fixed point y = sigma * violation (proportional, never
    capped), i.e. the quadratic-penalty problem with weight sigma.

Problem form (one knot's block, or any small QP):
    min_x  0.5 x^T H x + g^T x   s.t.  lo <= G x <= hi
with G a selection-style row matrix (the audit's pure-block rows). Equalities
are lo == hi rows; one-sided rows use +/-inf.
"""
import numpy as np


# ---- scalar twins of the shipped CUDA slack/AL math (rowgroups.cuh,
#      admm.cuh) — exact control-flow mirrors, gated by FD/seam tests --------

def admm_z_update(v, lo, hi, rho, sigma=0.0):
    """Twin of admm.cuh::admm_z_update: hard clamp, or smoothed projection
    with slope rho/(rho+sigma) past the bound when sigma > 0."""
    if sigma > 0.0:
        if v > hi:
            return hi + rho * (v - hi) / (rho + sigma)
        if v < lo:
            return lo + rho * (v - lo) / (rho + sigma)
        return v
    return min(max(v, lo), hi)


def al_hinge_value(c, lam, rho, sigma=0.0):
    """Twin of rowgroups.cuh::al_hinge_value (one hinge side, c signed)."""
    a = max(lam + rho * c, 0.0)
    if sigma > 0.0 and a > sigma:
        d = sigma - lam
        return sigma * c - d * d / (2.0 * rho)
    return (a * a - lam * lam) / (2.0 * rho)


def al_interval_value(g, lo, hi, lam_hi, lam_lo, rho, sigma=0.0):
    """Twin of rowgroups.cuh::al_interval_value (eq rows in the lam_hi slot)."""
    if np.isfinite(lo) and lo == hi:
        c = g - hi
        a = lam_hi + rho * c
        if sigma > 0.0 and a > sigma:
            d = sigma - lam_hi
            return sigma * c - d * d / (2.0 * rho)
        if sigma > 0.0 and a < -sigma:
            d = sigma + lam_hi
            return -sigma * c - d * d / (2.0 * rho)
        return lam_hi * c + 0.5 * rho * c * c
    v = 0.0
    if np.isfinite(hi):
        v += al_hinge_value(g - hi, lam_hi, rho, sigma)
    if np.isfinite(lo):
        v += al_hinge_value(lo - g, lam_lo, rho, sigma)
    return v


def al_interval_grad_hess(g, lo, hi, lam_hi, lam_lo, rho, sigma=0.0):
    """Twin of rowgroups.cuh::al_interval_grad_hess. Returns (gr, h)."""
    gr = 0.0
    h = 0.0
    soft = sigma > 0.0
    if np.isfinite(lo) and lo == hi:
        a = lam_hi + rho * (g - hi)
        if soft and a > sigma:
            return sigma, 0.0
        if soft and a < -sigma:
            return -sigma, 0.0
        return a, rho
    if np.isfinite(hi):
        a = lam_hi + rho * (g - hi)
        if a > 0.0:
            if soft and a > sigma:
                gr += sigma
            else:
                gr += a
                h += rho
    if np.isfinite(lo):
        a = lam_lo + rho * (lo - g)
        if a > 0.0:
            if soft and a > sigma:
                gr -= sigma
            else:
                gr -= a
                h += rho
    return gr, h


def admm_interval(H, g, G, lo, hi, rho=1.0, sigma=1e-6, iters=200,
                  eps_abs=1e-8, x0=None, y0=None, callback=None,
                  sigma_slack=0.0):
    """OSQP-style ADMM on the interval-constrained QP. Returns dict with
    x, z, y (dual), iterations run, and final primal/dual residuals.

    x0/y0 warm starts mirror the MPC shift semantics CL-1 needs to test.
    callback(k, x, z, y) lets tests capture per-iterate state (the CUDA
    kernel must match these iterates on the same data, same rho/sigma).
    sigma_slack > 0 = quadratic slack (smoothed z-projection, twin of
    RowGroupDesc.sigma on the ADMM path); <= 0 = exact hard clamp.
    """
    H = np.asarray(H, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n = H.shape[0]
    m = G.shape[0]

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    y = np.zeros(m) if y0 is None else np.asarray(y0, dtype=np.float64).copy()
    z = np.clip(G @ x, lo, hi)

    # fixed rho -> constant matrix -> factor once (the bdsv-reuse structure)
    M = H + sigma * np.eye(n) + rho * (G.T @ G)
    from scipy.linalg import cho_factor, cho_solve
    F = cho_factor(M)

    r_prim = r_dual = np.inf
    k = 0
    for k in range(1, iters + 1):
        z_prev = z.copy()
        rhs = sigma * x - g + G.T @ (rho * z - y)
        x = cho_solve(F, rhs)
        Gx = G @ x
        v = Gx + y / rho
        if sigma_slack > 0.0:                       # smoothed projection
            z = v.copy()
            over, under = v > hi, v < lo            # False on +/-inf bounds
            z[over] = hi[over] + rho * (v[over] - hi[over]) / (rho + sigma_slack)
            z[under] = lo[under] + rho * (v[under] - lo[under]) / (rho + sigma_slack)
        else:
            z = np.clip(v, lo, hi)                  # interval projection
        y = y + rho * (Gx - z)                      # dual update
        r_prim = np.abs(Gx - z).max() if m else 0.0
        r_dual = rho * np.abs(G.T @ (z - z_prev)).max() if m else 0.0
        if callback is not None:
            callback(k, x.copy(), z.copy(), y.copy())
        if r_prim < eps_abs and r_dual < eps_abs:
            break
    return dict(x=x, z=z, y=y, iters=k, r_prim=r_prim, r_dual=r_dual)


def al_phr(H, g, G, lo, hi, rho0=1.0, rho_factor=10.0, outer=20,
           inner_tol=1e-10, viol_tol=1e-9, lam0=None, sigma_slack=0.0):
    """PHR augmented Lagrangian on the same QP, interval rows split into
    hinge inequalities (G x - hi <= 0, lo - G x <= 0) and always-active
    equality rows (lo == hi). Inner minimization is exact per active set
    (Newton on the piecewise-quadratic AL — it IS quadratic per active set).
    Per-row grad/hess route through the al_interval_grad_hess scalar twin,
    so this solver and the shipped CUDA math cannot drift apart.

    Returns dict(x, lam_hi, lam_lo, lam_eq, outer_iters, true_violation).
    PDDP-settled semantics: acceptance measured on TRUE violation
    (sum of positive parts), duals warm-startable via lam0.
    sigma_slack > 0 = L1 elastic (activation saturates at sigma, outer
    update caps |lam| <= sigma; the soft problem's violation does NOT go
    to zero, so the outer loop exits on dual stationarity instead).
    """
    H = np.asarray(H, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n = H.shape[0]
    m = G.shape[0]
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)

    lam_hi = np.zeros(m) if lam0 is None else np.asarray(lam0[0], dtype=np.float64).copy()
    lam_lo = np.zeros(m) if lam0 is None else np.asarray(lam0[1], dtype=np.float64).copy()
    lam_eq = np.zeros(m) if lam0 is None else np.asarray(lam0[2], dtype=np.float64).copy()

    rho = rho0
    soft = sigma_slack > 0.0
    x = np.zeros(n)
    true_viol = np.inf
    it = 0
    for it in range(1, outer + 1):
        # inner: damped Newton on the AL (convex piecewise quadratic, C^1);
        # per-row grad/hess via the shipped-math scalar twin (eq rows pass
        # their multiplier in the lam_hi slot, matching the CUDA layout).
        # Backtracking on the AL VALUE is required in soft mode: saturated
        # rows have zero Hessian, so full Newton steps overshoot the narrow
        # quadratic pieces at large rho and cycle across active sets (the
        # shipped solver is immune — its line search plays this role).
        def al_value(xv):
            Gxv = G @ xv
            v = 0.5 * xv @ H @ xv + g @ xv
            for j in range(m):
                lam1 = lam_eq[j] if eq[j] else lam_hi[j]
                v += al_interval_value(Gxv[j], lo[j], hi[j],
                                       lam1, lam_lo[j], rho, sigma_slack)
            return v

        f = al_value(x)
        for _ in range(100):
            Gx = G @ x
            grad = H @ x + g
            Hess = H.copy()
            for j in range(m):
                lam1 = lam_eq[j] if eq[j] else lam_hi[j]
                gr, hh = al_interval_grad_hess(Gx[j], lo[j], hi[j],
                                               lam1, lam_lo[j], rho, sigma_slack)
                grad += gr * G[j]
                if hh != 0.0:
                    Hess += hh * np.outer(G[j], G[j])
            step = np.linalg.solve(Hess, -grad)
            alpha = 1.0
            for _ls in range(60):
                f_try = al_value(x + alpha * step)
                if f_try < f or alpha * np.abs(step).max() < inner_tol:
                    break
                alpha *= 0.5
            x = x + alpha * step
            f = min(f, f_try)
            if alpha * np.abs(step).max() < inner_tol:
                break
        # outer: PHR dual update on TRUE violations (soft: cap at sigma —
        # the elastic problem's multiplier bound, |lam| <= sigma for eq)
        Gx = G @ x
        prev = (lam_eq.copy(), lam_hi.copy(), lam_lo.copy())
        lam_eq = np.where(eq, lam_eq + rho * (Gx - hi), lam_eq)
        lam_hi = np.where(~eq & np.isfinite(hi), np.maximum(0.0, lam_hi + rho * (Gx - hi)), lam_hi)
        lam_lo = np.where(~eq & np.isfinite(lo), np.maximum(0.0, lam_lo + rho * (lo - Gx)), lam_lo)
        if soft:
            lam_eq = np.clip(lam_eq, -sigma_slack, sigma_slack)
            lam_hi = np.minimum(lam_hi, sigma_slack)
            lam_lo = np.minimum(lam_lo, sigma_slack)
        viol_hi = np.maximum(0.0, Gx - hi)[np.isfinite(hi) & ~eq]
        viol_lo = np.maximum(0.0, lo - Gx)[np.isfinite(lo) & ~eq]
        viol_eq = np.abs(Gx - hi)[eq]
        true_viol = sum(v.sum() for v in (viol_hi, viol_lo, viol_eq))
        if true_viol < viol_tol:
            break
        if soft:
            # engaged slack: violation stays positive — exit on dual
            # stationarity (the capped multipliers stop moving)
            dlam = max(np.abs(lam_eq - prev[0]).max(initial=0.0),
                       np.abs(lam_hi - prev[1]).max(initial=0.0),
                       np.abs(lam_lo - prev[2]).max(initial=0.0))
            if dlam < 1e-11:
                break
        rho = min(rho * rho_factor, 1e8)
    return dict(x=x, lam_hi=lam_hi, lam_lo=lam_lo, lam_eq=lam_eq,
                outer_iters=it, true_violation=float(true_viol))


# ---- relaxed-barrier scalars: numpy twin of gato/bsqp/rowgroups.cuh --------
# (same formulas; the pytest FD/continuity checks gate the shipped math)

def rb_value(d, mu, delta):
    d = np.asarray(d, dtype=np.float64)
    r = d / delta
    return np.where(d > delta, -mu * np.log(np.where(d > 0, d, 1.0)),
                    -mu * (np.log(delta) - 1.5 + 2.0 * r - 0.5 * r * r))


def rb_grad(d, mu, delta):
    d = np.asarray(d, dtype=np.float64)
    return np.where(d > delta, -mu / np.where(d != 0, d, 1.0),
                    -mu * (2.0 - d / delta) / delta)


def rb_hess(d, mu, delta):
    d = np.asarray(d, dtype=np.float64)
    return np.where(d > delta, mu / np.where(d != 0, d * d, 1.0),
                    mu / delta**2)
