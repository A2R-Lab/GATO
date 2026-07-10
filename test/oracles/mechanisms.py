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

Problem form (one knot's block, or any small QP):
    min_x  0.5 x^T H x + g^T x   s.t.  lo <= G x <= hi
with G a selection-style row matrix (the audit's pure-block rows). Equalities
are lo == hi rows; one-sided rows use +/-inf.
"""
import numpy as np


def admm_interval(H, g, G, lo, hi, rho=1.0, sigma=1e-6, iters=200,
                  eps_abs=1e-8, x0=None, y0=None, callback=None):
    """OSQP-style ADMM on the interval-constrained QP. Returns dict with
    x, z, y (dual), iterations run, and final primal/dual residuals.

    x0/y0 warm starts mirror the MPC shift semantics CL-1 needs to test.
    callback(k, x, z, y) lets tests capture per-iterate state (the CUDA
    kernel must match these iterates on the same data, same rho/sigma).
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
        z = np.clip(Gx + y / rho, lo, hi)          # interval projection
        y = y + rho * (Gx - z)                      # dual update
        r_prim = np.abs(Gx - z).max() if m else 0.0
        r_dual = rho * np.abs(G.T @ (z - z_prev)).max() if m else 0.0
        if callback is not None:
            callback(k, x.copy(), z.copy(), y.copy())
        if r_prim < eps_abs and r_dual < eps_abs:
            break
    return dict(x=x, z=z, y=y, iters=k, r_prim=r_prim, r_dual=r_dual)


def al_phr(H, g, G, lo, hi, rho0=1.0, rho_factor=10.0, outer=20,
           inner_tol=1e-10, viol_tol=1e-9, lam0=None):
    """PHR augmented Lagrangian on the same QP, interval rows split into
    hinge inequalities (G x - hi <= 0, lo - G x <= 0) and always-active
    equality rows (lo == hi). Inner minimization is exact per active set
    (Newton on the piecewise-quadratic AL — it IS quadratic per active set).

    Returns dict(x, lam_hi, lam_lo, lam_eq, outer_iters, true_violation).
    PDDP-settled semantics: acceptance measured on TRUE violation
    (sum of positive parts), duals warm-startable via lam0.
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
    x = np.zeros(n)
    true_viol = np.inf
    it = 0
    for it in range(1, outer + 1):
        # inner: Newton on the AL (piecewise quadratic; iterate active sets)
        for _ in range(100):
            Gx = G @ x
            grad = H @ x + g
            Hess = H.copy()
            # equality rows: always active
            for j in np.where(eq)[0]:
                c = Gx[j] - hi[j]
                grad += (lam_eq[j] + rho * c) * G[j]
                Hess += rho * np.outer(G[j], G[j])
            # inequality rows: PHR hinge activation (lam + rho*c > 0)
            for j in np.where(~eq)[0]:
                if np.isfinite(hi[j]):
                    c = Gx[j] - hi[j]
                    if lam_hi[j] + rho * c > 0:
                        grad += (lam_hi[j] + rho * c) * G[j]
                        Hess += rho * np.outer(G[j], G[j])
                if np.isfinite(lo[j]):
                    c = lo[j] - Gx[j]
                    if lam_lo[j] + rho * c > 0:
                        grad -= (lam_lo[j] + rho * c) * G[j]
                        Hess += rho * np.outer(G[j], G[j])
            step = np.linalg.solve(Hess, -grad)
            x = x + step
            if np.abs(step).max() < inner_tol:
                break
        # outer: PHR dual update on TRUE violations
        Gx = G @ x
        lam_eq = np.where(eq, lam_eq + rho * (Gx - hi), lam_eq)
        lam_hi = np.where(~eq & np.isfinite(hi), np.maximum(0.0, lam_hi + rho * (Gx - hi)), lam_hi)
        lam_lo = np.where(~eq & np.isfinite(lo), np.maximum(0.0, lam_lo + rho * (lo - Gx)), lam_lo)
        viol_hi = np.maximum(0.0, Gx - hi)[np.isfinite(hi) & ~eq]
        viol_lo = np.maximum(0.0, lo - Gx)[np.isfinite(lo) & ~eq]
        viol_eq = np.abs(Gx - hi)[eq]
        true_viol = sum(v.sum() for v in (viol_hi, viol_lo, viol_eq))
        if true_viol < viol_tol:
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
