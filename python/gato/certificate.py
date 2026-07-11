"""Solver-independent constraint certificate for GATO solves.

Port of TrajoptMPCReference's ``kkt_certificate.py`` (PDDP lane, commit
c700baa) onto GATO's row-group vocabulary: instead of an AL handler it takes
the installed row-group descriptors (``BSQP.get_row_groups()``) and a returned
trajectory row, and reports the same residual set:

  primal          = max over active (knot, row) of interval violation
  dual            = max(0, -min lambda)          [needs row duals — CL-1]
  complementarity = max |lambda_j * viol_j|      [needs row duals — CL-1]
  n_active        = rows within ``active_tol`` of (or beyond) a bound

CL-0 solves carry no row multipliers (telemetry / relaxed-barrier modes), so
``duals=None`` reports primal + n_active and leaves the dual axes None. The
ADMM / AL bindings (CL-1) pass their per-row dual state to activate them.
Numpy-only — usable as a pytest gate and from notebooks.
"""
import numpy as np

KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U, KIND_EE_POS = 0, 1, 2, 3


def row_values(group, xu, nx, nu, ee_fk=None):
    """(n_knots, n_rows) g values of one row-group over a flat trajectory row.

    ee_fk: callable q -> xyz for KIND_EE_POS rows (e.g. ``BSQP.ee_pos``);
    required when the group is an EE group."""
    nq = nx // 2
    step = nx + nu
    ks = range(group["knot_lo"], group["knot_hi"])
    kind = group["kind"]
    out = np.empty((len(ks), group["n_rows"]), dtype=np.float64)
    for j, k in enumerate(ks):
        base = k * step
        if kind == KIND_BOX_Q:
            out[j] = xu[base:base + nq]
        elif kind == KIND_BOX_QD:
            out[j] = xu[base + nq:base + nx]
        elif kind == KIND_BOX_U:
            out[j] = xu[base + nx:base + nx + nu]
        elif kind == KIND_EE_POS:
            if ee_fk is None:
                raise ValueError("KIND_EE_POS rows need ee_fk (q -> xyz), e.g. BSQP.ee_pos")
            out[j] = np.asarray(ee_fk(xu[base:base + nq]), dtype=np.float64)[:group["n_rows"]]
        else:
            raise ValueError(f"unknown row-group kind {kind}")
    return out


def kkt_residuals(groups, xu, nx, nu, duals=None, active_tol=1e-4, ee_fk=None):
    """Certificate for one trajectory row against the installed row-groups.

    duals: optional list (one per group) of (n_knots, n_rows) multipliers
    (CL-1 ADMM/AL state). ee_fk: q -> xyz for EE row-groups. Returns
    dict(primal, dual, complementarity, n_active, per_group) — dual axes are
    None without duals.
    """
    primal = 0.0
    dual = None if duals is None else 0.0
    comp = None if duals is None else 0.0
    n_active = 0
    per_group = []
    for gi, grp in enumerate(groups):
        g = row_values(grp, xu, nx, nu, ee_fk=ee_fk)
        lo = np.asarray(grp["lo"], dtype=np.float64)
        hi = np.asarray(grp["hi"], dtype=np.float64)
        viol = np.maximum(0.0, g - hi) + np.maximum(0.0, lo - g)
        g_primal = float(viol.max()) if viol.size else 0.0
        g_active = int(np.count_nonzero((g >= hi - active_tol) | (g <= lo + active_tol)))
        primal = max(primal, g_primal)
        n_active += g_active
        entry = dict(kind=grp["kind"], primal=g_primal, n_active=g_active)
        if duals is not None:
            lam = np.asarray(duals[gi], dtype=np.float64)
            g_dual = float(np.maximum(0.0, -lam).max()) if lam.size else 0.0
            g_comp = float(np.abs(lam * viol).max()) if lam.size else 0.0
            dual = max(dual, g_dual)
            comp = max(comp, g_comp)
            entry.update(dual=g_dual, complementarity=g_comp)
        per_group.append(entry)
    return dict(primal=primal, dual=dual, complementarity=comp,
                n_active=n_active, per_group=per_group)


def certify(result, groups, b=0, duals=None, primal_tol=1e-5, tol=1e-3, ee_fk=None):
    """Gate wrapper: residuals for batch entry ``b`` of a SolveResult + pass/fail.

    Passes when primal <= primal_tol and (if duals given) dual/complementarity
    <= tol — the arc's "approximately hard" claim, honestly measured.
    """
    r = kkt_residuals(groups, np.asarray(result.xu[b], dtype=np.float64),
                      result.nx, result.nu, duals=duals, ee_fk=ee_fk)
    ok = r["primal"] <= primal_tol
    if r["dual"] is not None:
        ok = ok and r["dual"] <= tol and r["complementarity"] <= tol
    return r, ok
