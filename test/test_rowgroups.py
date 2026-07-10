"""Constraint row-group layer (CL-0): telemetry-only mode gates.

- default OFF and telemetry ON produce bit-identical trajectories (the layer
  is off the solver path);
- installed limit bounds match the URDF <limit> tags with the vendored
  JOINT_LIMIT_MARGIN tightening;
- reported per-group violations match a numpy oracle recomputed from the
  returned trajectory and the descriptors' own bounds;
- telemetry is deterministic solve-to-solve.
"""
import numpy as np
import pytest

import gato
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}

# gato/dynamics/plant.cuh JOINT_LIMIT_MARGIN<T>() — limits are TIGHTENED by
# |margin| (lo - margin, hi + margin with margin = -0.1)
JOINT_LIMIT_MARGIN = -0.1

KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U = 0, 1, 2
BLOCK_X, BLOCK_U = 0, 1


def _inputs(plant, N, B):
    q0 = np.asarray(START[plant], dtype=np.float32)
    nq = q0.size
    x0 = np.concatenate([q0, np.zeros(nq, dtype=np.float32)])
    rng = np.random.default_rng(1234)
    X = np.tile(x0, (B, 1)) + rng.normal(0, 0.01, (B, 2 * nq)).astype(np.float32)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.35, 0.25, 0.5
    return X, goals


def _oracle_violations(xu, groups, nx, nu):
    """numpy recompute of {max, sum} true violation per group from the flat
    trajectory row and the group's own bounds (f32 throughout, like the kernel)."""
    nq = nx // 2
    step = nx + nu
    out = []
    for grp in groups:
        lo = np.asarray(grp["lo"], dtype=np.float32)
        hi = np.asarray(grp["hi"], dtype=np.float32)
        viols = []
        for k in range(grp["knot_lo"], grp["knot_hi"]):
            base = k * step
            if grp["kind"] == KIND_BOX_Q:
                g = xu[base:base + nq]
            elif grp["kind"] == KIND_BOX_QD:
                g = xu[base + nq:base + nx]
            else:
                g = xu[base + nx:base + nx + nu]
            g = g.astype(np.float32)
            viols.append(np.maximum(0, g - hi) + np.maximum(0, lo - g))
        v = np.concatenate(viols)
        out.append((v.max(), v.sum(dtype=np.float64)))
    return out


def test_telemetry_off_the_solver_path(make_solver, smallest_module):
    """Bit-parity: telemetry enabled vs disabled must not change the solve."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    ref = make_solver(plant, N, batch_size=B).solve(X, goals)
    assert ref.stats.row_max_violation is None

    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_telemetry()
    got = s.solve(X, goals)

    np.testing.assert_array_equal(ref.xu, got.xu)
    np.testing.assert_array_equal(ref.stats.sqp_iters, got.stats.sqp_iters)
    assert got.stats.row_max_violation.shape == (3, B)
    assert got.stats.row_sum_violation.shape == (3, B)

    # disable drops the stats fields again
    s.disable_row_groups()
    off = s.solve(X, goals)
    assert off.stats.row_max_violation is None


def test_limit_bounds_match_urdf(make_solver, smallest_module):
    """Descriptor bounds == URDF <limit> tags tightened by |JOINT_LIMIT_MARGIN|."""
    pytest.importorskip("pinocchio")

    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    s.enable_limit_telemetry()
    groups = s.get_row_groups()
    assert [g["kind"] for g in groups] == [KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U]
    assert [g["block"] for g in groups] == [BLOCK_X, BLOCK_X, BLOCK_U]

    model = s.model  # the pinocchio model BSQP already built from the URDF
    m = JOINT_LIMIT_MARGIN
    np.testing.assert_allclose(groups[0]["lo"], model.lowerPositionLimit - m, rtol=1e-6)
    np.testing.assert_allclose(groups[0]["hi"], model.upperPositionLimit + m, rtol=1e-6)
    np.testing.assert_allclose(groups[1]["lo"], -model.velocityLimit - m, rtol=1e-6)
    np.testing.assert_allclose(groups[1]["hi"], model.velocityLimit + m, rtol=1e-6)
    np.testing.assert_allclose(groups[2]["lo"], -model.effortLimit - m, rtol=1e-6)
    np.testing.assert_allclose(groups[2]["hi"], model.effortLimit + m, rtol=1e-6)

    # knot masks: state boxes span all knots; the terminal knot has no control
    assert groups[0]["knot_lo"] == 0 and groups[0]["knot_hi"] == N
    assert groups[2]["knot_hi"] == N - 1


def test_telemetry_matches_numpy_oracle(make_solver, smallest_module):
    plant, N = smallest_module
    B = 8
    X, goals = _inputs(plant, N, B)
    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_telemetry()
    res = s.solve(X, goals)
    groups = s.get_row_groups()

    for b in range(B):
        oracle = _oracle_violations(res.xu[b], groups, res.nx, res.nu)
        for g, (vmax, vsum) in enumerate(oracle):
            np.testing.assert_allclose(res.stats.row_max_violation[g, b], vmax,
                                       rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(res.stats.row_sum_violation[g, b], vsum,
                                       rtol=1e-5, atol=1e-6)


def test_relaxed_barrier_changes_solution_and_stays_finite(make_solver, smallest_module):
    """MECH_BARRIER_RELAXED actually enforces (solution moves) and the solver
    stays finite; telemetry keeps reporting."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    ref = make_solver(plant, N, batch_size=B).solve(X, goals)

    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_barrier(mu=1e-1, delta=0.1)
    got = s.solve(X, goals)

    assert np.isfinite(got.xu).all()
    assert got.stats.row_max_violation is not None
    assert np.abs(got.xu - ref.xu).max() > 1e-6  # the barrier is in the objective


def test_relaxed_barrier_infeasible_start_no_nan(make_solver, smallest_module):
    """The quadratic extension makes an out-of-bounds start well-defined: no
    NaN/inf anywhere (the clamped log barrier's failure mode)."""
    plant, N = smallest_module
    B = 2
    X, goals = _inputs(plant, N, B)

    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_barrier(mu=1e-2, delta=0.1)
    bounds = s.get_row_groups()[0]
    X_bad = X.copy()
    X_bad[:, 0] = np.float32(bounds["hi"][0] + 0.5)  # q0 well past its upper limit

    res = s.solve(X_bad, goals)
    assert np.isfinite(res.xu).all()
    assert np.isfinite(res.stats.final_merit).all()


def test_certificate_matches_telemetry(make_solver, smallest_module):
    """python/gato/certificate.py (the kkt_certificate port) must agree with
    the on-device telemetry on primal violation, per group and per batch."""
    from gato.certificate import kkt_residuals, certify

    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)
    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_telemetry()
    res = s.solve(X, goals)
    groups = s.get_row_groups()

    for b in range(B):
        r = kkt_residuals(groups, res.xu[b].astype(np.float64), res.nx, res.nu)
        assert r["dual"] is None  # no row duals until CL-1
        np.testing.assert_allclose(r["primal"], res.stats.row_max_violation[:, b].max(),
                                   rtol=1e-6, atol=1e-7)
        for g in range(len(groups)):
            np.testing.assert_allclose(r["per_group"][g]["primal"],
                                       res.stats.row_max_violation[g, b],
                                       rtol=1e-6, atol=1e-7)

    # gate wrapper runs end-to-end (pass/fail depends on the problem; just
    # exercise it and check the tuple shape)
    r, ok = certify(res, groups, b=0)
    assert isinstance(ok, (bool, np.bool_)) and "primal" in r


def test_admm_enforces_boxes(make_solver, smallest_module):
    """MECH_ADMM (fixed-budget interval projection on the reused bdsv factor)
    drives limit violations to ~0 on a problem whose unconstrained solution
    violates them badly; residuals are reported; solver stays finite."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    base = make_solver(plant, N, batch_size=B)
    base.enable_limit_telemetry()
    rb = base.solve(X, goals)

    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_admm(rho=10.0, iters=10)
    r = s.solve(X, goals)

    assert np.isfinite(r.xu).all()
    assert r.stats.admm_r_prim is not None and r.stats.admm_r_dual is not None
    if rb.stats.row_max_violation.max() > 1.0:  # the problem actually stresses the boxes
        assert r.stats.row_max_violation.max() < 1e-3 * rb.stats.row_max_violation.max()
    assert r.stats.row_max_violation.max() < 1e-4  # approximately-hard claim, measured


def test_admm_deterministic_and_warmstart_stable(make_solver, smallest_module):
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    def run():
        s = make_solver(plant, N, batch_size=B)
        s.enable_limit_admm(rho=10.0, iters=10)
        first = s.solve(X, goals)
        for _ in range(3):  # dual warm start across solves
            last = s.solve(X, goals)
        return first, last

    (a1, a2), (b1, b2) = run(), run()
    np.testing.assert_array_equal(a1.xu, b1.xu)  # bit-deterministic
    np.testing.assert_array_equal(a2.xu, b2.xu)
    assert a2.stats.row_max_violation.max() < 1e-4  # feasibility persists warm-started


def test_telemetry_deterministic(make_solver, smallest_module):
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    def run():
        s = make_solver(plant, N, batch_size=B)
        s.enable_limit_telemetry()
        return s.solve(X, goals)

    a, b = run(), run()
    np.testing.assert_array_equal(a.stats.row_max_violation, b.stats.row_max_violation)
    np.testing.assert_array_equal(a.stats.row_sum_violation, b.stats.row_sum_violation)
