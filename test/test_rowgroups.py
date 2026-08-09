"""Constraint row-group layer (CL-0/CL-1): telemetry, RB, ADMM, and AL gates.

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

    # knot masks: state boxes start at knot 1 — x_0 is DATA (pinned to the
    # measurement), so knot-0 state rows are unsatisfiable whenever the
    # measured state violates a limit (R1: AL winds up / freezes on them and
    # closed-loop MPC destabilizes). The terminal knot has no control.
    assert groups[0]["knot_lo"] == 1 and groups[0]["knot_hi"] == N
    assert groups[1]["knot_lo"] == 1
    assert groups[2]["knot_lo"] == 0 and groups[2]["knot_hi"] == N - 1


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


VCAP = 0.30  # tight velocity box (rad/s) that binds hard on the reach problem


def _tighten_qd(s):
    n = s.get_row_groups()[1]["n_rows"]  # BOX_QD
    s.set_row_group_bounds(1, -VCAP * np.ones(n), VCAP * np.ones(n))


def _qd_max(res, N):
    step = res.nx + res.nu
    return max(np.abs(res.xu[:, k * step + res.nx // 2:k * step + res.nx]).max()
               for k in range(N))


def test_al_enforces_boxes(make_solver, smallest_module):
    """MECH_AL (PHR augmented Lagrangian, dual update per solve) drives a
    hard-binding velocity box to feasibility across warm-started solves; the
    unconstrained solution violates it by orders of magnitude."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    base = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    base.enable_limit_telemetry()
    _tighten_qd(base)
    for _ in range(5):
        rb = base.solve(X, goals)

    s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    s.enable_limit_al(rho=100.0)
    _tighten_qd(s)
    for _ in range(10):
        r = s.solve(X, goals)

    assert np.isfinite(r.xu).all()
    assert rb.stats.row_max_violation.max() > 10 * VCAP  # the box really binds
    assert r.stats.row_max_violation.max() < 1e-4
    assert _qd_max(r, N) <= VCAP + 1e-4


def test_al_holds_binding_optimum(make_solver, smallest_module):
    """Seeded at ADMM's bound-riding solution, AL holds it: feasibility and
    merit are preserved (the mechanisms agree on the constrained optimum when
    started in the same basin — cross-mechanism consistency, GPU end-to-end;
    exact QP-level dual agreement is gated by test/oracles/mechanisms.py)."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    sad = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    sad.enable_limit_admm(rho=10.0, iters=10)
    _tighten_qd(sad)
    for _ in range(20):
        rad = sad.solve(X, goals)
    assert rad.stats.row_max_violation.max() < 1e-3  # fixed-budget residual

    sal = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    sal.enable_limit_al(rho=100.0)
    _tighten_qd(sal)
    ral = sal.solve(X, goals, XU_B=rad.xu.copy())
    for _ in range(10):
        ral = sal.solve(X, goals)

    assert ral.stats.row_max_violation.max() < 1e-3
    m_ad = np.asarray(rad.stats.final_merit, dtype=np.float64)
    m_al = np.asarray(ral.stats.final_merit, dtype=np.float64)
    np.testing.assert_allclose(m_al, m_ad, rtol=0.02)  # holds the optimum


def test_al_deterministic_and_duals(make_solver, smallest_module):
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    def run():
        s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
        s.enable_limit_al(rho=100.0)
        _tighten_qd(s)
        for _ in range(4):
            r = s.solve(X, goals)
        return r, s.get_row_duals()

    (a, da), (b, db) = run(), run()
    np.testing.assert_array_equal(a.xu, b.xu)  # bit-deterministic
    np.testing.assert_array_equal(da["lam_hi"], db["lam_hi"])
    # hinge multipliers are nonnegative by construction; state shape is the
    # dense row-state layout
    assert da["lam_hi"].min() >= 0 and da["lam_lo"].min() >= 0
    assert da["lam_hi"].shape[0] == B and da["lam_hi"].shape[2] == N
    assert np.isfinite(da["lam_hi"]).all() and np.isfinite(da["lam_lo"]).all()


def test_al_no_drift_at_insufficient_rho(make_solver, smallest_module):
    """True-violation acceptance gate: when a modest rho stalls the primal
    (plateau), lambda freezes instead of accumulating rho*viol every solve —
    the plateau must not drift upward over many warm solves (pre-gate this
    drifted 0.069 -> 0.57 on iiwa14)."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    s.enable_limit_al(rho=10.0)  # deliberately modest: may plateau
    _tighten_qd(s)
    for _ in range(2):
        r = s.solve(X, goals)
    plateau = r.stats.row_max_violation.max()
    worst = 0.0
    for _ in range(20):
        r = s.solve(X, goals)
        worst = max(worst, float(r.stats.row_max_violation.max()))
    assert worst <= max(1.05 * plateau, 1e-4)


def test_soft_toggle_elastic_and_hard_parity(make_solver, smallest_module):
    """set_row_group_soft (TurboMPC delta_xi). ADMM (cold path rides the
    tight velocity box): smoothed projection admits violation monotonically
    in 1/sigma (measured hard 1e-5 / sigma=1 0.016 / sigma=0.1 0.122) with
    the quadratic-slack dual fixed point y = sigma*violation. AL (seeded at
    the bound-riding optimum, true multiplier lam* ~ 0.03): sigma < lam*
    caps the multiplier at EXACTLY sigma and lets the bound ride out,
    trading violation for merit (measured viol 3e-4 -> 1.25, merit
    24.45 -> 24.35 at sigma = 0.01). sigma = 0 is the exact hard path
    (bit-identical trajectories)."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    # --- ADMM: smoothed projection on the cold bound-riding path ---
    def run_admm(sigma):
        s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
        s.enable_limit_admm(rho=10.0, iters=10)
        _tighten_qd(s)
        if sigma is not None:
            s.set_row_group_soft(1, sigma)
        for _ in range(8):
            r = s.solve(X, goals)
        return s, r

    _, a_hard = run_admm(None)
    _, a_zero = run_admm(0.0)
    s_soft, a_soft = run_admm(0.1)
    _, a_mid = run_admm(1.0)
    np.testing.assert_array_equal(a_hard.xu, a_zero.xu)  # sigma=0 == hard, bitwise

    av_hard = a_hard.stats.row_max_violation[1].max()
    av_mid = a_mid.stats.row_max_violation[1].max()
    av_soft = a_soft.stats.row_max_violation[1].max()
    assert np.isfinite(a_soft.xu).all()
    assert av_soft > av_mid > av_hard + 1e-4   # monotone in 1/sigma
    y_max = np.abs(s_soft.get_admm_state()["y"][:, 1]).max()
    assert y_max <= 0.1 * av_soft * 2.0        # quadratic slack: y = sigma*viol

    # --- AL: L1 elastic at the seeded binding optimum ---
    s0 = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    s0.enable_limit_admm(rho=10.0, iters=10)
    _tighten_qd(s0)
    for _ in range(20):
        rad = s0.solve(X, goals)

    def run_al(sigma):
        s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
        s.enable_limit_al(rho=100.0)
        _tighten_qd(s)
        if sigma is not None:
            s.set_row_group_soft(1, sigma)
        r = s.solve(X, goals, XU_B=rad.xu.copy())
        for _ in range(10):
            r = s.solve(X, goals)
        return s, r

    _, r_hard = run_al(None)
    _, r_zero = run_al(0.0)
    s_al, r_soft = run_al(0.01)  # below the measured multiplier lam* ~ 0.03
    np.testing.assert_array_equal(r_hard.xu, r_zero.xu)

    v_hard = r_hard.stats.row_max_violation[1].max()
    v_soft = r_soft.stats.row_max_violation[1].max()
    assert np.isfinite(r_soft.xu).all()
    assert v_soft > v_hard + 0.01              # elastic lets the bound ride out
    duals = s_al.get_row_duals()
    lam_max = max(duals["lam_hi"][:, 1].max(), duals["lam_lo"][:, 1].max())
    assert lam_max <= 0.01 + 1e-6              # multiplier caps at EXACTLY sigma
    # the trade buys merit: elastic must not be worse than hard
    m_hard = np.mean(np.asarray(r_hard.stats.final_merit, dtype=np.float64))
    m_soft = np.mean(np.asarray(r_soft.stats.final_merit, dtype=np.float64))
    assert m_soft <= m_hard + 1e-3


def test_certificate_dual_axes_with_al(make_solver, smallest_module):
    """certificate.py's dual/complementarity axes activate with real AL
    multipliers (per-group (n_knots, n_rows) slices of the dense state)."""
    from gato.certificate import kkt_residuals

    plant, N = smallest_module
    B = 2
    X, goals = _inputs(plant, N, B)
    s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    s.enable_limit_al(rho=100.0)
    _tighten_qd(s)
    for _ in range(6):
        res = s.solve(X, goals)
    groups = s.get_row_groups()
    d = s.get_row_duals()

    b = 0
    duals = [(d["lam_hi"] + d["lam_lo"])[b, g, grp["knot_lo"]:grp["knot_hi"], :grp["n_rows"]]
             for g, grp in enumerate(groups)]
    r = kkt_residuals(groups, res.xu[b].astype(np.float64), res.nx, res.nu, duals=duals)
    assert r["dual"] is not None and r["complementarity"] is not None
    assert r["dual"] <= 1e-6  # hinge duals can't be negative
    assert r["complementarity"] < 1e-2  # lam*viol small at the converged point


def test_set_row_group_bounds_roundtrip(make_solver, smallest_module):
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    s.enable_limit_telemetry()
    n = s.get_row_groups()[1]["n_rows"]
    lo, hi = -0.25 * np.ones(n, np.float32), 0.25 * np.ones(n, np.float32)
    s.set_row_group_bounds(1, lo, hi)
    g = s.get_row_groups()[1]
    np.testing.assert_array_equal(g["lo"], lo)
    np.testing.assert_array_equal(g["hi"], hi)
    with pytest.raises(Exception):
        s.set_row_group_bounds(7, lo, hi)  # out of range


def _solver_frame_target(s, plant, delta=(0.02, -0.02, -0.03)):
    """A nearby EE target in the SOLVER's frame (see BSQP.ee_pos)."""
    q0 = np.asarray(START[plant], dtype=np.float64)
    return (s.ee_pos(q0, frame="solver") + np.asarray(delta)).astype(np.float32)


def test_ee_telemetry_matches_pinocchio(make_solver, smallest_module):
    """EE_POS rows (the first non-selection kind, cooperative on-device FK):
    reported violation == |pin_ee(q_N) - target| in the SOLVER frame — since
    GRiD e31f7bd the device FK uses the named-target *_EE codegen (fixed-joint
    origin INCLUDED), so the solver frame IS the URDF ee_frame."""
    pytest.importorskip("pinocchio")
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)
    s = make_solver(plant, N, batch_size=B)
    target = _solver_frame_target(s, plant)
    s.enable_limit_telemetry()
    s.enable_ee_terminal_equality(target, rho=100.0)
    res = s.solve(X, goals)
    assert len(s.get_row_groups()) == 4 and s.get_row_groups()[3]["kind"] == 3

    step = res.nx + res.nu
    for b in range(B):
        qN = res.xu[b, (N - 1) * step:(N - 1) * step + res.nx // 2].astype(np.float64)
        oracle = np.abs(s.ee_pos(qN, frame="solver") - target.astype(np.float64)).max()
        np.testing.assert_allclose(res.stats.row_max_violation[3, b], oracle,
                                   rtol=1e-5, atol=1e-6)


def test_ee_equality_al_converges_when_reachable(make_solver, smallest_module):
    """AL-bound EE terminal equality: with the tracking goal AT the target
    (reachable), warm-started solves drive the EE violation from decimeters
    to centimeters (measured 0.186 -> 0.017 on indy7). Conflict-regime
    convergence (goal far from target) is an R1 outer-loop question — the
    acceptance gate freezes duals there rather than risk runaway."""
    pytest.importorskip("pinocchio")
    plant, N = smallest_module
    B = 4
    X, _ = _inputs(plant, N, B)
    s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    target = _solver_frame_target(s, plant)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = target[0], target[1], target[2]

    s.enable_limit_al(rho=1.0)
    s.enable_ee_terminal_equality(target, rho=1.0)
    first = s.solve(X, goals)
    for _ in range(10):
        r = s.solve(X, goals)

    assert np.isfinite(r.xu).all()
    assert r.stats.row_max_violation[3].max() < 0.05
    # warm solves never worsen it; when solve 1 lands far (indy7: 0.186) they
    # must improve it substantially (measured -> 0.017)
    v1, vk = first.stats.row_max_violation[3].max(), r.stats.row_max_violation[3].max()
    assert vk <= 1.05 * v1
    if v1 > 0.06:
        assert vk < 0.5 * v1


def test_ee_equality_admm_converges_when_reachable(make_solver, smallest_module):
    """ADMM-bound EE terminal equality (linearized inner-loop projection):
    with the tracking goal AT the target (reachable), warm-started solves
    keep the EE violation at centimeters (measured indy7: 0.028 vs 0.065
    pure-tracking, stable over 30 solves); z pins to the target exactly
    (degenerate-interval clip — an indexing sanity gate); and y stays at
    WITHIN-SOLVE scale — equality rows reinit (z, y) per solve, or a parked
    primal turns the warm-started dual into an unbounded violation
    integrator (measured: |y| grew ~11/solve without the reinit)."""
    pytest.importorskip("pinocchio")
    plant, N = smallest_module
    B = 4
    X, _ = _inputs(plant, N, B)
    s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    target = _solver_frame_target(s, plant)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = target[0], target[1], target[2]

    rho, iters = 10.0, 5
    s.enable_limit_admm(rho=rho, iters=iters)
    s.enable_ee_terminal_equality(target, rho=rho)
    assert s.get_row_groups()[3]["mech"] == 2  # MECH_ADMM
    first = s.solve(X, goals)
    for _ in range(10):
        r = s.solve(X, goals)

    assert np.isfinite(r.xu).all()
    assert r.stats.row_max_violation[3].max() < 0.05
    v1, vk = first.stats.row_max_violation[3].max(), r.stats.row_max_violation[3].max()
    assert vk <= 1.05 * v1
    if v1 > 0.06:
        assert vk < 0.5 * v1
    st = s.get_admm_state()
    z3 = st["z"][:, 3, N - 1, :3]
    np.testing.assert_array_equal(z3, np.broadcast_to(target, z3.shape))
    # one solve's worth of accumulation, not 11 solves' worth (windup guard)
    y_cap = rho * max(vk, 0.05) * iters * 8 * 1.5
    assert np.abs(st["y"][:, 3, N - 1, :3]).max() < y_cap


def test_ee_admm_certificate_and_determinism(make_solver, smallest_module):
    from gato.certificate import kkt_residuals

    pytest.importorskip("pinocchio")
    plant, N = smallest_module
    B = 2
    X, goals = _inputs(plant, N, B)

    def run():
        s = make_solver(plant, N, batch_size=B, max_sqp_iters=6)
        target = _solver_frame_target(s, plant)
        s.enable_limit_admm(rho=10.0, iters=5)
        s.enable_ee_terminal_equality(target, rho=10.0)
        for _ in range(3):
            r = s.solve(X, goals)
        return s, r

    (sa, ra), (sb, rb) = run(), run()
    np.testing.assert_array_equal(ra.xu, rb.xu)  # bit-deterministic ADMM-EE

    groups = sa.get_row_groups()
    fk = lambda q: sa.ee_pos(q, frame="solver")
    r = kkt_residuals(groups, ra.xu[0].astype(np.float64), ra.nx, ra.nu, ee_fk=fk)
    np.testing.assert_allclose(r["per_group"][3]["primal"],
                               ra.stats.row_max_violation[3, 0], rtol=1e-5, atol=1e-6)


def test_ee_certificate_and_determinism(make_solver, smallest_module):
    from gato.certificate import kkt_residuals

    pytest.importorskip("pinocchio")
    plant, N = smallest_module
    B = 2
    X, goals = _inputs(plant, N, B)

    def run():
        s = make_solver(plant, N, batch_size=B, max_sqp_iters=6)
        target = _solver_frame_target(s, plant)
        s.enable_limit_al(rho=1.0)
        s.enable_ee_terminal_equality(target, rho=1.0)
        for _ in range(3):
            r = s.solve(X, goals)
        return s, r

    (sa, ra), (sb, rb) = run(), run()
    np.testing.assert_array_equal(ra.xu, rb.xu)  # bit-deterministic with EE rows

    groups = sa.get_row_groups()
    fk = lambda q: sa.ee_pos(q, frame="solver")
    r = kkt_residuals(groups, ra.xu[0].astype(np.float64), ra.nx, ra.nu, ee_fk=fk)
    np.testing.assert_allclose(r["per_group"][3]["primal"],
                               ra.stats.row_max_violation[3, 0], rtol=1e-5, atol=1e-6)


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


# ---- CL-2: LIN_U rows + cone semantics -------------------------------------

def _u_at(xu, k, nx, nu):
    step = nx + nu
    return xu[k * step + nx:k * step + nx + nu].astype(np.float32)


def _lin_u_oracle(xu, grp, nx, nu):
    """numpy {max, sum} violation of one LIN_U group from the returned
    trajectory: cone margin (one scalar per knot) or interval violation."""
    C = np.asarray(grp["C"], dtype=np.float32)
    d = np.asarray(grp["d"], dtype=np.float32)
    viols = []
    for k in range(grp["knot_lo"], grp["knot_hi"]):
        g = C @ _u_at(xu, k, nx, nu) + d
        if grp["cone"]:
            viols.append(max(0.0, np.linalg.norm(g[1:].astype(np.float64)) - g[0]))
        else:
            lo = np.asarray(grp["lo"], dtype=np.float32)
            hi = np.asarray(grp["hi"], dtype=np.float32)
            viols.extend(np.maximum(0, g - hi) + np.maximum(0, lo - g))
    v = np.asarray(viols)
    return v.max(), v.sum(dtype=np.float64)


def _norm_cap_cone(nu, cap):
    """(C, d) for the torque-norm cone ||u|| <= cap (row 0 = constant axis)."""
    C = np.zeros((nu + 1, nu), dtype=np.float32)
    C[1:, :] = np.eye(nu, dtype=np.float32)
    d = np.zeros(nu + 1, dtype=np.float32)
    d[0] = cap
    return C, d


def _baseline_u_norm_max(res):
    return max(np.linalg.norm(_u_at(res.xu[b], k, res.nx, res.nu))
               for b in range(res.batch_size) for k in range(res.N - 1))


def test_lin_u_descriptor_roundtrip_and_telemetry(make_solver, smallest_module):
    """C/d round-trip through get_row_groups; cone + interval LIN_U telemetry
    match the numpy oracle on the returned trajectory."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_telemetry()
    rng = np.random.default_rng(7)
    C4 = rng.normal(size=(4, s.nu)).astype(np.float32)
    d4 = rng.normal(size=4).astype(np.float32)
    s.add_lin_u_rows(C4, d4, cone=True, mech="telemetry")
    C3 = rng.normal(size=(3, s.nu)).astype(np.float32)
    s.add_lin_u_rows(C3, lo=np.full(3, -np.inf), hi=np.zeros(3), mech="telemetry")
    groups = s.get_row_groups()
    assert len(groups) == 5
    # cone maps install NORMALIZED (uniform 1/||C||_2 — SOC-invariant; the rho
    # defaults are for unit-norm maps): the round-trip returns the scaled map
    s4 = np.linalg.norm(C4.astype(np.float64), 2)
    np.testing.assert_allclose(groups[3]["C"], C4 / s4, rtol=1e-6)
    np.testing.assert_allclose(groups[3]["d"], d4 / s4, rtol=1e-6)
    assert groups[3]["cone"] == 1 and groups[4]["cone"] == 0
    assert not np.isfinite(groups[3]["lo"]).any()

    r = s.solve(X, goals)
    np.testing.assert_array_equal(r.xu, make_solver(plant, N, batch_size=B).solve(X, goals).xu)  # telemetry off-path
    for gi in (3, 4):
        for b in range(B):
            vmax, vsum = _lin_u_oracle(r.xu[b], groups[gi], r.nx, r.nu)
            np.testing.assert_allclose(r.stats.row_max_violation[gi, b], vmax, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(r.stats.row_sum_violation[gi, b], vsum, rtol=1e-4, atol=1e-5)


def test_u_cone_normalization_is_scale_invariant(make_solver, smallest_module):
    """Cone maps normalize to unit spectral norm regardless of input scale
    (the rho-scale-law guard: an SOC is invariant under uniform scaling, but
    the admm fold's rho * C^T C is not); normalize=False installs verbatim."""
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    s.enable_limit_telemetry()
    rng = np.random.default_rng(11)
    C = rng.normal(size=(3, s.nu)).astype(np.float32)
    d = rng.normal(size=3).astype(np.float32)
    s.add_lin_u_rows(C, d, cone=True, mech="telemetry")
    s.add_lin_u_rows(100.0 * C, 100.0 * d, cone=True, mech="telemetry")
    s.add_lin_u_rows(100.0 * C, 100.0 * d, cone=True, mech="telemetry",
                     normalize=False)
    g1, g100, graw = s.get_row_groups()[-3:]
    np.testing.assert_allclose(np.linalg.norm(np.asarray(g1["C"]), 2), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(g100["C"]), np.asarray(g1["C"]), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(g100["d"]), np.asarray(g1["d"]), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(graw["C"]), 100.0 * C, rtol=1e-6)


def test_u_cone_soc_admm_enforced(make_solver, smallest_module):
    """admm_soc: a binding torque-norm cone (cap = 50% of the unconstrained
    max) is driven to ~0 margin violation; deterministic run-to-run."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    base = make_solver(plant, N, batch_size=B)
    base.enable_limit_telemetry()
    rb = base.solve(X, goals)
    cap = 0.5 * _baseline_u_norm_max(rb)

    def run():
        s = make_solver(plant, N, batch_size=B)
        s.enable_limit_telemetry()
        C, d = _norm_cap_cone(s.nu, cap)
        gi = s.enable_u_cone(C, d, mech="admm", rho=0.01, admm_iters=10)
        r = s.solve(X, goals)
        return gi, r

    (gi, r1), (_, r2) = run(), run()
    np.testing.assert_array_equal(r1.xu, r2.xu)  # bit-deterministic
    assert np.isfinite(r1.xu).all()
    assert r1.stats.row_max_violation[gi].max() < 0.02 * cap  # margin ~0


def test_u_cone_soc_al_enforced(make_solver, smallest_module):
    """conic-AL: same binding cone, dual vector projected onto K per solve;
    warm-started solves converge the margin; duals live in lam_hi and are
    cone-feasible."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    base = make_solver(plant, N, batch_size=B)
    base.enable_limit_telemetry()
    rb = base.solve(X, goals)
    cap = 0.5 * _baseline_u_norm_max(rb)

    s = make_solver(plant, N, batch_size=B, max_sqp_iters=8)
    s.enable_limit_telemetry()
    C, d = _norm_cap_cone(s.nu, cap)
    gi = s.enable_u_cone(C, d, mech="al", rho=1.0)
    for _ in range(8):
        r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    assert r.stats.row_max_violation[gi].max() < 0.05 * cap

    lam = s.get_row_duals()["lam_hi"][:, gi, :N - 1, :s.nu + 1]
    m = s.nu + 1
    for b in range(B):
        for k in range(N - 1):
            v = lam[b, k, :m].astype(np.float64)
            assert np.linalg.norm(v[1:]) <= v[0] + 1e-4  # lam in K (self-dual)


def test_u_cone_pyramid_enforced(make_solver, smallest_module):
    """Pyramid facets (inscribed) ride the interval machinery: facet violation
    -> ~0 under ADMM, and inscribed feasibility implies a small SOC margin."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    base = make_solver(plant, N, batch_size=B)
    base.enable_limit_telemetry()
    rb = base.solve(X, goals)
    cap = 0.5 * _baseline_u_norm_max(rb)

    # 3-row cone on the first two controls: ||(u0, u1)|| <= cap
    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_telemetry()
    C = np.zeros((3, s.nu), dtype=np.float32)
    C[1, 0] = 1.0
    C[2, 1] = 1.0
    d = np.array([cap, 0.0, 0.0], dtype=np.float32)
    gi = s.enable_u_cone(C, d, mech="admm", rho=0.01, form="pyramid", facets=8,
                         admm_iters=10)
    r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    assert r.stats.row_max_violation[gi].max() < 0.02 * cap  # facet rows ~feasible

    # facet-feasible (inscribed) => SOC margin feasible up to the facet slack
    worst = 0.0
    for b in range(B):
        for k in range(N - 1):
            u = _u_at(r.xu[b], k, r.nx, r.nu)
            worst = max(worst, float(np.linalg.norm(C[1:] @ u) - cap))
    assert worst < 0.03 * cap


def test_u_cone_barrier_reduces_margin(make_solver, smallest_module):
    """Relaxed-barrier-cone (margin barrier): infeasible-start safe, margin
    violation reduced vs the unconstrained baseline."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)

    base = make_solver(plant, N, batch_size=B)
    base.enable_limit_telemetry()
    rb = base.solve(X, goals)
    cap = 0.5 * _baseline_u_norm_max(rb)
    C, d = _norm_cap_cone(rb.nu, cap)

    def cone_viol(res):
        return max(_lin_u_oracle(res.xu[b],
                                 dict(C=C, d=d, cone=1, knot_lo=0, knot_hi=N - 1),
                                 res.nx, res.nu)[0] for b in range(B))

    s = make_solver(plant, N, batch_size=B)
    s.enable_limit_telemetry()
    gi = s.enable_u_cone(C, d, mech="barrier", rho=3e-3, delta=0.05)
    r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    assert cone_viol(r) < 0.5 * cone_viol(rb)  # soft mode: reduced, not exact
