"""Per-knot external-wrench band (CL-3 prep, 2026-08-01).

The GPU wrench buffer is per-(solve, knot): wrench k applies to dynamics
interval [k, k+1] in both the KKT linearization and the merit integrator
error; sim_forward uses knot 0's wrench. set_f_ext_B accepts the historic
per-solve shapes (broadcast over knots) and the new (B, N, ...) per-knot
shapes — a uniform per-knot upload must be bit-identical to the broadcast.
"""
import numpy as np
import pytest

from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}


def _inputs(plant, N, B):
    q0 = np.asarray(START[plant], dtype=np.float32)
    nq = q0.size
    x0 = np.concatenate([q0, np.zeros(nq, dtype=np.float32)])
    rng = np.random.default_rng(1234)
    X = np.tile(x0, (B, 1)) + rng.normal(0, 0.01, (B, 2 * nq)).astype(np.float32)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.35, 0.25, 0.5
    return X, goals


def _wrench(B, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((B, 6)) * 5.0).astype(np.float32)


def test_per_knot_uniform_matches_broadcast(make_solver, smallest_module):
    """(B, N, 6) filled with one wrench per solve == the (B, 6) broadcast, bitwise."""
    plant, N = smallest_module
    B = 4
    X, goals = _inputs(plant, N, B)
    f = _wrench(B)

    s_bcast = make_solver(plant, N, batch_size=B)
    s_bcast.set_f_ext_B(f)
    ref = s_bcast.solve(X, goals)

    s_knot = make_solver(plant, N, batch_size=B)
    s_knot.set_f_ext_B(np.repeat(f[:, None, :], N, axis=1))
    got = s_knot.solve(X, goals)

    np.testing.assert_array_equal(ref.xu, got.xu)


def test_per_knot_varying_wrench_changes_solution(make_solver, smallest_module):
    """A horizon-varying wrench must produce a different trajectory than holding
    knot 0's wrench constant (the band is actually consumed per knot)."""
    plant, N = smallest_module
    B = 2
    X, goals = _inputs(plant, N, B)
    f0 = _wrench(B)

    s_const = make_solver(plant, N, batch_size=B)
    s_const.set_f_ext_B(f0)
    ref = s_const.solve(X, goals)

    f_knots = np.repeat(f0[:, None, :], N, axis=1)
    f_knots[:, N // 2:, :] *= -1.0  # flip the wrench mid-horizon
    s_vary = make_solver(plant, N, batch_size=B)
    s_vary.set_f_ext_B(f_knots)
    got = s_vary.solve(X, goals)

    assert np.isfinite(got.xu).all()
    assert not np.array_equal(ref.xu, got.xu)


def test_per_knot_deterministic(make_solver, smallest_module):
    plant, N = smallest_module
    B = 2
    X, goals = _inputs(plant, N, B)
    f_knots = np.repeat(_wrench(B)[:, None, :], N, axis=1)
    f_knots[:, ::2, :] *= 0.5

    xs = []
    for _ in range(2):
        s = make_solver(plant, N, batch_size=B)
        s.set_f_ext_B(f_knots)
        xs.append(s.solve(X, goals).xu)
    np.testing.assert_array_equal(xs[0], xs[1])


# ---------------------------------------------------------------------------
# Contact-wrench chain (CL-3 prep): debug_contact_dynamics vs the project's own
# finite differences. The oracle kernel evaluates f_ext(q, f_c), qdd, dqdd/dq
# at FIXED f_ext, the composed dqdd/df_c (the future B-block columns), and the
# dfext/dq chain correction (the term a solver drops if it treats the applied
# wrench as q-independent).
# ---------------------------------------------------------------------------

def _contact_sample(s, seed=3):
    rng = np.random.default_rng(seed)
    nq = s.nq
    q = rng.uniform(-0.8, 0.8, nq).astype(np.float32)
    qd = rng.uniform(-0.5, 0.5, nq).astype(np.float32)
    u = rng.uniform(-5.0, 5.0, nq).astype(np.float32)
    fc = (rng.standard_normal(6) * 10.0).astype(np.float32)
    return q, qd, u, fc


def test_contact_dqdd_dfc_vs_fd(make_solver, smallest_module):
    """Analytic dqdd/df_c == central FD of qdd over f_c (device evaluations)."""
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    q, qd, u, fc = _contact_sample(s)
    d = s.solver.debug_contact_dynamics(q, qd, u, fc)
    ana = np.asarray(d["dqdd_dfc"], dtype=np.float64)

    h = np.float32(1e-2)
    fd = np.zeros_like(ana)
    for j in range(ana.shape[1]):
        fp, fm = fc.copy(), fc.copy()
        fp[j] += h
        fm[j] -= h
        qp = np.asarray(s.solver.debug_contact_dynamics(q, qd, u, fp)["qdd"], dtype=np.float64)
        qm = np.asarray(s.solver.debug_contact_dynamics(q, qd, u, fm)["qdd"], dtype=np.float64)
        fd[:, j] = (qp - qm) / (2.0 * float(h))

    scale = max(1.0, np.abs(ana).max())
    np.testing.assert_allclose(fd, ana, rtol=2e-2, atol=2e-3 * scale)


def test_contact_dqdd_dq_total_vs_fd(make_solver, smallest_module):
    """FD of qdd over q at FIXED f_c must match dqdd_dq(fixed f_ext) + the
    dfext/dq chain correction — i.e. the correction term is real and correct."""
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    q, qd, u, fc = _contact_sample(s, seed=11)
    d = s.solver.debug_contact_dynamics(q, qd, u, fc)
    ana_total = (np.asarray(d["dqdd_dq"], dtype=np.float64)
                 + np.asarray(d["dqdd_dq_corr"], dtype=np.float64))

    h = np.float32(1e-3)
    fd = np.zeros_like(ana_total)
    for j in range(ana_total.shape[1]):
        qp_, qm_ = q.copy(), q.copy()
        qp_[j] += h
        qm_[j] -= h
        qp = np.asarray(s.solver.debug_contact_dynamics(qp_, qd, u, fc)["qdd"], dtype=np.float64)
        qm = np.asarray(s.solver.debug_contact_dynamics(qm_, qd, u, fc)["qdd"], dtype=np.float64)
        fd[:, j] = (qp - qm) / (2.0 * float(h))

    # the correction must be non-trivial at a nonzero wrench (else this gate
    # would pass vacuously with a zeroed corr output)
    assert np.abs(np.asarray(d["dqdd_dq_corr"])).max() > 1e-6
    scale = max(1.0, np.abs(ana_total).max())
    np.testing.assert_allclose(fd, ana_total, rtol=5e-2, atol=5e-3 * scale)


def test_contact_zero_wrench_structure(make_solver, smallest_module):
    """f_c = 0: mapped f_ext is zero, and the dfext/dq correction vanishes
    (the map is linear in f_c); dqdd_dfc must still be finite and nonzero."""
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    q, qd, u, _ = _contact_sample(s, seed=5)
    d = s.solver.debug_contact_dynamics(q, qd, u, np.zeros(6, dtype=np.float32))
    assert np.abs(np.asarray(d["fext"])).max() == 0.0
    assert np.abs(np.asarray(d["dqdd_dq_corr"])).max() == 0.0
    assert np.isfinite(np.asarray(d["dqdd_dfc"])).all()
    assert np.abs(np.asarray(d["dqdd_dfc"])).max() > 0.0


def test_wrong_per_knot_shape_raises(make_solver, smallest_module):
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=2)
    with pytest.raises(ValueError):
        s.set_f_ext_B(np.zeros((2, N + 1, 6), dtype=np.float32))
    with pytest.raises(ValueError):
        s.set_f_ext_B(np.zeros((2, N, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# CL-3a contact-force builds (GATO_CONTACT_FORCES=1): the contact wrench f_c is
# the tail of every control (CONTROL_SIZE = ACTUATED_SIZE + FC_SIZE). These
# gates SKIP on default builds (n_fc == 0) — run the suite against the
# build_fc modules (.so-swap) to engage them.
# ---------------------------------------------------------------------------

def _fc_solver(make_solver, smallest_module, B=1, **kw):
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=B, **kw)
    if s.n_fc == 0:
        pytest.skip("default build (no fc slots) — run against build_fc modules")
    return plant, N, s


def test_fc_adapter_matches_oracle(make_solver, smallest_module):
    """The ADAPTER's B-block fc columns (the solver's in-plant composition) ==
    the oracle composition (independent *_device path). Same math, different
    code path — catches wiring/offset/scratch bugs in the hot path."""
    plant, N, s = _fc_solver(make_solver, smallest_module)
    q, qd, u, fc = _contact_sample(s, seed=13)
    d = s.solver.debug_contact_dynamics(q, qd, u, fc)
    assert "dqdd_dfc_adapter" in d
    oracle = np.asarray(d["dqdd_dfc"], dtype=np.float64)
    adapter = np.asarray(d["dqdd_dfc_adapter"], dtype=np.float64)
    scale = max(1.0, np.abs(oracle).max())
    np.testing.assert_allclose(adapter, oracle, rtol=1e-4, atol=1e-5 * scale)


def test_fc_adapter_dq_carries_chain_term(make_solver, smallest_module):
    """W2 exactness: the ADAPTER's A-block dq columns must CARRY the dfext/dq
    chain term — i.e. equal the oracle's fixed-f_ext gradient plus the correction
    (independent *_device composition). W1 dropped it; at a 10 N wrench the term
    is the same size as the whole gradient, so this gate is not cosmetic."""
    plant, N, s = _fc_solver(make_solver, smallest_module)
    q, qd, u, fc = _contact_sample(s, seed=17)
    d = s.solver.debug_contact_dynamics(q, qd, u, fc)
    assert "dqdd_dq_adapter" in d
    fixed = np.asarray(d["dqdd_dq"], dtype=np.float64)
    corr = np.asarray(d["dqdd_dq_corr"], dtype=np.float64)
    adapter = np.asarray(d["dqdd_dq_adapter"], dtype=np.float64)
    # non-triviality: without this the gate would pass on a zeroed correction
    assert np.abs(corr).max() > 1e-6
    scale = max(1.0, np.abs(fixed + corr).max())
    np.testing.assert_allclose(adapter, fixed + corr, rtol=1e-4, atol=1e-5 * scale)


def test_fc_adapter_dq_zero_wrench_bitwise(make_solver, smallest_module):
    """f_c = 0: the chain term is LINEAR in f_c so it is exactly zero, and the
    adapter's dq block is BITWISE the fixed-f_ext gradient — W2 cannot perturb a
    zero-wrench trajectory."""
    plant, N, s = _fc_solver(make_solver, smallest_module)
    q, qd, u, _ = _contact_sample(s, seed=19)
    d = s.solver.debug_contact_dynamics(q, qd, u, np.zeros(s.n_fc, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(d["dqdd_dq_adapter"]),
                                  np.asarray(d["dqdd_dq"]))


def test_fc_solve_finite_deterministic(make_solver, smallest_module):
    """fc-build solve with regularized fc slots: finite, run-twice bitwise,
    and fc_traj has the fc-build shape."""
    plant, N = smallest_module
    xus = []
    for _ in range(2):
        s = make_solver(plant, N, batch_size=2)
        if s.n_fc == 0:
            pytest.skip("default build (no fc slots) — run against build_fc modules")
        s.set_fc_cost(1e-3)
        X, goals = _inputs(plant, N, B=2)
        r = s.solve(X, goals)
        assert np.isfinite(r.xu).all()
        assert r.fc_traj(0).shape == (N - 1, s.n_fc)
        xus.append(np.asarray(r.xu).copy())
    np.testing.assert_array_equal(xus[0], xus[1])


def test_lin_u_rows_strided_input_roundtrip(make_solver, smallest_module):
    """REGRESSION (2026-08-02): a 0-stride np.broadcast_to view passed as lo/hi
    reached the binding's raw .ptr read — 1 valid element + heap garbage became
    per-process-random bounds (the fc-pin nondeterminism). The device-side group
    descriptor must round-trip exactly for strided inputs. Build-agnostic."""
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    m = s.nu
    C = np.eye(m, dtype=np.float32)
    s.add_lin_u_rows(C, lo=np.broadcast_to(np.float32(-3.5), (m,)),
                     hi=np.broadcast_to(np.float32(7.25), (m,)), mech="al")
    g = s.solver.get_row_groups()[-1]
    np.testing.assert_array_equal(np.asarray(g["lo"]), np.full(m, -3.5, np.float32))
    np.testing.assert_array_equal(np.asarray(g["hi"]), np.full(m, 7.25, np.float32))
    np.testing.assert_array_equal(np.asarray(g["C"]), C)


def test_fc_box_pin_zeroes_fc(make_solver, smallest_module):
    """LIN_U box rows [0,0] on all fc slots (AL mech) pin the wrench to ~0:
    the fc build with pinned fc must behave like a plain solver."""
    plant, N, s = _fc_solver(make_solver, smallest_module)
    s.set_fc_cost(1e-4)
    s.enable_limit_al()
    s.add_fc_box(0.0, 0.0, mech="al")
    g = s.solver.get_row_groups()[-1]
    np.testing.assert_array_equal(np.asarray(g["lo"]), np.zeros(s.n_fc, np.float32))
    np.testing.assert_array_equal(np.asarray(g["hi"]), np.zeros(s.n_fc, np.float32))
    X, goals = _inputs(plant, N, B=1)
    r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    fc = r.fc_traj(0)
    assert np.abs(fc).max() < 5e-2, f"pinned fc leaked: max|fc|={np.abs(fc).max()}"


def test_fc_ref_zero_is_bitwise_noop(make_solver, smallest_module):
    """set_fc_ref(zeros) and set_fc_ref(None) are BITWISE the historic pure
    regularization (the contact-task wave must not perturb existing fc pools)."""
    plant, N, _ = _fc_solver(make_solver, smallest_module)
    X, goals = _inputs(plant, N, B=1)
    xus = []
    for ref in ("unset", "zeros", "reset"):
        s = make_solver(plant, N, batch_size=1)
        if ref == "zeros":
            s.set_fc_ref(np.zeros(s.n_fc))
        elif ref == "reset":
            s.set_fc_ref(np.full(s.n_fc, 7.0))
            s.set_fc_ref(None)
        xus.append(np.asarray(s.solve(X, goals).xu).copy())
    np.testing.assert_array_equal(xus[0], xus[1])
    np.testing.assert_array_equal(xus[0], xus[2])


def test_fc_ref_pulls_fc_toward_reference(make_solver, smallest_module):
    """With a dominant fc_cost, the solved wrench tracks fc_ref (the force-
    SETPOINT mechanism of the contact task): both in the trajectory and through
    the line search — this is also the gate on the merit-tail fix (the line-
    search merit used to silently drop the fc/posture cost terms)."""
    plant, N, s = _fc_solver(make_solver, smallest_module)
    ref = np.zeros(s.n_fc, dtype=np.float32)
    ref[5] = 5.0  # [n; f] layout: slot 5 = world-z force on the first contact frame
    s.set_fc_cost(1e2)
    s.set_fc_ref(ref)
    X, goals = _inputs(plant, N, B=1)
    r = s.solve(X, goals)
    assert np.isfinite(r.xu).all()
    fc = r.fc_traj(0)  # (N-1, n_fc)
    err = np.linalg.norm(fc - ref[None, :], axis=1)
    assert err.mean() < 0.3 * np.linalg.norm(ref), \
        f"fc does not track fc_ref: mean|fc-ref|={err.mean():.3f} vs |ref|={np.linalg.norm(ref):.3f}"


def test_fc_ref_validation(make_solver, smallest_module):
    """Wrong-size fc_ref raises; on default builds set_fc_ref raises loudly."""
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    if s.n_fc == 0:
        with pytest.raises(RuntimeError, match="fc"):
            s.set_fc_ref(np.zeros(6))
    else:
        with pytest.raises(ValueError, match="fc_ref"):
            s.set_fc_ref(np.zeros(s.n_fc + 1))


# ---------------------------------------------------------------------------
# MPC_GATO fc plumbing (W3). These run on DEFAULT builds — the contract under
# test is that the driver separates the xu control STRIDE from the applied
# ACTUATED slice, and refuses fc_config loudly when the module has no fc slots.
# ---------------------------------------------------------------------------

def _mpc_gato(urdfs, plant, N, batch_size=1, **kw):
    pin = pytest.importorskip("pinocchio")
    from gato.mpc_gato import MPC_GATO
    urdf = str(urdfs[plant])
    model, _, _ = pin.buildModelsFromUrdf(urdf, str(urdfs[plant].parent) + "/")
    return MPC_GATO(model, model_path=urdf, N=N, dt=0.01, batch_size=batch_size,
                    plant_type=plant, **kw)


def test_mpc_gato_control_widths(urdfs, smallest_module):
    """nu is the xu STRIDE (== solver.nu) and nu_act is the applied-torque
    slice (== solver.n_actuated). They coincide on default builds and diverge
    on GATO_CONTACT_FORCES builds, where fc must never be played into the sim."""
    plant, N = smallest_module
    mpc = _mpc_gato(urdfs, plant, N)
    assert mpc.nu == mpc.solver.nu
    assert mpc.nu_act == mpc.solver.n_actuated
    assert mpc.nu == mpc.nu_act + mpc.solver.n_fc


def test_mpc_gato_fc_config_needs_fc_build(urdfs, smallest_module):
    """fc_config on a module without fc slots must FAIL LOUD — silently
    ignoring it would run the no-estimator baseline while the caller believes
    the solver is explaining the wrench (the W3 arm-2 confound)."""
    plant, N = smallest_module
    mpc = _mpc_gato(urdfs, plant, N)
    if mpc.solver.n_fc:
        pytest.skip("fc build — the raise is the default-build contract")
    with pytest.raises(RuntimeError, match="GATO_CONTACT_FORCES"):
        _mpc_gato(urdfs, plant, N, fc_config={"cost": 1e-2, "pin_torque_rows": True})


# ---------------------------------------------------------------------------
# Wrench IDENTIFICATION (W3 arm 2). The estimator is pure host math over the
# solver's own model, so these are exact-oracle gates, not statistical ones.
# ---------------------------------------------------------------------------

def test_wrench_identifier_recovers_known_wrench(urdfs, smallest_module):
    """A wrench applied through pin.aba must be recovered EXACTLY from the
    resulting motion. This pins the whole convention stack at once — the rnea
    residual sign, the LOCAL_WORLD_ALIGNED frame, and the [force; torque]
    ordering shared with world_wrench_to_joint_local. Any one of them being
    wrong silently turns the observer into a divergence generator."""
    pin = pytest.importorskip("pinocchio")
    from gato.estimators import OneStepWrenchIdentifier
    from gato.common import world_wrench_to_joint_local

    plant, _ = smallest_module
    urdf = str(urdfs[plant])
    model, _, _ = pin.buildModelsFromUrdf(urdf, str(urdfs[plant].parent) + "/")
    model.gravity.linear = np.array([0.0, 0.0, -9.81])
    data = model.createData()
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.8, 0.8, model.nq)
    dq = rng.uniform(-0.4, 0.4, model.nv)
    tau = rng.uniform(-8.0, 8.0, model.nv)
    truth = np.array([3.0, -5.0, -98.1, 0.0, 0.0, 0.0])   # world [force; torque]

    ident = OneStepWrenchIdentifier(model, ee_frame="EE", alpha=1.0, damping=0.0)
    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force.Zero())
    jid, Fj = world_wrench_to_joint_local(model, data, q, truth,
                                          model.getFrameId("EE"))
    fext[jid] = Fj
    ddq = pin.aba(model, data, q, dq, tau, fext)

    # tiny dt so (dq_next - dq)/dt is the instantaneous ddq: isolates the FIT
    dt = 1e-6
    w = ident.identify(q, dq, dq + dt * ddq, tau, dt)
    np.testing.assert_allclose(w, truth, atol=1e-4)
    assert ident.residual_norm < 1e-8, "an exact EE wrench must be fully explained"


def test_wrench_identifier_midpoint_beats_startpoint(urdfs, smallest_module):
    """Evaluating the dynamics at the interval MIDPOINT must beat evaluating at
    its start when ddq is a finite difference over a realistic interval. This
    is not cosmetic: at the control period the start-evaluated fit was ~9%
    off, and feeding that back destabilized the closed loop."""
    pin = pytest.importorskip("pinocchio")
    from gato.estimators import OneStepWrenchIdentifier
    from gato.common import world_wrench_to_joint_local, rk4

    plant, _ = smallest_module
    urdf = str(urdfs[plant])
    model, _, _ = pin.buildModelsFromUrdf(urdf, str(urdfs[plant].parent) + "/")
    model.gravity.linear = np.array([0.0, 0.0, -9.81])
    data = model.createData()
    rng = np.random.default_rng(3)
    q = rng.uniform(-0.8, 0.8, model.nq)
    dq = rng.uniform(-0.4, 0.4, model.nv)
    tau = rng.uniform(-40.0, 40.0, model.nv)
    truth = np.array([0.0, 0.0, -98.1, 0.0, 0.0, 0.0])
    fid = model.getFrameId("EE")

    qq, dqq = q.copy(), dq.copy()
    dt = 0.002
    for _ in range(2):
        fext = pin.StdVec_Force()
        for _ in range(model.njoints):
            fext.append(pin.Force.Zero())
        jid, Fj = world_wrench_to_joint_local(model, data, qq, truth, fid)
        fext[jid] = Fj
        qq, dqq = rk4(model, data, qq, dqq, tau, dt / 2)

    ident = OneStepWrenchIdentifier(model, ee_frame="EE", alpha=1.0, damping=0.0)
    e_start = np.linalg.norm(ident.identify(q, dq, dqq, tau, dt)[:3] - truth[:3])
    e_mid = np.linalg.norm(
        ident.identify(q, dq, dqq, tau, dt, q_next=qq)[:3] - truth[:3])
    assert e_mid < e_start, f"midpoint {e_mid:.3f} should beat start {e_start:.3f}"


def test_wrench_identifier_vertical_force_unobservable_at_zero_config(urdfs, smallest_module):
    """★ At the all-zeros configuration the arm is extended straight up, so a
    VERTICAL EE force produces exactly zero joint torque -- J^T w == 0 -- and no
    wrench observer can see it, however good its math. Both iiwa14 start configs
    ('zero' and 'home') ARE all-zeros, so the pick-place benchmark begins in this
    blind spot and the estimate necessarily reads 0 until the arm bends away.
    Pinned as a property of the task, not a bug: a validation run seeded at this
    config measures the singularity, not the estimator."""
    pin = pytest.importorskip("pinocchio")
    plant, _ = smallest_module
    urdf = str(urdfs[plant])
    model, _, _ = pin.buildModelsFromUrdf(urdf, str(urdfs[plant].parent) + "/")
    data = model.createData()
    fid = model.getFrameId("EE")
    w_vert = np.array([0.0, 0.0, -98.1, 0.0, 0.0, 0.0])
    J0 = pin.computeFrameJacobian(model, data, np.zeros(model.nq), fid,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    blind = np.linalg.norm(J0.T @ w_vert)
    assert blind < 1e-4, f"expected the vertical blind spot, got |J^T w|={blind:.3e}"
    # ... and that it IS observable once the arm bends away
    q = np.zeros(model.nq)
    q[1] = 0.6
    J1 = pin.computeFrameJacobian(model, data, q, fid,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    bent = np.linalg.norm(J1.T @ w_vert)
    assert bent > 1e5 * max(blind, 1e-12), (
        f"bending away must restore observability: {blind:.3e} -> {bent:.3e}")


def test_wrench_identifier_weight_mode_is_gravity_aligned(urdfs, smallest_module):
    """'weight' mode must emit a purely downward force and no moment: it exists
    to keep ONLY the horizon-constant part of the disturbance, because the full
    instantaneous wrench is dominated by the payload's inertial reaction to the
    arm's own motion and holding that fixed across the horizon measurably hurts."""
    pin = pytest.importorskip("pinocchio")
    from gato.estimators import OneStepWrenchIdentifier
    plant, _ = smallest_module
    urdf = str(urdfs[plant])
    model, _, _ = pin.buildModelsFromUrdf(urdf, str(urdfs[plant].parent) + "/")
    model.gravity.linear = np.array([0.0, 0.0, -9.81])
    rng = np.random.default_rng(5)
    ident = OneStepWrenchIdentifier(model, ee_frame="EE", alpha=1.0, mode="weight")
    # a generic (non-vertical) configuration: at all-zeros a vertical force is
    # unobservable, so seeding there would test the singularity, not the mode
    q = rng.uniform(-0.8, 0.8, model.nq)
    dq = rng.uniform(-0.4, 0.4, model.nv)
    w = ident.identify(q, dq, dq + 1e-3 * rng.normal(size=model.nv),
                       rng.uniform(-20, 20, model.nv), 1e-3, q_next=q)
    assert w[0] == 0.0 and w[1] == 0.0, "weight mode must emit no lateral force"
    assert w[2] <= 0.0, "weight mode must emit a downward (or zero) force"
    np.testing.assert_array_equal(w[3:], np.zeros(3))


def test_wrench_id_rejects_batch(urdfs, smallest_module):
    """wrench_id + B>1 must raise: it REPLACES the hypothesis batch, and
    silently running both would confound which mechanism explains the wrench."""
    pytest.importorskip("pinocchio")
    plant, N = smallest_module
    with pytest.raises(ValueError, match="batch_size == 1"):
        _mpc_gato(urdfs, plant, N, batch_size=4, wrench_id={})
