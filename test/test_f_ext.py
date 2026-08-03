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
