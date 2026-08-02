"""Posture-anchor (q_pos_cost / set_q_nom) functional gates — PDDP round-4.

The anchor adds 0.5*w*||q - q_nom||^2 to every knot's tracking cost. These
gates pin its KKT contribution DIRECTLY (debug_setup_kkt on a held
trajectory), because solve-level discriminators are confounded: +w on the
q-block Hessian diagonal regularizes the rank-3 GN EE Hessian even at zero
posture error, so an anchored solve legitimately converges deeper (lower
merit, larger honest torques) than an under-converged unanchored one —
"max|u| must not rise at x0 == q_nom" is NOT a property of correct code.
"""
import numpy as np
import pytest

from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

pytestmark = pytest.mark.gpu

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}


def _held(s, plant):
    """(x0, ref, xu_hold): start state, constant reachable EE ref, hold warm-start."""
    q0 = np.asarray(START[plant], dtype=np.float32)
    nq, nx, nu, N = q0.size, s.nx, s.nu, s.N
    x0 = np.concatenate([q0, np.zeros(nq, dtype=np.float32)])
    ref = np.zeros((N, 6), dtype=np.float32)
    ref[:, 0], ref[:, 1], ref[:, 2] = 0.35, 0.25, 0.5
    xu = np.zeros(N * (nx + nu) - nu, dtype=np.float32)
    for k in range(N):
        xu[k * (nx + nu):k * (nx + nu) + nx] = x0
    return x0, ref, xu


def _kkt(s, x0, ref, xu):
    d = s.solver.debug_setup_kkt(xu[None, :], np.float32(0.01), x0[None, :],
                                 ref.ravel()[None, :])
    N, nx = s.N, s.nx
    Q = np.asarray(d["Q"]).reshape(N, nx, nx)
    q = np.asarray(d["q"]).reshape(N, nx)
    return Q, q


def test_anchor_kkt_inert_at_q_nom(make_solver, smallest_module):
    """Held trajectory AT q_nom: gradient rows identical to w=0; Hessian diag
    gains exactly +w on the q rows and nothing anywhere else."""
    plant, N = smallest_module
    w = 0.05
    s0 = make_solver(plant, N, batch_size=1)
    x0, ref, xu = _held(s0, plant)
    Q0, q0 = _kkt(s0, x0, ref, xu)

    sw = make_solver(plant, N, batch_size=1)
    sw.set_q_pos_cost(w)
    sw.set_q_nom(x0[:s0.nq])
    Qw, qw = _kkt(sw, x0, ref, xu)

    np.testing.assert_array_equal(qw, q0)  # zero posture error -> zero gradient shift
    dQ = Qw - Q0
    nq = s0.nq
    for k in range(N):
        np.testing.assert_allclose(np.diag(dQ[k])[:nq], w, atol=1e-6)
        np.testing.assert_allclose(np.diag(dQ[k])[nq:], 0.0, atol=0.0)
    off = dQ.copy()
    for k in range(N):
        np.fill_diagonal(off[k], 0.0)
    np.testing.assert_allclose(off, 0.0, atol=0.0)


def test_anchor_gradient_sign(make_solver, smallest_module):
    """Held trajectory OFFSET from q_nom: q-row gradient delta == w*(q - q_nom)."""
    plant, N = smallest_module
    w, offset = 0.05, 0.3
    s0 = make_solver(plant, N, batch_size=1)
    x0, ref, xu = _held(s0, plant)
    _, q0 = _kkt(s0, x0, ref, xu)

    q_nom = x0[:s0.nq] + np.float32(offset)
    sw = make_solver(plant, N, batch_size=1)
    sw.set_q_pos_cost(w)
    sw.set_q_nom(q_nom)
    _, qw = _kkt(sw, x0, ref, xu)

    nq = s0.nq
    expect = w * (x0[:nq] - q_nom)
    for k in range(N):
        np.testing.assert_allclose(qw[k, :nq] - q0[k, :nq], expect, atol=1e-6)
    np.testing.assert_array_equal(qw[:, nq:], q0[:, nq:])  # qd rows untouched


def test_anchor_pulls_toward_q_nom(make_solver, smallest_module):
    """A/B pull direction: with everything else identical, the solve anchored
    at target A ends closer to A than the solve anchored at target B does —
    the anchors demonstrably steer the nullspace. (Comparing anchored vs
    UNanchored final postures is not a clean property: the anchor's Hessian
    regularization deepens convergence, which can move the solution more.)"""
    plant, N = smallest_module
    w = 0.5

    def run(q_nom):
        s = make_solver(plant, N, batch_size=1, max_sqp_iters=50)
        s.set_q_pos_cost(w)
        s.set_q_nom(q_nom)
        x0, ref, xu = _held(s, plant)
        r = s.solve(x0[None, :], ref.ravel()[None, :], xu[None, :])
        nq, stride = s.nq, s.nx + s.nu
        qN = np.asarray(r.xu[0])[(N - 1) * stride:(N - 1) * stride + nq]
        assert np.isfinite(np.asarray(r.xu)).all()
        return qN

    s_probe = make_solver(plant, N, batch_size=1)
    x0, _, _ = _held(s_probe, plant)
    nom_a = x0[:s_probe.nq].copy()
    nom_b = nom_a + np.float32(0.4)

    qN_a = run(nom_a)
    qN_b = run(nom_b)
    assert np.linalg.norm(qN_a - nom_a) < np.linalg.norm(qN_b - nom_a)
    assert np.linalg.norm(qN_b - nom_b) < np.linalg.norm(qN_a - nom_b)


def test_per_joint_cost_vectors_kkt(make_solver, smallest_module):
    """set_u_cost_vec / per-joint set_q_pos_cost land EXACTLY on the KKT
    diagonals (R actuated rows / Q q rows), per joint, every knot — the
    round-5 per-joint gain knobs. Empty/scalar resets restore the baseline."""
    plant, N = smallest_module
    s0 = make_solver(plant, N, batch_size=1)
    x0, ref, xu = _held(s0, plant)
    Q0, _ = _kkt(s0, x0, ref, xu)
    d0 = s0.solver.debug_setup_kkt(xu[None, :], np.float32(0.01), x0[None, :],
                                   ref.ravel()[None, :])
    R0 = np.asarray(d0["R"]).reshape(N, s0.nu, s0.nu)

    nq, na = s0.nq, s0.n_actuated
    uvec = (1e-6 + 1e-3 * np.arange(na)).astype(np.float32)
    wvec = (0.01 + 0.01 * np.arange(nq)).astype(np.float32)

    sv = make_solver(plant, N, batch_size=1)
    sv.set_u_cost_vec(uvec)
    sv.set_q_pos_cost(wvec)          # per-joint anchor stiffness
    sv.set_q_nom(x0[:nq])
    Qv, _ = _kkt(sv, x0, ref, xu)
    dv = sv.solver.debug_setup_kkt(xu[None, :], np.float32(0.01), x0[None, :],
                                   ref.ravel()[None, :])
    Rv = np.asarray(dv["R"]).reshape(N, sv.nu, sv.nu)

    for k in range(N - 1):           # terminal knot has no control block
        np.testing.assert_allclose(np.diag(Rv[k])[:na] - np.diag(R0[k])[:na],
                                   uvec - np.float32(1e-6), atol=1e-7)
    for k in range(N):
        np.testing.assert_allclose(np.diag(Qv[k])[:nq] - np.diag(Q0[k])[:nq],
                                   wvec, atol=1e-6)

    # resets restore the scalar path bitwise
    sv.set_u_cost_vec(None)
    sv.set_q_pos_cost(0.0)
    sv.set_q_nom(None)
    Qr, qr = _kkt(sv, x0, ref, xu)
    np.testing.assert_array_equal(Qr, Q0)
