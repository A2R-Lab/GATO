"""Convention locks for the CL-3 floating-base manifold ops (host-only).

``gato/dynamics/manifold.cuh`` (over ``glass::se3_retract/se3_difference``)
implements x ⊞ dz / x ⊟ x with the base tangent ordered **[v_lin; omega]**
(the pinocchio integrate/difference convention; quat stored xyzw at q[3:7]).
Device gates need a go2 module, which needs the whole floating solver path to
compile — so the math is locked HERE first: a numpy transliteration of the
device cores asserted against pinocchio ``integrate``/``difference`` on the
go2 free-flyer. If these pass, the device code is a checked transliteration;
the device-vs-numpy bit path gets its own gate at go2 bring-up (W3.5).
"""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from scipy.spatial.transform import Rotation

GO2_URDF = "external/GRiD/config/robot_assets/go2.urdf"
RNG = np.random.default_rng(0)


def _model():
    return pin.buildModelFromUrdf(GO2_URDF, pin.JointModelFreeFlyer())


def _rand_q(model):
    q = np.zeros(model.nq)
    q[:3] = RNG.standard_normal(3)
    quat = RNG.standard_normal(4)
    q[3:7] = quat / np.linalg.norm(quat)
    lo, hi = model.lowerPositionLimit[7:], model.upperPositionLimit[7:]
    q[7:] = RNG.uniform(lo, hi)
    return q


def _Jl(phi):
    """SO(3) left Jacobian (the SE(3) V matrix)."""
    th = np.linalg.norm(phi)
    S = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + 0.5 * S + S @ S / 6.0
    return (np.eye(3) + (1 - np.cos(th)) / th**2 * S
            + (th - np.sin(th)) / th**3 * S @ S)


def se3_difference_np(pose_from, pose_to):
    """Numpy mirror of glass::se3_difference (lie_detail::se3_difference_core)."""
    Rf = Rotation.from_quat(pose_from[3:7])          # xyzw
    phi = (Rf.inv() * Rotation.from_quat(pose_to[3:7])).as_rotvec()
    pl = Rf.as_matrix().T @ (pose_to[:3] - pose_from[:3])
    rho = np.linalg.solve(_Jl(phi), pl)
    return np.concatenate([rho, phi])


def se3_retract_np(pose, rho, phi):
    """Numpy mirror of glass::se3_retract / grid_integrate_floating_q."""
    Rq = Rotation.from_quat(pose[3:7])
    q_new = (Rq * Rotation.from_rotvec(phi)).as_quat()
    p_new = pose[:3] + Rq.apply(_Jl(phi) @ rho)
    return np.concatenate([p_new, q_new])


def _quat_align(q, ref):
    return -q if np.dot(q, ref) < 0 else q


def test_difference_matches_pinocchio():
    """x_to ⊟ x_from (base [v_lin; omega] + joint subtract) == pin.difference."""
    model = _model()
    for _ in range(50):
        q1, q2 = _rand_q(model), _rand_q(model)
        want = pin.difference(model, q1, q2)
        base = se3_difference_np(q1[:7], q2[:7])
        got = np.concatenate([base, q2[7:] - q1[7:]])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_retract_matches_pinocchio():
    """x ⊞ v (base SE(3) retract + joint add) == pin.integrate."""
    model = _model()
    for _ in range(50):
        q1 = _rand_q(model)
        v = RNG.standard_normal(model.nv)
        want = pin.integrate(model, q1, v)
        base = se3_retract_np(q1[:7], v[:3], v[3:6])
        got = np.concatenate([base[:3],
                              _quat_align(base[3:7], want[3:7]),
                              q1[7:] + v[6:]])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_difference_inverts_retract():
    """⊟ is the exact inverse of ⊞ on the composed go2 state."""
    model = _model()
    for _ in range(50):
        q1 = _rand_q(model)
        v = RNG.standard_normal(model.nv) * 0.5
        q2 = pin.integrate(model, q1, v)
        base = se3_difference_np(q1[:7], q2[:7])
        got = np.concatenate([base, q2[7:] - q1[7:]])
        np.testing.assert_allclose(got, v, rtol=1e-8, atol=1e-8)
