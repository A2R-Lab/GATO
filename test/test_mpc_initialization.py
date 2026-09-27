"""Host-only gates for controller seeds and spherical-payload state layout."""
from types import SimpleNamespace

import numpy as np
import pytest

from gato import MPCController


def _controller():
    solver = SimpleNamespace(N=3, nx=2, nu=1, nq=1, nv=1, n_actuated=1,
                             n_fc=0, batch_size=2, reset_dual=lambda: None,
                             reset_rho=lambda: None, set_linsys=lambda _: None)
    return MPCController(solver, linsys="pcg")


def test_explicit_controller_seed_is_copied_and_first_state_replaced():
    c = _controller()
    seed = np.arange(16, dtype=np.float32).reshape(2, 8)
    original = seed.copy()
    c.reset([2, 3], xu_warm=seed)
    np.testing.assert_array_equal(c._XU[:, 2:], original[:, 2:])
    np.testing.assert_array_equal(c._XU[:, :2], [[2, 3], [2, 3]])
    np.testing.assert_array_equal(seed, original)
    seed[:] = -1
    assert c._XU[1, -1] == 15
    c.reset([2, 3], xu_warm=original[0])
    np.testing.assert_array_equal(c._XU[0], c._XU[1])
    c.reset([2, 3])
    np.testing.assert_array_equal(c._XU[0], [2, 3, 0, 2, 3, 0, 2, 3])


@pytest.mark.parametrize("seed", [np.zeros(7), np.zeros((1, 8)), np.full(8, np.nan)])
def test_invalid_controller_seed_rejected(seed):
    with pytest.raises(ValueError, match="xu_warm"):
        _controller().reset([0, 0], xu_warm=seed)


def test_floating_explicit_seed_rejects_invalid_tail_quaternion():
    solver = SimpleNamespace(N=2, nx=13, nu=1, nq=7, nv=6, n_actuated=1,
                             n_fc=0, batch_size=1, floating_base=True,
                             reset_dual=lambda: None, reset_rho=lambda: None,
                             set_linsys=lambda _: None)
    c = MPCController(solver, linsys='pcg')
    x = np.zeros(13, np.float32)
    x[6] = 1
    seed = np.zeros(27, np.float32)
    with pytest.raises(ValueError, match='quaternion'):
        c.reset(x, xu_warm=seed)
    seed[14 + 6] = 1
    c.reset(x, xu_warm=seed)  # knot zero gets the measured, valid quaternion


@pytest.mark.parametrize("angle", [0.0, 0.3, [0.2, -0.3, 0.1]])
def test_pendulum_state_is_on_manifold_and_preserves_robot_velocity(angle):
    pin = pytest.importorskip("pinocchio")
    from gato.mpc_gato import MPC_GATO
    m = MPC_GATO.__new__(MPC_GATO)  # no solver or GPU needed for state assembly
    robot = pin.buildModelFromUrdf("examples/iiwa_description/iiwa14.urdf")
    m.pendulum_config = dict(initial_angle=angle)
    m.model = m._add_pendulum_to_model(robot.copy(), m.pendulum_config)
    m.nq_robot, m.nv_robot = robot.nq, robot.nv
    m.nq, m.nv, m.nx = m.model.nq, m.model.nv, robot.nq + robot.nv
    m.has_pendulum = True
    x = np.linspace(-0.2, 0.3, m.nx)
    q, v = m._initial_sim_state(x)
    np.testing.assert_array_equal(q[:robot.nq], x[:robot.nq])
    np.testing.assert_array_equal(v[:robot.nv], x[robot.nq:])
    np.testing.assert_array_equal(v[robot.nv:], 0)
    assert np.linalg.norm(q[robot.nq:]) == pytest.approx(1.0, abs=1e-14)
    neutral = pin.neutral(m.model)
    neutral[:robot.nq] = x[:robot.nq]
    expected = [angle, 0, 0] if np.ndim(angle) == 0 else angle
    np.testing.assert_allclose(pin.difference(m.model, neutral, q)[robot.nv:],
                               expected, atol=1e-14)
    m.pendulum_config['initial_angle'] = [0, np.nan, 0]
    with pytest.raises(ValueError, match="initial_angle"):
        m._initial_sim_state(x)
