"""Host-only numerics: figure8 generator, reference providers, warm-start layout."""
import numpy as np
import pytest

from gato.common import figure8, initialize_warm_start
from gato.policy import GoalReference, TrajectoryReference


def test_figure8_shape_and_tiling():
    dt, period, cycles = 0.01, 6, 5
    traj = figure8(dt, period=period, cycles=cycles)
    assert traj.ndim == 1 and traj.size % 6 == 0
    pts = traj.reshape(-1, 6)
    per_cycle = int(period / dt)
    assert len(pts) == per_cycle * cycles
    # orientation slots are zero; cycles are exact repeats
    assert np.all(pts[:, 3:] == 0.0)
    np.testing.assert_array_equal(pts[:per_cycle], pts[per_cycle:2 * per_cycle])


def test_figure8_rotation_is_pure_z():
    a = figure8(0.01, theta=0.0, cycles=1).reshape(-1, 6)
    b = figure8(0.01, theta=np.pi / 4, cycles=1).reshape(-1, 6)
    # z is invariant under the z-rotation; xy norms match
    np.testing.assert_allclose(a[:, 2], b[:, 2], rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(a[:, :2], axis=1),
                               np.linalg.norm(b[:, :2], axis=1), rtol=1e-9)


def test_trajectory_reference_window_and_clamp():
    N, dt = 4, 0.1
    T = 10
    traj = np.arange(T * 6, dtype=np.float32)
    ref = TrajectoryReference(traj, dt, N)
    np.testing.assert_array_equal(ref.window(0.0), traj[: 6 * N])
    np.testing.assert_array_equal(ref.window(2 * dt), traj[12: 12 + 6 * N])
    # clamps at the end: window start never runs past T - N
    last = traj[6 * (T - N): 6 * T]
    np.testing.assert_array_equal(ref.window(100.0), last)
    assert ref.window(100.0).size == 6 * N


def test_trajectory_reference_validation():
    with pytest.raises(ValueError):
        TrajectoryReference(np.zeros(7), 0.1, 1)   # not a multiple of 6
    with pytest.raises(ValueError):
        TrajectoryReference(np.zeros(6 * 3), 0.1, 4)  # shorter than one horizon


def test_goal_reference_constant():
    ref = GoalReference([0.1, 0.2, 0.3], N=5)
    w = ref.window(0.0)
    assert w.shape == (30,)
    np.testing.assert_array_equal(w, ref.window(123.0))
    np.testing.assert_allclose(w.reshape(5, 6)[:, :3], [[0.1, 0.2, 0.3]] * 5, rtol=1e-6)
    assert not ref.done(1e9)


def test_initialize_warm_start_layout():
    nx, nu, N = 4, 2, 3
    x = np.arange(nx, dtype=np.float64)
    xu = initialize_warm_start(x, N, nx, nu)
    assert xu.shape == (N * (nx + nu) - nu,)
    for k in range(N):
        s = k * (nx + nu)
        np.testing.assert_array_equal(xu[s: s + nx], x)
    # control slots stay zero
    assert xu.sum() == x.sum() * N
