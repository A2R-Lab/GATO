"""TrajectoryReference / GoalReference: window() and done() agree on the last
valid horizon (2026-09-20: done() mixed knot and element units — it fired
5N knots early)."""
import numpy as np
import pytest

from gato.policy import GoalReference, TrajectoryReference


def _traj(T):
    return np.repeat(np.arange(T, dtype=np.float32), 6)   # knot k = [k]*6


@pytest.mark.parametrize("T,N", [(4, 4), (5, 4), (40, 8), (13, 5)])
def test_done_fires_one_knot_after_the_last_full_window(T, N):
    ref = TrajectoryReference(_traj(T), dt_knot=0.1, N=N)
    last = T - N                     # start knot of the last full window
    for k in range(last + 1):
        assert not ref.done(k * 0.1), (k, last)
        w = ref.window(k * 0.1)
        assert w.shape == (6 * N,)
        assert w[0] == k              # window really starts at knot k
    assert ref.done((last + 1) * 0.1)
    # window() clamps to the last full window beyond the end
    assert ref.window((last + 3) * 0.1)[0] == last


def test_window_and_done_agree_for_exactly_one_horizon():
    ref = TrajectoryReference(_traj(6), dt_knot=0.05, N=6)
    assert not ref.done(0.0)
    assert ref.done(0.05)
    assert ref.window(0.0)[0] == 0 and ref.window(10.0)[0] == 0


def test_trajectory_reference_validation():
    with pytest.raises(ValueError):
        TrajectoryReference(np.zeros(7, np.float32), 0.1, 1)     # not a multiple of 6
    with pytest.raises(ValueError):
        TrajectoryReference(np.zeros(12, np.float32), 0.1, 3)    # shorter than one horizon


def test_goal_reference_constant_and_never_done():
    g = GoalReference([1.0, 2.0, 3.0], N=4)
    w = g.window(123.0)
    assert w.shape == (24,)
    np.testing.assert_array_equal(w[0:3], [1.0, 2.0, 3.0])
    assert not g.done(1e9)
