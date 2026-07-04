"""Input validation: wrong-width arrays must raise, not read out of bounds."""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def solver(make_solver, smallest_module):
    plant, N = smallest_module
    return make_solver(plant, N, batch_size=2)


def test_wrong_state_width_raises(solver):
    B, N, nx, nu = 2, solver.N, solver.nx, solver.nu
    XU = np.zeros((B, N * (nx + nu) - nu), dtype=np.float32)
    x_bad = np.zeros((B, nx + 1), dtype=np.float32)
    ref = np.zeros((B, N * 6), dtype=np.float32)
    with pytest.raises(ValueError):
        solver.solver.solve(XU, solver.dt, x_bad, ref)


def test_wrong_traj_width_raises(solver):
    B, N, nx, nu = 2, solver.N, solver.nx, solver.nu
    XU_bad = np.zeros((B, N * (nx + nu)), dtype=np.float32)  # one extra u block
    x = np.zeros((B, nx), dtype=np.float32)
    ref = np.zeros((B, N * 6), dtype=np.float32)
    with pytest.raises(ValueError):
        solver.solver.solve(XU_bad, solver.dt, x, ref)


def test_wrong_reference_width_raises(solver):
    B, N, nx, nu = 2, solver.N, solver.nx, solver.nu
    XU = np.zeros((B, N * (nx + nu) - nu), dtype=np.float32)
    x = np.zeros((B, nx), dtype=np.float32)
    ref_bad = np.zeros((B, N * 6 - 6), dtype=np.float32)
    with pytest.raises(ValueError):
        solver.solver.solve(XU, solver.dt, x, ref_bad)


def test_wrong_f_ext_width_raises(solver):
    with pytest.raises(ValueError):
        solver.set_f_ext_B(np.zeros((2, 5), dtype=np.float32))


def test_wrong_per_knot_weights_shape_raises(solver):
    with pytest.raises(ValueError):
        solver.set_cost_weights_per_knot(np.zeros((solver.N + 1, 3), dtype=np.float32))
