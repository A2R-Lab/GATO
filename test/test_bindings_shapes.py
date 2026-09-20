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


# ---- non-contiguous inputs (2026-09-20: reproduced 69 corrupted entries) ----
# The bindings copy .data() linearly; any array parameter must therefore be
# c_style|forcecast so a strided view / broadcast / f64 input is materialised
# contiguous BEFORE the copy, not read through its strides as garbage.

def _plain_inputs(solver, B):
    N, nx, nu = solver.N, solver.nx, solver.nu
    rng = np.random.default_rng(7)
    XU = rng.normal(0, 0.05, (B, N * (nx + nu) - nu)).astype(np.float32)
    x = rng.normal(0, 0.05, (B, nx)).astype(np.float32)
    ref = np.zeros((B, N * 6), dtype=np.float32)
    ref[:, 0::6], ref[:, 1::6], ref[:, 2::6] = 0.35, 0.25, 0.5
    return XU, x, ref


def test_strided_xu_view_matches_contiguous(make_solver, smallest_module):
    plant, N = smallest_module
    B = 2
    XU, x, ref = _plain_inputs(make_solver(plant, N, batch_size=B), B)
    # every-other-row view of a 2B stack: same element COUNT as the dense input,
    # but its memory is strided — the size check alone cannot catch it
    stack = np.repeat(XU, 2, axis=0)
    strided = stack[::2]
    assert not strided.flags.c_contiguous
    a = make_solver(plant, N, batch_size=B).solver.solve(np.ascontiguousarray(strided), 0.01, x, ref)
    b = make_solver(plant, N, batch_size=B).solver.solve(strided, 0.01, x, ref)
    np.testing.assert_array_equal(a["XU"], b["XU"])


def test_broadcast_reference_and_f64_state_match_contiguous(make_solver, smallest_module):
    plant, N = smallest_module
    B = 2
    XU, x, ref = _plain_inputs(make_solver(plant, N, batch_size=B), B)
    ref_bcast = np.broadcast_to(ref[0], (B, ref.shape[1]))   # zero-stride rows
    assert ref_bcast.strides[0] == 0
    a = make_solver(plant, N, batch_size=B).solver.solve(XU, 0.01, x, ref)
    b = make_solver(plant, N, batch_size=B).solver.solve(XU, 0.01, x.astype(np.float64), ref_bcast)
    np.testing.assert_array_equal(a["XU"], b["XU"])


def test_fortran_order_f_ext_matches_c_order(make_solver, smallest_module):
    plant, N = smallest_module
    B = 2
    s = make_solver(plant, N, batch_size=B)
    nb = s.n_bodies or s.nv
    f = np.random.default_rng(3).normal(0, 1.0, (B, N, nb * 6)).astype(np.float32)
    XU, x, ref = _plain_inputs(s, B)
    s.solver.set_f_ext_knot_batch(np.ascontiguousarray(f))
    a = s.solver.solve(XU, 0.01, x, ref)
    s2 = make_solver(plant, N, batch_size=B)
    s2.solver.set_f_ext_knot_batch(np.asfortranarray(f))   # same bytes, transposed strides
    b = s2.solver.solve(XU, 0.01, x, ref)
    np.testing.assert_array_equal(a["XU"], b["XU"])
