"""numpy / torch / JAX at the solver boundary. Host tier: kind detection and
host conversions (CPU tensors). GPU tier: the device path (CUDA tensors read in
place, results back through DLPack) is bit-identical to the host path, cold and
warm-started, and the controller returns ``u`` in the caller's kind."""
import numpy as np
import pytest

import gato
from gato import _arrays
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


def test_kind_and_host_conversions():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    t = torch.from_numpy(a)
    j = jax.device_put(jnp.asarray(a), jax.devices("cpu")[0])
    assert _arrays.kind(a) == "numpy" and _arrays.kind(t) == "torch" and _arrays.kind(j) == "jax"
    assert not _arrays.is_cuda(a) and not _arrays.is_cuda(t) and not _arrays.is_cuda(j)
    for x in (a, t, j):
        h = _arrays.to_numpy(x)
        assert h.dtype == np.float32 and h.flags.c_contiguous and np.array_equal(h, a)
    assert isinstance(_arrays.like(a.astype(np.float32), "torch"), torch.Tensor)
    assert _arrays.kind(_arrays.like(a.astype(np.float32), "jax")) == "jax"
    assert _arrays.like(a, "numpy") is a


def _problem(B, N):
    x0 = np.tile(np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(6))).astype(np.float32), (B, 1))
    x0[1, 0] += 0.1
    ref = np.tile(figure8(0.01, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32), (B, 1))
    return x0, ref


def _indy7_solver(N):
    return gato.BSQP(gato.robot_urdf("indy7"), batch_size=2, N=N, dt=0.01, plant_type="indy7",
                     max_sqp_iters=3)


@pytest.fixture(scope="module")
def indy7_N():
    combos = [n for p, n in gato.available() if p == "indy7"]
    if not combos:
        pytest.fail("an indy7 receipt-profile module must be built")
    return min(combos)


@pytest.mark.gpu
def test_device_path_matches_host_path(indy7_N):
    if not torch.cuda.is_available():
        pytest.fail("the device path needs a CUDA torch build in the receipt environment")
    N = indy7_N
    x0, ref = _problem(2, N)
    host = _indy7_solver(N).solve(x0, ref)
    xt, rt = torch.from_numpy(x0).cuda(), torch.from_numpy(ref).cuda()
    dev = _indy7_solver(N).solve(xt, rt)
    assert isinstance(dev.xu, torch.Tensor) and dev.xu.is_cuda and dev.xu.dtype == torch.float32
    assert dev.xu.shape == (2, host.xu.shape[1])
    np.testing.assert_array_equal(dev.xu.cpu().numpy(), host.xu)
    assert dev.u0().shape == (6,)
    # a non-float32, non-contiguous CUDA input is converted, not rejected
    dev64 = _indy7_solver(N).solve(xt.double().t().contiguous().t(), rt.double())
    np.testing.assert_array_equal(dev64.xu.cpu().numpy(), host.xu)
    # CPU tensors take the host path and come back on the CPU
    cpu = _indy7_solver(N).solve(torch.from_numpy(x0), torch.from_numpy(ref))
    assert isinstance(cpu.xu, torch.Tensor) and not cpu.xu.is_cuda
    np.testing.assert_array_equal(cpu.xu.numpy(), host.xu)


@pytest.mark.gpu
def test_device_path_jax(indy7_N):
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.fail("the device path needs a CUDA JAX build in the receipt environment")
    N = indy7_N
    x0, ref = _problem(2, N)
    host = _indy7_solver(N).solve(x0, ref)
    res = _indy7_solver(N).solve(jnp.asarray(x0), jnp.asarray(ref))
    assert _arrays.kind(res.xu) == "jax" and _arrays.is_cuda(res.xu)
    np.testing.assert_array_equal(np.asarray(res.xu), host.xu)


@pytest.mark.gpu
def test_device_warm_start_and_device_mismatch(indy7_N):
    N = indy7_N
    x0, ref = _problem(2, N)
    sh, sd = _indy7_solver(N), _indy7_solver(N)
    h1 = sh.solve(x0, ref)
    h2 = sh.solve(x0, ref, xu_warm=h1.xu)
    xt, rt = torch.from_numpy(x0).cuda(), torch.from_numpy(ref).cuda()
    d1 = sd.solve(xt, rt)
    d2 = sd.solve(xt, rt, xu_warm=d1.xu)
    np.testing.assert_array_equal(d2.xu.cpu().numpy(), h2.xu)
    assert d2.stats.sqp_iters.tolist() == h2.stats.sqp_iters.tolist()
    with pytest.raises(TypeError, match="same device"):
        sd.solve(xt, rt, xu_warm=d1.xu.cpu())
    with pytest.raises(ValueError, match="expected"):
        sd.solve(xt[:, :3], rt)


@pytest.mark.gpu
def test_controller_returns_callers_kind(indy7_N):
    N = indy7_N
    x0, ref = _problem(2, N)
    c = gato.MPCController(_indy7_solver(N))
    c.reset(x0[0])
    c.warmup(x0[0], ref[0])
    u_np = c.step(x0[0], ref[0]).u
    c2 = gato.MPCController(_indy7_solver(N))
    c2.reset(x0[0])
    c2.warmup(x0[0], ref[0])
    u_t = c2.step(torch.from_numpy(x0[0]).cuda(), torch.from_numpy(ref[0]).cuda()).u
    assert isinstance(u_t, torch.Tensor) and u_t.is_cuda
    np.testing.assert_array_equal(u_t.cpu().numpy(), u_np)
