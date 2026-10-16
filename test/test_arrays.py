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


# ---- MPCController on CUDA tensors: the resident warm start ----------------

def _kinematic_loop(ctrl_a, ctrl_b, N, wrap_a, wrap_b, ticks=30, kick_at=12, budget_s=None):
    """Run two controllers on the same kinematic plant (next state = the plan's
    knot-1 state; a velocity kick at ``kick_at`` exercises the reseed path) and
    return the per-tick (u_a, u_b, StepResult_a, StepResult_b)."""
    fig = figure8(0.01, **FIG8_DEFAULT_PARAMS).astype(np.float32)
    x0, _ = _problem(2, N)
    x = x0[0].copy()
    for c, wrap in ((ctrl_a, wrap_a), (ctrl_b, wrap_b)):
        c.reset(wrap(x))
        c.warmup(wrap(x), wrap(fig[: 6 * N]))
    out = []
    for t in range(ticks):
        ref = np.ascontiguousarray(fig[6 * t: 6 * (t + N)])
        ra = ctrl_a.step(wrap_a(x), wrap_a(ref), budget_s=budget_s)
        rb = ctrl_b.step(wrap_b(x), wrap_b(ref), budget_s=budget_s)
        out.append((_arrays.to_numpy(ra.u), _arrays.to_numpy(rb.u), ra, rb))
        nxt = _arrays.to_numpy(ra.xu_best)[ctrl_a.nx + ctrl_a.nu: 2 * ctrl_a.nx + ctrl_a.nu].copy()
        if t == kick_at:
            nxt[ctrl_a.nq:] += 2.0          # a kick the model did not predict: pred_err jumps, the warm start reseeds
        x = nxt
    return out


def _controller(N, **kw):
    # B=2 with no hypothesis batch: both rows solve the same problem (exercises the batch tiling)
    return gato.MPCController(_indy7_solver(N), linsys="pcg", reseed_threshold=0.5, **kw)


def _cuda(a):
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


@pytest.mark.gpu
@pytest.mark.parametrize("warm_start", ["shift", "hold"])
def test_controller_device_path_is_the_host_path(indy7_N, warm_start):
    """CUDA x and reference, no hypotheses: the step runs on the resident
    device warm start and returns device tensors; the plan is bit-identical to
    the host path tick for tick, through a reseed."""
    N = indy7_N
    host, dev = _controller(N, warm_start=warm_start), _controller(N, warm_start=warm_start)
    rows = _kinematic_loop(host, dev, N, lambda a: a, _cuda)
    assert any(rb.reseeded for _, _, _, rb in rows), "the kick must trigger the reseed path"
    for ua, ub, ra, rb in rows:
        assert isinstance(rb.u, torch.Tensor) and rb.u.is_cuda and rb.xu_best.is_cuda and rb.solve.xu.is_cuda
        np.testing.assert_array_equal(ua, ub)
        np.testing.assert_array_equal(_arrays.to_numpy(ra.xu_best), _arrays.to_numpy(rb.xu_best))
        assert ra.reseeded == rb.reseeded and ra.iterations == rb.iterations
        assert rb.pred_err == pytest.approx(ra.pred_err, rel=1e-5, abs=1e-6)
        assert ra.limits == rb.limits == "admm" and np.array_equal(ra.row_viol, rb.row_viol)
    assert dev._xu_where == "device"


@pytest.mark.gpu
def test_controller_device_path_budget_and_kind_switch(indy7_N):
    """A step budget passes through to the resident solve; the warm start follows
    the inputs when a controller alternates host and device calls."""
    N = indy7_N
    host, dev = _controller(N), _controller(N)
    rows = _kinematic_loop(host, dev, N, lambda a: a, _cuda, ticks=6, kick_at=99, budget_s=10.0)
    for ua, ub, ra, rb in rows:
        np.testing.assert_array_equal(ua, ub)
        assert not ra.budget_hit and not rb.budget_hit and ra.iterations == rb.iterations == 3
    fig = figure8(0.01, **FIG8_DEFAULT_PARAMS).astype(np.float32)
    x = _arrays.to_numpy(rows[-1][2].xu_best)[: host.nx]
    # device controller, host inputs: the warm start comes back to the host
    ra, rb = host.step(x, fig[36: 36 + 6 * N]), dev.step(x, fig[36: 36 + 6 * N])
    assert dev._xu_where == "host" and isinstance(rb.u, np.ndarray)
    np.testing.assert_array_equal(ra.u, rb.u)
    # and goes up again on the next device call
    ra, rb = host.step(x, fig[42: 42 + 6 * N]), dev.step(_cuda(x), _cuda(fig[42: 42 + 6 * N]))
    assert dev._xu_where == "device" and rb.u.is_cuda
    np.testing.assert_array_equal(ra.u, rb.u.cpu().numpy())
    # mixed kinds (CUDA x, numpy reference) take the host path
    rb = dev.step(_cuda(x), fig[48: 48 + 6 * N])
    assert dev._xu_where == "host" and rb.u.is_cuda      # u still comes back in the caller's kind


@pytest.mark.gpu
def test_controller_device_path_jax(indy7_N):
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.fail("the receipt environment needs a CUDA JAX (jax[cuda13])")
    N = indy7_N
    host, dev = _controller(N), _controller(N)
    rows = _kinematic_loop(host, dev, N, lambda a: a, lambda a: jnp.asarray(np.ascontiguousarray(a)), ticks=5, kick_at=2)
    for ua, ub, ra, rb in rows:
        assert _arrays.kind(rb.u) == "jax" and _arrays.is_cuda(rb.u)
        np.testing.assert_array_equal(ua, ub)
    assert dev._xu_where == "device"


@pytest.mark.gpu
def test_resident_warm_start_api(indy7_N):
    """set_warm / get_warm round-trip (host and device seeds), hold_warm, the
    solve from the resident buffer, the shift, and the validation errors."""
    N = indy7_N
    s = _indy7_solver(N)
    x0, ref = _problem(2, N)
    seed = np.random.default_rng(0).normal(size=(2, s.xu_size)).astype(np.float32)
    s.set_warm(seed)
    np.testing.assert_array_equal(s.get_warm(), seed)
    s.set_warm(_cuda(seed[0]))                                  # one device row broadcast to the batch
    np.testing.assert_array_equal(s.get_warm(), np.tile(seed[0], (2, 1)))
    s.hold_warm(_cuda(x0))
    hold = s.get_warm()
    stride = s.nx + s.nu
    for b in range(2):
        for k in range(N):
            np.testing.assert_array_equal(hold[b, k * stride: k * stride + s.nx], x0[b])
    assert not hold[:, s.nx: stride].any()
    assert s.pred_err_warm(_cuda(x0[0])) == pytest.approx(0.0)   # knot 0 of the hold seed is x0
    # resident solve from the hold seed == host solve from the same (cold) seed
    host = _indy7_solver(N).solve(x0, ref)
    res = s.solve_warm(_cuda(x0), _cuda(ref))
    assert res.xu.is_cuda
    np.testing.assert_array_equal(res.xu.cpu().numpy(), host.xu)
    np.testing.assert_array_equal(s.get_warm(), host.xu)         # the solution stays resident
    # the SQP step moves knot 0 too (the initial-state constraint is in the merit): the signal reads it
    assert s.pred_err_warm(_cuda(x0[0])) == pytest.approx(float(np.linalg.norm(x0[0] - host.xu[0, : s.nx])), rel=1e-5, abs=1e-6)
    s.shift_warm(1, hold=False)
    want = np.concatenate([host.xu[1, stride:], host.xu[1, -stride:]])
    np.testing.assert_array_equal(s.get_warm(), np.tile(want, (2, 1)))
    s.shift_warm(0, hold=True)
    np.testing.assert_array_equal(s.get_warm(), np.tile(want, (2, 1)))   # row 0 held as is
    with pytest.raises(TypeError, match="CUDA"):
        s.solve_warm(x0, ref)
    with pytest.raises(ValueError, match="expected"):
        s.solve_warm(_cuda(x0[:, :3]), _cuda(ref))
    with pytest.raises(ValueError, match="best_id"):
        s.shift_warm(2)
    with pytest.raises(ValueError, match="shape"):
        s.set_warm(seed[:, :5])
