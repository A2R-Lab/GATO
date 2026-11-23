"""The inverse-dynamics gradient spill (0.2.3): a robot whose KKT assembly does
not fit one block's shared memory keeps that gradient's scratch pool in global memory.
Only the storage moves, so a robot forced onto the spill gives the same bits."""
import numpy as np
import pytest

import gato

N, DT, B = 8, 0.01, 2


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from gato import _cache
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    _cache.toolchain_identity.cache_clear()
    _cache._native_source_id.cache_clear()
    return tmp_path / "cache"


def test_spill_setting_is_validated_and_keys_the_module(monkeypatch):
    from gato import _cache
    monkeypatch.delenv("GATO_ID_GRAD_SPILL", raising=False)
    assert _cache.id_grad_spill() is None
    monkeypatch.setenv("GATO_ID_GRAD_SPILL", "1")
    assert _cache.id_grad_spill() == 1
    monkeypatch.setenv("GATO_ID_GRAD_SPILL", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _cache.id_grad_spill()


@pytest.mark.gpu
@pytest.mark.slow
def test_forced_spill_is_bit_identical(cache, urdfs, monkeypatch):
    def solver(name, spill):
        if spill:
            monkeypatch.setenv("GATO_ID_GRAD_SPILL", "1")
        else:
            monkeypatch.delenv("GATO_ID_GRAD_SPILL", raising=False)
        gato.register_robot(name, urdfs["indy7"], N=[N], build=True)
        return gato.BSQP(str(urdfs["indy7"]), batch_size=B, N=N, dt=DT, plant_type=name)

    shared, spilled = solver("indy7_shared", False), solver("indy7_spilled", True)
    assert shared.lib.ID_GRAD_SPILL_COUNT == 0 and spilled.lib.ID_GRAD_SPILL_COUNT > 0
    assert spilled.lib.KKT_SMEM_BYTES < shared.lib.KKT_SMEM_BYTES

    rng = np.random.default_rng(0)
    nq = shared.nq
    x = rng.uniform(-0.8, 0.8, (B, 2 * nq)).astype(np.float32)
    u = rng.uniform(-5.0, 5.0, (B, nq)).astype(np.float32)
    xu = np.concatenate([np.tile(np.c_[x, u], (1, N))[:, :-nq]], axis=1)   # a different state per batch member
    goals = np.tile(np.array([0.4, 0.1, 0.5, 0, 0, 0], np.float32), (B, N))
    a, b = shared.debug_setup_kkt(xu, x, goals), spilled.debug_setup_kkt(xu, x, goals)
    for block in ("A", "B", "c", "Q", "R", "q", "r"):
        got, want = np.asarray(b[block]), np.asarray(a[block])
        if block in ("A", "B", "R", "r"):   # one block per interval; the slot of the last knot is never written
            got, want = (m.reshape(B, N, -1)[:, :N - 1] for m in (got, want))
        np.testing.assert_array_equal(got, want, err_msg=block)
    ra, rb = shared.solve(x, goals), spilled.solve(x, goals)
    np.testing.assert_array_equal(np.asarray(ra.xu), np.asarray(rb.xu))
