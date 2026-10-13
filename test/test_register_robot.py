"""gato.register_robot: the pip-install path. Codegen into a per-host cache
(host tier), then the first solver for the robot compiles its module once and
later constructions hit the cache (gpu/slow tier). Every test uses its own
cache directory via GATO_CACHE_DIR."""
import json
import logging

import numpy as np
import pytest

import gato
from gato import _cache

NAME = "cachebot_indy7"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    _cache.toolchain_identity.cache_clear()
    _cache._native_source_id.cache_clear()
    return tmp_path / "cache"


def test_register_writes_headers_and_manifest(cache, urdfs):
    meta = gato.register_robot(NAME, urdfs["indy7"], N=[8, 16])
    assert meta["nq"] == meta["nv"] == 6 and meta["ee_frame"] == "EE"
    entry = _cache.robot_entry(NAME)
    dyn = cache / "robots" / entry["codegen_key"] / NAME
    assert (dyn / "grid.cuh").exists() and (dyn / "limits.cuh").exists()
    assert entry["N"] == [8, 16] and entry["urdf_sha256"]
    assert gato.robot_info(NAME)["codegen_key"] == entry["codegen_key"]
    # the same URDF + options re-registers without regenerating (content-keyed)
    stamp = (dyn / "grid.cuh").stat().st_mtime_ns
    gato.register_robot(NAME, urdfs["indy7"], N=[8])
    assert (dyn / "grid.cuh").stat().st_mtime_ns == stamp
    assert _cache.robot_entry(NAME)["N"] == [8]
    # nothing compiled yet: no module, and discovery does not list it
    assert _cache.find_module(NAME, 8) is None
    assert (NAME, 8) not in gato.available()


def test_register_rejects_bad_name_and_missing_urdf(cache, urdfs, tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        gato.register_robot("not a name", urdfs["indy7"])
    with pytest.raises(FileNotFoundError):
        gato.register_robot("ghost", tmp_path / "missing.urdf")


def test_unregistered_robot_raises_with_register_hint(cache, urdfs):
    with pytest.raises(ValueError, match="register_robot"):
        gato.BSQP(str(urdfs["indy7"]), batch_size=1, N=8, dt=0.01, plant_type="nobody_registered_this")


def test_module_key_tracks_inputs(cache, urdfs):
    gato.register_robot(NAME, urdfs["indy7"], N=[8])
    entry = _cache.robot_entry(NAME)
    root = cache / "robots" / entry["codegen_key"]
    k1, inputs = _cache.module_key(entry, 8, "default", root, "120")
    assert set(inputs) == {"codegen_key", "N", "variant", "native_source_id", "toolchain", "schema"}
    assert _cache.module_key(entry, 8, "default", root, "120")[0] == k1
    assert _cache.module_key(entry, 16, "default", root, "120")[0] != k1      # horizon
    assert _cache.module_key(entry, 8, "fc", root, "120")[0] != k1           # variant
    assert _cache.module_key(entry, 8, "default", root, "89")[0] != k1       # GPU architecture
    other = dict(entry, codegen_key="0" * 16)
    assert _cache.module_key(other, 8, "default", root, "120")[0] != k1      # URDF / options / generator


@pytest.mark.gpu
@pytest.mark.slow
def test_first_solver_compiles_once_then_hits_cache(cache, urdfs, caplog):
    pytest.importorskip("pinocchio")
    urdf = str(urdfs["indy7"])
    gato.register_robot(NAME, urdf, N=[8])
    with caplog.at_level(logging.WARNING, logger="gato"):
        s = gato.BSQP(urdf, batch_size=2, N=8, dt=0.01, plant_type=NAME, max_sqp_iters=3)
    assert any("one-time" in r.getMessage() and f"bsqpN8_{NAME}" in r.getMessage() for r in caplog.records)
    so = _cache.find_module(NAME, 8)
    assert so is not None and so.parent.parent.parent == cache / "modules"
    meta = json.loads((so.parent / "meta.json").read_text())
    assert meta["inputs"]["N"] == 8 and meta["seconds"] > 0
    assert not (cache / "build" / NAME).exists() or not any((cache / "build" / NAME).iterdir())
    x = np.zeros((2, s.nx), np.float32)
    goals = np.zeros((2, 8 * 6), np.float32)
    goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
    assert np.isfinite(s.solve(x, goals).xu).all()
    assert (NAME, 8) in gato.available()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gato"):
        gato.BSQP(urdf, batch_size=1, N=8, dt=0.01, plant_type=NAME)
    assert not [r for r in caplog.records if "compiling" in r.getMessage()], "second construction must hit the cache"


@pytest.mark.gpu
@pytest.mark.slow
def test_vendored_robot_compiles_from_bundled_headers(cache, urdfs, caplog):
    """The robots GATO ships headers for need no register_robot call: a horizon
    with no package module compiles from the bundled headers into the cache."""
    N = 12   # not in the receipt profile, so never a package module
    assert ("indy7", N) not in gato.available()
    urdf = str(urdfs["indy7"])
    with caplog.at_level(logging.WARNING, logger="gato"):
        s = gato.BSQP(urdf, batch_size=1, N=N, dt=0.01, plant_type="indy7", max_sqp_iters=2)
    assert any(f"bsqpN{N}_indy7" in r.getMessage() for r in caplog.records)
    assert s.N == N and _cache.find_module("indy7", N) is not None
    assert ("indy7", N) in gato.available()
    x = np.zeros((1, s.nx), np.float32)
    goals = np.zeros((1, N * 6), np.float32)
    goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
    assert np.isfinite(s.solve(x, goals).xu).all()
