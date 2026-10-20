"""Correctness gates for the release timing harness; never collect durations."""
import importlib
import subprocess

import numpy as np
import pytest


@pytest.fixture
def replay(repo_root, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root / "examples/benchmarks"))
    return importlib.import_module("controller_replay")


@pytest.mark.parametrize("plant", ["indy7", "iiwa14", "fr3", "gen3", "ur5e"])
def test_replay_inputs_are_repeatable(replay, plant):
    x, refs, digest = replay.replay_inputs(plant, 7)
    x2, refs2, digest2 = replay.replay_inputs(plant, 7)
    assert digest == digest2
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(refs, refs2)
    assert x.dtype == refs.dtype == np.float32
    assert x.flags.c_contiguous and refs.flags.c_contiguous
    assert refs.shape == (7, 64 * 6)
    assert np.isfinite(x).all() and np.isfinite(refs).all()
    assert not np.array_equal(x[0], x[-1])


def test_replay_refuses_unapproved_or_dirty_timing(replay, monkeypatch):
    monkeypatch.delenv("GATO_QUIET_WINDOW", raising=False)
    with pytest.raises(RuntimeError, match="assigned window"):
        replay.validate_timing_state(dict(dirty=False, sha="abc"))
    monkeypatch.setenv("GATO_QUIET_WINDOW", "1")
    with pytest.raises(RuntimeError, match="clean source"):
        replay.validate_timing_state(dict(dirty=True, sha="abc"))
    monkeypatch.setattr(replay, "git_provenance", lambda: dict(dirty=False, sha="def"))
    with pytest.raises(RuntimeError, match="clean source"):
        replay.validate_timing_state(dict(dirty=False, sha="abc"))


def test_replay_parity_refuses_mismatch(replay):
    rows = [dict(kind=k, plan_telemetry_sha256="same", pred_err=[1.0]) for k in replay.KINDS]
    replay.check_parity(rows)
    rows[1]["plan_telemetry_sha256"] = "different"
    with pytest.raises(RuntimeError, match="parity failed"):
        replay.check_parity(rows)


def test_replay_launcher_preview_and_window_guard(repo_root, monkeypatch):
    path = repo_root / "examples/benchmarks/run_controller_replay.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    r = subprocess.run(["bash", str(path), "--dry-run"], check=True, capture_output=True, text=True)
    assert "180 result rows" in r.stdout
    monkeypatch.delenv("GATO_QUIET_WINDOW", raising=False)
    r = subprocess.run(["bash", str(path)], capture_output=True, text=True)
    assert r.returncode == 2 and "assigned quiet window" in r.stderr


@pytest.mark.gpu
@pytest.mark.parametrize("plant", ["indy7", "iiwa14", "fr3", "gen3", "ur5e"])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("limits", ["off", "admm"])
def test_replay_host_torch_jax_parity_without_timings(replay, plant, batch, limits):
    x, refs, _ = replay.replay_inputs(plant, 7)
    rows = [replay.run_path(plant, batch, limits, kind, (x, refs), 2) for kind in replay.KINDS]
    replay.check_parity(rows)
    for row in rows:
        assert not any("_ms" in key for key in row)
        assert len(row["iterations"]) == 7
        assert row["native_source_id"]
        assert row["effective_limits"] == ("" if limits == "off" else "admm")
        assert row["admm_exit"] == ([0., 0., 1] if limits == "off" else [0., .01, 1])
