"""Actuator saturation experiment plumbing; production defaults stay unchanged."""
import importlib
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def experiment(repo_root, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root / "examples/benchmarks"))
    return importlib.import_module("torque_clip_ab")


@pytest.mark.parametrize("clip", [False, True])
def test_substep_audit_clips_only_applied_torque_and_feeds_observer(experiment, clip):
    model = SimpleNamespace(nq=2, nv=2, effortLimit=np.array([2., 4.]),
                            velocityLimit=np.array([1., 2.]),
                            lowerPositionLimit=np.array([-1., -1.]),
                            upperPositionLimit=np.array([1., 1.]))
    received, observed = [], []

    def world_step(q, dq, u, dt):
        received.append(np.array(u))
        return np.array([1.2, 0.]), np.array([0., 3.])

    mpc = SimpleNamespace(world=SimpleNamespace(step=world_step),
                          _observe_substep=lambda *args: observed.append(args[4].copy()))
    audit = experiment.ActuationAudit(model, clip)
    audit.attach(mpc)
    u = np.array([3., -6.])
    q, dq = np.zeros(2), np.array([2., 0.])
    q_next, dq_next = mpc.world.step(q, dq, u, .001)
    mpc._observe_substep(q, dq, q_next, dq_next, u, .001)
    expected = np.array([2., -4.]) if clip else u
    np.testing.assert_array_equal(received[0], expected)
    np.testing.assert_array_equal(observed[0], expected)
    np.testing.assert_array_equal(u, [3., -6.])
    stats = audit.metrics()
    assert stats["samples"] == 1 and stats["simulated_s"] == .001
    assert stats["commanded_u_ratio"] == 1.5
    assert stats["applied_u_ratio"] == (1 if clip else 1.5)
    assert stats["clipped_samples"] == int(clip)
    assert stats["qd_ratio"] == 2  # initial state is included
    assert stats["velocity_over_samples"] == 1  # post-step endpoints only
    assert stats["q_out_rad"] == pytest.approx(.2)
    with pytest.raises(FloatingPointError, match="command"):
        mpc.world.step(q, dq, np.array([np.nan, 0]), .001)


def test_clip_report_rejects_missing_duplicate_and_mismatched_pairs(experiment):
    rows = [dict(protocol=dict(plants=["fr3"], tasks=["pickplace"], n=1), source="frozen",
                 plant="fr3", task="pickplace", scenario=0, scenario_def=dict(q0=[0]),
                 clip=clip, quotable_times=False) for clip in (False, True)]
    experiment.validate(rows)
    for broken in (rows[:1], rows + rows[:1]):
        with pytest.raises(ValueError, match="inventory"):
            experiment.validate(broken)
    rows[1]["scenario_def"] = dict(q0=[1])
    with pytest.raises(ValueError, match="scenarios differ"):
        experiment.validate(rows)


def test_clip_collection_refuses_existing_output(experiment, tmp_path, monkeypatch):
    import sys
    out = tmp_path / "results.jsonl"
    out.write_text("keep existing results\n")
    monkeypatch.setattr(sys, "argv", ["torque_clip_ab", "--out", str(out)])
    monkeypatch.setattr(experiment, "git_provenance", lambda: dict(dirty=False))
    with pytest.raises(FileExistsError):
        experiment.main()
    assert out.read_text() == "keep existing results\n"


@pytest.mark.parametrize("seed,digest", [
    (1, "6afcb5bd3e95b7af104fe605a8c987b335def27f7d7d970403c2742e39ac37d2"),
    (2, "5b2f104a2e14ee6b3d6c870856148fef10a1e26d7d9c506282d57111ea16c8df"),
], ids=["seed1", "seed2"])
def test_clip_development_data_integrity(experiment, repo_root, seed, digest):
    path = repo_root / f"examples/benchmarks/data/limits_holdout/torque_clip_dev_seed{seed}_20261018.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    protocol = experiment.validate(rows)
    assert len(rows) == 240 and protocol["seed"] == seed and protocol["n"] == 20
    for row in rows:
        assert row["status"] == "ok" and not row["source"]["dirty"]
        assert row["source"]["sha"] == "1c92dcbb7108037d196425353176070cf3ba9f5f"
        result = row["result"]
        assert "wall_s" not in result and "solve_ms" not in result
        sub = result["substeps"]
        assert sub["samples"] > 0
        if row["clip"]:
            assert sub["applied_u_ratio"] <= 1
        else:
            assert sub["commanded_u_ratio"] == sub["applied_u_ratio"]


@pytest.mark.gpu
def test_unclipped_audit_preserves_episode_and_covers_final_substep(experiment):
    model = experiment.pin_model("fr3")
    sc = experiment.scenarios(model, "fr3", "pickplace", 1, 1)[0]
    plain = experiment.run_episode(model, "fr3", 64, sc, "admm")
    audit = experiment.ActuationAudit(model, False)
    measured = experiment.run_episode(model, "fr3", 64, sc, "admm", audit=audit)
    for key, value in plain.items():
        if key not in ("wall_s", "solve_ms"):
            assert measured[key] == value, key
    sub = measured["substeps"]
    assert measured["reached"] == 3
    assert sub["samples"] == round(measured["time_to_all"] / experiment.SIM_DT)
    assert sub["samples"] == (measured["steps"] + 1) * 10
    assert sub["commanded_u_ratio"] == sub["applied_u_ratio"]
    assert sub["clipped_samples"] == 0
    assert sub["simulated_s"] == pytest.approx(measured["goal_events"][-1]["time"])
