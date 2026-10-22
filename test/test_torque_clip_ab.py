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


def test_actuator_derating_validates_scale_and_preserves_nominal_limits(experiment):
    model = SimpleNamespace(nq=1, nv=1, effortLimit=np.array([4.]),
                            velocityLimit=np.array([1.]),
                            lowerPositionLimit=np.array([-1.]), upperPositionLimit=np.array([1.]))
    for scale in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="effort_scale"):
            experiment.ActuationAudit(model, True, effort_scale=scale)
    audit = experiment.ActuationAudit(model, True, effort_scale=.25)
    mpc = SimpleNamespace(world=SimpleNamespace(step=lambda q, dq, u, dt: (q, dq)),
                          _observe_substep=lambda *a: None)
    audit.attach(mpc)
    mpc.world.step(np.zeros(1), np.zeros(1), np.array([3.]), .001)
    np.testing.assert_array_equal(audit.last_applied, [1.])
    assert audit.metrics()["applied_u_ratio"] == .25
    assert audit.metrics()["commanded_u_ratio"] == .75
    np.testing.assert_array_equal(model.effortLimit, [4.])


def test_stress_inventory_is_fixed_and_rejects_missing_cases(experiment):
    stress = importlib.import_module("torque_stress")
    cases = stress.cases()
    assert len(cases) == 34
    assert len({c["case"] for c in cases}) == 34
    rows = [dict(**c, source="frozen", protocol={}, quotable_times=False, status="ok") for c in cases]
    stress.validate(rows)
    with pytest.raises(ValueError, match="inventory"):
        stress.validate(rows[:-1])
    rows[-1]["scale"] = .5
    with pytest.raises(ValueError, match="changed stress case"):
        stress.validate(rows)


def test_stress_streak_is_per_joint_and_peak_keeps_producing_plan(experiment):
    stress = importlib.import_module("torque_stress")
    model = SimpleNamespace(nq=2, nv=2, effortLimit=np.array([2., 4.]),
                            velocityLimit=np.array([1., 2.]),
                            lowerPositionLimit=np.array([-1., -1.]), upperPositionLimit=np.array([1., 1.]))
    solve = SimpleNamespace(xu=np.zeros((1, 10)), stats=SimpleNamespace(
        row_max_violation=np.array([[.1], [.2], [.3]]), admm_iters=np.array([1])))
    mpc = SimpleNamespace(nx=4, nu=2, nq_robot=2, N=2,
                          world=SimpleNamespace(step=lambda q, dq, u, dt: (q, dq)),
                          controller=SimpleNamespace(warmup=lambda *a: solve, step=lambda *a, **kw: None),
                          _observe_substep=lambda *a: None)
    audit = stress.StressAudit(model, .5)
    audit.attach(mpc)
    mpc.controller.warmup(np.zeros(4), None)
    for u in ([2., 0.], [2., 0.], [0., 4.]):
        mpc.world.step(np.zeros(2), np.array([1.1, 0.]), np.array(u), .001)
    np.testing.assert_allclose(audit.longest, [.002, .001])
    event = audit.metrics()["peak_velocity_event"]
    assert event["plan"]["phase"] == "warmup" and event["time_s"] == .001
    assert event["plan"]["row_viol"] == [.1, .2, .3]
    mpc.controller.warmup(np.ones(4), None)
    assert event["plan"]["measured_qd"] == [0., 0.]
    assert audit.first_over == event
    with pytest.raises(FloatingPointError, match="escape guard"):
        mpc.world.step(np.zeros(2), np.array([6., 0.]), np.zeros(2), .001)


@pytest.mark.gpu
def test_stress_diagnostics_preserve_unclipped_development_tail(experiment):
    stress = importlib.import_module("torque_stress")
    model = experiment.pin_model("gen3")
    sc = next(c["scenario"] for c in stress.cases() if c["case"] == "dev-15-None")
    baseline = experiment.run_episode(model, "gen3", 64, sc, "admm")
    audit = stress.StressAudit(model, None)
    traced = experiment.run_episode(model, "gen3", 64, sc, "admm", audit=audit)
    for key, value in baseline.items():
        if key not in ("wall_s", "solve_ms"):
            assert traced[key] == value, key
    assert audit.peak_event["qd_ratio"] > 1
    assert audit.endpoint_event is not None and audit.endpoint_error >= 0
    assert audit.peak_event["plan"]["time_s"] < audit.peak_event["time_s"]
    parts = stress.decompose_endpoint(model, audit.endpoint_event)
    np.testing.assert_allclose(np.array(parts["planned_dynamics_gap"]) + parts["initial_state_gap"]
                               + np.array(parts["integration_and_actuation_gap"]),
                               parts["actual_minus_plan"], atol=1e-12)
    event = dict(audit.endpoint_event, time_s=audit.endpoint_event["time_s"] + .001)
    with pytest.raises(ValueError, match="complete control interval"):
        stress.decompose_endpoint(model, event)


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["dev-15-None", "holdout-94-replay"])
def test_stress_velocity_gap_matches_native_discrete_dynamics(experiment, repo_root, case):
    """The traced discrepancy is not a Pinocchio-only forward-dynamics artifact."""
    import gato
    import pinocchio as pin
    stress = importlib.import_module("torque_stress")
    rows = [json.loads(line) for line in (repo_root /
        "examples/benchmarks/data/limits_holdout/gen3_torque_stress_20261018.jsonl").read_text().splitlines()]
    row, = [r for r in rows if r["case"] == case]
    model = experiment.pin_model("gen3")
    from _bench import urdf_path
    solver = gato.BSQP(urdf_path("gen3"), batch_size=1, N=64, dt=stress.DT, plant_type="gen3")
    for key in ("peak_velocity_event", "worst_endpoint_event"):
        event = row["audit"][key]
        plan = event["plan"]
        for prefix in ("planned", "measured"):
            q = np.array(plan["planned_q0" if prefix == "planned" else "measured_q"])
            v = np.array(plan["planned_qd0" if prefix == "planned" else "measured_qd"])
            u = np.array(event["commanded_u"])
            expected_v = v + stress.DT * pin.aba(model, model.createData(), q, v, u)
            native = solver.sim_forward(np.r_[q, v], u, stress.DT)[0, model.nq:]
            np.testing.assert_allclose(native, expected_v, atol=5e-4, rtol=0)


def test_stress_data_preserves_failures_and_frozen_protocol(experiment, repo_root):
    stress = importlib.import_module("torque_stress")
    path = repo_root / "examples/benchmarks/data/limits_holdout/gen3_torque_stress_20261018.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "6db5897fbd8b6d169d87d77bee6a3cadc4c640f674ea32571764c9071ad3149d"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    stress.validate(rows)
    assert sum(r["status"] == "ok" for r in rows) == 23
    assert sum(r["status"] == "error" for r in rows) == 11
    for row in rows:
        assert row["source"]["sha"] == "539126066448b6cd622f64b4ee8e1fdc85f4f5cc"
        assert not row["source"]["dirty"]
        for p in (stress.DEV, stress.HOLDOUT):
            assert row["protocol"]["inputs"][p.name] == hashlib.sha256(p.read_bytes()).hexdigest()
        if row["status"] == "error":
            assert "stress escape guard" in row["error"]
            assert row["audit"]["qd_ratio"] > 5 or row["audit"]["q_out_rad"] > 1
        else:
            assert row["result"]["reached"] == 3
            assert "wall_s" not in row["result"] and "solve_ms" not in row["result"]
        if row["scale"] is not None:
            assert row["audit"]["applied_u_ratio"] <= row["scale"]
        assert len(row["audit"]["peak_velocity_event"]["plan"]["row_viol"]) == 3


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


def test_defect_policy_report_renders_and_rejects_unknown_policy(experiment, repo_root, tmp_path, capsys):
    """defect_policy_dev: the report renders every (plant, task, actuator) cell
    per policy from saved rows, keeps error rows out of the metrics, and the
    collector refuses a policy that limits_holdout does not define."""
    study = importlib.import_module("defect_policy_dev")

    def row(mech, clip, reached, defect, iters, status="ok"):
        r = dict(set="dev", seed=1, plant="fr3", task="pickplace", scenario=0, scenario_def={},
                 mech=mech, clip=clip, N=64, sha="abc", date="d", quotable_times=False, status=status)
        if status == "ok":
            r["result"] = dict(reached=reached, time_to_all=1.5, iters_mean=iters, iters_max=iters,
                               defect=(dict(p50=defect, p90=defect, p99=defect, max=defect, mean=defect)
                                       if defect is not None else None),
                               substeps=dict(qd_ratio=.9, applied_u_ratio=1.0, commanded_u_ratio=1.2,
                                             clipped_samples=3 if clip else 0))
        return r
    rows = [row("admm-defect-tele", False, 3, 2e-3, 1), row("admm-defect-tele", True, 3, 2e-3, 1),
            row("admm-sqp4-defect1e-3", False, 3, 8e-4, 2.5), row("admm-sqp4-defect1e-3", True, 3, 8e-4, 2.5),
            row("admm-sqp4-defect1e-3", True, 0, None, 0, status="error")]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    study.report([str(path)])
    out = capsys.readouterr().out
    assert "fr3 pickplace clipped  (1 errors)" in out and "fr3 pickplace unclipped  (0 errors)" in out
    assert "| admm-sqp4-defect1e-3 | 1 | 3.00 | 1.50 | 2.50 / 2.5 | 8.00e-04" in out
    assert "| admm-defect-tele | 1 | 3.00 | 1.50 | 1.00 / 1 | 2.00e-03" in out
    assert out.count("| 0.900 | 1.000 | 1.200 |") == 4

    args = SimpleNamespace(out=str(tmp_path / "x.jsonl"), mechs="nobody", plants="fr3", tasks="pickplace",
                           n=1, seed=1, N=64, clip="0", scenarios=None)
    with pytest.raises(SystemExit, match="unknown mechanism"):
        study.collect(args)


def test_defect_policy_datasets_are_frozen(repo_root):
    """The October 21 defect-policy development set (clipped, 240 episodes) and the
    fresh holdout (unclipped, seed 20261021, 1200 episodes): digests, complete
    inventories, zero errors, one clean source, every goal reached."""
    data = repo_root / "examples/benchmarks/data/limits_holdout"
    dev = data / "defect_policy_dev_20261021.jsonl"
    assert hashlib.sha256(dev.read_bytes()).hexdigest() == "a849c6aca0e957a58a99debbecf7e6c6e99e958dc517d4db8e7647a0b7c537d6"
    rows = [json.loads(line) for line in dev.read_text().splitlines()]
    assert len(rows) == 240
    assert {(r["plant"], r["task"], r["mech"], r["clip"]) for r in rows} == {
        (p, "pickplace", m, True) for p in ("fr3", "gen3", "ur5e")
        for m in ("admm-defect-tele", "admm-sqp4", "admm-sqp4-defect1e-2", "admm-sqp4-defect1e-3")}
    for r in rows:
        assert r["status"] == "ok" and r["sha"] == "34bdcaf" and r["quotable_times"] is False
        assert r["result"]["reached"] == 3 and r["result"]["substeps"]["applied_u_ratio"] <= 1
        assert "wall_s" not in r["result"] and "solve_ms" not in r["result"]

    hold = data / "defect_holdout_20261021.jsonl"
    assert hashlib.sha256(hold.read_bytes()).hexdigest() == "ef9ac9bbc75a94091c3968a45ae659a87b63b148487fdf753ed13138389ea7f4"
    rows = [json.loads(line) for line in hold.read_text().splitlines()]
    assert len(rows) == 1200
    keys = {(r["plant"], r["task"], r["mech"], r["scenario"]) for r in rows}
    assert len(keys) == 1200 and {r["seed"] for r in rows} == {20261021} and {r["set"] for r in rows} == {"holdout"}
    assert {r["mech"] for r in rows} == {"admm", "admm-sqp4-defect1e-2"}
    for r in rows:
        assert r["sha"] == "34bdcaf" and r["quotable_times"] is False and r["N"] == 64
        if r["task"] == "pickplace":
            assert r["reached"] == 3
        else:
            assert r["track_mean"] > 0


def test_derated_planner_datasets_are_frozen(repo_root):
    """The October 21 derated-planner studies: the clipped development set (u_scale
    0.9 and 0.8, 120 episodes) and the clipped fresh holdout (seed 20261021,
    baseline and u_scale 0.8, both tasks, 1200 episodes): digests, inventories,
    zero errors, one clean source, every pick-and-place goal reached."""
    data = repo_root / "examples/benchmarks/data/limits_holdout"
    dev = data / "derated_dev_20261021.jsonl"
    assert hashlib.sha256(dev.read_bytes()).hexdigest() == "4f00af4589fd6ab34be6f293acc633d6a6a95bcc48c71c2873dd087d5d4b16d8"
    rows = [json.loads(line) for line in dev.read_text().splitlines()]
    assert len(rows) == 120 and {r["mech"] for r in rows} == {"admm-u0.9", "admm-u0.8"}
    for r in rows:
        assert r["status"] == "ok" and r["sha"] == "ce0945c" and r["clip"] and r["result"]["reached"] == 3

    hold = data / "derated_holdout_clipped_20261021.jsonl"
    assert hashlib.sha256(hold.read_bytes()).hexdigest() == "45194dad5234adf76e11636a4a2521ecb82e92de37371246d6d01a33b8bcc461"
    rows = [json.loads(line) for line in hold.read_text().splitlines()]
    assert len(rows) == 1200 and {r["seed"] for r in rows} == {20261021}
    keys = {(r["plant"], r["task"], r["mech"], r["scenario"]) for r in rows}
    assert len(keys) == 1200 and {k[2] for k in keys} == {"admm-defect-tele", "admm-u0.8"}
    assert {k[1] for k in keys} == {"fig8", "pickplace"}
    for r in rows:
        assert r["status"] == "ok" and r["sha"] == "ce0945c" and r["clip"] and r["quotable_times"] is False
        assert r["result"]["substeps"]["applied_u_ratio"] <= 1
        if r["task"] == "pickplace":
            assert r["result"]["reached"] == 3


def test_merit_timing_report_pairs_launches(experiment, repo_root, tmp_path, capsys):
    """merit_fusion_timing: the report pairs the two launches per (plant, N, B, block
    size) with the fused/unfused ratio and skips unpaired rows; errors are counted."""
    timing = importlib.import_module("merit_fusion_timing")

    def row(fused, merit, status="ok", threads="default"):
        r = dict(condition="5090", plant="indy7", N=32, B=8, threads=threads, fused_requested=fused,
                 sha="abc", quotable_times=True, status=status)
        if status == "ok":
            r["result"] = dict(merit_fused=bool(fused), solve_us=dict(p50=1000.0 + merit, p90=1100.0, mean=1000.0, max=1200.0),
                               stages_us={"merit": dict(p50=merit, p90=merit + 5, mean=merit, max=merit + 9)})
        return r
    rows = [row(0, 200.0), row(1, 100.0), row(0, 300.0, threads="64"), row(1, 0.0, status="error", threads="64")]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    timing.report([str(path)])
    out = capsys.readouterr().out
    assert "3 rows ok, 1 errors" in out and "### condition 5090" in out
    assert "| indy7 | 32 | 8 | default | 200.0 [205.0] | 100.0 [105.0] | 0.50 | 1200.0 | 1100.0 |" in out
    assert "| indy7 | 32 | 8 | 64 |" not in out
