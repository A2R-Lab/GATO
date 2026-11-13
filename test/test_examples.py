"""Check the introductory entry points without importing or running GPU code."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_intro_example_syntax(path):
    ast.parse(path.read_text(), filename=str(path))


def test_constraint_example_exports_correctness_without_timings(repo_root, tmp_path, monkeypatch):
    import numpy as np
    path = repo_root / "examples/12_constraint_mechanisms.py"
    spec = importlib.util.spec_from_file_location("constraint_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "run", lambda *args: dict(
        qd=np.float64(1.1), qd_over=0.01, u=0.8, plan_viol=0.02,
        reached=2, t_all=None, solve_ms=12345.0))
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--json", str(out)])
    mod.main()
    data = json.loads(out.read_text())
    assert data["quotable_times"] is False and data["seed"] == 1
    assert data["task_settings"]["pace_by_solve_time"] is False
    assert len(data["scenario"]["goals"]) == 3
    assert {(r["N"], r["mechanism"]) for r in data["results"]} == {
        (N, mech) for N in mod.HORIZONS for mech in mod.MECHS}
    assert len(data["results"]) == 12
    for row in data["results"]:
        assert "solve_ms" not in row and row["t_all"] is None
        assert row["qd"] == 1.1 and row["reached"] == 2


def test_merge_checkpoint_dry_run(repo_root, tmp_path):
    script = repo_root / "examples/benchmarks/run_merge_checkpoint.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script), "--dry-run"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert "no GPU queries" in result.stdout
    assert "three process repeats" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_merge_checkpoint_requires_quiet_declaration(repo_root, tmp_path):
    env = dict(os.environ)
    env.pop("GATO_QUIET_WINDOW", None)
    result = subprocess.run(["bash", str(repo_root / "examples/benchmarks/run_merge_checkpoint.sh")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "user-declared quiet window" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('suite', ['seed-ab', 'boundaries'])
def test_seed_ab_preview_and_refusal(repo_root, tmp_path, suite):
    script = repo_root / 'examples/benchmarks/run_timing_handoff.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    preview = subprocess.run(['bash', str(script), '--suite', suite, '--dry-run'],
                              cwd=tmp_path, capture_output=True, text=True, check=True)
    assert 'same frozen reference' in preview.stdout
    assert 'controller-step wall' in preview.stdout
    env = dict(os.environ)
    env.pop('GATO_QUIET_WINDOW', None)
    refused = subprocess.run(['bash', str(script), '--suite', suite], cwd=tmp_path,
                              env=env, capture_output=True, text=True)
    assert refused.returncode == 2
    assert 'user-declared quiet window' in refused.stderr
    assert list(tmp_path.iterdir()) == []


def test_call_boundary_refuses_unassigned_window(repo_root, tmp_path):
    env = {k:v for k,v in os.environ.items() if k != 'GATO_QUIET_WINDOW'}
    command = [sys.executable,str(repo_root/'examples/benchmarks/sweep_batch_iiwa_fig8.py'),
               '--record-call-boundary','--out',str(tmp_path/'samples.csv')]
    result = subprocess.run(command,env=env,capture_output=True,text=True)
    assert result.returncode == 2 and 'assigned quiet window' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cpu_baseline_uses_selected_interpreter(repo_root, tmp_path, monkeypatch):
    figures = repo_root/'examples/paper-figures'
    monkeypatch.syspath_prepend(str(figures))
    spec = importlib.util.spec_from_file_location('fig3_cpu_runner', figures/'reproduce_fig3_fair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bench = tmp_path/'benchmark path with spaces'
    (bench/'baselines').mkdir(parents=True)
    (bench/'baselines/sqpcpu_env.sh').write_text('export GATO_CPU_PYTHON=/bin/echo\n')
    monkeypatch.setattr(module.C,'BENCH_DIR',str(bench))
    commands = []
    monkeypatch.setattr(module,'_run',commands.append)
    module.run_bt([8],[2],.1)
    command, = commands
    output = subprocess.check_output([str(arg) for arg in command],text=True)
    assert output.strip() == f'{bench}/baselines/track_iiwa_fig8_bt.py 0.1 2 8 {module.BT_CSV} --robot iiwa14'


@pytest.mark.gpu
def test_call_boundary_check_only_has_no_timing_output(repo_root, tmp_path):
    result = subprocess.run([sys.executable,str(repo_root/'examples/benchmarks/sweep_batch_iiwa_fig8.py'),
        '--N','64','--batches','1,8','--solves','12','--record-call-boundary','--check-only',
        '--out',str(tmp_path/'samples.csv')],capture_output=True,text=True,check=True)
    assert 'bitwise raw/controller trajectory and PCG parity' in result.stdout
    assert 'median_ms' not in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.gpu
def test_mujoco_example_correctness_only(repo_root):
    result = subprocess.run([sys.executable, str(repo_root / "examples/14_mujoco_closed_loop.py"),
                             "--correctness-only", "--periods", "1"],
                            capture_output=True, text=True, check=True)
    assert "budget_s=None" in result.stdout
    assert "step wall time" not in result.stdout
    assert "end-effector error" in result.stdout


def _rotor_example(repo_root):
    pytest.importorskip("pinocchio")
    spec = importlib.util.spec_from_file_location("rotor_example", repo_root / "examples/21_rotor_armature.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rotor_example_rejects_bad_parameter_rows(repo_root, tmp_path):
    """Example 21 matches the CSV to the model by joint name: a joint named
    twice, a missing joint and an unknown joint are errors, never a silent pick."""
    import numpy as np
    import pinocchio as pin
    mod = _rotor_example(repo_root)
    d = repo_root / "examples/tiago_pro_description"
    model = pin.buildModelFromUrdf(str(d / "tiago_pro_right_arm.urdf"))
    good = (d / "tiago_pro_right_arm_parameters.csv").read_text().splitlines()
    names, armature = mod.measured_armature(model, d / "tiago_pro_right_arm_parameters.csv")
    assert names == [model.names[j] for j in range(1, model.njoints)]
    np.testing.assert_array_equal(armature, [.43, .43, .19, .17, .21, .17, .18])
    cases = [
        ("more than once", good + [good[3].rsplit(",", 1)[0] + ",9.9"]),   # joint 3 again, another value
        ("not the model's joints", good[:-1]),
        ("not the model's joints", good[:-1] + [good[-1].replace("arm_right_7_joint", "elbow")]),
    ]
    for match, lines in cases:
        bad = tmp_path / "bad.csv"
        bad.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=match):
            mod.measured_armature(model, bad)


@pytest.mark.gpu
@pytest.mark.slow
def test_rotor_example_closed_loop_numbers(repo_root, tmp_path, monkeypatch):
    """Example 21 end to end on the vendored arm, in its own cache: every
    reported number is finite, the device step agrees with Pinocchio, and the
    plan that knows the rotors tracks closer, with a smaller dynamics defect
    and no velocity excursion, than the plan that does not. One simulated
    reach; the bounds are wide of the recorded run (5.6 against 14.2 mm, defect
    0.2 against 35, velocity excursion 0 against 0.38 rad/s)."""
    import numpy as np
    from gato import _cache
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    _cache.toolchain_identity.cache_clear()
    _cache._native_source_id.cache_clear()
    out = _rotor_example(repo_root).run()
    rotor, rigid = out["with armature"], out["without"]
    for r in (rotor, rigid):
        assert len(r["err"]) == len(r["defect"]) == 100
        for key in ("err", "defect", "over", "tau_rotor", "tau_link"):
            assert np.isfinite(r[key]).all(), key
        assert (r["over"] >= 0).all() and r["over"][0] == 0     # neither run leaves the position limits
    assert np.isfinite(out["device_step_diff"]) and out["device_step_diff"] < 2e-5
    assert rotor["err"][-1] < 0.010 and rotor["err"][-1] < 0.7 * rigid["err"][-1]
    assert rotor["defect"].max() < 1.0 and rigid["defect"].max() > 10 * rotor["defect"].max()
    assert rotor["over"][1] < 0.01 and rigid["over"][1] > 0.1
    # on the first joint the rotor carries most of the planned inertial torque
    assert rotor["tau_rotor"][0] > 2 * rotor["tau_link"][0]
