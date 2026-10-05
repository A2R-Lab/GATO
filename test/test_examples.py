"""Check the introductory entry points without importing or running GPU code."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_intro_example_syntax(path):
    ast.parse(path.read_text(), filename=str(path))


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


@pytest.mark.gpu
def test_call_boundary_check_only_has_no_timing_output(repo_root, tmp_path):
    result = subprocess.run([sys.executable,str(repo_root/'examples/benchmarks/sweep_batch_iiwa_fig8.py'),
        '--N','64','--batches','1,8','--solves','12','--record-call-boundary','--check-only',
        '--out',str(tmp_path/'samples.csv')],capture_output=True,text=True,check=True)
    assert 'bitwise raw/controller trajectory and PCG parity' in result.stdout
    assert 'median_ms' not in result.stdout
    assert list(tmp_path.iterdir()) == []
