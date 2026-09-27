"""Check the introductory entry points without importing or running GPU code."""
import ast
import os
import subprocess
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
