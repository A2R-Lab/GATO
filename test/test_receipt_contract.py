"""Fail-closed receipt policy and native source identity gates."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_identity', ROOT/'tools/native_identity.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_receipt_scope_and_fail_closed_ci():
    config = tomllib.loads((ROOT/'pyproject.toml').read_text())['tool']['gpu_proof']
    policy = yaml.safe_load((ROOT/'test/gpu-proof-policy.yaml').read_text())
    required = {'python', 'gato', 'tools', 'examples', 'test', 'CMakeLists.txt',
                'external/GRiD', 'external/GLASS', '.github', '.gitmodules', 'pyproject.toml'}
    assert required <= set(config['fingerprint_paths']) == set(policy['required_fingerprint_paths'])
    assert policy['required_test_manifest'] == 'test/expected_tests.txt'
    workflow = (ROOT/'.github/workflows/verify-gpu-proof.yml').read_text()
    assert "steps.receipt.outputs.present" not in workflow


@pytest.mark.parametrize('args,addopts', [(['-k', 'smoke'], ''), ([], '-k smoke')])
def test_receipt_refuses_selection(args, addopts):
    result = subprocess.run(['bash', 'test/run_gpu_proof.sh', *args], cwd=ROOT,
                            env={**os.environ, 'PYTEST_ADDOPTS': addopts}, capture_output=True, text=True)
    assert result.returncode != 0 and 'full suite' in result.stderr


def test_expected_receipt_collection():
    output = subprocess.check_output([sys.executable, '-m', 'pytest', 'test/', '--collect-only', '-q'],
                                     cwd=ROOT, text=True, env={**os.environ, 'PYTEST_ADDOPTS': ''})
    actual = {line for line in output.splitlines() if line.startswith('test/') and '::' in line}
    expected = set((ROOT/'test/expected_tests.txt').read_text().splitlines())
    assert actual == expected


def test_native_identity_is_content_based(tmp_path):
    files = ['CMakeLists.txt', 'python/bindings.cu', 'tools/native_identity.py',
             'gato/constants.h', 'gato/dynamics/plant.cuh', 'gato/bsqp/bsqp.cuh',
             'gato/utils/cuda.cuh', 'gato/dynamics/testbot/grid.cuh',
             'external/GLASS/glass.cuh', 'external/GRiD/grid_codegen/collision/geometry.cuh']
    for name in files:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    original = native.native_identity(tmp_path, 'testbot')
    for name in files:
        path = tmp_path/name
        before = path.stat()
        path.write_text('modified')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert native.native_identity(tmp_path, 'testbot') != original, name
        path.write_text('original')
    assert native.native_identity(tmp_path, 'testbot') == original


@pytest.mark.gpu
def test_receipt_native_modules_match_sources():
    native.verify_receipt_modules(ROOT)
