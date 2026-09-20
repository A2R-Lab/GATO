"""Distribution-artifact truth gate (plan D10 / Wave 0.5).

Builds the wheel + sdist from a CLEAN export of the tracked tree (git archive)
and asserts the honest contract:
  * the wheel is pure-python (universal tag) and carries NO native solver
    modules — they are CMake/arch/CUDA-specific and must come from a source
    tree build;
  * the sdist is a buildable source tree: CMakeLists.txt, gato/, the bindings
    TU, tools/build.sh and the vendored robot descriptions are inside.
Needs the `build` package (pip install build) and network for the isolated
build env; slow, host-only.
"""
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def artifacts(repo_root, tmp_path_factory):
    pytest.importorskip("build", reason="pip install build")
    src = tmp_path_factory.mktemp("src")
    subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=repo_root,
                   stdout=open(src / "tree.tar", "wb"), check=True)
    with tarfile.open(src / "tree.tar") as t:
        t.extractall(src / "tree")
    out = tmp_path_factory.mktemp("dist")
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", str(out), str(src / "tree")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]
    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, (wheels, sdists)
    return wheels[0], sdists[0]


def test_wheel_is_pure_python_without_native_modules(artifacts):
    whl, _ = artifacts
    assert whl.name.endswith("py3-none-any.whl"), whl.name
    names = zipfile.ZipFile(whl).namelist()
    native = [n for n in names if n.endswith((".so", ".pyd", ".dylib"))]
    assert not native, f"native modules must never ride a universal wheel: {native}"
    assert any(n.endswith("gato/_registry.json") for n in names)
    assert any(n.endswith("gato/interface.py") for n in names)
    assert whl.stat().st_size < 2_000_000, "wheel should be the python layer only"


def test_sdist_is_a_buildable_source_tree(artifacts):
    _, sdist = artifacts
    names = [Path(*Path(n).parts[1:]).as_posix() for n in tarfile.open(sdist).getnames()]
    required = ["CMakeLists.txt", "python/bindings.cu", "tools/build.sh", "tools/regen_grid.py",
                "gato/settings.h", "gato/bsqp/bsqp.cuh", "gato/dynamics/plant.cuh",
                "gato/dynamics/indy7/grid.cuh", "gato/dynamics/iiwa14/grid.cuh", "gato/dynamics/go2/grid.cuh",
                "examples/indy7_description/indy7.urdf", "test/receipt_modules.txt"]
    missing = [r for r in required if r not in names]
    assert not missing, f"sdist cannot build the solver without: {missing}"
    assert not [n for n in names if n.endswith(".so")]
