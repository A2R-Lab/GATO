"""The C++/CUDA tier: `python -m gato.paths` resolves everything a translation
unit needs for a (robot, N, variant), and the standalone example under
examples/cpp configures, builds and runs against the installed package
(gpu/slow: one nvcc compile of the Indy7 plant at N8)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import gato
from gato import paths

REPO = Path(__file__).resolve().parents[1]


def test_paths_resolve_vendored_robot():
    s = paths.settings("indy7", 16)
    assert s["defines"][:3] == ["KNOT_POINTS=16", "GATO_PLANT_NAME=indy7", 'GATO_PLANT_HEADER="dynamics/plant.cuh"']
    inc = [Path(d) for d in s["include_dirs"]]
    assert (inc[0] / "bsqp" / "bsqp.cuh").is_file()            # GATO's header root
    assert (inc[1] / "grid.cuh").is_file() and inc[1].name == "indy7"
    assert (inc[2] / "glass.cuh").is_file()                    # GLASS from the grid-rbd wheel
    assert (inc[3] / "grid_collision_geometry.cuh").is_file()  # GRiD collision primitives
    assert (Path(s["example"]) / "batched_solve.cu").is_file() and (Path(s["example"]) / "CMakeLists.txt").is_file()
    assert paths.settings("indy7", 16, "fc")["defines"][-1] == "GATO_CONTACT_FORCES=1"
    assert paths.settings("indy7", 16, "eh")["defines"][-1] == "USE_EXACT_HESSIAN=1"
    with pytest.raises(ValueError, match="register_robot"):
        paths.settings("nobody_registered_this", 16)


def test_paths_cli_formats():
    out = subprocess.check_output([sys.executable, "-m", "gato.paths", "--robot", "indy7", "--N", "32", "--format", "cmake"],
                                  text=True, cwd=REPO).strip()
    inc, defs, flags = out.split("|")
    assert "KNOT_POINTS=32" in defs.split(",") and "-use_fast_math" in flags.split(",")
    assert all(Path(d).is_dir() for d in inc.split(","))
    nvcc = subprocess.check_output([sys.executable, "-m", "gato.paths", "--robot", "indy7", "--N", "32", "--format", "nvcc"],
                                   text=True, cwd=REPO)
    assert "-DKNOT_POINTS=32" in nvcc and "-I" in nvcc
    js = json.loads(subprocess.check_output([sys.executable, "-m", "gato.paths", "--robot", "indy7", "--N", "32"],
                                            text=True, cwd=REPO))
    assert js["N"] == 32 and js["robot"] == "indy7"


@pytest.mark.gpu
@pytest.mark.slow
def test_standalone_cpp_example_builds_and_runs(tmp_path):
    if shutil.which("nvcc") is None or shutil.which("cmake") is None:
        pytest.fail("the C++ example gate needs nvcc and cmake")
    build = tmp_path / "build"
    env = {**os.environ, "CUDAARCHS": os.environ.get("CUDAARCHS", "native")}
    subprocess.run(["cmake", "-S", str(REPO / "examples" / "cpp"), "-B", str(build), "-DGATO_ROBOT=indy7", "-DGATO_N=8",
                    f"-DPython3_EXECUTABLE={sys.executable}"], check=True, env=env, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build)], check=True, env=env, capture_output=True, text=True)
    out = subprocess.run([str(build / "batched_solve")], check=True, capture_output=True, text=True).stdout
    assert "plant indy7, N=8, B=8" in out and "trajectories finite: yes" in out, out
