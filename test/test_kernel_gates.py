"""The five standalone single-block kernel harnesses under test/cuda/, built
and run from pytest (plan 2.D — they used to be orphaned: documented nvcc lines,
never executed for the receipt). Each harness exits non-zero on any FAIL.

-DNDEBUG is required (gpuAssert ODR: CLAUDE.md) and -arch=native is REQUIRED
for the real-kernel harnesses (default-arch PTX JIT miscompiled GATO's bdsv
kernels on CUDA 13.2 / RTX 5090 — see the bdsv_factor_solve.cu header). No
fast-math: the tolerances are meaningful. ~30-60 s of nvcc per harness -> slow.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMMON = ["-std=c++17", "-O2", "-DNDEBUG", "-arch=native", "-I", str(REPO / "external" / "GLASS")]
HARNESSES = {
    "pcg_vs_cpu": [],
    "bdsv_vs_pcg": [],
    "gamma_parity": ["-DKNOT_POINTS=16", "-DGATO_PLANT_HEADER=\"plant_shim.cuh\"",
                     "-I", str(REPO / "gato"), "-I", str(HERE / "cuda")],
    "bdsv_factor_solve": ["-DKNOT_POINTS=64", "-DGATO_PLANT_HEADER=\"plant_shim.cuh\"",
                          "-I", str(REPO / "gato"), "-I", str(HERE / "cuda")],
    "ee_rows": ["-DKNOT_POINTS=8", "-DGATO_PLANT_HEADER=\"dynamics/plant.cuh\"",
                "-I", str(REPO / "gato"), "-I", str(REPO / "gato" / "dynamics" / "indy7"),
                "-I", str(REPO / "external" / "GRiD" / "grid_codegen" / "collision"), "-I", str(HERE / "cuda")],
}


@pytest.mark.parametrize("name", sorted(HARNESSES))
def test_kernel_gate(name, tmp_path):
    if shutil.which("nvcc") is None:
        pytest.fail("nvcc not on PATH — the kernel gates need the CUDA toolkit")
    exe = tmp_path / name
    src = HERE / "cuda" / f"{name}.cu"
    build = subprocess.run(["nvcc", *COMMON, *HARNESSES[name], str(src), "-o", str(exe)],
                           capture_output=True, text=True, cwd=REPO)
    assert build.returncode == 0, f"{name} failed to compile:\n{build.stderr[-4000:]}"
    run = subprocess.run([str(exe)], capture_output=True, text=True, cwd=tmp_path, timeout=600)
    tail = run.stdout[-3000:] + run.stderr[-1000:]
    assert run.returncode == 0, f"{name} reported FAIL (rc={run.returncode}):\n{tail}"
    assert "FAIL" not in run.stdout, tail
