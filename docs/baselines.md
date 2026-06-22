# Competitive baselines (Fig 3)

Provenance + reproduction for the paper's comparison baselines. The harnesses live on old branches
(see [archaeology.md](archaeology.md)); this records how to build/run them today.

## CPU / Pinocchio-sim MPC baseline (Indy7) — ported, runs
- **What:** drives the GATO BSQP solver in closed loop with a Pinocchio RK4 simulator + Pinocchio FK
  (a Pinocchio-physics MPC baseline — *not* a CPU-solver competitor; it uses the GPU solver).
- **File:** [examples/benchmark_pinocchio.py](../examples/benchmark_pinocchio.py) (ported from
  `origin/a2rlab03:benchmark_pinocchio.py`: fixed URDF path, dropped dead `f_ext_B_std` /
  `f_ext_resample_std` ctor kwargs, added `plant_type='indy7'`). Goal set
  `examples/points1000.npy` recovered from `origin/a2rlab03`.
- **Run:** build an Indy7 module (e.g. `KNOTS=64 PLANT=indy7`), then
  `python examples/benchmark_pinocchio.py` from the repo root (writes `data/benchmark_stats*`).
- **Status:** construction-validated (solver + model build, goal set loads). Runs on Indy7 (the
  migrated Indy7 dynamics are validated). iiwa14 is blocked by the FD-NaN bug (see archaeology).

## MPCGPU / GBD-PCG baseline (iiwa14) — builds on sm_120, runtime port pending
The cited GPU competitor ([arXiv:2309.08079](https://arxiv.org/abs/2309.08079)). Build it from its
**frozen pins** (do NOT point it at the migrated GRiD): canonical source `origin/adu/multisolve-v1`
(submodules: MPCGPU `0efde8c`, GRiD `032ed027`, GBD-PCG `0b4bd64`, GLASS `90a7a21`, qdldl `12dbdf0`).

**Reproduce the build (verified 2026-06-21 on RTX 5090 / sm_120 / CUDA 13.2):**
```bash
git worktree add --detach /tmp/gato_mpcgpu origin/adu/multisolve-v1
cd /tmp/gato_mpcgpu && git submodule update --init --recursive
```
Then apply three compat patches (the frozen code targets sm_75/86/89 + CUDA 12.2):
1. `CMakeLists.txt`: `CMAKE_CUDA_ARCHITECTURES 75 86 89` → `120`; hardcoded
   `CMAKE_CUDA_COMPILER .../cuda-12.2/bin/nvcc` → `/usr/local/cuda/bin/nvcc`; strip `-G` (device
   debug — also wrecks timing) from `CMAKE_CUDA_FLAGS` and the flags list.
2. `gato/utils/experiment.cuh`: stub `prop.memoryClockRate` and `prop.deviceOverlap` (both removed
   from `cudaDeviceProp` in CUDA 13) — cosmetic device-info prints only.
3. Build: `cmake -S . -B build_mpcgpu -DCMAKE_BUILD_TYPE=Release && cmake --build build_mpcgpu --target MPCGPU-pcg -j4`.

**Run:** `mkdir -p build/results && ./build_mpcgpu/MPCGPU-pcg` from the worktree root (reads
`data/trajfiles/{start}_{goal}_{eepos.traj,traj.csv}`, writes `build/results/<prefix>_overall_stats.csv`:
Average/Std/Min/Max/Median/Q1/Q3 solve times). Config: iiwa14, N=32, PCG, 20 SQP iters, cooperative launch.

**Status — BUILD ✅, RUNTIME ⛔ on sm_120.** The binary builds clean and launches, but
`compute_merit_kernel` throws `cudaErrorIllegalAddress` (`gato/solvers/sqp/sqp_pcg.cuh:190` catches
the launch at :183) on Blackwell. This is an in-kernel out-of-bounds specific to sm_120 (the 2024
code predates Blackwell), **not** a build error — so collecting MPCGPU timing needs a focused sm_120
runtime port (e.g. cooperative-launch occupancy / SM-count assumptions, or smem opt-in). This is the
"clean up MPCGPU" follow-up; the build recipe above is the starting point. (The old GRiD `032ed027`
iiwa14 dynamics themselves are fine — the old GATO `sim_forward` is finite — so the crash is in the
MPCGPU solver kernels, not the dynamics.)
