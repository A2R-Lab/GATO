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

## OSQP / CPU baseline (Indy7) — runs
- **What:** the paper's CPU/OSQP competitor — a pure-Python single-solve SQP (pinocchio +
  `scipy.sparse` + OSQP), the `Thneed` class in the `sqpcpu` submodule
  (`baselines/sqpcpu/pinocchio_template.py`). A genuine CPU *solver* competitor (unlike the
  Pinocchio-sim baseline above, which uses the GPU solver). No C++/pybind build needed.
- **File:** [baselines/run_osqp_fig8.py](../baselines/run_osqp_fig8.py) — drives `Thneed` through the
  same Indy7 figure-8 MPC loop as `examples/benchmark_fig8.py` (same trajectory, dt, sim_dt).
- **Deps:** pinocchio + scipy + osqp. The GATO `.venv` lacks pinocchio; use the GRiD venv
  (`../GRiD/.venv`) — `pip install osqp` into it if missing (the only extra over pinocchio/scipy).
- **Run:** `../GRiD/.venv/bin/python baselines/run_osqp_fig8.py --N 8,16,32,64 --sim-time 5`.
- **Status:** runs. CPU solve time scales cleanly with horizon: **N=8/16/32/64 → 5.7/12.5/32/72
  ms/solve** at `max_qp_iters=5` (real-time budget). Tracking is tight at N≤16 (~0.01–0.02 m); long
  horizons need more SQP iters to converge (N=32 reaches <0.02 m only at ~20 iters / ~135 ms),
  itself an honest illustration of the CPU/GPU gap the figure shows.

## MPCGPU / GBD-PCG baseline (iiwa14) — builds AND runs on sm_120 (ported)
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

**Status — BUILD ✅, RUNTIME ✅ on sm_120 (ported 2026-06-22, branch `fix/sm120-runtime-smem`).**
The illegal-access was two genuine shared-memory sizing bugs (silently absorbed on sm_75/86/89,
fatal on Blackwell's tighter bounds), **not** a cooperative-launch/occupancy issue:
1. **Plant `forwardDynamics` split `s_XITemp` at twice the XImats offset** (`iiwa14_plant.cuh`
   `[1008]`→`[504]`, `indy7_plant.cuh` `[864]`→`[432]`) — pushing `forward_dynamics_inner`'s scratch
   ~500 floats past the `FD_DYNAMIC_SHARED_MEM_COUNT=1444` budget. (Matches each plant's own
   `forwardDynamicsAndGradient` s_vaf offset and the canonical `grid::forward_dynamics_device`.)
   This was the `compute_merit_kernel` crash (merit calls `integratorError`→`forwardDynamics`).
2. **`end_effector_positions_kernel` launched with no dynamic-smem argument** (`mpcsim.cuh:172/216/323`,
   `mpcsim_n.cuh:168/234`) — the kernel uses `extern __shared__` (`EE_POS_SHARED_MEM_COUNT` floats);
   the grid host wrapper passes that size, these launches did not. This was the second crash at
   `mpcsim.cuh:216`.

Both fixes are one-liners (committed on `fix/sm120-runtime-smem` off `adu/multisolve-v1`).
**Verified run** (iiwa14, N=32, PCG, 20 SQP iters): tracking err avg 0.071 m / final 0.0053 m;
**linsys (PCG) solve time median 74–113 µs** (Q1 ~23–35 µs, depends on PCG exit tol), written to
`build/results/32_PCG_*_overall_stats.csv` (rows = tracking-error stats, then linsys-time stats).
These are the GPU-competitor bars for the Fig-3 comparison.
