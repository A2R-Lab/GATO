# Competitive baselines (Fig 3)

Provenance + reproduction for the paper's comparison baselines. The harnesses live on old branches
(see [archaeology.md](archaeology.md)); this records how to build/run them today.

## What the paper's Fig-3 actually is (IV-B scalability)
**Robot Indy7 (6-DoF), figure-8 tracking, N=64, h=0.01, batch M∈[1,2,4,…,128], 1 SQP iter** — GATO
(GPU, batched) vs **OSQP (CPU)** and **MPCGPU (GPU)** single-solve baselines. Reported speedup
**18–21× over CPU, 1.4–16× over GPU** as batch grows. So the fair comparison runs ALL THREE on Indy7
fig8 N=64 (NOT iiwa14 — that robot is only Fig-4/Table-I/hardware). Per-control-step solve time is the
metric. Assemble with [baselines/assemble_fig3.py](../baselines/assemble_fig3.py) once all three are
collected.

### Raw timings collected 2026-06-23 (quiet RTX 5090 / sm_120), indy7 fig8 N=64, 1 SQP iter
| solver | per-step solve time | closed-loop tracking | notes |
|---|---|---|---|
| **GATO** batch=1 | **0.133 ms** | 0.056 m | stable; batch 4→0.449, 8→0.657, 16→0.841, 32→1.151, 64→1.795, 128→3.140 ms (5.4× throughput @128). batch=2 unsupported ("must be >3 for exploitation") |
| **OSQP (CPU)** | **~37–74 ms** (±high) | **~0.9 m (does NOT lock on)** | conditioning-bound; see fairness note |
| **MPCGPU (GPU)** | **0.293 ms** *(linsys median; NOT full-solve)* | **~0.61 m** | GRAVITY=0; reported stat is PCG-linsys only, full sqp_times collected but not printed |

### ✅ VERIFICATION #1 (2026-06-24): the tracking gap is REAL, not a measurement artifact
Driving GATO batch=1 under the OSQP baseline's *exact* conditions — fixed-`dt` pacing (not real-time
solve-time pacing), knot-0 goal metric, identical 500 control steps, same robot/traj/N=64/dt/`rk4`
rollout — GATO tracks **0.0593 m** (vs 0.0600 m under real-time pacing; pacing & metric-alignment are
negligible). OSQP is ~0.9 m under the same harness. So the ~15× gap is **genuine solver quality**
(trapezoidal integrator + adaptive rho 1e-3→10 + the batched solve), NOT harness/pacing/metric
unfairness. Implication: the baselines need not match GATO's *tracking* — but their solve *TIME* must be
an honest cost (issues 1–3 below are about the TIME, now that the quality gap is confirmed real).
Repro: `run_mpc_fig8(..., pace_by_solve_time=False)` reports `goal_distances_knot0`.

### ✅ RESOLVED 2026-06-24: OSQP/CPU baseline now CONVERGES at 1 SQP iter (root cause = mismatched weights)
The earlier "~0.9 m, never locks on" was **NOT a solver-quality gap** — `Thneed`'s default cost weights
(Q_cost=100/dQ_cost=0.01, ratio 10000) weight EE-tracking ~50× more aggressively than GATO's indy7 fig8
weights (q_cost=2/qd_cost=0.01, ratio 200), so the closed loop commanded violent torques (|u|≈120–180 Nm),
joint velocities exploded (|dq|→134), the line search rejected steps (α→0), and it diverged. **Fix
(fairness-correct): use GATO's OWN weights** — `run_osqp_fig8.py` now imports `DEFAULT_SOLVER_PARAMS`
(q_cost/qd_cost/u_cost/N_cost/q_lim_cost/rho) and passes them to `Thneed`, so the CPU baseline solves the
*identical weighted problem*. Result at **1 SQP iter** (matching GATO): **tracking 0.090 m** (final ~0.056 m,
matching GATO's 0.059 m), **11.96 ± 1.86 ms/solve** (stable; was 37–74 ±46). So GATO's win is **speed, not
quality** — exactly the Fig-3 point. (Decomposition of the 11.96 ms full `sqp()`: pinocchio matrix assembly
~2.5 ms + `osqp.solve()` ~0.7 ms [25 ADMM iters, well-conditioned] + Python line-search ~5.6 ms; the line
search is interpreter-bound, so a C++ impl would be faster — note when citing the CPU bar.)

### ⚠️ REMAINING FAIRNESS ISSUES
1. ~~Baselines' solve TIME not honest~~ — **OSQP DONE** (above). MPCGPU still open (issues below).
2. **OSQP is conditioning-bound, not just "slow CPU".** Thneed's EE-pos Hessian is the rank-1
   `Q·(Jᵀr)(Jᵀr)ᵀ` outer product (`pinocchio_template.py:257`) — nearly singular. At 1 SQP iter with
   OSQP's default `sigma`=1e-6 the KKT goes indefinite ("not quasidefinite" → NaN). Adding `sigma`=0.01
   (matching GATO's rho; now threaded through `Thneed`/`run_osqp_fig8 --sigma`) removes the NaN but
   tracking is still ~0.9 m and time is high-variance (OSQP grinds many internal ADMM iters). Result:
   ~280–550× "speedup" vs the paper's 18–21× — i.e. our OSQP is mis-tuned, not a fair CPU bar.
   Integrator-consistency (euler vs rk4 rollout) does NOT fix it — error grows either way.
3. **MPCGPU stat is linsys-only.** 0.293 ms is the PCG solve, not the full SQP step (GATO's 0.133 ms
   IS full-step). `mpcsim.cuh` collects `sqp_times` too (line 281) — need to print/return those (or
   build TIME_LINSYS=0) for a parity metric. Also GRAVITY=0 on the indy7 port.

**Until these are resolved the three numbers are recorded but NOT assembled into a published Fig-3.**
Raw artifacts: `baselines/benchmark_fig8_64N.pkl` (GATO), `baselines/osqp_fig8_results.pkl` (OSQP),
`baselines/mpcgpu_indy7_fig8_N64.csv` (MPCGPU, linsys).

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

## MPCGPU / GBD-PCG baseline — builds AND runs on sm_120 (ported), iiwa14 AND indy7
The cited GPU competitor ([arXiv:2309.08079](https://arxiv.org/abs/2309.08079)). Build it from its
**frozen pins** (do NOT point it at the migrated GRiD): canonical source `origin/adu/multisolve-v1`
(submodules: MPCGPU `0efde8c`, GRiD `032ed027`, GBD-PCG `0b4bd64`, GLASS `90a7a21`, qdldl `12dbdf0`).

**For Fig-3 (indy7 fig8 N=64) — branch `fig3/indy7-mpcgpu`** (off the sm_120-fix branch). MPCGPU ships
iiwa14-only (`track_iiwa_pcg.cu`; `rbd_plant.cuh` has indy7 commented). The port (commit `94fdbe3`):
switch `rbd_plant.cuh`→indy7; `gato.cuh` `KNOT_POINTS=64`, `TIMESTEP=0.01`; add iiwa14-era constant
aliases `EE_POS_SHARED_MEM_COUNT`/`DEE_POS_SHARED_MEM_COUNT` to `indy7_grid.cuh` (indy7 grid names them
`*_DYNAMIC_*`); `indy7_plant.cuh` `#include "cost_settings.h"` (was missing `settings.cuh`) + rename
`trackingcost`→`trackingCost` (merit kernel needs the capital-C value fn); `SQP_MAX_ITER=1` (match
GATO fig8); `mpcsim.cuh` returns full per-step `sqp_times` (not just linsys) for a fair
per-control-step metric. The 0_0 trajfiles are the GATO `figure8()` indy7 ref (regen:
`data/trajfiles/gen_indy7_fig8.py`; iiwa14 originals in `*.iiwa14.bak`). Build OK (links clean,
sm_120). **CAVEAT:** indy7 plant `GRAVITY()=0.0` (self-consistent between MPCGPU's solver+sim, doesn't
affect solve TIME — the Fig-3 metric); tracking is loose at 1 SQP iter (expected — that's the
real-time budget; GATO gets quality from batching, not iters). **Timing run HELD until GPU quiet.**

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
