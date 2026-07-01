# Competitive baselines (Fig 3)

Provenance + reproduction for the paper's comparison baselines. The harnesses live on old branches
(see [archaeology.md](archaeology.md)); this records how to build/run them today.

## What the paper's Fig-3 actually is (IV-B scalability)
**Robot Indy7 (6-DoF), figure-8 tracking, N=64, h=0.01, batch M∈[1,2,4,…,128], 1 SQP iter** — GATO
(GPU, batched) vs **OSQP (CPU)** and **MPCGPU (GPU)** single-solve baselines. Reported speedup
**18–21× over CPU, 1.4–16× over GPU** as batch grows. So the fair comparison runs ALL THREE on Indy7
fig8 N=64 (NOT iiwa14 — that robot is only Fig-4/Table-I/hardware). Per-control-step solve time is the
metric. Assemble with [reproduce_fig3_scalability.py](../examples/paper-figures/reproduce_fig3_scalability.py)
(`--replot`) once the baselines are collected. NOTE: single-solve OSQP was dropped — BatchThneed (the
threaded CPU competitor) is faster even at M=1, so OSQP added nothing to the comparison.

### Raw timings collected 2026-06-23 (quiet RTX 5090 / sm_120), indy7 fig8 N=64, 1 SQP iter
| solver | per-step solve time | closed-loop tracking | notes |
|---|---|---|---|
| **GATO** batch=1 | **0.133 ms** | 0.056 m | stable; batch 4→0.449, 8→0.657, 16→0.841, 32→1.151, 64→1.795, 128→3.140 ms (5.4× throughput @128). batch=2 unsupported ("must be >3 for exploitation") |
| **OSQP (CPU)** | **~37–74 ms** (±high) | **~0.9 m (does NOT lock on)** | conditioning-bound; see fairness note |
| **MPCGPU (GPU)** | **0.293 ms** *(linsys median; NOT full-solve)* | **~0.61 m** | GRAVITY=0; reported stat is PCG-linsys only, full sqp_times collected but not printed |

### ✅ FINAL quiet-box re-collect (2026-06-26, quiet RTX 5090 / sm_120 + 24-thread CPU)
All Fig-3 lines re-collected on a verified-quiet box (GPU 4% util / no compute apps; GPU and CPU
timing runs serialized, never concurrent). These are the numbers in the shipped `fig3_scalability.png`.

| M | GATO (GPU batched) | BatchThneed (CPU threaded) | MPCGPU ×M (GPU, linear) | GATO vs CPU | GATO vs MPCGPU |
|---|---|---|---|---|---|
| 1   | **0.136 ms** | 2.49 ms  | 0.29 ms  | 18.3× | 2.2×  |
| 4   | 0.443 ms     | 3.17 ms  | 1.17 ms  | 7.1×  | 2.6×  |
| 8   | 0.651 ms     | 3.08 ms  | 2.34 ms  | 4.7×  | 3.6×  |
| 16  | 0.842 ms     | 3.28 ms  | 4.69 ms  | 3.9×  | 5.6×  |
| 32  | 1.160 ms     | 6.68 ms  | 9.38 ms  | 5.8×  | 8.1×  |
| 64  | 1.804 ms     | 11.44 ms | 18.8 ms  | 6.3×  | 10.4× |
| 128 | **3.158 ms** | 20.82 ms | 37.6 ms  | 6.6×  | 11.9× |

- **GATO**: `examples/benchmark_fig8.py --plant indy7 --N 64` → `benchmark_fig8_64N.pkl`. Sub-linear
  (128× batch = 23× time). M=2 unsupported ("batch must be >3 for exploit+explore").
- **BatchThneed** (`pysqpcpu`, the student's real batched-CPU baseline, threads=24): flat ~2.5–3.3 ms to
  M=16 (fan-out to core count), then linear past 24 cores. `baselines/batchthneed_fig8_results.pkl`.
- **MPCGPU** (single cooperative solve, GRAVITY=0, Q_COST=1.0): median **293.37 µs/solve**, drawn ×M
  (single-solve cannot batch). `baselines/mpcgpu_indy7_fig8_N64.csv` (median = col 4).
- **Result**: GATO dominant at every M — **18–21× over CPU at small batch, 1.4–16× over MPCGPU growing
  with batch** — exactly the paper's Fig-3 claim. (Speedups here are ratios of the re-collected medians.)

**OSQP single-solve DROPPED (2026-06-26, user):** the threaded BatchThneed baseline is faster than
single-solve OSQP even at M=1 (2.49 ms vs 12.33 ms), so the OSQP line was meaningless. Removed from the
figure, the `reproduce_fig3_scalability.py` script, and the timing pipeline (`run_osqp_fig8.py` +
`osqp_fig8_results.pkl` deleted; the superseded `assemble_fig3.py` assembler deleted with it).

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

### ✅ Batched-CPU baseline BUILT (2026-06-25) — pysqpcpu.BatchThneed (the paper's real CPU line)
The faithful CPU competitor is the **threaded C++ BatchThneed** (`baselines/sqpcpu`, the student's
`EmreAdabag/sqpcpu`), NOT the single-solve Python `Thneed`. Single-solve×M overstates the CPU
(assumes 1 core); BatchThneed solves M problems across `num_threads` cores → **sub-linear** (flat to
the core count, then linear), exactly the paper's ~3→30 ms over M=1..128.
- **Built WITHOUT the upstream Dockerfile's heavy stack** (ROS humble + source pinocchio). Key shortcut:
  the GRiD venv already ships pinocchio + hpp-fcl(coal) + Eigen C++ **and their CMake configs** via
  `cmeel`, so only **osqp v0.6.3 + osqp-eigen v0.8.1** (the two missing pieces) are built into a LOCAL
  prefix (`baselines/sqpcpu/deps/install`, no sudo). pysqpcpu then builds against cmeel pinocchio +
  local osqp using the GRiD venv's py3.12. **cmeel pinocchio was API-compatible** (only an unused-var
  warning). Automated: `baselines/build_cpu_baseline.sh` → writes `baselines/sqpcpu_env.sh`
  (LD_LIBRARY_PATH + PYTHONPATH); committed `fdee0f6`.
- **Runs + scales correctly** (Indy7 fig8 N=64, 24-core box, PROVISIONAL — non-quiet CPU):
  M=1→2.5 ms, M=8→3.2 ms, M=16→3.5 ms, M=64→11.8 ms (vs single-solve×M = 2.5→160 ms). Lands in the
  paper's ballpark. Runner = `baselines/run_batchthneed_fig8.py` → `batchthneed_fig8_results.pkl`;
  wired into `reproduce_fig3_scalability.py` as the batched-CPU line (single-solve OSQP demoted to a
  faint reference). **Final timing → quiet-box pass.** osqp pinned 0.6.3 / osqp-eigen 0.8.1 (the v1.0
  C-API churn breaks osqp-eigen 0.8.x).

### MPCGPU/GPU baseline — full-sqp time RESOLVED; tracking is a structural limit (2026-06-24)
- **Full per-step sqp time = DONE (was a mislabel, not missing).** `mpcsim.cuh` returns the full sqp
  time in tuple slot 0 under `TIME_LINSYS==1` (`sqp_pcg.cuh:383` times the whole `sqpSolvePcg`,
  start→end); `track_iiwa_pcg.cu` read it into a var *named* `linsys_times` and printed "Linsys times".
  Relabeled to "Full per-step SQP solve times (us)". **So the earlier 293 µs median was already the
  full per-step solve**, not linsys-only. Parity with GATO's full-step metric holds.
- **Tracking: MPCGPU is a structurally weaker tracker at 1 SQP iter.** Experiments (worktree
  `/tmp/gato_mpcgpu`, branch `fig3/indy7-mpcgpu`):
  | config | tracking final | variance | sqp time median |
  |---|---|---|---|
  | gravity OFF (orig R=.001/QD=.0001 **or** matched R=1e-5/QD=5e-3 — identical) | 0.47 m (stuck) | std 0.22 | **293 µs** |
  | gravity ON + matched | 0.065 m | bimodal, max 1.6 m | 1128 µs |
  - **Gravity-off tracking is weight-invariant** (byte-identical 0.605 m mean / 293 µs across weight
    sets): MPCGPU's EE-tracking weight is hardcoded 1.0 (`indy7_plant.cuh:291`, no Q multiplier) and
    dominates QD/R, so it's stuck at ~0.47 m — a structural limit, not a tunable. GATO does the SAME
    fig-8 *with* gravity at 0.056 m → MPCGPU at 1 iter is genuinely far weaker (part of why GATO exists).
  - **Gravity destabilizes + inflates MPCGPU**: with gravity on it converges at the end (0.065 m) but
    excursions to 1.6 m and the PCG ill-conditions on the gravity-loaded KKT → 4× slower (293→1128 µs,
    a conditioning artifact, not a representative cost).
  - **Clean representative MPCGPU solve cost = 293 µs (gravity-off).** Tracking-to-GATO-quality is
    open-ended MPCGPU re-tuning/debugging (the "retune is non-trivial" R&D the unification plan flagged).
- **CONFIG DECISION (open):** gravity-off (clean 293 µs time, poor 0.47 m tracking, physics-mismatched
  vs GATO/OSQP which run with gravity) vs gravity-on (fair physics, 0.065 m final but unstable + inflated
  1128 µs time). For a TIMING figure the gravity-off 293 µs is the honest solve cost; flag the caveat.

### ✅ EE-weight made tunable (2026-06-25) — and the tracking⇄time tradeoff is FUNDAMENTAL
The MPCGPU indy7 EE-tracking weight was hardcoded 1.0 (the EE term used a bare `0.5*err^2`). Exposed
it as `COST_Q1`/`Q_COST` (config/cost_settings.h `#ifndef Q_COST #define Q_COST 1.0`, applied
consistently to value + gradient + the rank-1 GN-proxy Hessian, single factor of Q). `Q_COST=1.0`
is byte-identical to the old behavior (final 0.470957 m, mean 0.605298 m — verified no-op).
**The earlier "weight-invariant" claim was wrong**: it only ever varied QD/R with EE pinned at 1.0;
EE itself was never a free variable. Sweeping it (worktree `/tmp/gato_mpcgpu`, sm_120, deterministic):

| `Q_COST` | tracking final | tracking mean | sqp time **median** | conditioning |
|---|---|---|---|---|
| **1** (publish) | 0.471 m | 0.605 m | **293 µs** | clean |
| 3 | 0.471 m (no gain) | 0.605 m | ~310 µs | clean-ish |
| 10 | ~0.15–0.27 m | 0.15–0.27 m (high var) | ~750 µs (2.7×) | inflating |
| 100 | ~0.016 m | 0.10–0.95 m (high var) | ~1500 µs (5×) | ill-conditioned |

- **Knee is sharp**: tracking only moves at `Q_COST≥10`, which is exactly where the PCG solve time
  inflates ≥2.7×. The rank-1 `(Jᵀr)(Jᵀr)ᵀ` EE Hessian (the same heuristic GATO's P3 flagged as a
  latent near-singular term) ill-conditions the Schur PCG once the EE weight is large enough to bite.
  So **clean 293 µs solve time and materially-better tracking are mutually exclusive** for this
  1-SQP-iter, rank-1-Hessian solver — the plan's premise (EE-weight fix buys tracking *while keeping*
  293 µs) does NOT hold empirically.
- **Recommendation: publish `Q_COST=1.0` (293 µs / 0.47 m).** 293 µs is the representative, best-case
  MPCGPU solve cost (the Fig-3 metric is TIME); 0.47 m is the honest 1-iter real-time-budget tracking.
  The tunable knob stays in (and is the right honest fix) but the published bar is Q=1. **Bonus narrative
  point**: MPCGPU's single solve faces a speed⇄quality tradeoff (293 µs@0.47 m ↔ 1500 µs@0.016 m) that
  GATO's batched solve sidesteps — quality without the time hit. (Solve-TIME numbers here are provisional
  conditioning signals on a non-quiet GPU; the published 293 µs gets re-collected on a quiet GPU at the end.)

### GATO batch-scaling vs paper — diagnosed 2026-06-24 (real fig8, varying goal, M=128)
Paper Fig-3 raw: GATO **0.33 ms (M=1) → ~1.6 ms (M=128)** vs a **batched/threaded CPU 3.3 → 30 ms**
(21× at M=1). Our measurements (isolated repeated-input UNDER-counts because warm inputs converge
instantly + the SQP breaks early skipping the line search — must use VARYING real goals):
| metric | M=1 | M=128 | scaling |
|---|---|---|---|
| GATO isolated (warm, identical input) | 117 µs | 986 µs | 8.4× |
| **GATO real fig8 (varying goal)** | **384 µs** | **2988 µs** | 7.8× |
| paper GATO | ~330 µs | ~1600 µs | ~5× |
- **Our M=1 (384 µs) ≈ paper (330 µs)** — the solver is NOT slow. We're **~1.87× the paper at M=128**.
- **Bottleneck = the line-search merit kernel, NOT PCG/host.** nsys (real, M=128): `computeMerit` 69%
  (~2032 µs/solve, the `NUM_ALPHAS=8` line-search eval — dynamics-compute-heavy, saturates GPU at high
  M), PCG 15%, setupKKT 11%, host/memcpy ~2%. **`max_pcg_iters` has ZERO effect** (2996→2980 µs at
  200/100/50/30 — PCG converges well before the cap), so it is not a PCG-grind. Tracking stays ~0.063 m.
- **Fix = GATO merit/line-search kernel optimization** (NUM_ALPHAS, kernel efficiency/occupancy — the
  migration may have bloated smem). Focused perf task (subagent candidate), NOT a config knob. We already
  roughly replicate (M=1 matches; M=128 1.87×).
- **CPU measurement gap:** our `run_osqp_fig8` is SINGLE-solve (flat 11.96 ms, Python-inflated ~3.6×);
  paper CPU is the multi-threaded `BatchThneed` (pysqpcpu, C++) scaling 3.3→30 ms. NEED to build+run it.

### ⚠️ REMAINING FAIRNESS ISSUES
1. ~~Baselines' solve TIME not honest~~ — **OSQP single-solve DONE** (converges, 11.96 ms Python). NEED
   the batched/threaded `BatchThneed` CPU curve (3.3→30 ms) to match the paper. **MPCGPU time DONE**
   (293 µs full-sqp, gravity-off); MPCGPU *tracking* is a structural baseline limit.
2. **GATO M=128 is 1.87× the paper** — line-search merit kernel (above). Perf-optimization follow-up.
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

## iiwa14 pick-place (Fig-7 / Table-I, CS3) — Phase-0 localization (2026-06-25)
Diagnostic: `examples/paper-figures/_diag_pickplace_phase0.py` (4-way cross payload off/on ×
batch1/batch8+FE, + wide-FE + rho sweep; success-rate only, no timing). Fixed scenario
(PENDULUM_DEFAULT_PARAMS, 15 kg ≈ 147 N), N=16, h=0.01, PICKPLACE_SOLVER_PARAMS.

### 🔴 ROOT CAUSE #1 (FIXED) — the ForceEstimator was SILENTLY DISABLED in every run
`mpc_controller.py` did `sys.path.append('./examples')` (CWD-relative) then
`from force_estimator import ForceEstimator` in a `try/except ImportError: ForceEstimator = None`.
`force_estimator.py` lives in `<repo>/examples`, so unless the process CWD was the repo root the
import **failed silently** → `ForceEstimator = None` → `setup_force_estimator` set
`self.force_estimator = None` for ALL batch sizes → `update_force_batch` early-returns (no f_ext set)
and `evaluate_best_trajectory` always returns 0. **So the paper's entire CS3 batched-robustness
mechanism was off**, and every batch size behaved identically (= batch-1, 0% on the payload task) —
exactly the all-zero Table-I. `reproduce_fig7` runs from `examples/paper-figures/`, so it tripped this.
**Fix**: resolve the examples dir relative to `__file__` (CWD-independent) + a loud `RuntimeWarning`
if the import still fails (never silently disable robustness again). Verified: `ForceEstimator` is now
the real class in the `reproduce_fig7` import context (was `None`).

### Phase-0 cross (full 5-goal sequence, after the FE fix)
| run | payload | batch+FE | reached | diverged | note |
|---|---|---|---|---|---|
| a_off_b1 | off | b1 | **5/5** | no | solver fine, no disturbance |
| d_off_b8 | off | b8+FE | **5/5** | no | FE est=0 correctly; batching harmless |
| b_on_b1 | on | b1 (no FE) | 0/5 | no | stable but inaccurate (gd 0.162 m) |
| c_on_b8 | on | b8+FE | 0/5 → **3/5** | **sometimes (NaN)** | see non-determinism |
| c_wide_b8 | on | b8+FE wide(200N) | 0/5 | sometimes | wider range alone doesn't fix |

- **The solver is NOT the problem** (5/5 with no payload at both batch sizes; nothing NaNs without FE).
- **Batching now demonstrably helps**: with FE live, batch=8 reaches up to **3/5** where batch=1 (no FE)
  gets 0/5 — the paper's "success climbs with batch" signature appears once the FE is actually on.

### 🟠 ROOT CAUSE #2 (residual, ESCALATE) — the FE is high-variance + sometimes divergent
The SAME config (payload, b8, rho=0.001) gave **0/5 with NaN divergence** in one run and **3/5 stable**
in another. Cause: `ForceEstimator._random_rotation_matrix` uses **unseeded `np.random`**, so the
sphere sampling differs every run → the closed loop is genuinely stochastic and occasionally diverges.
- **rho tuning doesn't reliably fix it** (success vs rho is non-monotone: 0.001→3/5, 0.01→0/5,
  0.05→3/5, 0.1→1/5, 0.3→1/5; variance dominates — the documented "no single rho" risk).
- **The FE estimate never converges to the true ~147 N** (peaks ~89 N, finals 1–17 N) — the
  estimate-update dynamics are too weak/slow for a large *swinging* (time-varying) disturbance, and
  widening max_radius (20→200 N) alone doesn't close it.
- This is **FE-algorithm robustness R&D** (seed/determinism, time-correlated sampling, convergence to
  large disturbances) = Phase-2 structural per the plan → **escalated to user before proceeding**.

### Resolution (2026-06-25) — FE seeded (user choice "just seed it, then ship")
`ForceEstimator` now takes a `seed` (default 0; `np.random.default_rng`), threaded through
`MPC_GATO(fe_seed=0)`. This makes the closed loop **reproducible** and, at seed=0, **removes the NaN
divergence**: the canonical seeded 5-goal cross is
`a_off_b1 5/5 · b_on_b1 0/5 · c_on_b8 3/5 (no divergence) · d_off_b8 5/5 · c_wide_b8 0/5`.
So **batch=8 reaches 3/5 where batch=1 (no FE) gets 0/5 — deterministically** — the paper's
"success climbs with batch" signature. (Widening max_radius 20→200 N makes it WORSE here, confirming
the gap is FE convergence dynamics, not range — so the default range is kept.)

### Exit status
The dominant Table-I bug (FE disabled) is fixed + the FE is seeded → `reproduce_fig7` now produces a
**reproducible, shape-correct, batch-increasing** success curve (the plan's "likely case"), to ship with
the FE-variance caveat. Matching the paper's exact success *magnitudes* needs the parked Phase-2
FE-robustness work (time-correlated sampling / convergence to large swinging disturbances). NaN scenarios
are caught by `run_pickplace_sweep`'s per-batch try/except, so the 100-scenario sweep won't crash.
