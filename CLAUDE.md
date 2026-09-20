# CLAUDE.md — orientation for AI agents (and humans) working on GATO

GATO is a **GPU-accelerated batched trajectory optimizer**: it solves tens-to-low-hundreds
of optimal-control problems simultaneously in real time, one CUDA block per solve. It is the
top layer of the A2R-Lab stack — it owns **only the SQP / batch / Schur structure** and defers
everything else to its submodules:

- **dynamics, integrators, and plant costs → [GRiD](https://github.com/A2R-Lab/GRiD)** via the
  generated `grid.cuh` and the `grid_plant::` cost/step surface.
- **all block-wide linear algebra → [GLASS](https://github.com/A2R-Lab/GLASS)** (`glass::gemm`,
  `glass::pcg`, `glass::invertMatrix`, …).

So GATO carries no hand-rolled dynamics, integrators, or BLAS — if you find yourself writing any,
stop and check whether GRiD/GLASS already provide it (and if a *generic* primitive is missing,
add it upstream rather than locally — that is how `glass::invertMatrix` fused overloads and
`grid_plant::tracking_cost` came to be).

## The mental model

**One CUDA block per problem instance in the batch.** A batched solve launches a grid of
`(work_dim, batch_size)` blocks; each block runs the single-block algorithm over data already in
shared/global memory. The batch size is **runtime** (a `BSQP` constructor argument — buffers are
sized to it, kernels read it from `gridDim.y`); modules are compiled per `(plant, knot_points)`
only, and any batch size works. Block-scoped kernels must be **thread-count invariant** (strided
`for (i = rank; i < n; i += blockDim)` loops, barriers between a write phase and a dependent read).

## The BSQP solve pipeline (one SQP iteration)

Each `(plant, N)` module is a pybind extension `bsqpN{N}_{plant}`. One SQP iteration runs these
kernels (all in `gato/bsqp/kernels/`):

1. **`setup_kkt.cuh`** — per knot: linearize dynamics (`grid_plant::compute_linearized_dynamics`
   → A,B,c) + build the cost gradient/Hessian blocks (`grid_plant::tracking_cost_grad_hess` →
   Q,q,R,r). Terminal block uses `x_{k+1}` + `N_cost` (PR #17 fix).
2. **`schur_linsys.cuh`** — form the Schur complement system (`form_schur_system_batched_kernel1/2`):
   invert Q_k/Q_kp1/R_k (`glass::invertMatrix` fused), build S (block-tridiagonal) + Pinv + gamma.
3. **`pcg.cuh`** — solve `S·λ = γ` with `glass::pcg<T, STATE_SIZE, KNOT_POINTS>` (block-tridiagonal
   preconditioned conjugate gradient; row-major `[L|D|R]` strips + padded `(KP+2)·STATE_SIZE` vecs).
   Alternative: **`bdsv.cuh`** — direct block-Cholesky (`glass::bdsv`) on the same buffers; GATO
   stores the NEGATED Schur complement, so the bdsv kernel solves `(−S)λ = (−γ)` (negates in place —
   safe, formSchur rewrites every slot next iteration). Host-side `set_linsys_mode` picks per solve
   (0 = pcg default/bit-identical, 1 = bdsv, 2 = bdsv on SQP iter 0 then pcg); python
   `SolverParams(linsys=...)`, controller `linsys="auto"` switches on `‖x_meas − x_pred‖`. Plan + gates:
   `docs/open-tasks/hybrid_pcg_bdsv_plan_2026-07-07.md`.
   Optional exact-Hessian (SO-SQP) path: built with `-DGATO_EXACT_HESSIAN=ON` and toggled per
   solver (`SolverParams(exact_hessian=True)` / `set_exact_hessian`, needs the `_eh` variant module), setup_kkt assembles the
   (nx+nu)² stage block, adds the lagged-λ `λᵀ∇²F` contraction (grid `fdsva_so`), and
   PSD-projects it (`glass::psd_project`, eps = 1e-5·(1+max|diag|)). Default builds are
   preprocessor-identical (bitwise). Forces bdsv; needs rho ≥ 1e-4 (f32). Verdict + traps:
   `docs/open-tasks/so_sqp_device/RESULTS_2026-07-30.md`.
4. **`schur_linsys.cuh::compute_dz_batched_kernel`** — recover the primal step `dz` from λ.
5. **`merit.cuh` / `line_search.cuh`** — evaluate the merit (`grid_plant::tracking_cost_value` +
   `compute_integrator_error`), pick a step.
6. **`sim.cuh`** — roll the chosen control forward (`grid_plant::sim_step`).

External wrenches are a per-(solve, **knot**) band since 2026-08-01 (`d_f_ext_batch_`,
body-major 6·NUM_BODIES per knot; wrench k applies to dynamics interval [k, k+1],
`sim_forward` uses knot 0). `set_f_ext_B` accepts the historic per-solve shapes
(broadcast over knots, bit-identical to the old behavior) and per-knot `(B, N, ...)`
arrays. CL-3 prep: `kernels/contact_debug.cuh` + `debug_contact_dynamics` expose the
contact-frame chain (f_c → f_ext, dqdd/df_c B-block columns, the dfext/dq A-block
chain correction), FD-gated in `test/test_f_ext.py`.

**Contact-force builds (CL-3a)**: `-DGATO_CONTACT_FORCES=ON` grows the control to
`CONTROL_SIZE = ACTUATED_SIZE + FC_SIZE` (FC_SIZE = 6·NUM_CONTACT_FRAMES wrench slots
[n; f], world-aligned at the baked contact frame); the fc tail feeds `f_ext_body` into
the dynamics, the B-block gains the dqdd/dfc columns, and the A-block carries the
**dfext/dq chain term** (`dqdd_dfext · dfext_dq`) — f_c is world-aligned, so the mapped
wrench rotates with the arm and dropping the term leaves an inexact Jacobian (it is
~1.5× the whole fixed-f_ext gradient at 1 N and ~14× at 10 N on iiwa14; linear in f_c, so
zero-wrench trajectories are bitwise unchanged). Default builds are
preprocessor-identical (bitwise). fc is a module VARIANT (`bsqpN{N}_{plant}_fc.so`,
built side by side with the defaults: `-DMODULES="iiwa14:16:fc"` / `--variant fc` /
`gato.build(..., contact_forces=True)`; loaded with `BSQP(..., variant="fc")`; no
.so-swap). Python reads the module attrs CONTROL_SIZE/
ACTUATED_SIZE/FC_SIZE: `SolveResult.u0/control_at` return ACTUATED control only,
`fc_at/fc_traj` the wrench slots; `add_fc_box` pins/caps fc slots via LIN_U rows.
⚠ fc_cost defaults to 1e-2 on fc builds — 0 makes fc a free wrench actuator and the
solve destabilizes (pin with fc_cost≈1e6 or box rows instead). ⚠ Cross-build
comparisons must not add AL rows to only one arm: ANY AL group freezes trust-region
adaptation and legitimately changes the trajectory.

`gato/utils/linalg.cuh` keeps only GATO-specific helpers: `block::reduce` (kept — several
consumers), the `getOffset*` batch-layout accessors, and printers. All other `block::` linalg was
migrated to `glass::`.

## Source layout

- `gato/bsqp/` — the BSQP solver: `bsqp.cuh` (host orchestration), `rowgroups.cuh` (the
  constraint row-group layer: box / EE / LIN_U / collision rows, barrier / ADMM / AL folds),
  `kernels/{setup_kkt, schur_linsys, pcg, bdsv, merit, line_search, sim, admm, contact_debug}.cuh`.
  Every kernel's shared-memory layout is one `*Smem` struct read by both the host sizer and the
  device carve. Naming: snake_case for every hand-written device/host function.
- `gato/dynamics/` — `plant.cuh` (ONE shared robot-agnostic adapter: dimension aliases from
  `grid::` constants, dynamics + tracking-cost wrappers over `grid_plant::`), `integrator.cuh`,
  `grid_plant_step.cuh` (floating-base step/linearization via the emitted arena carves),
  `manifold.cuh` (SE(3) retract/difference) + per-robot `<name>/{grid.cuh, limits.cuh}` (both
  generated; `indy7`, `iiwa14`, `go2`). CMake injects `-DGATO_PLANT_HEADER="dynamics/plant.cuh"`
  and puts the robot dir on the include path. Regen emits only the consumed GRiD families
  (`gato.builder.GATO_ALGORITHMS`).
- `gato/utils/` — `linalg.cuh` (batch-layout accessors), `cuda.cuh` (error macros).
- `gato/{constants.h, settings.h, types.cuh}` — dims, build flags, KKT/Schur structs.
- `python/gato/` — the Python package (`import gato`):
  - `config.py` — `SolverParams` (THE solver configuration) + robot start configs.
  - `interface.py` — `BSQP` (stateless `solve(x, ref, xu_warm)` → `SolveResult`/`SolverStats`),
    `available(variant)`, `module_name`, `robot_info()`.
  - `controller.py` — task-agnostic `MPCController` (owns the warm start; shift/hold, linsys
    "auto", reseed, hypothesis hooks); `StepResult.u` is the ACTUATED control.
  - `hypotheses.py` / `estimators.py` — `HypothesisBatch` ABC, `ForceHypothesisBatch`,
    `ForceEstimator`/`CEMForceEstimator` (batch-as-identity API).
  - `policy.py` / `envs.py` — `MPCPolicy` + reference providers (numpy-only) / gymnasium
    `ArmTrackEnv` (lazy import).
  - `worlds.py` — `PinocchioWorld` / `MuJoCoWorld` simulators; `certificate.py` (KKT certificate);
    `linsys_autotune.py` (resolve_linsys + the tuned-entry table); `fingerprint.py`.
  - `mpc_gato.py` — `MPC_GATO`, the closed-loop SIM DRIVER (`run_mpc_fig8`, `run_mpc_goals`).
  - `builder.py` — `gato.build(urdf, ..., floating_base=, contact_frames=, contact_forces=,
    exact_hessian=)` codegen+compile; `_registry.json` (tracked) holds robot metadata.
  - `common.py` — figure8 / rk4 / manifold helpers.
- `examples/` — intro demos `01..08` + `explore.ipynb` + robot URDFs; `paper-figures/`
  (`reproduce_fig*.py`, `_common.py`, `_pickplace_runner.py`); `benchmarks/` (`_bench.py` shared
  helpers, `iiwa_fig8_shared.py`, `constraint_eval.py`, `_linsys_probe.py`/`linsys_auto_cdf.py`,
  `run_timing_night.sh`, `baselines/` incl. the opt-in `sqpcpu` submodule); `contact-task/`
  (the wipe task); `archive/` (superseded June-era scripts).
- `test/` — pytest suite (`gpu`/`slow` markers; see Validation): goldens in `test/golden/`,
  the receipt module set in `test/receipt_modules.txt`, kernel harnesses in `test/cuda/` (run by
  `test_kernel_gates.py`), `conftest.py` shared fixtures (TEST_PARAMS, go2 helpers).
- `external/GRiD` (pinned to `modernizing-tests`), `external/GLASS` (pinned to `main`).
- `tools/` — `regen_grid.py` (CLI over `gato.builder.codegen`), `build.sh` (`--profile receipt`,
  `--variant fc|eh`), `install.sh` (`--test --dev`), `autotune_linsys.py`, `gen_dynamics_fingerprint.py`.

## Build & run

Modules are built by CMake as a `(PLANT × KNOTS)` matrix, or an explicit `-DMODULES="indy7:8,16;go2:16"`
list (go2 is N16-ONLY — never cross it with the arm horizons), or `-DGATO_RECEIPT_PROFILE=ON` =
exactly `test/receipt_modules.txt` (what `run_gpu_proof.sh` preflights) → `bsqpN{knot}_{plant}.so`:

```bash
cmake -S . -B build -DKNOTS=64 -DPLANT=indy7 -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python \
      -Dpybind11_DIR=$($PWD/.venv/bin/python -m pybind11 --cmakedir) \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel 2      # cap jobs: each TU pulls the large grid.cuh (go2 ~7 GB)
```

`tools/install.sh` sets up a project-local `.venv` + submodules + regen. `KNOTS`/`PLANT` accept
semicolon lists. The project `.venv` is the ONLY python for this repo: `./tools/install.sh --test --dev`
gives it pinocchio + mujoco + scipy + gymnasium + pytest-gpu-proof, which is what `test/run_gpu_proof.sh`
requires (it refuses a python missing them — a skip-laden receipt fails CI). Do NOT sign receipts
from other repos' venvs.

New robots: `gato.build("robot.urdf", name=..., N=[32], ee_frame="EE", floating_base=False,
contact_frames=None, contact_forces=False, exact_hessian=False)` runs codegen (grid.cuh +
limits.cuh + registry; skipped when the inputs are unchanged) and compiles the modules in one
call (serial chains, fixed OR floating base, bounded `<limit>` tags; `ee_frame` must be a URDF
fixed joint). NOTE: it reconfigures the `build/` tree for its own MODULES request — re-run your
usual configure (`./tools/build.sh --profile receipt`) afterwards.

Constraint-layer rulings/provenance (moved out of the docstrings): [`docs/constraints.md`](docs/constraints.md).

**External consumers start at [`docs/consumer_contract.md`](docs/consumer_contract.md)** —
the conventions cross-repo integrations depend on (EE frame, JOINT_LIMIT_MARGIN, cost
semantics, xu layout, receipt scope) + the **dynamics fingerprint**
(`test/dynamics_fingerprint.json` + `gato.fingerprint.check`): verify any external
simulator is the same robot as our model in one screen of code (per-joint
inertia-response ratios; regenerate with `tools/gen_dynamics_fingerprint.py` whenever
the URDFs or dynamics semantics change — the pytest gate enforces staleness).

## Conventions & gotchas (learned the hard way)

- **Two GLASS namespaces coexist.** `grid.cuh` inlines an older GLASS into **`grid::glass::`**;
  the top-level `external/GLASS` (`#include "glass.cuh"`) is global **`glass::`**. They do not
  clash — use `glass::` in the bsqp kernels.
- **`gpuAssert`/`gpuErrchk` are defined in BOTH `grid.cuh` (unconditional) and `utils/cuda.cuh`
  (`#ifndef NDEBUG`).** Release builds (`-DNDEBUG`) are clean; `cuda.cuh` is guarded with
  `#ifndef gpuErrchk` to defer to grid's copy. A non-NDEBUG standalone TU that includes both will
  ODR-clash — compile such harnesses with `-DNDEBUG`.
- **No file-scope `using namespace gato::plant`** in kernel headers — they share one TU
  (`bsqp.cuh` includes them all), so a plant directive leaks `plant::` symbols across headers (it
  caused an `EE_POS_SIZE` ambiguity). Qualify plant calls explicitly.
- **`glass::` L1 ops take `T*` (not `const T*`)** — `const_cast` device-const inputs at the call.
- **Schur matrix inversion uses the augmented `[A | I]` convention** (each buffer is `2·dim·dim`;
  `loadIdentity` sets the right half; the inverse lands there). `glass::invertMatrix` (single +
  fused 2-/3-matrix) all follow it.
- **The real-time MPC loop is wall-clock paced** — `run_mpc_fig8`/`run_mpc_goals` do a
  *timing-dependent* number of solves, so step counts vary run-to-run while the per-solve result is
  deterministic. Do not mistake step-count variation for non-determinism; under GPU contention the
  loop does fewer solves (can look like an "early stop"). Regression-gate with **fixed pacing**
  (`pace_by_solve_time=False`) — those runs are bit-deterministic.
- **pybind classes must be `py::module_local()`** — every `bsqpN*_*` module defines the same C++
  `PyBSQP<T>` type; a global registration makes the SECOND module imported in a process fail with
  "type already registered".
- **`gato.build` is a lazy FUNCTION export from `gato/builder.py`** — the module is deliberately
  not named `build.py`: `from gato.build import ...` would import the submodule and permanently
  shadow the `gato.build()` callable on the package (PEP 562).
- **Merit is deterministic since 2026-08-01** (two-pass fixed-order reduction; the old atomicAdd
  accumulation had ±1 ulp schedule jitter that could flip line-search TIES and break trajectory
  bit-determinism — it did, intermittently, after any SASS-shifting recompile). Everything is now
  bit-exact run-to-run: `final_merit`, `xu`, iteration counts, PCG counts. Numbers from pre-fix
  binaries may differ by ulps (the atomic sum order was unspecified) — re-baseline, don't mix.
- **GATO is the unique dual-namespace GLASS consumer** — the same TU holds the vendored
  `grid::glass` (inside grid.cuh) AND external `glass::`. Any *preprocessor guard* in a GLASS
  base header breaks this (namespace-blind: whichever copy is included first claims the macro
  and the other namespace loses the block — the 2026-07-05 tile4-helpers parse error). Fixed
  upstream: GCG namespaces vendored `GLASS_*` guards as `GRID_VENDORED_GLASS_*`. If a future
  GLASS bump reintroduces a bare guard, the demo `examples/bsqp.cu` TU is the canary.

## Validation

`test/` is the pytest suite — markers select the tier:

```bash
pytest -m "not gpu"           # host-only: packaging, math, codegen-vs-vendored determinism
pytest -m "gpu and not slow"  # smoke solves, bit-determinism, shape validation, controller math
pytest                        # + slow: codegen diff both robots, gato.build dogfood
```

**GPU CI = pytest-gpu-proof** (PyPI package, in the `[dev]` extra): run
`./test/run_gpu_proof.sh` on the GPU box (clean tree, a python WITH pinocchio so
nothing skips) → signed `gpu-proof.json` at the repo root → commit it; the
`verify-gpu-proof` workflow checks it CPU-only (and skips gracefully when no
receipt is committed yet, so code can push before a receipt lands). Config in
`pyproject [tool.gpu_proof]` + `test/gpu-proof-policy.yaml`. Any change to
`gato/`, `python/gato`, `test/`, or `CMakeLists.txt` changes the fingerprint —
regenerate the receipt with (or right after) such a push. The workflow's
`cpu-lane` job runs `-m "not gpu and not slow"` directly in CI.

`test/test_parity_golden.py` is THE bit-parity gate for kernel changes: 34 goldens (pcg + bdsv
per receipt module, an arm ADMM/EE-row case, the fc/eh variants) — a kernel edit that keeps
them bitwise is safe; one that changes them needs `GATO_GOLDEN_REBASELINE=1` and the reason in
the commit message. `test/test_kernel_gates.py` builds + runs the five `test/cuda/` harnesses
(`-DNDEBUG`, `-arch=native`, no fast-math). Also `compute-sanitizer --tool racecheck` on the demo
`build/bsqp` for shared-memory changes.

## Commit style

Short, single-line commit messages; no `Co-Authored-By` footer. Work on branches → PRs. The
submodule merge chain for codegen changes is GRiDCodeGenerator → GRiD → GATO (bump pins up the
chain); GLASS is standalone.
