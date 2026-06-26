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
`(work_dim, BatchSize)` blocks; each block runs the single-block algorithm over data already in
shared/global memory. `BatchSize` is a **compile-time template parameter** (`<T, BatchSize>`) on
every kernel — that is why modules are built per `(plant, knot_points)` and instantiate a fixed
set of batch sizes. Block-scoped kernels must be **thread-count invariant** (strided
`for (i = rank; i < n; i += blockDim)` loops, barriers between a write phase and a dependent read).

## The BSQP solve pipeline (one SQP iteration)

Each `(plant, N)` module is a pybind extension `bsqpN{N}_{plant}`. One SQP iteration runs these
kernels (all in `gato/bsqp/kernels/`):

1. **`setup_kkt.cuh`** — per knot: linearize dynamics (`grid_plant::compute_linearized_dynamics`
   → A,B,c) + build the cost gradient/Hessian blocks (`grid_plant::trackingCostGradHess` →
   Q,q,R,r). Terminal block uses `x_{k+1}` + `N_cost` (PR #17 fix).
2. **`schur_linsys.cuh`** — form the Schur complement system (`formSchurSystemBatchedKernel1/2`):
   invert Q_k/Q_kp1/R_k (`glass::invertMatrix` fused), build S (block-tridiagonal) + Pinv + gamma.
3. **`pcg.cuh`** — solve `S·λ = γ` with `glass::pcg<T, STATE_SIZE, KNOT_POINTS>` (block-tridiagonal
   preconditioned conjugate gradient; row-major `[L|D|R]` strips + padded `(KP+2)·STATE_SIZE` vecs).
4. **`schur_linsys.cuh::computeDzBatchedKernel`** — recover the primal step `dz` from λ.
5. **`merit.cuh` / `line_search.cuh`** — evaluate the merit (`grid_plant::trackingCostValue` +
   `compute_integrator_error`), pick a step.
6. **`sim.cuh`** — roll the chosen control forward (`grid_plant::sim_step`).

`gato/utils/linalg.cuh` keeps only GATO-specific helpers: `block::reduce` (kept — several
consumers), the `getOffset*` batch-layout accessors, and printers. All other `block::` linalg was
migrated to `glass::`.

## Source layout

- `gato/bsqp/` — the BSQP solver: `bsqp.cuh` (host orchestration) + `kernels/*.cuh`.
- `gato/dynamics/{indy7,iiwa14}/` — per-robot `grid.cuh` (generated, vendored) + `*_plant.cuh`
  (thin adapters: dimension aliases, EE-cost/dynamics wrappers calling grid's `_inner` fns).
- `gato/utils/` — `linalg.cuh` (GATO helpers), `cuda.cuh` (error macros).
- `gato/{constants.h, settings.h, types.cuh}` — dims, build flags, KKT/Schur structs.
- `python/bsqp/` — `mpc_controller.py` (`MPC_GATO`: `run_mpc_fig8`, `run_mpc_goals`,
  `setup_external_forces`), `config.py` (robot/experiment configs), `experiment_runner.py`.
- `examples/` — user-facing demos (`01/02/03`, `explore.ipynb`) + robot URDFs + `force_estimator*.py`.
  `examples/paper-figures/` — `reproduce_fig*.py` + `visualizations.ipynb`. `examples/benchmarks/` —
  `benchmark_fig8.py`/`benchmark_pinocchio.py` + `baselines/` (incl. `sqpcpu` submodule) + `data/` + `plots/`.
- `external/GRiD` (pinned to `modernizing-tests`), `external/GLASS` (pinned to `main`).
- `tools/regen_grid.py` — regenerate the vendored `grid.cuh` from `external/GRiD`.

## Build & run

Modules are built by CMake as a `(PLANT × KNOTS)` matrix → `bsqpN{knot}_{plant}.so`:

```bash
cmake -S . -B build -DKNOTS=64 -DPLANT=indy7 -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python \
      -Dpybind11_DIR=$($PWD/.venv/bin/python -m pybind11 --cmakedir) \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel 4      # cap jobs: each TU pulls the large grid.cuh
```

`tools/install.sh` sets up a project-local `.venv` + submodules + regen. `KNOTS`/`PLANT` accept
semicolon lists. Use the GRiD venv (`../GRiD/.venv`) for anything needing pinocchio (the GATO venv
is the lean codegen/build set).

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
  loop does fewer solves (can look like an "early stop").

## Validation

There is no C++ test dir; validate with (a) a minimal standalone `.cu` that includes a kernel
header and launches one block (compile `-DNDEBUG`, no fast-math for tight tolerances), and (b)
end-to-end via the Python API on `examples/` (fig8 tracking error, goal success rate). Prefer
device-parity diffs (old vs new kernel on random inputs) for migration changes.

## Commit style

Short, single-line commit messages; no `Co-Authored-By` footer. Work on branches → PRs. The
submodule merge chain for codegen changes is GRiDCodeGenerator → GRiD → GATO (bump pins up the
chain); GLASS is standalone.
