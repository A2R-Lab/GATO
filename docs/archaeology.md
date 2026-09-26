# Branch archaeology & provenance

> **DATED SNAPSHOT (2026-06-24) — read as history.** Since then: `cleanup-modernization`
> IS pushed to A2R-Lab/GATO (CI-verified receipts); the Python package moved
> `python/bsqp/` → `python/gato/`; the recovered data lives under
> `examples/benchmarks/data/`; the June fig3 chain (`benchmark_fig8.py`,
> `benchmark_pinocchio.py`, `reproduce_fig3_{scalability,heatmap}.py`) is in
> `examples/archive/` — the paper's Fig-3 path is `examples/paper-figures/reproduce_fig3_fair.py`;
> the notebooks named below were folded into `examples/explore.ipynb` /
> `paper-figures/visualizations.ipynb`; `ImprovedForceEstimator`/`CEMForceEstimator` were
> re-imported (`python/gato/estimators.py`). The deletion gate below was never
> executed: items 1 (dynamics validated) and 3 (branch pushed) are cleared, item 2
> (fair Fig-3) landed 07-30, item 4 (explicit approval) is still open.

This repo's `main`/`ICRA-26` (the clean, GRiD/GLASS-vendored, migrated solver) is an **unrelated
git history** to the older `Alex-Du`-era development branches (empty merge-base — they share no
common ancestor). Those older branches are each a full *pre-migration-API* world that carried the
paper's experiment harnesses, notebooks, and **measured result data** which were never consolidated
onto `main`.

This document records, per branch, **what unique content it held and where that content now lives**,
so the stale branches can be deleted with confidence. It is the output of a full read-only audit of
all 24 branches (2026-06-21). Paper = *GATO: GPU-Accelerated Batched Trajectory Optimization*
([arXiv:2510.07625](https://arxiv.org/abs/2510.07625)), fixed-base Indy7 (6-DoF) + iiwa14 (7-DoF).

## Paper experiment → canonical source → where it lives now

| Paper element | Canonical source branch:path | Measured data recovered? | Now in this repo |
|---|---|---|---|
| **CS1 hyperparameter** (Fig 4): iiwa14 per-batch ρ, normalized-merit-vs-SQP-iter | clean nb `case_study_1:examples/explore.ipynb` | **YES** — `batch_rho:examples/gato_hparam_batch_results_adaptive_rho_2.pkl` (84 KB) | `examples/explore.ipynb` + `examples/gato_hparam_batch_results.pkl` (re-plots Fig 4, **no GPU needed**) |
| **Fig 3 scalability** (Indy7 batch×N solve-time) | `experiment_plots:benchmark_fig8.py` (fig8, modern API, batch≤1024) **and** `a2rlab03:benchmark.py` (`class Benchmark`, mujoco point-to-point — produced the surviving data) | **PARTIAL** — 23/24 point-to-point cells (missing `batch128_N64`); fig8-heatmap input data **lost** | `data/fig3_scalability_p2p/` (p2p grid); harness consolidation pending |
| **Fig 3 heatmap** | `experiment_plots:plots/fig8_benchmark_heatmap.ipynb` (+ rendered PNG) | input `benchmark_fig8_*.pkl` **lost** → re-run | `examples/paper-figures/` (the reproduce_fig*.py renders) + `examples/paper-figures/` (the reproduce_fig*.py renders) |
| **Fig 3 CPU baseline** | `a2rlab03:benchmark_pinocchio.py` (pinocchio-sim MPC driving the GPU solver — **not** OSQP) | none | pending (port: dead ctor kwargs `f_ext_B_std=` to remove) |
| **CS2 disturbance** (Fig 5): Indy7 fig8 under random external force | old `benchmark.py` (`usefext`/`f_ext_std` random-force batch) + modern `MPC_GATO.setup_external_forces` | **none** (figures only) → re-run from scratch | pending |
| **CS3a pick-place / Table I**: iiwa14 + pendulum, success-rate-vs-batch | `hardware:examples/gato_pick&place.ipynb` (batch sweep) + `_cem.ipynb` (success-aggregation figure) + `_sept_9.ipynb` (cleanest MPC class) | none → re-run | `examples/paper-figures/reproduce_fig7_pickplace.py` (basic demo); batch-sweep + Table-I driver pending |
| **CS3b hardware** (physical robot) | `demo_flexiv:python/bsqp/hardware_controller.py::MPCHardwareController` (robot-agnostic dual-thread driver = the reusable bit) | n/a | pending (hardware-blocked) |
| **MPCGPU baseline** (Fig 3 GPU competitor) | `adu/multisolve-v1` (only branch pinning `dependencies/MPCGPU` + harnesses + CMake + README + citation) | `bchol-integration:benchmark_results/` (bchol batch-SQP, secondary) | pending (frozen-pin build) |

**Success metric (CS3a), identical across the old notebooks and the modern API:** a goal is
`reached` iff `‖ee − goal‖ < 0.05 m` **and** `L1(Δq) < 1.0` before a per-goal timeout, else
`timeout`. The modern `MPC_GATO.run_mpc_goals` already exposes this as `stats['goal_outcomes']`.

**Lost in migration, to re-import for CS3a fidelity:** `ImprovedForceEstimator` (fibonacci-sphere)
and `CEMForceEstimator` — only the generic `ForceEstimator` survived. Source: `hardware`/
`iiwa14_demo` `force_estimator*.py`.

**API note:** the current `python/gato/interface.py` still exposes the old `BSQP` surface
(`solve`/`stats`/`reset_dual`/`set_f_ext_B`), so most harness ports are trivial renames
(e.g. `set_f_ext_batch` → `set_f_ext_B`), not rewrites.

## Recovered measured data (so figures re-plot without a GPU)
- `examples/gato_hparam_batch_results.pkl` — CS1 (Fig 4), 84 KB. *(was a 5-byte empty stub on `main`.)*
- `data/fig3_scalability_p2p/` — 23/24 Indy7 point-to-point solve-time cells. See `data/README.md`.
- `data/legacy_mpcgpu_solvetime_csv/` — 11 early SQP solve-time CSVs (unique to `dev`/`ROS_dev`/`adu/multisolve-v2`).

