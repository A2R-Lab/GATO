# Branch archaeology & provenance

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
| **CS1 hyperparameter** (Fig 4): iiwa14 per-batch ρ, normalized-merit-vs-SQP-iter | clean nb `case_study_1:examples/gato_hparam_batch.ipynb` | **YES** — `batch_rho:examples/gato_hparam_batch_results_adaptive_rho_2.pkl` (84 KB) | `examples/gato_hparam_batch.ipynb` + `examples/gato_hparam_batch_results.pkl` (re-plots Fig 4, **no GPU needed**) |
| **Fig 3 scalability** (Indy7 batch×N solve-time) | `experiment_plots:benchmark_fig8.py` (fig8, modern API, batch≤1024) **and** `a2rlab03:benchmark.py` (`class Benchmark`, mujoco point-to-point — produced the surviving data) | **PARTIAL** — 23/24 point-to-point cells (missing `batch128_N64`); fig8-heatmap input data **lost** | `data/fig3_scalability_p2p/` (p2p grid); harness consolidation pending |
| **Fig 3 heatmap** | `experiment_plots:plots/fig8_benchmark_heatmap.ipynb` (+ rendered PNG) | input `benchmark_fig8_*.pkl` **lost** → re-run | `plots/fig8_benchmark_heatmap.ipynb` + `plots/gato_solve_time_heatmap.png` |
| **Fig 3 CPU baseline** | `a2rlab03:benchmark_pinocchio.py` (pinocchio-sim MPC driving the GPU solver — **not** OSQP) | none | pending (port: dead ctor kwargs `f_ext_B_std=` to remove) |
| **CS2 disturbance** (Fig 5): Indy7 fig8 under random external force | old `benchmark.py` (`usefext`/`f_ext_std` random-force batch) + modern `MPC_GATO.setup_external_forces` | **none** (figures only) → re-run from scratch | pending |
| **CS3a pick-place / Table I**: iiwa14 + pendulum, success-rate-vs-batch | `hardware:examples/gato_pick&place.ipynb` (batch sweep) + `_cem.ipynb` (success-aggregation figure) + `_sept_9.ipynb` (cleanest MPC class) | none → re-run | `examples/gato_pickplace.ipynb` (basic demo); batch-sweep + Table-I driver pending |
| **CS3b hardware** (physical robot) | `demo_flexiv:python/bsqp/hardware_controller.py::MPCHardwareController` (robot-agnostic dual-thread driver = the reusable bit) | n/a | pending (hardware-blocked) |
| **MPCGPU baseline** (Fig 3 GPU competitor) | `adu/multisolve-v1` (only branch pinning `dependencies/MPCGPU` + harnesses + CMake + README + citation) | `bchol-integration:benchmark_results/` (bchol batch-SQP, secondary) | pending (frozen-pin build) |

**Success metric (CS3a), identical across the old notebooks and the modern API:** a goal is
`reached` iff `‖ee − goal‖ < 0.05 m` **and** `L1(Δq) < 1.0` before a per-goal timeout, else
`timeout`. The modern `MPC_GATO.run_mpc_goals` already exposes this as `stats['goal_outcomes']`.

**Lost in migration, to re-import for CS3a fidelity:** `ImprovedForceEstimator` (fibonacci-sphere)
and `CEMForceEstimator` — only the generic `ForceEstimator` survived. Source: `hardware`/
`iiwa14_demo` `force_estimator*.py`.

**API note:** the current `python/bsqp/interface.py` still exposes the old `BSQP` surface
(`solve`/`get_stats`/`reset_dual`/`set_f_ext_B`), so most harness ports are trivial renames
(e.g. `set_f_ext_batch` → `set_f_ext_B`), not rewrites.

## Recovered measured data (so figures re-plot without a GPU)
- `examples/gato_hparam_batch_results.pkl` — CS1 (Fig 4), 84 KB. *(was a 5-byte empty stub on `main`.)*
- `data/fig3_scalability_p2p/` — 23/24 Indy7 point-to-point solve-time cells. See `data/README.md`.
- `data/legacy_mpcgpu_solvetime_csv/` — 11 early SQP solve-time CSVs (unique to `dev`/`ROS_dev`/`adu/multisolve-v2`).

## Branch disposition (24 branches)

### KEEP (do not delete)
| Branch | Why |
|---|---|
| `main` / `ICRA-26` | the migrated solver trunk (this history) |
| `gh-pages` | **live** ICRA-2026 paper website — unique demo media (disturbance CDF, hardware photo, fig8/heatmap plots, an 18.9 MB demo video) + deploy workflow; deleting it takes the site down |

### PORT-SOURCE → DROP after consolidation lands & is verified
Harness/data mined; nothing unique remains once Phase-1 consolidation is committed.
| Branch | Unique content (now mined) |
|---|---|
| `batch_rho` | CS1 84 KB results pickle (recovered) + full demo notebook suite |
| `case_study_1` | clean CS1 notebook (no embedded outputs) — superset's portable copy |
| `experiment_plots` | Fig3 `benchmark_fig8.py` (only copy) + heatmap notebook + CS2 disturbance figures |
| `a2rlab03` | Fig3 measured p2p data (recovered) + `benchmark.py` + `benchmark_pinocchio.py` baseline |
| `a2rlab-02` | byte-identical Fig3 data (redundant with a2rlab03) |
| `hardware` | CS3a pick-place trio + `force_estimator*.py` (Improved/CEM) |
| `iiwa14_demo` | CS3a + C++ force-estimator path (`force_estimator.hpp`, `batch_utils.cuh`) |
| `demo_flexiv` | CS3b `MPCHardwareController` (reusable) + Flexiv Rizon one-off port |

### MPCGPU baseline — KEEP until the baseline is built
| Branch | Why |
|---|---|
| `adu/multisolve-v1` | **canonical** MPCGPU baseline (submodule pin `MPCGPU @ 0efde8c`, old `GRiD @ 032ed027`, harnesses, CMake, citation arXiv:2309.08079) |
| `bchol-integration`, `point-mass-example`, `docs` | byte-identical `track_iiwa_{pcg,qdldl}.cu` harnesses (redundant once v1 is built) |

> MPCGPU build = **frozen pins**, not the migrated GRiD. Bounded porting: add CUDA arch `120`,
> CUDA ≥ 12.8 (pin hardcodes 12.2), strip `-G` debug flag before timing, fix CWD/trajfile paths.

### DROP-safe (verified — nothing unique lost)
| Branch | Verdict |
|---|---|
| `benchmark-ckpt`, `working-dev`, `a2rlab-01` | pre-`main` multisolve dev checkpoints; notebooks/data byte-identical to experiment-branch copies, or source superseded by `main` |
| `ROS_dev` | **no ROS code here** — the ROS/MuJoCo integration is the external `A2R-Lab/indy7-mpc` submodule (only a pointer was pinned); unique CSVs salvaged to `data/legacy_mpcgpu_solvetime_csv/` |
| `dev`, `adu/multisolve-v2` | earlier dev checkpoints superseded by `main`; their unique CSV set salvaged (above) |

### USER-CALL (low stakes)
| Branch | Note |
|---|---|
| `main-deprecated` | genuine pre-migration "old main" + a unique `experiments/MPCGPU/` harness (already mined). Tag-then-drop candidate, or keep as a historical marker. |

## Staging method (why not literal `git cherry-pick`)
The old branches are **unrelated histories with disjoint file trees** (old pre-migration `gato/`
source vs the migrated tree), so a per-commit cherry-pick produces conflicts on nearly every file.
Instead, consolidation brings the canonical artifacts onto `cleanup-modernization` as
**provenance-credited file ports** — each commit names its source `branch:SHA`, and this document is
the history record. The old branches remain on `origin` until consolidation is verified and deletion
is explicitly approved.
