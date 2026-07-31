# Measured result data (recovered from pre-migration branches)

> **⚠ 2026-07-31 rotation:** `stale_L7_frame_pre20260730/` holds the 07-08
> fig3-fair sweep CSVs + benchmark_fig8 pkls measured at the old **L7** metric
> frame (pre named-target `_EE` regen) — see its README. Do not mix with fresh
> EE-frame rows; the fig3 assembler takes the last row per (N,B). The recovered
> archives below (fig3_scalability_p2p, legacy CSVs) are older still and were
> already quarantined by provenance.

These are **measured** GPU/solver results recovered during the cross-branch consolidation
(see [`docs/archaeology.md`](../docs/archaeology.md)). They let the paper figures re-plot
**without re-running on a GPU**. Provenance is recorded per directory below.

## `fig3_scalability_p2p/` — Fig 3 scalability grid (Indy7, point-to-point)
- **What:** 23 of 24 cells of the batch×knot solve-time sweep — batch ∈ {1,2,4,8,16,32,64,128}
  × N ∈ {16,32,64}. Each cell = `benchmark_statsgpu_batch{B}_N{N}_stats_final.pkl` (the
  `allstats` dict: `failed/avg_solve_time/max_solve_time/steps/cumulative_dist/total_ctrl/total_vel`,
  over ≤50 point-pairs) + `..._benchmark_config.pkl` (run config).
- **Producer:** `benchmark.py` `class Benchmark` (mujoco point-to-point, indy7, `usefext=False`,
  `realtime=True`, `dt=0.01`, `max_qp_iters=5`), run 2025-05-18.
- **Missing cell:** `batch128_N64` (only the config pickle survived — no `stats_final`).
- **Provenance:** `origin/a2rlab03:data/` (byte-identical copy also on `origin/a2rlab-02`).
- **⚠️ Caveat:** this is the **point-to-point** scalability experiment. The Fig-3 **fig8 heatmap**
  (`plots/fig8_benchmark_heatmap.ipynb` + `plots/gato_solve_time_heatmap.png`) was produced from a
  *separate* `benchmark_fig8_*.pkl` that did **not** survive on any branch — that heatmap must be
  re-run to regenerate its input data.

## `legacy_mpcgpu_solvetime_csv/` — early SQP solve-time stats (unique, salvaged)
- **What:** 11 CSVs of measured SQP solve-time statistics, swept over PCG exit tol {1e-4, 1e-5} ×
  knot points {8,16,32,64,128} × batch sizes.
- **Provenance:** `origin/dev:benchmark_results/` (byte-identical on `origin/ROS_dev`,
  `origin/adu/multisolve-v2`). **Exists on no other branch** (not on main, not on any experiment
  branch) — salvaged so those dev branches are safe to delete.
- **Status:** secondary/early reference (predates the paper harnesses); kept for completeness.

## Recovered elsewhere (not in this directory)
- **CS1 hyperparameter results** (Fig 4): `examples/gato_hparam_batch_results.pkl` (84 KB, the
  `agg` dict of normalized merit-vs-SQP-iter curves per batch size) — restored into `examples/`
  next to its notebook `gato_hparam_batch.ipynb`, which re-plots Fig 4 directly from it.
