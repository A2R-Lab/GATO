# examples/archive — superseded scripts, kept for provenance

Nothing here is maintained or run by the paper/benchmark pipelines; each file
carries an `ARCHIVED` banner naming what replaced it. They still import the live
helpers by path (`_paths.py`) so they can be re-run for archaeology.

| File | Superseded by |
|---|---|
| `reproduce_fig3_scalability.py` | `paper-figures/reproduce_fig3_fair.py` — the iiwa14 parity harness (2026-07); this was the June indy7 Fig-3-left assembler |
| `reproduce_fig3_heatmap.py` | `paper-figures/reproduce_fig3_fair.py` (its N×B heat map) |
| `benchmark_fig8.py` | `benchmarks/sweep_batch_iiwa_fig8.py` — the fig3-fair GATO data generator (this was the closed-loop indy7 batch sweep behind the June fig3 chain) |
| `run_batchthneed_fig8.py` | `benchmarks/baselines/track_iiwa_fig8_bt.py` — the fair BatchThneed CPU arm on the shared iiwa14 problem |
| `mpcgpu_indy7_fig8_N64.csv` | `benchmarks/data/sweep_fig8_mpcgpu.csv` via MPCGPU `tools/time_persolve.sh`; this CSV predates the 2026-07-06 MPCGPU terminal-cost fix — never mix it with fair-path numbers |
| `benchmark_pinocchio.py` + `points1000.npy` | the fair fig3 chain; this is the recovered pre-migration pinocchio-sim MPC baseline (`docs/archaeology.md`, `docs/baselines.md`) |
| `_diag_pickplace_phase0.py` | its Phase-0 findings are folded into `paper-figures/reproduce_fig7_pickplace.py` (header) and `docs/baselines.md` |
| `bdsv_timing_session.py` | `benchmarks/linsys_auto_cdf.py` (the 08-12 CDF study) + `tools/autotune_linsys.py`; its measured tables stay in `benchmarks/data/bdsv_timing/` |
