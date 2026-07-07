# Reproducing the GATO paper figures

Committed, runnable scripts that regenerate the data and figures from the GATO paper
([arXiv:2510.07625](https://arxiv.org/abs/2510.07625)). Each script regenerates its
data on the GPU **by default** and re-renders from saved/recovered data with
`--replot`; `--quick` runs a tiny wiring smoke (not paper numbers). Run everything
from the **repo root**.

```bash
# EVERYTHING, overnight, on a quiet box (fig3 fair sweeps -> gate runners -> fig5/fig4
# regen -> full 100-scenario fig7; ~9-10 h; logs+SUMMARY in overnight_logs/<stamp>/)
examples/paper-figures/run_all_overnight.sh

# one figure
python examples/paper-figures/reproduce_fig4_hparam.py            # GPU re-run
python examples/paper-figures/reproduce_fig4_hparam.py --replot   # no GPU, bundled data
python examples/paper-figures/reproduce_fig4_hparam.py --quick    # fast smoke

# everything
python examples/paper-figures/make_all.py --quick                 # smoke all
python examples/paper-figures/make_all.py                         # full regen
```

Use the GRiD venv for the Python deps (pinocchio etc.): `../GRiD/.venv/bin/python …`.

## Figures

| Script | Paper | What it does | Status |
|---|---|---|---|
| **`reproduce_fig3_fair.py`** | **Fig-3 (both)** | **FAIR iiwa14 fig8 parity harness (2026-07): identical problem for GATO / BatchThneed-CPU / MPCGPU-GPU; left = B∈[1..128] total-time table + GATO speedups at every B; right = GATO N×B heat map (B to 512). THE current data path.** | ✅ (see below) |
| `reproduce_fig3_scalability.py` | Fig-3 left | Indy7 fig-8 solve time vs batch M (June pipeline) | superseded by `reproduce_fig3_fair.py` — kept for indy7 provenance |
| `reproduce_fig3_heatmap.py` | Fig-3 right | GATO solve-time heat map over (N, M) (June pipeline) | superseded by `reproduce_fig3_fair.py` |
| `reproduce_fig4_hparam.py` | Fig-4 (CS1) | iiwa14 online ρ sweep; normalized merit vs SQP iter per batch. **Regenerates by default**; `--replot` uses bundled `examples/gato_hparam_batch_results.pkl` | ✅ |
| `reproduce_fig5_disturbance.py` | Fig-5 (CS2) | Indy7 fig-8 + EE disturbance; tracking err + joint vel vs force, and EE trajectories at 50 N | ✅ |
| `reproduce_fig7_pickplace.py` | Fig-7 + Table-I (CS3) | iiwa14 pick-place + 15 kg pendulum; success rate + completion-time CDF | ⚠️ **gated** (see below) |

Not reproducible in software: **Fig-6** (meshcat sim snapshot) and **Fig-8 / Table-II**
(physical-hardware pick-place). Documented for completeness only.

## Reproducibility tiers
- **Tier A — no GPU (`--replot`):** Fig-4 from the bundled pkl; Fig-3 reports from saved
  pkls. Render-only.
- **Tier B — GPU re-run (default):** every script regenerates its data on the GPU. This
  is the canonical path (Fig-4 regenerates by default per the project requirement).
- **Tier C — hardware-only (not reproducible):** Fig-6, Fig-8, Table-II.

## Build matrix
Each `(plant, N)` is a compile-time module `python/gato/bsqpN{N}_{plant}.so`. Build all
the suite needs in one shot from the repo root:

```bash
cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="8;16;32;64;128" \
      -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 4
```

| Figure | Modules |
|---|---|
| Fig-3 left / Fig-5 | indy7 N64 |
| Fig-3 heatmap | indy7 N∈{8,16,32,64,128} |
| Fig-4 | iiwa14 N64 |
| Fig-7 | iiwa14 N16 |

Scripts emit a clear "module not built" error naming the cmake line if a module is missing.

## The FAIR Fig-3 path (2026-07, current)
`reproduce_fig3_fair.py` replaces the June indy7 data path with the parity harness: all
three solvers solve the IDENTICAL iiwa14 fig8 problem (`examples/benchmarks/
iiwa_fig8_shared.py` — same goal file, same L7 metric frame, same costs/warm-start) under
the matched config (SQP=1, PCG≤200 rel 1e-4, ρ=0.01; MPCGPU = `GATO_REG_PATTERN` + native
exit). Provenance + measured tables: `MPCGPU docs/benchmark_3way_2026-07-06.md`. Data
generators (each stage is a TIMING run — quiet box, one at a time):
- GATO: `examples/benchmarks/sweep_batch_iiwa_fig8.py --N {8..128} --batches 1..512`
- BatchThneed: `examples/benchmarks/baselines/track_iiwa_fig8_bt.py <sim> <B> <N> <csv>`
- MPCGPU: `MPCGPU tools/time_persolve.sh <N> pcg 3 <csv>` (per-solve; no batch axis → ×B)
CSVs land in `examples/benchmarks/data/sweep_fig8_{gato,bt,mpcgpu}.csv`; the assembler
(default, no GPU) writes `fig3_fair_scalability.{txt,png}` + `fig3_fair_heatmap.{txt,png}`.
NOTE the robot delta vs the published figure: the paper used **Indy7**; the fair harness
is **iiwa14** (all indy7 N-modules are still built if a faithful indy7 rerun is wanted).

## Known caveats (honest reproduction status)
- **MPCGPU line (old June Fig-3 path):** the stale `mpcgpu_indy7_fig8_N64.csv` predates the
  2026-07-06 terminal-cost fix (MPCGPU kkt.cuh 88c3853) and the fair config — do NOT mix it
  with fair-path numbers. The fair path times MPCGPU via `tools/time_persolve.sh`.
- **OSQP CPU bar (old June path):** superseded — the fair path uses the paper's real
  threaded C++ `BatchThneed` (`baselines/build_cpu_baseline.sh`).
- **Fig-7 / Table-I (iiwa14 pick-place):** UNBLOCKED 2026-07-07 — the failures were an
  f_ext frame-convention bug (hypothesis wrenches uploaded with swapped [angular;linear]
  halves and a wrong frame chain; fixed in gato.common.world_wrench_to_joint_local et
  al.). Post-fix quick probe: batch=8 reaches 10/10 goals vs 0/10 at batch=1 — the
  paper's Table-I shape. The full 100-scenario × batch sweep still needs a long quiet-GPU
  window (~hours) to produce citable numbers. NOTE the same bug invalidates any Fig-5
  disturbance data generated before 2026-07-07 (the sim applied the world force wrong)
  — Fig-5 must be regenerated.
- **Fig-4 grid:** our bundled/recovered figure used 50 random targets × a 24-combo Q/R
  grid; the paper text states 100 runs × 81 Q/R values. Defaults reproduce *our* bundled
  data; `--num-targets` overrides. The exact paper Q/R grid is pending confirmation.

## Hardware / config delta
Paper: RTX 4090, Ryzen 9 7900X (24-core), Ubuntu 22.04, CUDA 12.6, g++ 11.4, `-O3
-use_fast_math`, timed with Python `timeit` around the wrappers. Reproductions on other
GPU/CPU/CUDA versions will differ in absolute numbers; the scaling trends should hold.

## Data provenance
Regenerated pkls land in `data/` (gitignored). `_common.load_data` prefers a regenerated
pkl, else falls back to a bundled/recovered dataset (e.g. Fig-4's
`examples/gato_hparam_batch_results.pkl`), printing which source it used. See
`docs/archaeology.md` for the provenance of each recovered dataset.
