# Reproducing the GATO paper figures

Committed, runnable scripts that regenerate the data and figures from the GATO paper
([arXiv:2510.07625](https://arxiv.org/abs/2510.07625)). Each script regenerates its
data on the GPU **by default** and re-renders from saved/recovered data with
`--replot`; `--quick` runs a tiny wiring smoke (not paper numbers). Run everything
from the **repo root**.

```bash
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
| `reproduce_fig3_scalability.py` | Fig-3 left | Indy7 fig-8 solve time vs batch M: GATO vs OSQP-CPU vs MPCGPU-GPU | GATO + OSQP ✅; MPCGPU line optional (see below) |
| `reproduce_fig3_heatmap.py` | Fig-3 right | GATO solve-time heat map over (N, M) | ✅ (needs indy7 N∈{8..128}) |
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
Each `(plant, N)` is a compile-time module `python/bsqp/bsqpN{N}_{plant}.so`. Build all
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

## Known caveats (honest reproduction status)
- **MPCGPU line (Fig-3):** the GPU baseline is built separately from frozen pins (see
  `docs/baselines.md`); its indy7 tracking fix + PR are deferred until Fig-3 is otherwise
  complete. `reproduce_fig3_scalability.py` plots it only if
  `benchmarks/baselines/mpcgpu_indy7_fig8_N64.csv` is present, and degrades gracefully otherwise.
- **OSQP CPU bar:** the committed baseline is the single-solve Python `Thneed`
  (interpreter-bound; conservative). The paper's CPU bar is a multi-threaded C++
  `BatchThneed` (needs osqp/OsqpEigen/pinocchio-C++ — a backlog item).
- **Fig-7 / Table-I (iiwa14 pick-place):** a known closed-loop instability is parked
  solver-robustness R&D. The harness, Table, and CDF are correct in structure but the
  success numbers may not match the paper until that fix lands.
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
