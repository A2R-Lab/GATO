# GATO examples

Two kinds of examples live here:

## Intro demos — "how to use GATO"
Minimal, self-contained scripts that show the core API. Run them from the repo root
after building the `bsqpN64_indy7` module (see the root README's build instructions).

| Script | Shows |
|---|---|
| [`01_single_solve.py`](01_single_solve.py) | Construct a `BSQP` solver and run one trajectory-optimization solve; print the stats. |
| [`02_batched_solve.py`](02_batched_solve.py) | Solve M=8 problems in **one** GPU launch, each with a different damping `rho`; report which batch member converged best. GATO's headline feature. |
| [`03_mpc_loop.py`](03_mpc_loop.py) | A closed-loop `MPC_GATO` figure-8 tracking loop; print average tracking error + per-step solve time. |

```bash
python examples/01_single_solve.py
python examples/02_batched_solve.py
python examples/03_mpc_loop.py
```

For a live, interactive tour of the same APIs (plus a no-GPU Fig-4 re-plot), open
[`explore.ipynb`](explore.ipynb) — it wraps the three demos above and points at the
paper-figure scripts.

For *qualitative* paper visualizations (figure-8 EE tracking + 3D pick-place trajectories
that the headless scripts don't render), see
[`paper-figures/visualizations.ipynb`](paper-figures/visualizations.ipynb). The committed,
CLI-runnable reproduction path is the scripts in `paper-figures/` (below).

## Paper reproduction — `paper-figures/`
Committed scripts that regenerate the data and figures from the paper
([arXiv:2510.07625](https://arxiv.org/abs/2510.07625)). Each `reproduce_figN_*.py`
runs the experiment on the GPU by default and re-renders from saved/recovered data
with `--replot`; `--quick` runs a tiny wiring smoke. See
[`paper-figures/README.md`](paper-figures/README.md) for the full list, the build
matrix each figure needs, the reproducibility tiers, and the hardware/config delta
vs the paper.

```bash
python examples/paper-figures/reproduce_fig4_hparam.py --replot   # no GPU, bundled data
python examples/paper-figures/reproduce_fig3_scalability.py        # GPU re-run
python examples/paper-figures/make_all.py --quick                  # smoke all figures
```
