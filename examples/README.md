# GATO examples

Three kinds of things live here: intro demos, the paper-reproduction scripts, and
the benchmark / timing harnesses. Every script imports the installed `gato` package
(`./tools/install.sh` does `pip install -e .`) and runs from any cwd.

## Intro demos — "how to use GATO"
Minimal, self-contained scripts that show the core API. They need the `bsqpN64_indy7`
module (`MODULES="indy7:64" ./tools/build.sh`, or `./tools/build.sh --profile receipt`
for everything) and, from 03 on, a python with pinocchio (`./tools/install.sh --test`).

| Script | Shows |
|---|---|
| [`01_single_solve.py`](01_single_solve.py) | Construct a `BSQP` solver, run one solve (cold, then warm-started from `res.xu`); print the stats. |
| [`02_batched_solve.py`](02_batched_solve.py) | Solve M=8 problems in **one** GPU launch, each with a different damping `rho`; report which batch member converged best. GATO's headline feature. |
| [`03_mpc_loop.py`](03_mpc_loop.py) | A closed-loop `MPC_GATO` figure-8 tracking loop; print average tracking error + per-step solve time. |
| [`04_gym_mpc.py`](04_gym_mpc.py) | MPC as a **gymnasium policy** (`MPCPolicy` + `MPCController` + `ArmTrackEnv`), tracking under an unmodeled EE force with a B=16 force-hypothesis batch vs plain B=1. Needs the `[examples]` extra (gymnasium). |
| [`05_build_your_robot.py`](05_build_your_robot.py) | `gato.build(urdf, ...)`: codegen + compile a solver module for your own URDF, then a smoke solve. |
| [`06_constraints.py`](06_constraints.py) | The constraint layer: URDF limit boxes under ADMM, extra linear control rows, a collision obstacle; per-row-group violation telemetry. |
| [`07_go2_floating.py`](07_go2_floating.py) | A floating-base robot (go2) in closed loop on a MuJoCo ground (or the device integrator): standing keyframe anchor, imu goal, the floating controller defaults. |
| [`08_linsys_and_autotune.py`](08_linsys_and_autotune.py) | The `pcg` / `bdsv` / `bdsv_first` linear-system paths, cold vs warm-started, and where `tools/autotune_linsys.py` fits in. |

```bash
python examples/01_single_solve.py
python examples/02_batched_solve.py
python examples/03_mpc_loop.py
python examples/04_gym_mpc.py
python examples/05_build_your_robot.py      # builds a module (minutes; reconfigures build/)
python examples/06_constraints.py
python examples/07_go2_floating.py          # needs bsqpN16_go2
python examples/08_linsys_and_autotune.py
```

For a live, interactive tour of the same APIs (plus a no-GPU Fig-4 re-plot), open
[`explore.ipynb`](explore.ipynb) — it wraps the first three demos and points at the
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
python examples/paper-figures/reproduce_fig3_fair.py              # assemble fig3 from the sweep CSVs
python examples/paper-figures/make_all.py --quick                 # smoke all figures
```

## Benchmarks / timing — `benchmarks/`, `contact-task/`
The timing harnesses (fig3-fair sweeps, the linsys CDF study, the constraint-eval
matrix, the go2 linsys sweep) share [`benchmarks/_bench.py`](benchmarks/_bench.py)
(URDF/model lookup, the quiet-GPU guard, provenance, the kicked-arm MPC probe) and
refuse to time on a busy box. The one staged overnight driver is
[`benchmarks/run_timing_night.sh`](benchmarks/run_timing_night.sh). `contact-task/`
is the fc-vs-baselines contact-wipe study. Superseded scripts are kept for provenance
in [`archive/`](archive/README.md).
