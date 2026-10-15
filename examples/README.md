# GATO examples

Three kinds of things live here: intro demos, the paper-reproduction scripts, and
the benchmark / timing harnesses. Every script imports the installed `gato` package
(`./tools/install.sh` does `pip install -e .`) and runs from any cwd.

## Intro demos — "how to use GATO"
Minimal, self-contained scripts that show the core API. 01–04, 06, 08, 09 and 10
use the Indy7 at N64: in a checkout build it once (`MODULES="indy7:64" JOBS=2
./tools/build.sh`), or let the first run compile it into the per-host cache
(a few minutes, logged). 05 registers and compiles the requested robot itself;
07 needs go2 N16 (the `_fc` variant for standing). `JOBS=2 ./tools/build.sh
--profile receipt` covers the shipped variants. Install `--test` for pinocchio, MuJoCo and gymnasium; `--examples`
adds plotting/notebook dependencies. These are correctness demos, not benchmarks.
They demonstrate API use and simulated behavior, not guaranteed joint-limit
feasibility or real-time deadlines (a step budget bounds the solver's
iterations, not your loop). Inspect the printed violation telemetry and
the independent simulated trajectory before interpreting a task as feasible.

| Script | Shows |
|---|---|
| [`01_single_solve.py`](01_single_solve.py) | Construct a `BSQP` solver, run one solve (cold, then warm-started from `res.xu`); print the stats. |
| [`02_batched_solve.py`](02_batched_solve.py) | Solve M=8 problems in one batched solver call, each with a different damping `rho`; report which batch member converged best. |
| [`03_mpc_loop.py`](03_mpc_loop.py) | A closed-loop `MPC_GATO` figure-8 loop with fixed simulation pacing; print tracking error and diagnostic durations. |
| [`04_gym_mpc.py`](04_gym_mpc.py) | MPC as a **gymnasium policy** (`MPCPolicy` + `MPCController` + `ArmTrackEnv`), tracking under an unmodeled EE force with a B=16 force-hypothesis batch vs plain B=1. Needs the `[examples]` extra (gymnasium). |
| [`05_build_your_robot.py`](05_build_your_robot.py) | `gato.register_robot(name, urdf, ...)`: codegen into the per-host cache, the first solver compiles the module once, then a smoke solve — the pip-install path (checkouts can also `gato.build`). |
| [`06_constraints.py`](06_constraints.py) | The constraint layer: URDF limit boxes under ADMM, extra linear control rows, a collision obstacle; per-row-group violation telemetry. |
| [`07_go2_floating.py`](07_go2_floating.py) | A floating-base robot (go2) in closed loop on a MuJoCo ground (or the device integrator): standing keyframe anchor, imu goal, the floating controller defaults. |
| [`08_linsys_and_autotune.py`](08_linsys_and_autotune.py) | The `pcg` / `bdsv` / `bdsv_first` linear-system paths, cold vs warm-started, and where `tools/autotune_linsys.py` fits in. |
| [`09_mpc_with_limits.py`](09_mpc_with_limits.py) | Joint limits in closed loop: the ADMM box mechanism holds a velocity cap the figure-8 cannot meet; the simulated plant, not the planner, is graded. |
| [`10_torch_jax_arrays.py`](10_torch_jax_arrays.py) | The solver as a batched tensor op: torch / JAX CUDA arrays in, the same kind out through DLPack, bit-identical to the numpy path; warm starts stay on the device. |
| [`11_deadline_step.py`](11_deadline_step.py) | The deadline-aware step: `step(x, ref, budget_s=...)` runs SQP iterations while the next one still fits and returns the best plan so far; iterations, budget hits and wall time per tick. |

```bash
python examples/01_single_solve.py
python examples/02_batched_solve.py
python examples/03_mpc_loop.py
python examples/04_gym_mpc.py
python examples/05_build_your_robot.py      # registers a robot; its first solver compiles a module into the cache (minutes)
python examples/06_constraints.py
python examples/07_go2_floating.py --fc     # standing; needs bsqpN16_go2_fc
python examples/08_linsys_and_autotune.py
python examples/09_mpc_with_limits.py
python examples/10_torch_jax_arrays.py
python examples/11_deadline_step.py
```

07 without `--fc` demonstrates contactless-model plumbing, not standing or
walking. See [feature status](../docs/status.md) before using the gait surface.
Do not compare printed solve times while other processes use the GPU.

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
runs the experiment on the GPU by default (except Fig-3, which assembles saved
CSVs) and re-renders from saved/recovered data
with `--replot`; `--quick` runs a tiny wiring smoke. See
[`paper-figures/README.md`](paper-figures/README.md) for the full list, the build
matrix each figure needs, the reproducibility tiers, and the hardware/config delta
vs the paper.

```bash
python examples/paper-figures/reproduce_fig4_hparam.py --replot   # no GPU, bundled data
python examples/paper-figures/reproduce_fig3_fair.py              # assemble fig3 from the sweep CSVs
python examples/paper-figures/make_all.py --quick                 # smoke all figures
```

## C++/CUDA — `cpp/`
[`cpp/batched_solve.cu`](cpp/batched_solve.cu) runs the same batched figure-8
solve straight on the header-only `BSQP<T>` class the Python package wraps, with
[`cpp/CMakeLists.txt`](cpp/CMakeLists.txt) as a standalone build that works from
a wheel or a checkout (`cmake -S examples/cpp -B build-cpp -DGATO_ROBOT=indy7
-DGATO_N=32`; `python -m gato.paths` resolves the include directories and
defines). The in-tree `build/bsqp` target compiles the same file.

## Benchmarks / timing — `benchmarks/`, `contact-task/`
The timing harnesses (fig3-fair sweeps, the linsys CDF study, the constraint-eval
matrix, the go2 linsys sweep) share [`benchmarks/_bench.py`](benchmarks/_bench.py)
(URDF/model lookup, the quiet-GPU guard, provenance, the kicked-arm MPC probe) and
provide quiet-GPU checks, but not every direct script invocation enforces one.
Use the guarded [merge checkpoint](benchmarks/run_merge_checkpoint.sh) for the
focused batch; it requires an explicit quiet-window declaration.
For a coordinating agent, `run_timing_handoff.sh` adds a
shared lock and selectable isolated compile/calibration legs through
[`run_timing_handoff.sh`](benchmarks/run_timing_handoff.sh). The older
[`run_timing_night.sh`](benchmarks/run_timing_night.sh) is a broad research run,
not the merge checklist: it also accesses a sibling MPCGPU checkout and reruns
Fig-7. Do not launch it blindly on a shared machine. `contact-task/`
is the [fc-vs-baselines contact-wipe study](contact-task/README.md). The superseded June-era
scripts (the indy7 fig3 chain, the pinocchio-sim baseline, the bdsv timing session) were
retired on 2026-10-01; recover them from git history (`git log -- examples/archive`).
