# GATO examples

Three kinds of things live here: intro demos, the paper-reproduction scripts, and
the benchmark / timing harnesses. Every script imports the installed `gato` package
(`./tools/install.sh` does `pip install -e .`) and runs from any cwd.

## Intro demos — "how to use GATO"
Minimal, self-contained scripts that show the core API. 01–04, 06, 08, 09, 10
and 11 use the Indy7 at N64: in a checkout build it once (`MODULES="indy7:64"
JOBS=2 ./tools/build.sh`), or let the first run compile it into the per-host
cache (a few minutes, logged). 05 registers and compiles the requested robot
itself; 07 needs go2 N16 (the `_fc` variant for standing); 12 needs the Indy7 at
N16, N32 and N64; 13 needs the FR3, Gen3 and UR5e at N64; 14 the iiwa14 at N64 with mujoco. `JOBS=2
./tools/build.sh --profile receipt` covers all of them. Install `--test` for pinocchio, MuJoCo and gymnasium; `--examples`
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
| [`09_mpc_with_limits.py`](09_mpc_with_limits.py) | Joint limits in closed loop through `MPCController(limits=...)`: telemetry only, ADMM with a fixed inner budget, and the controller default (ADMM with the residual exit and the inner linear solver by horizon) hold a velocity cap the figure-8 cannot meet; the simulated plant, not the planner, is graded; inner iterations spent per tick. |
| [`10_torch_jax_arrays.py`](10_torch_jax_arrays.py) | The solver as a batched tensor op: torch / JAX CUDA arrays in, the same kind out through DLPack, bit-identical to the numpy path; warm starts stay on the device. The controller on CUDA tensors: the warm start resident on the device, `u` back as a device tensor, the same plan as the numpy controller. |
| [`11_deadline_step.py`](11_deadline_step.py) | The deadline-aware step: `step(x, ref, budget_s=...)` runs SQP iterations while the next one still fits and returns the best plan so far; iterations, budget hits and wall time per tick. |
| [`12_constraint_mechanisms.py`](12_constraint_mechanisms.py) | The four limit mechanisms (`limits="telemetry"`, `"barrier"`, `"al"`, `"admm"`) on one pick-and-place task with a step reference, at N16/32/64: sampled plant velocity and commanded torque against URDF limits, planner violation, goals reached and simulated task time. `--plot` and `--json` save correctness results without solve timings; [figure and interpretation](../docs/constraints.md#mechanism-comparison-example). |
| [`13_more_arms.py`](13_more_arms.py) | The arms added in 0.1.2 (FR3, Gen3, UR5e) under the controller defaults on figure-8s sized from their own URDF limits (`gato.tasks`): a feasible reference and one that asks for more than the velocity limit, with the telemetry-only run for contrast; the simulated plant's peak velocity and torque against the limits, the planner's violation, ADMM inner iterations. |
| [`14_mujoco_closed_loop.py`](14_mujoco_closed_loop.py) | GATO inside a MuJoCo loop: the MuJoCo model from the same URDF, the dynamics fingerprint check on MuJoCo's forward dynamics, the controller defaults with a step budget, torques through `qfrc_applied` and substeps; the iiwa14 on a figure-8 sized from its velocity limits. |
| [`15_solver_stats.py`](15_solver_stats.py) | Reading `SolveResult.stats` and `summary()`: the device time of each stage per SQP iteration (KKT setup, Schur assembly, linear system, merit, line search) as a share of the solve, cold and warm, and the knobs that move each stage. |
| [`16_cost_menu.py`](16_cost_menu.py) | The cost menu: the tracking objective spelled as a menu and checked against it, then per-solve goals, a pseudo-Huber posture term, targets that move from tick to tick, the per-term readout; [the page](../docs/cost_menu.md). |
| [`17_tool_pose_and_posture.py`](17_tool_pose_and_posture.py) | A tool pose: an orientation target (the log-map residual), a linear feature holding the elbow, an input cost measured against the gravity torque at a reference configuration; the cost split per term along the horizon. |
| [`18_robust_reference.py`](18_robust_reference.py) | A reference with corrupted knots: the quadratic and the pseudo-Huber tracking term on the same problem, the planned end-effector path against the clean reference and the outliers. |
| [`19_go2_standing.py`](19_go2_standing.py) | A standing reference for the Go2 through the posture anchor's base-pose weights (the floating-base menu is the next GRiD batch): a lowered and rolled base returns to the stance. |

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
python examples/12_constraint_mechanisms.py
python examples/13_more_arms.py             # needs bsqpN64_{fr3,gen3,ur5e} (the receipt profile)
python examples/14_mujoco_closed_loop.py    # needs bsqpN64_iiwa14 and mujoco
python examples/14_mujoco_closed_loop.py --correctness-only --periods 1  # no clock-dependent budget or timing report
python examples/15_solver_stats.py
python examples/16_cost_menu.py
python examples/17_tool_pose_and_posture.py   # needs pinocchio
python examples/18_robust_reference.py        # needs pinocchio
python examples/19_go2_standing.py            # needs bsqpN16_go2 and pinocchio
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

The [controller replay](benchmarks/controller_replay.py) compares NumPy,
CUDA PyTorch and CUDA JAX on identical moving-state inputs for all five arms
at N64, batches 1 and 8, with limits off and default ADMM. Its default mode
checks full-plan and telemetry parity without collecting durations.
The [assigned-window launcher](benchmarks/run_controller_replay.sh) records
three rotated-order repeats, external call-through-device-completion latency
and native solve time separately. This kinematic replay measures controller
overhead, not closed-loop task quality or a complete robot-cycle deadline.
The [October 20 measurements](../docs/consumer_contract.md#host-and-device-replay-on-october-20)
retain two full collections: device inputs do not reduce latency in this replay.

The [torque clipping A/B](benchmarks/torque_clip_ab.py) is a fixed-pacing
correctness experiment, not a timing benchmark. It pairs identical development
tasks with simulator torque clipping off/on and measures every integration
endpoint, including warmup and the final transition. It does not change
controller defaults. See the [protocol and results](../docs/constraints.md#torque-saturation-development-comparison).

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
