# GATO
> GPU-Accelerated Trajectory Optimization

Numerical experiments and the open-source solver from ["GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control"](https://arxiv.org/abs/2510.07625).

[Documentation](docs/README.md) · [Feature status and limitations](docs/status.md) ·
[API migration](docs/consumer_contract.md#6-migrating-from-the-paper-era-api) ·
[Examples](examples/README.md) · [Paper reproduction](examples/paper-figures/README.md)

The software has grown beyond the paper. Paper figures describe the published
experiments, not performance guarantees for the latest code. Floating-base
standing and contact-row examples are available; closed-loop walking is still
in development. See the feature-status page for the tested scope.
Constraint mechanisms and simulated goal arrival do not by themselves establish
joint-limit feasibility. Check solver residuals and the physical trajectory;
the current-default arm holdouts and timing results are described in
[constraints](docs/constraints.md). The [dated experiment report](docs/figure-refresh-2026-10-01.md)
covers the paper-comparison and payload studies at their recorded revisions.

## Quick start

Prerequisites (Linux):

- an NVIDIA driver + **CUDA toolkit ≥ 12.x matching your GPU architecture**
  (e.g. RTX 50xx / sm_120 needs CUDA ≥ 12.8) with `nvcc` on `PATH`
- `apt install build-essential cmake python3-dev python3-venv git`
  (CMake ≥ 3.24 recommended so the build auto-detects your GPU arch; older
  CMake requires an explicit architecture, e.g. `ARCH=120` for sm_120)
- Python ≥ 3.10

### pip install

The commands below install the published version. Features listed under
`Unreleased` in [CHANGELOG.md](CHANGELOG.md) require the corresponding source
revision until the next package release.

```sh
pip install gato-mpc        # the import name is gato
```

```python
import numpy as np, gato

# the vendored robots (indy7, iiwa14, fr3, gen3, ur5e, go2) need no registration; your own URDF does:
# gato.register_robot("myarm", "myarm.urdf", N=[32], ee_frame="EE")      # codegen, seconds
solver = gato.BSQP(gato.robot_urdf("indy7"), batch_size=8, N=32, dt=0.01, plant_type="indy7")
```

`gato.robot_urdf(name)` returns the bundled description of a vendored robot
(URDF + collision meshes), which you can copy and edit as a starting point.
The first solver constructed for a (robot, horizon) on this machine compiles
its CUDA module — a one-time build of a few minutes that is logged when it
starts — into `~/.cache/gato` (`GATO_CACHE_DIR` to move it); every later
process loads it from there. `gato.register_robot(..., build=True)` compiles
up front. Solver modules are specific to your GPU architecture, CUDA toolkit
and Python; the cache key covers all of them, so nothing stale is ever loaded.
Inputs may be numpy, torch or JAX arrays — see [Usage](#usage).

### From a checkout (development, the receipt profile, the paper figures)

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
./tools/install.sh
MODULES="indy7:64" JOBS=2 ./tools/build.sh
source .venv/bin/activate
python examples/01_single_solve.py
python examples/02_batched_solve.py
```

This builds one solver module into `python/gato/`, sufficient for the two
introductory demos and the usage snippet below (any other horizon compiles into
the cache on first use). Each demo reports solve results; printed durations on
a shared GPU are not benchmark measurements.

GATO installs host-native into a project-local `.venv`, using only the Python
standard-library `venv` + `pip` — no Docker, no `uv` (same lightweight model as
GRiD's `base_install.sh`). The install script runs a preflight check (nvcc,
GPU, cmake) and tells you exactly what's missing. Optional install choices
(pick the extras you need; there is no need to run every line):

```sh
./tools/install.sh            # lean: codegen + build deps (incl. the pinned grid-rbd) + regen grid.cuh
./tools/install.sh --examples #   + runtime to run the MPC/benchmark examples (torch, pinocchio, mujoco, viz)
./tools/install.sh --test     #   + full test-suite deps (pinocchio, mujoco, scipy, gymnasium)
./tools/install.sh --dev      #   + test tooling (pytest, pytest-gpu-proof)
./tools/install.sh --all      #   + examples + test + dev
```

GRiD (the code generator and its URDF parser) and the GLASS and collision headers
come from one released wheel, [`grid-rbd`](https://pypi.org/project/grid-rbd/),
pinned to an exact version in `pyproject.toml`; no submodule is needed to build.
(The paper's CPU baseline, `examples/benchmarks/baselines/sqpcpu`, is the one
optional submodule; `examples/benchmarks/baselines/build_cpu_baseline.sh` fetches
and compiles it — osqp and osqp-eigen into a local prefix — only when you reproduce Fig-3.) The lean default is all you need to generate code, build the solver, and run
`gato.BSQP` (runtime: numpy + the built module). The lean installer also supplies
trimesh and SciPy for collision-mesh code generation. Missing voxelization
dependencies must not silently substitute a different collision geometry.
Pinocchio and MuJoCo are needed only by
the simulation worlds (`gato.worlds`), the FK helpers, the examples and the full
test suite — `--test` pulls exactly those in; `--examples` adds torch and the
viz stack. To run the fixed-pacing MPC and gym demos, add the test runtime:

```sh
./tools/install.sh --test
python examples/03_mpc_loop.py
python examples/04_gym_mpc.py
```

Docker (`./tools/docker.sh`) is an **optional prerequisites image** — CUDA
toolkit, CMake, Python — that drops you into a shell with the repo mounted; you
then run the same `tools/install.sh` + `tools/build.sh` inside it. It does not
build GATO for you.

**Distribution:** both the wheel and a Git checkout support local CUDA builds.
The universal wheel carries Python, GATO's native sources and bundled robot
descriptions, but no compiled solver modules; these are GPU-arch/CUDA/ABI-specific
CMake products built locally. The sdist also contains the build sources.
Both paths obtain the GRiD code generator and GLASS/collision headers from the
pinned `grid-rbd` wheel.

### Build Options

You can control which Python extension modules are built by selecting plant models and horizon lengths at CMake configure time:

```sh
cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="8;32;128"   # PLANT x KNOTS cross-product
cmake -S . -B build -DMODULES="indy7:8,32;go2:16"                # explicit per-plant horizons
cmake -S . -B build -DGATO_RECEIPT_PROFILE=ON                     # exactly test/receipt_modules.txt
cmake --build build --parallel 2                                # RAM-bound: roughly 1-7 GB per job
```

- `PLANT`: semicolon-separated plant targets (`indy7`, `iiwa14`, `fr3`, `gen3`, `ur5e`, `go2`).
- `KNOTS`: semicolon-separated horizon lengths — crossed with EVERY plant, so
  use `MODULES` when plants differ (go2 is **N16-only**; longer floating-base
  horizons are compile blowups).
- `./tools/build.sh --profile receipt` builds the set the signed GPU receipt
  attests (`test/receipt_modules.txt`).
- `GATO_MERIT_THREADS` (a CMake cache variable, or the environment variable for
  `register_robot` and first-use compiles): the merit kernel's block size, a
  multiple of 32; the default is 64, the measured optimum on the RTX 5090 and
  on few-SM conditions ([docs/merit_timing.md](docs/merit_timing.md)). Results
  are bit-identical at any size.
- `GATO_MERIT_FUSED=1` (environment, read at run time): the fused merit launch,
  one block per knot evaluating every candidate; measured slower in every
  condition, kept for other hardware. Results are bit-identical either way.
- `tools/build.sh` defaults to four jobs; use `JOBS=1` on the shared lab box or a RAM-limited
  machine. CMake caches `MODULES`: an old explicit list overrides `PLANT`/`KNOTS`.
  Clear it with `-DMODULES=""` when returning to the cross-product. The receipt
  profile takes precedence over both. `--clean` deletes `build/`, not just its cache.

Built Python modules are written to `python/gato/` as `bsqpN{N}_{plant}[_{variant}].so`.
**Variants** are separate ABIs that live side by side: `_fc` (contact-force
controls appended to `u`; `-DGATO_CONTACT_FORCES=ON`, `./tools/build.sh --variant fc`,
`gato.build(..., contact_forces=True)`, `-DMODULES="iiwa14:16:fc"`), `_eh`
(exact-Hessian SO-SQP; `GATO_EXACT_HESSIAN`) and `_rk4` (the classical
fourth-order step on the generated grid integrator instead of the default
trapezoidal step; `gato.build(..., integrator="rk4")`, `-DMODULES="iiwa14:16:rk4"`;
four dynamics evaluations per knot, so a wider stable step at a higher cost per
iteration). Load one with `gato.BSQP(..., variant="fc")`; list them with
`gato.available("fc")`; `BSQP.integrator` names the step a module runs.

### Requirements

- Linux (developed on Ubuntu 22.04/24.04)
- CUDA toolkit ≥ 12.x matching your GPU arch (sm_120 needs ≥ 12.8)
- A C++17 host compiler (gcc 11+)
- CMake ≥ 3.22 (≥ 3.24 recommended for automatic GPU-arch detection)
- Python ≥ 3.10
- Docker (optional — only for the containerized build)

## Usage

```python
import numpy as np
import gato

# one batched solve: B trajectories across the solver's CUDA kernel pipeline.
# Configuration is ONE object (gato.SolverParams; max_sqp_iters=1 by default);
# keyword overrides of its fields are accepted directly.
solver = gato.BSQP(model_path="examples/indy7_description/indy7.urdf",
                   batch_size=8, N=64, dt=0.01, plant_type="indy7",
                   params=gato.SolverParams(max_sqp_iters=10))
x0 = np.zeros((8, solver.nx), dtype=np.float32)          # [q, dq] per batch entry
goals = np.zeros((8, 64 * 6), dtype=np.float32)          # (x,y,z,0,0,0) per knot
goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
res = solver.solve(x0, goals)                            # -> SolveResult (cold start: hold at x0)
res = solver.solve(x0, goals, xu_warm=res.xu)            # warm-started from the previous solution
print(res.u0(0), res.stats.sqp_iters, res.solve_time_us)
```

The same call takes torch or JAX arrays and returns the same kind. CUDA tensors
never leave the device: inputs are read in place and `res.xu` comes back
through DLPack, bit-identical to the numpy path.

```python
import torch
x0_t = torch.from_numpy(x0).cuda()
goals_t = torch.from_numpy(goals).cuda()
res = solver.solve(x0_t, goals_t)                        # res.xu: torch.float32 tensor on cuda:0
res = solver.solve(x0_t, goals_t, xu_warm=res.xu)        # warm start stays on the device
```

For closed-loop control, wrap the solver in the task-agnostic `MPCController`
(warm-start shifting, best-of-batch hypothesis selection) or go straight to a
gymnasium policy:

```python
from gato import MPCController, MPCPolicy, TrajectoryReference
from gato.envs import ArmTrackEnv   # needs the [examples] extra (gymnasium)
```

The intro demos in [examples/](examples/) walk the whole surface:
`01_single_solve.py`, `02_batched_solve.py` (per-batch hyperparameters),
`03_mpc_loop.py`, `04_gym_mpc.py` (MPC-as-policy + force-hypothesis batch),
`05_build_your_robot.py` (`gato.register_robot` / `gato.build` on your URDF), `06_constraints.py`,
`07_go2_floating.py` (quadruped, floating base), `08_linsys_and_autotune.py`,
`09_mpc_with_limits.py` (limit mechanisms and simulated closed-loop violations),
`10_torch_jax_arrays.py` (the solver and the controller on torch / JAX device tensors),
`11_deadline_step.py` (the step budget), `12_constraint_mechanisms.py` (the four limit
mechanisms on one task), `13_more_arms.py` (the FR3, Gen3 and UR5e on figure-8s sized
from their own limits), `14_mujoco_closed_loop.py` (an independent MuJoCo loop),
`15_solver_stats.py` (where a solve's time goes, stage by stage), `20_integrators.py` (the
trapezoidal and the RK4 step side by side), `16_cost_menu.py` to
`19_go2_standing.py` (the cost menu: orientation targets, robust and barrier activations, linear
features, a gravity-relative effort cost, the Go2 base-pose weights; [docs/cost_menu.md](docs/cost_menu.md)).
See [examples/cpp](examples/cpp/) for the same batched solve in C++/CUDA (below).

### C++/CUDA interface (power users)

The Python package is a thin wrapper: the solver is the header-only `BSQP<T>`
class in `gato/bsqp/bsqp.cuh`, and everything it reads or writes is device
memory you own (`solve(d_xu, inputs)` with `ProblemInputs{dt, d_x_s_batch,
d_reference_traj_batch}`; the same constraint, cost and warm-start surface the
Python class exposes). Python is needed once, to generate a robot's headers
(`gato.register_robot`, or the vendored robots as shipped); after that a
translation unit needs four include directories and three defines, which
`python -m gato.paths` prints for any (robot, horizon, variant):

```sh
pip install gato-mpc                                   # headers ship in the wheel
cmake -S examples/cpp -B build-cpp -DGATO_ROBOT=indy7 -DGATO_N=32   # python -m gato.paths resolves the rest
cmake --build build-cpp && ./build-cpp/batched_solve   # B figure-8 solves + a warm start, straight on BSQP<T>
```

`examples/cpp/batched_solve.cu` is the template (60 lines of setup around two
`solve` calls); `examples/cpp/CMakeLists.txt` is a standalone build that works
from a wheel or a checkout. Plant, horizon and variant are compile-time
(`KNOT_POINTS`, `GATO_PLANT_NAME`, `GATO_PLANT_HEADER`, plus
`GATO_CONTACT_FORCES=1` / `USE_EXACT_HESSIAN=1`), so one binary is one
(robot, N, variant) — exactly how the Python modules are built.

### Python API at a glance

| object | owns | you call it for |
|---|---|---|
| `gato.SolverParams` | THE solver configuration (frozen dataclass, `.replace()`) | every knob: SQP/PCG budgets, `mu`, `rho`, cost weights, `linsys`, `exact_hessian`, `defect_exit` |
| `gato.BSQP` | one compiled module + its device buffers; **stateless in the trajectory**, stateful in duals / adapted rho | `solve(x, ref, xu_warm=None)`, the `set_*` cost/constraint surface, `ee_pos`, `sim_forward` |
| `gato.MPCController` | the warm-start buffer (shift/hold), the per-step linsys policy, hypothesis batches, re-seeding, the URDF limit rows (ADMM by default on arms without existing rows; `limits="off"` adds no rows and preserves existing ones) | `reset(x0)`, `warmup`, `step(x, ref, budget_s=None) -> StepResult` (`u` = ACTUATED control to apply; `limits` / `row_viol` = the mechanism in force and the plan's violation per row group; a budget limits SQP work at iteration boundaries but can overrun) |
| `gato.MPCPolicy` + `TrajectoryReference` / `GoalReference` | the clock and reference window | `policy(obs) -> action` in any gym-style loop |
| `gato.MPC_GATO` | the paper's closed-loop SIMULATION driver (pinocchio-RK4 / MuJoCo world, pacing, task loops) | `run_mpc_fig8`, `run_mpc_goals` — not a controller |
| `gato.worlds.{PinocchioWorld, MuJoCoWorld}` | an independent simulator to close the loop against | contact / disturbance experiments |
| `gato.ForceHypothesisBatch` + `ForceEstimator` / `CEMForceEstimator` | the batch-as-identity disturbance layer | each batch entry solves under its own wrench hypothesis; reality picks the winner |
| `gato.build` / `gato.available` / `gato.robot_info` / `gato.fingerprint` (a submodule: `import gato.fingerprint as fp`) | codegen + compile, module discovery, registry, the dynamics fingerprint | adding robots, checking that an external simulator is the same robot |

### Linear system: pcg / bdsv / auto

The Schur system `S λ = γ` is solved either iteratively (`pcg`, warm-start
friendly) or directly (`bdsv`, block-Cholesky, iteration-count free).
`SolverParams(linsys=...)` pins the raw solver's path (None = pcg fixed base /
bdsv floating base); `MPCController(linsys="auto", bdsv_threshold=0.1)` (the
fixed-base default) picks per step from warm-startedness: a cold step (large
`‖x_measured − x_predicted‖`, or a PCG cap-out) takes one exact solve. The
threshold is per-workload: `tools/autotune_linsys.py` probes a task and persists
a tuned entry in `linsys_tuning.json` (written at runtime next to the registry, gitignored; timing runs — quiet box only).

### Floating base (quadruped)

`go2` is vendored as a floating-base plant (quaternion free-flyer root, SE(3)
step/linearization from GRiD's `grid_plant`, semi-implicit Euler integrator (RK4 as the `_rk4` variant), tangent-space
state cost; N16-only module). The stored state is `[p(3); quat xyzw(4); q_j; qd]`
and every knot's tangent is `[v_lin; omega; Δq_j; Δqd]` (2·nv); `gato.common.state_difference`
/ `check_floating_state` are the manifold helpers, `gato.worlds.MuJoCoWorld(floating=True)`
the ground-plane simulator.

**Contact forces on the feet** (`bsqpN16_go2_fc`, `gato.build(..., floating_base=True,
contact_forces=True)`): the fc variant appends one world-aligned wrench `[n; f]` per
baked foot frame to every control (`FC_SIZE = 24`; `solver.contact_frames`,
`solver.fc_slots(frame, "n"|"f")`). The wrenches are the solver's contact
EXPLANATION, not commands to the world: a standing loop with `fc_ref = mg/4` up per
foot and the moment rows pinned (`add_fc_box(0, 0, slots=..., mech="al")`) holds
the stance height on MuJoCo with each foot carrying mg/4 — the contactless model
collapses from the same start (`test_floating_worlds.py`). Two traps: the standing
keyframe (base z 0.35) has the feet 8.5 cm in the air — derive the stance pose from
FK (feet touch at z ≈ 0.287); and an asymmetric `fc_ref` does NOT shift weight — the
world's split follows the centre of mass, so the solver plans for an imagined split
and can tip over. Weight shift / foot lift also need a base-tracking cost and
a pre-liftoff support plan; contact rows alone do not establish walking.
`python examples/07_go2_floating.py --fc` runs the standing recipe.

**Foot position rows and the gait oracle** (CL-4): `add_contact_pos_rows` puts
per-knot residual rows `p_f(q_k) - tgt_k` on every baked contact frame (on-device
FK over GRiD's `contact_frame_positions` surface) — an equality per knot pins a
stance foot where it touched down, a one-sided z row is a swing clearance, and a
per-knot target table (`set_row_group_targets`) tracks a swing curve. The gait is
GIVEN, never discovered: `gato.gait.GaitSchedule` (trot/bound/pace/walk/stand,
period, duty, phase offsets) rolls the stance mask over the horizon and
`GaitProgrammer` writes it into the solver every tick — swing-foot wrench pins,
per-knot normal-force reference, stance-masked friction cones (`install_cones`) and,
with `install_foot_rows`, the stance/swing foot rows (`apply(t, q)`). Operating
point: the rows fold onto the Q block against the standing costs, so solver-level
enforcement needs AL rho ~1e3 (ADMM ~1e2; rho 10 is dominated) — verified solver-only
(feet held < 1 mm, a 3 cm foot lift in the horizon, `test_contact_rows.py`). In the
closed loop, static standing with AL stance rows now passes its regression gate
after non-PD factor recovery and measured-foot target re-anchoring. Moving targets
constrain each prediction horizon; they do not prove zero accumulated physical
slip. S2 weight shift and S3 foot lift remain open work; see
[feature status](docs/status.md) and [constraints](docs/constraints.md).

### Same robot? The dynamics fingerprint

Before comparing controllers across simulators, run
`import gato.fingerprint as fp; fp.check(my_qdd_fn, "iiwa14")` — per-joint inertia-response
ratios against `test/dynamics_fingerprint.json` (pinned URDF sha + probes).
A ratio away from 1 is a model mismatch, not a solver bug
([docs/consumer_contract.md](docs/consumer_contract.md)).

### Constraints

Beyond the tracking cost, `BSQP` carries a **constraint row-group layer**:
joint position/velocity/torque boxes (from the URDF `<limit>` tables), an EE
terminal-position equality, and linear-map / second-order-cone rows on the
controls. Groups bind to one of four mechanisms:

| mechanism | call | character |
|---|---|---|
| telemetry | `enable_limit_telemetry()` | report-only — violation stats every solve, solver path untouched (bit-identical) |
| relaxed barrier | `enable_limit_barrier(mu, delta)` | soft interior penalty, infeasible-start safe |
| ADMM projection | `enable_limit_admm(rho, iters, eps_rel=...)` | inner splitting loop per SQP step, tight transients; `set_admm_exit` stops the loop on its residuals |
| augmented Lagrangian | `enable_limit_al(rho)` | PHR outer duals — exact at convergence, made for warm-started MPC |

```python
solver.enable_limit_al(rho=1.0)              # boxes from the URDF limit tables
res = solver.solve(x0, goals)
res.stats.row_max_violation                  # (group, batch) telemetry, always on once a row group exists

# EE terminal-position equality on top of the boxes (reach-to-point)
solver.enable_ee_terminal_equality(target_xyz, rho=10.0)

# cone on a mapped control quantity g = C @ u + d — e.g. an EE contact-force
# friction cone with C = S @ pinv(J(q).T), frozen at the contact config q
solver.enable_u_cone(C, d, mech="admm", rho=0.01)       # exact second-order cone
solver.enable_u_cone(C, d, form="pyramid", facets=8)    # linear-facet approximation
```

`enable_u_cone` enforces `‖g[1:]‖ <= g[0]` (row 0 = the cone axis).
`form="soc"` is exact: ADMM projects onto the cone each inner iteration, AL
runs the conic PHR update (dual vector projected onto the cone), and
`mech="barrier"` penalizes the margin `g[0] - ‖g[1:]‖`. `form="pyramid"`
replaces the cone with one-sided linear facets riding the ordinary interval
machinery (`facet_scale="inscribed"` is conservative: facet-feasible implies
cone-feasible). Arbitrary affine control rows are the same surface one level
down: `add_lin_u_rows(C, d, lo=..., hi=...)` appends interval rows
`lo <= C @ u + d <= hi`.

Mechanisms mix across groups (e.g. AL boxes + an ADMM cone). Two rules:
call `add_lin_u_rows`/`enable_u_cone` **after** `enable_limit_*` (mechanism
enables reinstall the canonical groups and therefore reject that ordering), and when the
map `C` is large, scale `rho` down by `‖C‖²` — the fold lands `rho * CᵀC` on
the control Hessian block. Per-solve duals and ADMM state are inspectable via
`get_row_duals()` / `get_admm_state()`; `set_row_group_soft(g, sigma)` turns a
hard group into a slack-penalized one. The full parameter surface is in the
[docs/constraints.md](docs/constraints.md) (measured defaults and provenance per method).

### Adding a robot

The dynamics layer's file map is in [gato/dynamics/README.md](gato/dynamics/README.md).

One call generates the dynamics code (via GRiD), the limit tables, and compiles
the solver modules from a URDF:

```python
import gato
gato.build("path/to/robot.urdf", name="myrobot", N=[32, 64], ee_frame="EE")
gato.build("path/to/quad.urdf", name="quad", N=[16], ee_frame="imu_joint",
           floating_base=True, contact_frames=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"])
gato.build("path/to/robot.urdf", name="myrobot", N=[16], contact_forces=True)   # the "fc" variant
# then: gato.BSQP(model_path="path/to/robot.urdf", N=32, plant_type="myrobot", ...)
```

From a wheel (no checkout), the same robot is registered once and compiled on
first use into the per-host cache:

```python
gato.register_robot("myrobot", "path/to/robot.urdf", N=[32, 64], ee_frame="EE")
gato.register_robot("quad", "path/to/quad.urdf", N=[16], ee_frame="imu_joint", floating_base=True,
                    contact_frames=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"])
solver = gato.BSQP("path/to/robot.urdf", N=32, plant_type="myrobot", ...)   # compiles bsqpN32_myrobot once
```

`ee_frame` must be a **fixed joint** in the URDF (the EE target frame the cost
tracks; on a floating base, the base-pose target frame); every actuated joint
needs bounded `<limit>` tags (the barrier cost uses them). `joint_dynamics=True`
bakes the URDF's `<dynamics damping>` and `friction` into the dynamics and
its derivatives, the friction as GRiD's smooth `tanh` model with
`friction_width` (rad/s, default 0.03; match it to the plan step). `cost_frames=[...]`
bakes extra URDF joints as cost frames for the cost menu's second position term (the
end-effector target is always cost frame 0; [docs/cost_menu.md](docs/cost_menu.md)). Scope: serial chains,
fixed or floating base (quaternion free-flyer). Codegen is skipped when nothing
that feeds it changed (URDF, the grid-rbd release, algorithm set, options). The same path is
exposed as a CLI for the vendored robots: `python tools/regen_grid.py`. Built
modules and robot metadata are discoverable via `gato.available()` /
`gato.robot_info(name)` / `gato.robot_urdf(name)`; the registry
(`python/gato/_registry.json`) is tracked and gated for freshness by
`test/test_codegen.py`; registered robots live in the cache manifest.

Constraint layer (limit boxes, EE rows, cones, collision; barrier / ADMM / AL
mechanisms, measured defaults and provenance): [docs/constraints.md](docs/constraints.md).

## Tests

The suite needs the `[test]` and `[dev]` extras (`./tools/install.sh --test --dev`): a missing
dependency is a broken environment, never a skip (`test_floating_*` import pinocchio at collection).

```sh
pytest -m "not gpu"           # host-only: packaging, math, codegen determinism
pytest -m "gpu and not slow"  # GPU: smoke solves, determinism, shapes, controller
pytest                        # everything (slow adds codegen diff + a build dogfood)
```

The five standalone single-block kernel harnesses in [test/cuda/](test/cuda/)
are built and run by `test/test_kernel_gates.py` (slow); `test/test_parity_golden.py`
pins bitwise goldens for every receipt module (`test/golden/`, re-baseline with
`GATO_GOLDEN_REBASELINE=1` and say why in the commit).

**GPU CI** uses [pytest-gpu-proof](https://github.com/A2R-Lab/pytest-gpu-proof):
the full suite runs on a real GPU via `./test/run_gpu_proof.sh`, which emits a
**signed receipt** (`gpu-proof.json`) binding the git SHA, a source fingerprint,
and per-test outcomes; a CPU-only GitHub Action verifies the signature against
the signer's public GitHub keys on pushes to `main`,
pull requests and a weekly cron (receipts expire after 30 days). The same
workflow runs the host-only tier in CI with the full `[test]` deps, and fails
on ANY host-tier skip. Sign receipts from the project `.venv` (`--test --dev`).
The step-by-step developer workflow (capped builds, goldens, receipt, verify, the
quiet-window timing rule) is [docs/development.md](docs/development.md).

## Reproducing the paper

Committed scripts that regenerate the data and figures from
[the paper](https://arxiv.org/abs/2510.07625) live in
**[examples/paper-figures/](examples/paper-figures/)** — one `reproduce_figN_*.py` per figure. Fig-4/5/7
regenerate their data on the GPU **by default**, re-render from saved/recovered data with `--replot`,
and run a fast smoke with `--quick`; Fig-3 assembles the table and plots from the committed sweep CSVs
(no GPU) and re-runs a lane only on request (`--run-gato`, `--run-bt`; MPCGPU cells via `--mpcgpu-timing-dir`). Run from the repo root:

```bash
python examples/paper-figures/reproduce_fig4_hparam.py --replot   # Tier A: no GPU, bundled data
python examples/paper-figures/reproduce_fig3_fair.py --run-gato    # Tier B: GPU re-run (timing: quiet box)
python examples/paper-figures/make_all.py --quick                  # smoke every figure
```

Build the module set first: `./tools/build.sh --profile receipt` (the arms at
every paper horizon + go2 N16 + the fc/eh variants).

| Paper element | Script | Notes |
|---|---|---|
| **Fig-3** scalability (iiwa14 fig-8, GATO vs a multi-threaded QDLDL-based CPU solver vs MPCGPU) | `reproduce_fig3_fair.py` (+ `benchmarks/iiwa_fig8_shared.py`, `sweep_batch_iiwa_fig8.py`) | the fair 3-way protocol (1 SQP iter, EE-frame metric); `--robot indy7` runs the GATO and CPU lanes on the paper's arm |
| **Fig-4** (CS1) iiwa14 online ρ convergence | `reproduce_fig4_hparam.py` | regenerates by default; `--replot` uses bundled `examples/paper-figures/gato_hparam_batch_results.pkl` |
| **Fig-5** (CS2) Indy7 disturbance rejection | `reproduce_fig5_disturbance.py` | fixed-pacing force sweep + EE trajectories; not a reproduction of latency-induced degradation |
| **Fig-7 + Table-I** (CS3) iiwa14 pick-place | `reproduce_fig7_pickplace.py` | exploratory 2026-10-03 arm-arrival results: `--task stop` (ramp and dwell) 15 → 100 % and `--task pass-through` 26 → 97 % from B = 1 to 8–32; neither tests payload placement or joint-limit feasibility |

See [examples/paper-figures/README.md](examples/paper-figures/README.md) for the full build matrix,
reproducibility tiers, hardware/config delta, and caveats (MPCGPU/CPU baselines). Fig-6 (sim
snapshot) and Fig-8 / Table-II (hardware) are not reproducible in software. Timing legs are
QUIET-BOX runs — never on a shared machine. Provenance for every recovered dataset is in
[docs/archaeology.md](docs/archaeology.md) (a dated 2026-06 snapshot).

## Related

GATO is the top of a chain of A²R Lab libraries, each released and pinned on its own:
[GLASS](https://github.com/A2R-Lab/GLASS) (block-local GPU linear algebra) is bundled by
[GRiD](https://github.com/A2R-Lab/GRiD) (rigid-body dynamics, kinematics and collision code
generated per robot from its URDF, with analytical derivatives; the `grid-rbd` wheel GATO pins
in `pyproject.toml` and `python/gato/_grid.py`), and GATO adds the batched SQP, the
controller and the constraint layer on top. [MPCGPU](https://github.com/A2R-Lab/MPCGPU) is the
single-solve GPU MPC solver this project grew from. The documentation site is
<https://a2r-lab.org/GATO/docs/>; the project page with the paper's results is
<https://a2r-lab.org/GATO/>.

## Cite
```bibtex
@inproceedings{du2026gato,
    title={GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control}, 
    author={Alexander Du and Emre Adabag and Gabriel Bravo and Brian Plancher},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)}, 
    year={2026},
    month={June}
}
```

## Funding acknowledgement

This material is based upon work supported by the National Science Foundation
(under Awards [2411369](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2411369)
and [2246022](https://www.nsf.gov/awardsearch/show-award?AWD_ID=2246022)). Any opinions,
findings, conclusions, or recommendations expressed in this material are those
of the authors and do not necessarily reflect those of the funding organizations.
