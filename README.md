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

## Quick start (host-native — no Docker needed)

Prerequisites (Linux):

- an NVIDIA driver + **CUDA toolkit ≥ 12.x matching your GPU architecture**
  (e.g. RTX 50xx / sm_120 needs CUDA ≥ 12.8) with `nvcc` on `PATH`
- `apt install build-essential cmake python3-dev python3-venv git`
  (CMake ≥ 3.24 recommended so the build auto-detects your GPU arch; older
  CMake requires an explicit architecture, e.g. `ARCH=120` for sm_120)
- Python ≥ 3.10

```sh
git clone --branch cleanup-modernization https://github.com/A2R-Lab/GATO.git
cd GATO
./tools/install.sh
MODULES="indy7:64" JOBS=2 ./tools/build.sh
source .venv/bin/activate
python examples/01_single_solve.py
python examples/02_batched_solve.py
```

This builds one solver module, sufficient for the two introductory demos and
the usage snippet below. Each demo reports solve results; printed durations on
a shared GPU are not benchmark measurements. The branch selection is temporary
until this API is merged into `main`.

GATO installs host-native into a project-local `.venv`, using only the Python
standard-library `venv` + `pip` — no Docker, no `uv` (same lightweight model as
GRiD's `base_install.sh`). The install script runs a preflight check (nvcc,
GPU, cmake) and tells you exactly what's missing. Optional install choices
(pick the extras you need; there is no need to run every line):

```sh
./tools/install.sh            # lean: codegen + build deps + submodules + regen grid.cuh
./tools/install.sh --examples #   + runtime to run the MPC/benchmark examples (torch, pinocchio, mujoco, viz)
./tools/install.sh --test     #   + full test-suite deps (pinocchio, mujoco, scipy, gymnasium)
./tools/install.sh --dev      #   + test tooling (pytest, pytest-gpu-proof)
./tools/install.sh --all      #   + examples + test + dev
```

(The paper's CPU baseline, `examples/benchmarks/baselines/sqpcpu`, is fetched
like every other submodule; `examples/benchmarks/baselines/build_cpu_baseline.sh`
compiles it — osqp and osqp-eigen into a local prefix — only when you reproduce Fig-3.) The lean default is all you need to generate code, build the solver, and run
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

**Distribution:** the supported native build path is a Git checkout with its
pinned submodules initialized (the install script does this). The pure-python
wheel contains no solver modules; these are GPU-arch/CUDA/ABI-specific CMake
products. The sdist contains GATO's own build sources but does not bundle the
external GRiD/GLASS trees, so it is NOT a standalone native-build distribution.
See [merge/release boundaries](docs/merge-readiness.md).

### Build Options

You can control which Python extension modules are built by selecting plant models and horizon lengths at CMake configure time:

```sh
cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="8;32;128"   # PLANT x KNOTS cross-product
cmake -S . -B build -DMODULES="indy7:8,32;go2:16"                # explicit per-plant horizons
cmake -S . -B build -DGATO_RECEIPT_PROFILE=ON                     # exactly test/receipt_modules.txt
cmake --build build --parallel 2                                # RAM-bound: roughly 1-7 GB per job
```

- `PLANT`: semicolon-separated plant targets (`indy7`, `iiwa14`, `go2`).
- `KNOTS`: semicolon-separated horizon lengths — crossed with EVERY plant, so
  use `MODULES` when plants differ (go2 is **N16-only**; longer floating-base
  horizons are compile blowups).
- `./tools/build.sh --profile receipt` builds the set the signed GPU receipt
  attests (`test/receipt_modules.txt`).
- `tools/build.sh` defaults to four jobs; use `JOBS=2` on a shared or RAM-limited
  machine. CMake caches `MODULES`: an old explicit list overrides `PLANT`/`KNOTS`.
  Clear it with `-DMODULES=""` when returning to the cross-product. The receipt
  profile takes precedence over both. `--clean` deletes `build/`, not just its cache.

Built Python modules are written to `python/gato/` as `bsqpN{N}_{plant}[_{variant}].so`.
**Variants** are separate ABIs that live side by side: `_fc` (contact-force
controls appended to `u`; `-DGATO_CONTACT_FORCES=ON`, `./tools/build.sh --variant fc`,
`gato.build(..., contact_forces=True)`, `-DMODULES="iiwa14:16:fc"`) and `_eh`
(exact-Hessian SO-SQP; `GATO_EXACT_HESSIAN`). Load one with
`gato.BSQP(..., variant="fc")`; list them with `gato.available("fc")`.

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
`05_build_your_robot.py` (`gato.build` on your URDF), `06_constraints.py`,
`07_go2_floating.py` (quadruped, floating base), `08_linsys_and_autotune.py`.
See [bsqp.cu](examples/bsqp.cu) for a minimal C++/CUDA batched solve.

### Python API at a glance

| object | owns | you call it for |
|---|---|---|
| `gato.SolverParams` | THE solver configuration (frozen dataclass, `.replace()`) | every knob: SQP/PCG budgets, `mu`, `rho`, cost weights, `linsys`, `exact_hessian` |
| `gato.BSQP` | one compiled module + its device buffers; **stateless in the trajectory**, stateful in duals / adapted rho | `solve(x, ref, xu_warm=None)`, the `set_*` cost/constraint surface, `ee_pos`, `sim_forward` |
| `gato.MPCController` | the warm-start buffer (shift/hold), the per-step linsys policy, hypothesis batches, re-seeding | `reset(x0)`, `warmup`, `step(x, ref) -> StepResult` (`u` = ACTUATED control to apply) |
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
step/linearization from GRiD's `grid_plant`, SI-Euler integrator, tangent-space
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
| ADMM projection | `enable_limit_admm(rho, iters)` | inner splitting loop per SQP step, tight transients |
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

`ee_frame` must be a **fixed joint** in the URDF (the EE target frame the cost
tracks; on a floating base, the base-pose target frame); every actuated joint
needs bounded `<limit>` tags (the barrier cost uses them). Scope: serial chains,
fixed or floating base (quaternion free-flyer). Codegen is skipped when nothing
that feeds it changed (URDF, GRiD pin, algorithm set, options). The same path is
exposed as a CLI for the vendored robots: `python tools/regen_grid.py`. Built
modules and robot metadata are discoverable via `gato.available()` /
`gato.robot_info(name)`; the registry (`python/gato/_registry.json`) is tracked
and gated for freshness by `test/test_codegen.py`.

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
the signer's public GitHub keys on pushes to `main`/`cleanup-modernization`,
pull requests and a weekly cron (receipts expire after 30 days). The same
workflow runs the host-only tier in CI with the full `[test]` deps, and fails
on ANY host-tier skip. Sign receipts from the project `.venv` (`--test --dev`).

## Reproducing the paper

Committed scripts that regenerate the data and figures from
[the paper](https://arxiv.org/abs/2510.07625) live in
**[examples/paper-figures/](examples/paper-figures/)** — one `reproduce_figN_*.py` per figure. Fig-4/5/7
regenerate their data on the GPU **by default**, re-render from saved/recovered data with `--replot`,
and run a fast smoke with `--quick`; Fig-3 assembles the table and plots from the committed sweep CSVs
(no GPU) and re-runs a lane only on request (`--run-gato`, `--run-bt`, `--run-mpcgpu`). Run from the repo root:

```bash
python examples/paper-figures/reproduce_fig4_hparam.py --replot   # Tier A: no GPU, bundled data
python examples/paper-figures/reproduce_fig3_fair.py --run-gato    # Tier B: GPU re-run (timing: quiet box)
python examples/paper-figures/make_all.py --quick                  # smoke every figure
```

Build the module set first: `./tools/build.sh --profile receipt` (the arms at
every paper horizon + go2 N16 + the fc/eh variants).

| Paper element | Script | Notes |
|---|---|---|
| **Fig-3** scalability (iiwa14 fig-8, GATO vs BatchThneed-CPU vs MPCGPU) | `reproduce_fig3_fair.py` (+ `benchmarks/iiwa_fig8_shared.py`, `sweep_batch_iiwa_fig8.py`) | the fair 3-way protocol (1 SQP iter, EE-frame metric); the June single-robot chain is in `examples/archive/` |
| **Fig-4** (CS1) iiwa14 online ρ convergence | `reproduce_fig4_hparam.py` | regenerates by default; `--replot` uses bundled `examples/gato_hparam_batch_results.pkl` |
| **Fig-5** (CS2) Indy7 disturbance rejection | `reproduce_fig5_disturbance.py` | fixed-pacing force sweep + EE trajectories; not a reproduction of latency-induced degradation |
| **Fig-7 + Table-I** (CS3) iiwa14 pick-place | `reproduce_fig7_pickplace.py` | runnable; success magnitudes and protocol/metric provenance remain unresolved; full refresh deferred |

See [examples/paper-figures/README.md](examples/paper-figures/README.md) for the full build matrix,
reproducibility tiers, hardware/config delta, and caveats (MPCGPU/CPU baselines). Fig-6 (sim
snapshot) and Fig-8 / Table-II (hardware) are not reproducible in software. Timing legs are
QUIET-BOX runs — never on a shared machine. Provenance for every recovered dataset is in
[docs/archaeology.md](docs/archaeology.md) (a dated 2026-06 snapshot).

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients

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
