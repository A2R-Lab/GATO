# GATO
> GPU-Accelerated Trajectory Optimization

Numerical experiments and the open-source solver from  ["GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control"](https://arxiv.org/abs/2510.07625)

## Installation (host-native — no Docker needed)

Prerequisites (Linux):

- an NVIDIA driver + **CUDA toolkit ≥ 12.x matching your GPU architecture**
  (e.g. RTX 50xx / sm_120 needs CUDA ≥ 12.8) with `nvcc` on `PATH`
- `apt install build-essential cmake python3-dev python3-venv git`
  (CMake ≥ 3.24 recommended so the build auto-detects your GPU arch; older
  CMake falls back to a fixed arch list — override with
  `-DCMAKE_CUDA_ARCHITECTURES=...`)
- Python ≥ 3.10

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
```

GATO installs host-native into a project-local `.venv`, using only the Python
standard-library `venv` + `pip` — no Docker, no `uv` (same lightweight model as
GRiD's `base_install.sh`). The install script runs a preflight check (nvcc,
GPU, cmake) and tells you exactly what's missing.

```sh
./tools/install.sh            # lean: codegen + build deps + submodules + regen grid.cuh
./tools/install.sh --examples #   + runtime to run the MPC/benchmark examples (torch, pinocchio, viz)
./tools/install.sh --dev      #   + test tooling (pytest)
./tools/install.sh --all      #   + examples + dev
```

The lean default is all you need to generate code and build the solver. The
heavy runtime (torch, pinocchio, Qt/meshcat viz) is only pulled in by
`--examples`. Then activate and build:

```sh
source .venv/bin/activate
./tools/build.sh              # incremental; --clean to reconfigure; PLANT=/KNOTS=/ARCH= to subset
```

Docker remains available as an **optional** fallback for reproducible builds
(`./tools/docker.sh` — a thin image that wraps the same `tools/install.sh`).

### Build Options

You can control which Python extension modules are built by selecting plant models and horizon lengths at CMake configure time:

```sh
mkdir -p build && cd build
cmake -DPLANT="indy7;iiwa14" -DKNOTS="8;32;128" ..
cmake --build . --parallel
```

- `PLANT`: semicolon-separated list of plant targets (`indy7`, `iiwa14`).
- `KNOTS`: semicolon-separated list of horizon lengths.

Built Python modules are written to `python/gato/` as `bsqpN{N}_{plant}.so`.

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

# one batched solve: B trajectories in a single GPU launch
solver = gato.BSQP(model_path="examples/indy7_description/indy7.urdf",
                   batch_size=8, N=32, dt=0.01, plant_type="indy7")
x0 = np.zeros((8, solver.nx), dtype=np.float32)          # [q, dq] per batch entry
goals = np.zeros((8, 32 * 6), dtype=np.float32)          # (x,y,z,0,0,0) per knot
goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
res = solver.solve(x0, goals)                            # -> SolveResult
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
`03_mpc_loop.py`, `04_gym_mpc.py` (MPC-as-policy + force-hypothesis batch).
See [bsqp.cu](examples/bsqp.cu) for a minimal C++/CUDA batched solve.

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
res.stats.row_max_violation                  # (group, batch) telemetry, always on

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
enables reinstall the canonical groups, dropping appended ones), and when the
map `C` is large, scale `rho` down by `‖C‖²` — the fold lands `rho * CᵀC` on
the control Hessian block. Per-solve duals and ADMM state are inspectable via
`get_row_duals()` / `get_admm_state()`; `set_row_group_soft(g, sigma)` turns a
hard group into a slack-penalized one. The full parameter surface is in the
`BSQP` docstrings.

### Adding a robot

One call generates the dynamics code (via GRiD), the limit tables, and compiles
the solver modules from a fixed-base URDF:

```python
import gato
gato.build("path/to/robot.urdf", name="myrobot", N=[32, 64], ee_frame="EE")
# then: gato.BSQP(model_path="path/to/robot.urdf", N=32, plant_type="myrobot", ...)
```

`ee_frame` must be a **fixed joint** in the URDF (the EE target frame the cost
tracks); every actuated joint needs bounded `<limit>` tags (the barrier cost
uses them). Current scope: fixed-base serial chains. The same path is exposed as
a CLI for the vendored robots: `python tools/regen_grid.py`. Built modules and
robot metadata are discoverable via `gato.available()` / `gato.robot_info(name)`.

## Tests

```sh
pytest -m "not gpu"           # host-only: packaging, math, codegen determinism
pytest -m "gpu and not slow"  # GPU: smoke solves, determinism, shapes, controller
pytest                        # everything (slow adds codegen diff + a build dogfood)
```

There is also a standalone single-block PCG-vs-CPU harness in
[test/cuda/](test/cuda/) (build command in the file header).

**GPU CI** uses [pytest-gpu-proof](https://github.com/A2R-Lab/pytest-gpu-proof):
the full suite runs on a real GPU via `./test/run_gpu_proof.sh`, which emits a
**signed receipt** (`gpu-proof.json`) binding the git SHA, a source fingerprint,
and per-test outcomes; a CPU-only GitHub Action verifies the signature against
the signer's public GitHub keys on every push (no cloud GPUs). The same workflow
also runs the host-only test tier directly in CI.

## Reproducing the paper

Committed scripts that regenerate the data and figures from
[the paper](https://arxiv.org/abs/2510.07625) live in
**[examples/paper-figures/](examples/paper-figures/)** — one `reproduce_figN_*.py` per figure. Each
regenerates its data on the GPU **by default**, re-renders from saved/recovered data with `--replot`,
and runs a fast smoke with `--quick`. Run from the repo root:

```bash
python examples/paper-figures/reproduce_fig4_hparam.py --replot   # Tier A: no GPU, bundled data
python examples/paper-figures/reproduce_fig3_scalability.py        # Tier B: GPU re-run (default)
python examples/paper-figures/make_all.py --quick                  # smoke every figure
```

Build the needed `(plant, N)` modules first (one shot):
`cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="8;16;32;64;128" -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 4`.

| Paper element | Script | Notes |
|---|---|---|
| **Fig-3 left** scalability (Indy7 fig-8, GATO vs OSQP/MPCGPU) | `reproduce_fig3_scalability.py` | GATO + OSQP-CPU; MPCGPU line optional |
| **Fig-3 right** GATO (N×M) heat map | `reproduce_fig3_heatmap.py` | needs indy7 N∈{8..128} |
| **Fig-4** (CS1) iiwa14 online ρ convergence | `reproduce_fig4_hparam.py` | regenerates by default; `--replot` uses bundled `examples/gato_hparam_batch_results.pkl` |
| **Fig-5** (CS2) Indy7 disturbance rejection | `reproduce_fig5_disturbance.py` | force sweep + EE trajectories |
| **Fig-7 + Table-I** (CS3) iiwa14 pick-place | `reproduce_fig7_pickplace.py` | ⚠️ gated on a known iiwa14 instability (see below) |

See [examples/paper-figures/README.md](examples/paper-figures/README.md) for the full build matrix,
reproducibility tiers, hardware/config delta, and honest caveats (MPCGPU/CPU baselines, the Fig-7
instability). Fig-6 (sim snapshot) and Fig-8 / Table-II (hardware) are not reproducible in software.
Provenance for every recovered dataset is in [docs/archaeology.md](docs/archaeology.md).

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients

## Cite

```bibtex
@misc{du2025gatogpuacceleratedbatchedtrajectory,
      title={GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control}, 
      author={Alexander Du and Emre Adabag and Gabriel Bravo and Brian Plancher},
      year={2025},
      eprint={2510.07625},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07625}, 
}
```
