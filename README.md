# GATO
> GPU-Accelerated Trajectory Optimization

Numerical experiments and the open-source solver from  ["GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control"](https://arxiv.org/abs/2510.07625)

## Installation

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
```

GATO installs host-native (assumes CUDA is installed on the host) into a
project-local `.venv`, using only the Python standard-library `venv` + `pip`
— no Docker, no `uv` (same lightweight model as GRiD's `base_install.sh`).

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
./tools/build.sh
```

Docker is still available (optional, for reproducible builds) via `./tools/docker.sh`.

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

- Ubuntu 22.04
- CUDA 12.6
- C++17
- gcc 11.4.0
- Python >= 3.10
- Docker 28.1.0 (optional — only for the containerized build)

## Usage

See [bsqp.cu](examples/bsqp.cu) for a minimal C++/CUDA batched solve, and the intro Python demos in
[examples/](examples/) (`01_single_solve.py`, `02_batched_solve.py`, `03_mpc_loop.py`).

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
