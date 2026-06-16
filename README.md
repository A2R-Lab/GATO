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

Built Python modules are written to `python/bsqp/` as `bsqpN{N}_{plant}.so`.

### Requirements

- Ubuntu 22.04
- CUDA 12.6
- C++17
- gcc 11.4.0
- Python >= 3.10
- Docker 28.1.0 (optional — only for the containerized build)

## Usage

See [batch_sqp.cu](examples/bsqp.cu) for a minimal example of a batched trajectory optimization solve in C++/CUDA. Example Jupyter notebooks using GATO for MPC are in [examples/](examples/)

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
