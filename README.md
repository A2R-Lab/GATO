# GPU-Accelerated Trajectory Optimization for Robotics

This is a batched version of the trajectory optimization solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079).

## Installation

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
```

Docker is used for containerization and [uv](https://docs.astral.sh/uv/) is used as a Python package/project manager.

Setup

```sh
./tools/install.sh
```

Docker

```sh
./tools/docker.sh
'''

Manual Installation
'''sh
git submodule update --init --recursive
uv sync
source .venv/bin/activate
docker compose up -d # build and run
docker compose exec dev bash # enter container
docker compose exec -w /workspace dev bash #enter container in the workspace directory

docker down # stop and remove
```

GATO

```sh
./tools/build.sh
```

### Requirements

GATO works with:

- Ubuntu 22.04
- CUDA v12.6
- C++17
- gcc 11.4.0
- Python 3.10.12
- Docker 28.1.0

## Usage

See [batch_sqp.cu](examples/batch_sqp.cu) for a minimal example of a batched trajectory optimization solve in C++/CUDA. Example Jupyter notebooks using GATO for MPC are in [examples/](examples/)

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients
