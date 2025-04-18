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
# setup dependencies, build container, and make
./tools/install.sh
```

Docker

```sh
# build + run + enter container
./tools/docker.sh

# manually
docker compose up -d # build and run
docker compose exec dev bash # enter container
docker down # stop and remove
```

GATO

```sh
# examples, benchmark, and bindings
make build

# bindings only
make build-bindings

# clean
make clean
```

### Requirements

GATO works with:

- Ubuntu 22.04
- CUDA v12.2
- C++17
- Python 3.10.12
- Docker 28.1.0

## Usage

See [batch_sqp.cu](examples/batch_sqp.cu) for an example of a batch solve in C++/CUDA, and [batch_sqp.py](examples/batch_sqp.py) for an example using Python bindings. Examples of MPC with GATO are in [indy7-mpc/](examples/indy7-mpc/)

## Nomenclature

TODO

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients
