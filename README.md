# GPU-Accelerated Trajectory Optimization for Robotics

This is a batched version of the trajectory optimization solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079).

## Setup & Installation
[uv](https://docs.astral.sh/uv/) is used as a Python package and project manager.

```sh
# clone repo and submodules
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
git submodule update --init --recursive

# python virtual env and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# setup docker container
docker-compose up -d
docker-compose exec dev bash

make build
```

Make bindings only:

```sh
make build-bindings
```

To stop the docker container:

```sh
docker-compose down
```

<!--
Using Regular Docker (not docker-compose)
```sh
docker build -t gato .
docker run --gpus all -it -v $(pwd):/app gato
``` -->

### Requirements

GATO works with:

- Ubuntu 22.04.5
- C++17
- CUDA v12.2
- Python 3.10.12
- Docker 28.1.0

## Usage

See [batch_sqp.cu](examples/batch_sqp.cu) for an example of a batch solve in C++/CUDA, and [batch_sqp.py](examples/batch_sqp.py) for an example using Python bindings. Examples of MPC with GATO are in [indy7-mpc/](examples/indy7-mpc/)

## Nomenclature

TODO

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients
