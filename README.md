# GPU-Accelerated Trajectory Optimization for Robotics

This is a batched version of the trajectory optimization solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079).

## Installation


### Clone the repository

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
```

Docker is used for containerization and [uv](https://docs.astral.sh/uv/) is used as a Python package/project manager.

### Setup (Docker, recommended)

```sh
# setup dependencies, build container, and make
./tools/install.sh
```

#### Manual Build (without Docker)

Ensure you have CUDA 12.6 installed and available on your system. Set the environment variables so that the correct CUDA toolkit is used:

```sh
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

Check that the correct nvcc is being used:

```sh
which nvcc
nvcc --version
# Should show CUDA 12.6
```

Then build the project:

```sh
rm -rf build
mkdir build
cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc ..
cmake --build . --parallel
```

### Docker

```sh
# build + run + enter container
./tools/docker.sh

# manually
docker compose up -d # build and run
docker compose exec dev bash # enter container
docker compose exec -w /workspace dev bash #enter container in the workspace directory

docker compose down # stop and remove
```

### GATO Make Targets

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
- CUDA v12.6 (required for Hopper/H100, sm_89)
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
