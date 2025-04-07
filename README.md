# GPU-Accelerated Trajectory Optimization for Robotics

This is a batched version of the trajectory optimization solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079). 

## Installation
```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
git submodule update --init --recursive
docker-compose up -d
docker-compose exec dev bash
make build #builds examples and bindings
```
Bindings only
```sh
make build-bindings
```

### Requirements
GATO works with:
* Ubuntu 22.04
* CUDA v12.2
* C++17
* Python 3.10.12
* Docker 26.1.3

## Usage
See [batch_sqp.cu](examples/batch_sqp.cu) for an example of a batch solve in C++/CUDA, and [batch_sqp.py](examples/batch_sqp.py) for an example using Python bindings. Examples of MPC with GATO are in [indy7-mpc/](examples/indy7-mpc/)

## Nomenclature
TODO

## Related
* The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
* [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients