# GATO
> GPU-Accelerated Trajectory Optimization

Numerical experiments and the open-source solver from  ["GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control"](https://arxiv.org/abs/2510.07625)

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
docker build -t gato . # build image
docker run -d -it --gpus all --network host -e DISPLAY=:0 -v $(pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix --name gato-container gato # run container
docker exec -it gato-container bash # enter container
docker exec -it --workdir /workspace gato-container bash # enter container in the workspace directory

docker stop gato-container && docker rm gato-container # stop and remove
```

GATO

```sh
./tools/build.sh
```

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
- Python 3.10.12
- Docker 28.1.0

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
