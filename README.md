# GATO
GPU-Accelerated Trajectory Optimization for Robotics.


**---CURRENTLY UNDER CONSTRUCTION---**


## Directory structure:
Read notes on SQP solvers and source code structure in [gato/](gato).

- [bindings/](bindings/): Python bindings
- [config/](config/): Settings
    - [cost_settings.cuh](config/cost_settings.h): cost function settings
    - [sim_settings.cuh](config/sim_settings.h): mpc simulation settings
    - [solver_settings.cuh](config/solver_settings.h): solver settings
- [data/](data/): Data (e.g. trajectories)
- [dependencies/](dependencies/): External libraries and submodules
- [docs/](docs/): Documentation
- [dynamics/](dynamics/): Dynamics (plant files/grid files)
- [examples/](examples/): Examples
- [experiments/](experiments/): Experiments for papers
- [gato/](gato/): Library source
    - [sim/](gato/sim/): Simulation
    - [solvers/](gato/solvers/): Trjactory optimization solvers
    - [utils/](gato/utils/): Utilities
    - [gato.cuh](gato/gato.cuh): header file for gato toolbox
- [tests/](tests/): Tests
- [tools/](tools/): Tools

## Settings:
Tweak settings in [config/](config/)

## Python bindings:
See [bindings](bindings) to build and use Python bindings

## Building:
```
git clone https://github.com/A2R-Lab/gato.git
cd gato
git submodule update --init --recursive
chmod +x tools/build.sh
chmod +x tools/cleanup.sh
./tools/build.sh
```

## Running:
Single mpc simulation:
```
./build/mpc
```

Batched sqp solve:
```
./build/multi-sqp
```

Track with iiwa14 (pcg):
```
./build/MPCGPU-pcg
```

Track with iiwa14 (qdldl):
```
./build/MPCGPU-qdldl
```
