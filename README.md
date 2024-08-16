# GATO
GPU-Accelerated Trajectory Optimization for Robotics.



**CURRENTLY UNDER CONSTRUCTION**



## Directory structure:
- bindings/: Python bindings
- config/: Settings
    - cost_settings.cuh: cost function settings
    - sim_settings.cuh: mpc simulation settings
    - solver_settings.cuh: solver settings
- data/: Data (e.g. trajectories)
- dependencies/: External libraries and submodules
- docs/: Documentation
- dynamics/: Dynamics (plant files/grid files)
- examples/: Examples
- experiments/: Experiments for papers
- gato/: Library source
    - sim/: Simulation
    - solvers/: Trjactory optimization solvers
    - utils/: Utilities
    - gato.cuh: header file for gato toolbox
- tests/: Tests
- tools/: Tools

## Settings:
Tweak settings in config/


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
