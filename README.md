# GATO
GPU-Accelerated Trajectory Optimization for Robotics.




## Directory structure:
- config/: Settings
    - cost_settings.cuh: cost function settings
    - sim_settings.cuh: mpc simulation settings
    - solver_settings.cuh: solver settings
- data/: Data (e.g. trajectories)
- docs/: Documentation
- dynamics/: Dynamics (plant files/grid files)
- examples/: Examples
- experiments/: Experiments for papers
- external/: External libraries and submodules
- src/: Sources and private headers
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
./build_examples.sh
./build_MPCGPU.sh
```

## Running:
Single mpc simulation:
```
./examples/build/pcg
```

Track with iiwa14 (pcg):
```
./experiments/MPCGPU/sqp_pcg
```

Track with iiwa14 (qdldl):
```
./experiments/MPCGPU/sqp_qdldl
```
