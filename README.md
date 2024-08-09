# GPU-Accelerated Trajectory Optimization for Robotics


## Directory structure:
- data/: Data (e.g. trajectories)
- docs/: Documentation
- dynamics/: Dynamics (plant files/grid files)
- examples/: Examples
- experiments/: Experiments for papers
- external/: External libraries and submodules
- include/: Public headers
- src/: Sources and private headers
    - sim/: Simulation
    - solvers/: Trjactory optimization solvers
    - utils/: Utilities
    - gato.cuh: header file for gato toolbox
    - cost_settings.cuh: cost function settings
    - sim_settings.cuh: mpc simulation settings
    - solver_settings.cuh: solver settings
- tests/: Tests
- tools/: Tools


## Namespace:

- gato
  - plant