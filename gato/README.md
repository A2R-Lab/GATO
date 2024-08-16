# GATO Source Code

## Notes on structure:

Pretty self-explanatory, but here's a quick overview:
- sim/ contains everythng related to MPC simulation
- solvers/ contains trajectory optimization solvers
- utils/ contains utility functions used across the project

Settings/constants have been moved to config/ and split up into the following files for future extensibility:
- cost_settings.h
- sim_settings.h
- solver_settings.h

Submodules have been moved to dependencies/

### solvers/

The core of GATO is the solvers/ directory. 

Within solvers/sqp, you can find GPU-accelerated solvers for sequential quadratic programming (SQP) through preconditioned conjugate gradient (PCG) and QDLDL.
- sqp_pcg.cuh: single PCG solve function
- sqp_pcg_n.cuh: batched PCG solve function
- sqp_qdldl.cuh: QDLDL solve function

In our examples and experiments, we can call these functions:
- to directly solve a trajectory optimization problem from main()
- for MPC through simulation
- in Python with pybind11 bindings

In batched SQP, we introduce a new parameter "solve_count", which is the number of trajectories to solve in parallel. Device memory for each trajectory is allocated contiguously, and kernels are modified to parallelize across another grid dimension and use the correct device pointers for each respective trajectory. 

Examples for batched trajectory optimization:
- examples/sqp_n.cu
- examples/mpc_n.cu

### solvers/kernels/

This directory contains custom CUDA kernels for each step of the SQP solver. Kernels for batched SQP are denoted with "_n" in their names.

We split the SQP solver into the following kernels:
1. Setup KKT system through computing gradients and hessians (-> G, C, g, c)
    - setup_kkt.cuh/setup_kkt_n.cuh
2. Form Schur Complement system (-> S, Pinv, gamma)
    - setup_schur_pcg.cuh/setup_schur_pcg_n.cuh/setup_schur_qdldl.cuh
3. Solve linear system with PCG or QDLDL (-> lambda)
    - pcg.cuh/pcg_n.cuh/qdldl.cuh
4. Compute dz with KKT system (-> dz)
    - compute_dz.cuh
5. Compute merit function for different alphas
    - compute_merit.cuh/compute_merit_n.cuh
6. Line search to find best alpha
    - line_search_n.cuh (only implemented for batched SQP)

### sim/

This directory contains code to simulate model predictive control (MPC) through solving successive trajectory optimization problems.

- mpcsim.cuh
    - Given input end effector position trajectory, X and U trajectories, and initial state, in a loop until end of trajectory:
        - solve trajectory optimization problem
        - simulate with control from optimized trajectory
        - shift trajectories to account for time shift
        - record tracking error and other stats
- simulator.cuh
    - Code for simulating robot with GRiD and shifting trajectories
