#pragma once

#include <vector>
#include <cstdint>

// ----- Needed at compile time -----
#include "dynamics/indy7/indy7_plant.cuh" // robot dynamics model
using T = float; // float(32)  double(64)
constexpr uint32_t INTEGRATOR_TYPE = 2; // 0: euler, 1: semi-implicit euler, 2: trapezoidal

// forward declarations 
struct Solver_settings;
struct Cost_settings;
struct Cuda_settings;

// ----- Runtime configs -----
struct Settings {
    Solver_settings solver;
    Cost_settings cost;
    Cuda_settings cuda;
};


struct Solver_settings {
    uint32_t batch_size;
    uint32_t knot_points; // N (number of steps in horizon)
    T dt; // time step


    uint32_t sqp_max_iter;
    T sqp_tol;

    uint32_t pcg_max_iter;
    T pcg_tol;

    uint32_t num_alphas; // step sizes for line search

    uint32_t f_ext_horizon; // number of knots to apply simulated external wrench
};

struct Cost_settings {
    std::vector<T> ee_pos_weights;
    T ee_pos_reg;
    T ee_pos_reg_terminal;

    std::vector<T> x_reg_weights;
    T x_reg;
    T x_reg_terminal;

    std::vector<T> u_reg_weights;
    T u_reg;
    T u_reg_terminal;

    T reg_factor;
    T reg_max;
    T reg_min;
    
    T joint_limit_barrier;
};

struct Cuda_settings {
    const uint32_t kkt_threads;
    const uint32_t schur_threads;
    const uint32_t pcg_threads;
    const uint32_t dz_threads;
    const uint32_t merit_threads;
    const uint32_t line_search_threads;
    const uint32_t sim_forward_threads;
};