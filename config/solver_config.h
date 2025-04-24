#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "settings.h" // Include settings for the base type T

namespace sqp {

// Struct to hold configuration parameters loaded from YAML
struct SolverConfig {
    // Solver settings
    std::string plant_path = "dynamics/indy7/indy7_plant.cuh";
    uint32_t integrator_type = INTEGRATOR_TYPE; // Keep using settings.h for now
    uint32_t batch_size = 32; // Needs careful handling if used for allocation sizes
    uint32_t N_h = KNOT_POINTS; // Keep using settings.h for now
    T dt = 0.01;
    uint32_t f_ext_horizon = 64; // Example, may or may not be directly used by solver yet

    uint32_t sqp_max_iter = SQP_MAX_ITER;
    // T sqp_tol = 1.0e-5; // TODO: Currently unused in solver code

    uint32_t pcg_max_iter = PCG_MAX_ITER;
    T pcg_tol = PCG_TOLERANCE;

    uint32_t num_alphas = NUM_ALPHAS; // Keep using settings.h for now

    // Cost parameters
    std::vector<T> ee_pos_weights = {1.0, 1.0, 1.0};
    T ee_pos_reg = 5.0;
    T ee_pos_reg_terminal = 50.0;
    T joint_limit_barrier = BARRIER_COST;

    std::vector<T> x_reg_weights = {10., 10., 10., 5., 5., 5., 1., 1., 1., 1., 1., 1.};
    T x_reg = VELOCITY_COST; // Corresponds to VELOCITY_COST in settings.h? Needs review.
    T x_reg_terminal = VELOCITY_COST; // Assume same for now

    std::vector<T> u_reg_weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    T u_reg = CONTROL_COST;
    T u_reg_terminal = CONTROL_COST; // Assume same for now

    // Regularization update parameters (rho related)
    T rho_init = RHO_INIT;
    T rho_factor = RHO_FACTOR;
    T rho_max = RHO_MAX;
    T rho_min = RHO_MIN;
    // reg_factor, reg_max, reg_min seem related to rho? Clarify usage.
    T reg_factor = 10.0;
    T reg_max = 1.0e9;
    T reg_min = 1.0e-9;

    // Kernel thread counts (keep using settings.h for now)
    uint32_t kkt_threads = KKT_THREADS;
    uint32_t schur_threads = SCHUR_THREADS;
    uint32_t pcg_threads = PCG_THREADS;
    uint32_t dz_threads = DZ_THREADS;
    uint32_t merit_threads = MERIT_THREADS;
    uint32_t line_search_threads = LINE_SEARCH_THREADS;
    uint32_t sim_forward_threads = SIM_FORWARD_THREADS;

    // TODO: Add loading logic from YAML
};

} // namespace sqp 