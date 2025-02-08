#pragma once

#include <cstdint>
#include "config/settings.h"
#include "config/constants.h"
#include "utils/cuda_utils.cuh"

using namespace sqp;
using namespace gato::constants;

// --------------------------------------------------

template <typename T, uint32_t BatchSize>
struct ProblemInputs {
    T timestep;
    T *d_x_s_batch; // STATE_SIZE * batch_size
    T *d_reference_traj_batch; // grid::EE_POS_SIZE * KNOT_POINTS * batch_size
    void *d_GRiD_mem; 
};

// --------------------------------------------------

struct PCGStats { //TODO: use template
    double solve_time_us; // using cudaEventElapsedTime

    // if rho_max_reached for a solve, num_iterations = 0, converged = false
    std::vector<int> num_iterations;
    std::vector<int> converged; // 1 if converged, 0 if not
};

// --------------------------------------------------

template <typename T>
struct LineSearchStats {
    bool all_rho_max_reached;

    // if rho_max_reached for a solve, step_size = -1
    std::vector<T> min_merit; //min merit
    std::vector<T> step_size; //argmin of line search
};

// --------------------------------------------------

template <typename T, uint32_t BatchSize>
struct SQPStats {
    double solve_time_us; // using std::chrono::high_resolution_clock

    // for each solve
    std::vector<int> sqp_iterations; 
    std::vector<int> rho_max_reached; // 1 if reached, 0 if not
    std::vector<double> pcg_solve_times; //TODO: not used
    
    // for each SQP iteration
    std::vector<PCGStats> pcg_stats;
    std::vector<LineSearchStats<T>> line_search_stats;

    SQPStats() :
        sqp_iterations(BatchSize, 0),
        rho_max_reached(BatchSize, 0),
        pcg_solve_times(BatchSize, 0.0) {}
};

// --------------------------------------------------

template <typename T, uint32_t BatchSize>
struct KKTSystem {
    T *d_Q_batch;
    T *d_R_batch;
    T *d_q_batch;
    T *d_r_batch;
    T *d_A_batch;
    T *d_B_batch;
    T *d_c_batch;
};

// --------------------------------------------------

template <typename T, uint32_t BatchSize>
struct SchurSystem {
    T *d_S_batch;
    T *d_P_inv_batch;
    T *d_gamma_batch;
};

// --------------------------------------------------
