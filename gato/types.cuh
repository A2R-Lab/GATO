#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"

using namespace sqp;
using namespace gato::constants;

// --------------------------------------------------

template<typename T, uint32_t BatchSize>
struct ProblemInputs {
        T     timestep;
        T*    d_x_s_batch;             // STATE_SIZE * batch_size
        T*    d_reference_traj_batch;  // grid::EE_POS_SIZE * KNOT_POINTS * batch_size
        void* d_GRiD_mem;
};

// --------------------------------------------------

template<uint32_t BatchSize>
struct PCGStats {
        double solve_time_us;

        std::vector<int> num_iterations;
        std::vector<int> converged;  // 1 if converged (pcg exit tol), 0 if not

        PCGStats() : num_iterations(BatchSize, 0), converged(BatchSize, 0) {}
};

// --------------------------------------------------

template<typename T, uint32_t BatchSize>
struct LineSearchStats {
        // if line search failure, step_size = -1
        std::vector<T> min_merit;  // min merit
        std::vector<T> step_size;  // argmin of line search

        LineSearchStats() : min_merit(BatchSize, 0.0), step_size(BatchSize, 0.0) {}
};

// --------------------------------------------------

template<typename T, uint32_t BatchSize>
struct SQPStats {
        double solve_time_us;  // using std::chrono::high_resolution_clock

        // for each solve
        std::vector<int> sqp_iterations;
        std::vector<int> kkt_converged;  // 1 if converged, 0 if not

        // for each SQP iteration
        std::vector<PCGStats<BatchSize>>           pcg_stats;
        std::vector<LineSearchStats<T, BatchSize>> line_search_stats;

        SQPStats() : sqp_iterations(BatchSize, 0), kkt_converged(BatchSize, 0) {}
};

// --------------------------------------------------

template<typename T, uint32_t BatchSize>
struct KKTSystem {
        T* d_Q_batch;
        T* d_R_batch;
        T* d_q_batch;
        T* d_r_batch;
        T* d_A_batch;
        T* d_B_batch;
        T* d_c_batch;
};

// --------------------------------------------------

template<typename T, uint32_t BatchSize>
struct SchurSystem {
        T* d_S_batch;
        T* d_P_inv_batch;
        T* d_gamma_batch;
};

// --------------------------------------------------
