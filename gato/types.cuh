#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"

using namespace sqp;
using namespace gato::constants;

// --------------------------------------------------

template<typename T>
struct ProblemInputs {
        T     timestep;
        T*    d_x_s_batch;             // STATE_SIZE * batch_size
        T*    d_reference_traj_batch;  // grid::EE_POS_SIZE * KNOT_POINTS * batch_size
        void* d_GRiD_mem;
};

// --------------------------------------------------

struct PCGStats {
        double solve_time_us;

        std::vector<int> num_iterations;
        std::vector<int> converged;  // 1 if converged (pcg exit tol), 0 if not

        explicit PCGStats(uint32_t batch_size) : num_iterations(batch_size, 0), converged(batch_size, 0) {}
};

// --------------------------------------------------

template<typename T>
struct LineSearchStats {
        // if line search failure, step_size = -1
        std::vector<T> min_merit;  // min merit
        std::vector<T> step_size;  // argmin of line search

        explicit LineSearchStats(uint32_t batch_size) : min_merit(batch_size, 0.0), step_size(batch_size, 0.0) {}
};

// --------------------------------------------------

template<typename T>
struct SQPStats {
        double solve_time_us;  // using std::chrono::high_resolution_clock

        // for each solve
        std::vector<int> sqp_iterations;
        std::vector<int> kkt_converged;  // 1 if converged, 0 if not

        // for each SQP iteration
        std::vector<PCGStats>           pcg_stats;
        std::vector<LineSearchStats<T>> line_search_stats;

        explicit SQPStats(uint32_t batch_size) : sqp_iterations(batch_size, 0), kkt_converged(batch_size, 0) {}
};

// --------------------------------------------------

template<typename T>
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

template<typename T>
struct SchurSystem {
        T* d_S_batch;
        T* d_P_inv_batch;
        T* d_gamma_batch;
};

// --------------------------------------------------
