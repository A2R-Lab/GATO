#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/linalg.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

template<typename T, uint32_t NumAlphas>
__global__ __launch_bounds__(LINE_SEARCH_THREADS) void lineSearchAndUpdateBatchedKernel(T* d_xu_traj_batch, T* d_dz_batch, T* d_merit_batch, T* d_merit_initial_batch, T* d_step_size_batch, T* d_rho_penalty_batch, T* d_drho_batch, int adapt_rho, const int32_t* __restrict__ d_kkt_converged_batch)
{
        // launched with batch_size blocks
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        // converged solve: freeze trajectory AND rho/drho adaptation (its merit slots were
        // not written this iteration — the merit kernel skips converged solves too)
        if (d_kkt_converged_batch[solve_idx]) return;

        __shared__ T        s_merit[NumAlphas];
        __shared__ uint32_t s_step_idx[NumAlphas];

        // Initialize for parallel min reduction
        T        local_min_merit = static_cast<T>(1e38);  // max float
        uint32_t local_step_idx = 0;

        // Each thread handles multiple alphas if needed
        for (uint32_t i = tid; i < NumAlphas; i += blockDim.x) {
                T merit = d_merit_batch[solve_idx * NumAlphas + i];
                // printf("alpha: %d, merit: %4f  ", tid, merit);
                d_merit_batch[solve_idx * NumAlphas + i] = 0;  // reset merit to 0
                if (merit < local_min_merit) {
                        local_min_merit = merit;
                        local_step_idx = i;
                }
        }
        __syncthreads();

        // Store to shared memory
        if (tid < NumAlphas) {
                s_merit[tid] = local_min_merit;
                s_step_idx[tid] = local_step_idx;
        }
        __syncthreads();

        // Parallel reduction in shared memory
        for (uint32_t s = 1; s < NumAlphas; s *= 2) {
                uint32_t index = 2 * s * tid;
                if (index + s < NumAlphas) {
                        if (s_merit[index + s] < s_merit[index]) {
                                s_merit[index] = s_merit[index + s];
                                s_step_idx[index] = s_step_idx[index + s];
                        }
                }
                __syncthreads();
        }

        T min_merit = s_merit[0];

        bool line_search_success = (min_merit < d_merit_initial_batch[solve_idx]);

        // Thread 0 handles step size computation and rho update
        if (tid == 0) {

                // Update rho (only if adaptation is enabled)
                if (adapt_rho) {
                        T rho_multiplier = line_search_success ?  // 1 / RHO_FACTOR : RHO_FACTOR;
                                               min(d_drho_batch[solve_idx] / RHO_FACTOR, 1 / RHO_FACTOR)
                                                               :                                       // decrease on success
                                               max(d_drho_batch[solve_idx] * RHO_FACTOR, RHO_FACTOR);  // increase on failure

                        d_drho_batch[solve_idx] = rho_multiplier;
                        d_rho_penalty_batch[solve_idx] = max(d_rho_penalty_batch[solve_idx] * rho_multiplier, RHO_MIN);
                        d_rho_penalty_batch[solve_idx] = min(d_rho_penalty_batch[solve_idx], RHO_MAX);
                }

                if (!line_search_success) {
                        // rho saturated at RHO_MAX with a failing line search: reset so the
                        // solver can escape (the clamp above makes `>` unreachable; `>=` is
                        // the intended saturation test)
                        if (adapt_rho && d_rho_penalty_batch[solve_idx] >= RHO_MAX) {
                                d_rho_penalty_batch[solve_idx] = RHO_INIT;  // reset rho for next sqp solve
                        }
                        d_step_size_batch[solve_idx] = -1;
                } else {
                        // Compute step size and store in shared memory for all threads to use
                        s_merit[0] = 1.0 / (T)(1 << s_step_idx[0]);
                        d_merit_initial_batch[solve_idx] = min_merit;
                        d_step_size_batch[solve_idx] = s_merit[0];
                }
        }
        __syncthreads();

        // Only proceed with trajectory update if line search was successful
        if (line_search_success) {
                const T step_size = s_merit[0];
                T*      d_xu_traj = getOffsetTraj<T>(d_xu_traj_batch, solve_idx, 0);
                T*      d_dz = getOffsetTraj<T>(d_dz_batch, solve_idx, 0);
#pragma unroll
                for (uint32_t i = threadIdx.x; i < TRAJ_SIZE; i += blockDim.x) { d_xu_traj[i] += step_size * d_dz[i]; }
        }
}

template<typename T, uint32_t NumAlphas>
__host__ void lineSearchAndUpdateBatched(uint32_t batch_size, T* d_xu_traj_batch, T* d_dz_batch, T* d_merit_batch, T* d_merit_initial_batch, T* d_step_size_batch, T* d_rho_penalty_batch, T* d_drho_batch, int adapt_rho, const int32_t* d_kkt_converged_batch)
{
        dim3 grid(batch_size);
        dim3 thread_block(LINE_SEARCH_THREADS);
        // the kernel's s_merit/s_step_idx arrays are static __shared__ — no dynamic smem needed

        lineSearchAndUpdateBatchedKernel<T, NumAlphas>
            <<<grid, thread_block>>>(d_xu_traj_batch, d_dz_batch, d_merit_batch, d_merit_initial_batch, d_step_size_batch, d_rho_penalty_batch, d_drho_batch, adapt_rho, d_kkt_converged_batch);
}
