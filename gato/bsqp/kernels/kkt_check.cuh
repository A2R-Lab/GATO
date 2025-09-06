#pragma once

#include <cstdint>
#include "settings.h"
#include <cmath>
#include "constants.h"
#include "utils/linalg.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

template<typename T, uint32_t BatchSize>
__global__ void checkKKTConvergenceBatchedKernel(const T* __restrict__ d_q_batch,
                                                 const T* __restrict__ d_c_batch,
                                                 const uint32_t* __restrict__ d_pcg_iterations,
                                                 T kkt_tol,
                                                 int32_t* d_kkt_converged_batch)
{
        const uint32_t solve_idx = blockIdx.x;

        // Shared reduction buffers
        extern __shared__ T s_mem[];
        T* s_q_absmax = s_mem;                 // per-thread partial max for q
        T* s_c_absmax = s_q_absmax + blockDim.x; // per-thread partial max for c

        // Initialize
        T local_q_max = T(0);
        T local_c_max = T(0);

        // Scan q residuals (STATE_SIZE * KNOT_POINTS)
        const T* q_ptr = d_q_batch + solve_idx * STATE_P_KNOTS;
        for (uint32_t i = threadIdx.x; i < STATE_P_KNOTS; i += blockDim.x) {
                T v = fabs(q_ptr[i]);
                if (v > local_q_max) local_q_max = v;
        }

        // Scan c residuals (STATE_SIZE * KNOT_POINTS)
        const T* c_ptr = d_c_batch + solve_idx * STATE_P_KNOTS;
        for (uint32_t i = threadIdx.x; i < STATE_P_KNOTS; i += blockDim.x) {
                T v = fabs(c_ptr[i]);
                if (v > local_c_max) local_c_max = v;
        }

        // Write partials
        s_q_absmax[threadIdx.x] = local_q_max;
        s_c_absmax[threadIdx.x] = local_c_max;
        __syncthreads();

        // Block reduction to get maxima
        // Simple power-of-two style reduction
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                        T a = s_q_absmax[threadIdx.x + stride];
                        if (a > s_q_absmax[threadIdx.x]) s_q_absmax[threadIdx.x] = a;
                        T b = s_c_absmax[threadIdx.x + stride];
                        if (b > s_c_absmax[threadIdx.x]) s_c_absmax[threadIdx.x] = b;
                }
                __syncthreads();
        }

        if (threadIdx.x == 0) {
                const bool pcg_zero = (d_pcg_iterations[solve_idx] == 0);
                const bool kkt_ok = (s_q_absmax[0] < kkt_tol) && (s_c_absmax[0] < kkt_tol);
                d_kkt_converged_batch[solve_idx] = (pcg_zero || kkt_ok) ? 1 : 0;
        }
}

template<typename T, uint32_t BatchSize>
__host__ inline void checkKKTConvergenceBatched(const T* d_q_batch,
                                                const T* d_c_batch,
                                                const uint32_t* d_pcg_iterations,
                                                T kkt_tol,
                                                int32_t* d_kkt_converged_batch)
{
        dim3 grid(BatchSize);
        dim3 block(128);
        size_t smem = sizeof(T) * 2 * block.x;
        checkKKTConvergenceBatchedKernel<T, BatchSize><<<grid, block, smem>>>(
            d_q_batch, d_c_batch, d_pcg_iterations, kkt_tol, d_kkt_converged_batch);
}
