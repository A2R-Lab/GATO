#pragma once

#include <cstdint>
#include <cmath>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

template<typename U>
__device__ __forceinline__ bool finite(U v) {
        return !(isnan(v) || isinf(v));
}

template<typename T, uint32_t BatchSize>
__global__ __launch_bounds__(PCG_THREADS) void solvePCGBatchedKernel(uint32_t* __restrict__ d_iterations,
                                      T* __restrict__        d_x_batch,                 // (lambda) updated in-place
                                      const T* __restrict__ d_A_batch,     // (S)
                                      const T* __restrict__ d_M_inv_batch, // (P_inv)
                                      const T* __restrict__ d_b_batch,     // (gamma)
                                      const T* __restrict__ d_epsilon_batch,
                                      uint32_t  max_pcg_iters,
                                      int32_t* __restrict__  d_kkt_converged_batch)
{
        const uint32_t solve_idx = blockIdx.x;

        const T abs_tol = 1e-6;
        const T tiny    = 1e-20;  // guard against divide-by-zero / denormals
        const T epsilon = d_epsilon_batch[solve_idx];

        // skip solve if rho_max_reached
        if (d_kkt_converged_batch[solve_idx]) {
                if (threadIdx.x == 0) { d_iterations[solve_idx] = 0; }
                return;
        }

        // ----- Shared Memory -----
        // 5 vectors + 32 + 4
        extern __shared__ T s_mem[];
        block::zeroSharedMemory<T, 5 * VEC_SIZE_PADDED>(s_mem);

        // vectors
        T* s_A_p_vector = s_mem;
        T* s_x_vector = s_A_p_vector + VEC_SIZE_PADDED;
        T* s_r_vector = s_x_vector + VEC_SIZE_PADDED;
        T* s_z_vector = s_r_vector + VEC_SIZE_PADDED;
        T* s_p_vector = s_z_vector + VEC_SIZE_PADDED;

        // scratch for dot product
        T* s_scratch = s_p_vector + VEC_SIZE_PADDED;

        // scalars
        __shared__ T s_rho, s_rho_new, s_alpha, s_beta, s_rho_init;

        uint32_t iterations = 0;

        __syncthreads();

        // get A, M_inv, b, x pointers for current batch
        const T* d_A_matrix = getOffsetBlockRowPadded<T, BatchSize>(d_A_batch, solve_idx, 0);
        const T* d_M_inv_matrix = getOffsetBlockRowPadded<T, BatchSize>(d_M_inv_batch, solve_idx, 0);

        // getOffsetStatePadded points to the start of data, we want to point to the start of padding
        const T* d_b_vector = getOffsetStatePadded<T, BatchSize>(d_b_batch, solve_idx, 0) - STATE_SIZE;  // TODO: consider using shared memory for b
        T* d_x_vector = getOffsetStatePadded<T, BatchSize>(d_x_batch, solve_idx, 0) - STATE_SIZE;

        // copy x to shared memory
        block::copy<T, VEC_SIZE_PADDED>(s_x_vector, d_x_vector);
        __syncthreads();

        // ----- Init PCG -----

        // r = b - A * x
        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_r_vector, d_A_matrix, s_x_vector);
        __syncthreads();

        block::vecSub<T, VEC_SIZE_PADDED>(s_r_vector, d_b_vector, s_r_vector);
        __syncthreads();

        // z, p = M^-1 * r
        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_z_vector, s_p_vector, d_M_inv_matrix, s_r_vector);
        __syncthreads();

        // rho = r^T * z
        block::dot<T>(&s_rho, s_r_vector, s_z_vector, s_scratch, VEC_SIZE_PADDED);
        __syncthreads();

        // Guard: invalid or tiny initial residual
        if (!finite(s_rho) || fabs(s_rho) < abs_tol) {
                if (threadIdx.x == 0) { d_iterations[solve_idx] = 0; }
                __syncthreads();
                return;
        }

        // initial residual norm for relative tolerance
        if (threadIdx.x == 0) { s_rho_init = fabs(s_rho); }
        __syncthreads();

        // ----- PCG Loop -----
        for (uint32_t i = 0; i < max_pcg_iters; i++) {
                iterations++;

                // A_p = A * p
                block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_A_p_vector, d_A_matrix, s_p_vector);
                __syncthreads();

                // alpha = rho / (p^T * A_p)
                block::dot<T>(&s_alpha, s_p_vector, s_A_p_vector, s_scratch, VEC_SIZE_PADDED);
                __syncthreads();
                // Guard denominator: p^T A p should be > 0 for SPD; handle non-SPD/tiny/invalid
                if (threadIdx.x == 0) {
                        if (!finite(s_alpha) || fabs(s_alpha) <= tiny) {
                                // early exit to avoid NaNs; leave current x
                                iterations--; // this iteration produced no valid update
                        } else {
                                s_alpha = s_rho / s_alpha;
                        }
                }
                __syncthreads();
                // If denominator was invalid, break now
                if (!finite(s_alpha) || fabs(s_alpha) <= tiny) { break; }
                __syncthreads();

                // x = x + alpha * p
                // r = r - alpha * A_p
#pragma unroll
                for (uint32_t j = threadIdx.x; j < VEC_SIZE_PADDED; j += blockDim.x) {
                        s_x_vector[j] += s_alpha * s_p_vector[j];
                        s_r_vector[j] -= s_alpha * s_A_p_vector[j];
                }
                __syncthreads();

                // z = M^-1 * r
                block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_z_vector, d_M_inv_matrix, s_r_vector);
                __syncthreads();

                // rho_new = r^T * z
                block::dot<T>(&s_rho_new, s_r_vector, s_z_vector, s_scratch, VEC_SIZE_PADDED);
                __syncthreads();

                // Guard: invalid residuals
                if (!finite(s_rho_new)) { break; }

                // check for convergence using absolute and relative tolerance
                if (fabs(s_rho_new) < (abs_tol + epsilon * s_rho_init)) { break; }

                // beta = rho_new / rho
                // rho = rho_new
                if (threadIdx.x == 0) {
                        if (!finite(s_rho) || fabs(s_rho) <= tiny) {
                                s_beta = T(0);
                        } else {
                                s_beta = s_rho_new / s_rho;
                        }
                        s_rho = s_rho_new;
                }
                __syncthreads();

                // Guard: invalid beta leads to NaNs quickly
                if (!finite(s_beta)) { break; }

                // p = z + beta * p
#pragma unroll
                for (uint32_t j = threadIdx.x; j < VEC_SIZE_PADDED; j += blockDim.x) { s_p_vector[j] = s_z_vector[j] + s_beta * s_p_vector[j]; }
                __syncthreads();
        }
        // ----- End PCG -----

        // save stats for current batch
        if (threadIdx.x == 0) { d_iterations[solve_idx] = iterations; }

        block::copy<T, VEC_SIZE_PADDED>(d_x_vector, s_x_vector);
}

template<typename T>
__host__ size_t getSolvePCGBatchedSMemSize()
{
        size_t size = sizeof(T) * (5 * VEC_SIZE_PADDED + 32 + 5 + PCG_THREADS);
        return size;
}

template<typename T, uint32_t BatchSize>
__host__ void solvePCGBatched(T* d_lambda_batch, SchurSystem<T, BatchSize> schur, const T* d_epsilon_batch, uint32_t max_pcg_iters, int32_t* d_kkt_converged_batch, uint32_t* d_iterations)
{
        dim3           grid(BatchSize);
        dim3           thread_block(PCG_THREADS);
        const uint32_t s_mem_size = getSolvePCGBatchedSMemSize<T>();

        solvePCGBatchedKernel<T, BatchSize>
            <<<grid, thread_block, s_mem_size>>>(d_iterations, d_lambda_batch, schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch, d_epsilon_batch, max_pcg_iters, d_kkt_converged_batch);
}
