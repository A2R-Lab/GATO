#pragma once

#include <cstdint>
#include "config/settings.h"
#include "config/constants.h"
#include "utils/cuda_utils.cuh"
#include "utils/linalg.cuh"
#include "utils/types.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

/*

template <typename T>
__global__ 
void solvePCGBatchedKernel()

template <typename T>
__host__
size_t getSolvePCGBatchedSMemSize()

template <typename T>
__host__
void solvePCGBatched()

*/

template <typename T, uint32_t BatchSize>
__global__ 
void solvePCGBatchedKernel(
    int32_t* d_converged,
    uint32_t* d_iterations,
    T *d_x_batch, // (lambda) updated in-place
    T *d_A_batch, // (S)
    T *d_M_inv_batch, // (P_inv)
    T *d_b_batch, // (gamma)
    T epsilon,
    int32_t *d_rho_max_reached_batch
) {
    const uint32_t solve_idx = blockIdx.x;
    
    // skip solve if rho_max_reached
    if (d_rho_max_reached_batch[solve_idx]) {
        if (threadIdx.x == 0) {
            d_converged[solve_idx] = false;
            d_iterations[solve_idx] = 0;
        }
        return;
    }

    // ----- Shared Memory -----
    // 5 vectors + 32 + 4 
    extern __shared__ T s_mem[];

    // vectors
    T *s_A_p_vector = s_mem;
    T *s_x_vector = s_A_p_vector + VEC_SIZE_PADDED;
    T *s_r_vector = s_x_vector + VEC_SIZE_PADDED;
    T *s_z_vector = s_r_vector + VEC_SIZE_PADDED;
    T *s_p_vector = s_z_vector + VEC_SIZE_PADDED;

    // scratch for dot product
    T *s_scratch = s_p_vector + VEC_SIZE_PADDED;

    // scalars
    __shared__ T s_rho, s_rho_new, s_alpha, s_beta;

    bool converged = false;
    uint32_t iterations = 0;

    block::zeroSharedMemory<T, 5 * VEC_SIZE_PADDED>(s_mem);
    __syncthreads();

    // get A, M_inv, b, x pointers for current batch
    T *d_A_matrix = getOffsetBlockRowPadded<T, BatchSize>(d_A_batch, solve_idx, 0);
    T *d_M_inv_matrix = getOffsetBlockRowPadded<T, BatchSize>(d_M_inv_batch, solve_idx, 0);

    // getOffsetStatePadded points to the start of data, we want to point to the start of padding
    T *d_b_vector = getOffsetStatePadded<T, BatchSize>(d_b_batch, solve_idx, 0) - STATE_SIZE; // TODO: consider using shared memory for b
    T *d_x_vector = getOffsetStatePadded<T, BatchSize>(d_x_batch, solve_idx, 0) - STATE_SIZE;

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
    block::dot<T>(&s_rho, s_r_vector, s_z_vector, s_scratch, VEC_SIZE_PADDED); //TODO: need to make sure that vector padding is not included in dot product
    __syncthreads();
    

    // ----- PCG Loop -----
    for (uint32_t i = 0; i < PCG_MAX_ITER; i++) {
        iterations++;

        // A_p = A * p
        block::btdMatrixVectorProduct<T, KNOT_POINTS, STATE_SIZE>(s_A_p_vector, d_A_matrix, s_p_vector);
        __syncthreads();
        
        // alpha = rho / (p^T * A_p)
        block::dot<T>(&s_alpha, s_p_vector, s_A_p_vector, s_scratch, VEC_SIZE_PADDED);
        __syncthreads();
        if (threadIdx.x == 0) {
            s_alpha = s_rho / s_alpha;
        }
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

        // check for convergence
        if (abs(s_rho_new) < epsilon) {
            converged = true;
            break;
        }
        __syncthreads();

        // beta = rho_new / rho
        // rho = rho_new
        if (threadIdx.x == 0) {
            s_beta = s_rho_new / s_rho;
            s_rho = s_rho_new;
        }
        __syncthreads();

        // p = z + beta * p
        #pragma unroll
        for (uint32_t j = threadIdx.x; j < VEC_SIZE_PADDED; j += blockDim.x) {
            s_p_vector[j] = s_z_vector[j] + s_beta * s_p_vector[j];
        }
        __syncthreads();
    }
    // ----- End PCG -----
    
    // save stats for current batch
    if (threadIdx.x == 0) {
        d_converged[solve_idx] = 1 * converged;
        d_iterations[solve_idx] = iterations;
    }
    
    block::copy<T, VEC_SIZE_PADDED>(d_x_vector, s_x_vector);
}

template <typename T>
__host__
size_t getSolvePCGBatchedSMemSize() {
    size_t size = sizeof(T) * (
        5 * VEC_SIZE_PADDED + 32 + 4 + PCG_THREADS //TODO: check if this is correct
    );
    return size;
}

template <typename T, uint32_t BatchSize>
__host__
void solvePCGBatched(
    T *d_lambda_batch,
    SchurSystem<T, BatchSize> schur,
    T epsilon,
    int32_t *d_rho_max_reached_batch,
    int32_t *d_converged,
    uint32_t *d_iterations
) {
    dim3 grid(BatchSize);
    dim3 thread_block(PCG_THREADS);
    const uint32_t s_mem_size = getSolvePCGBatchedSMemSize<T>();

    solvePCGBatchedKernel<T, BatchSize><<<grid, thread_block, s_mem_size>>>(
        d_converged,
        d_iterations,
        d_lambda_batch,
        schur.d_S_batch,
        schur.d_P_inv_batch,
        schur.d_gamma_batch,
        epsilon,
        d_rho_max_reached_batch
    );
}