#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

/*

template <typename T, uint32_t NumAlphas>
__global__
void lineSearchAndUpdateBatchedKernel()

template <typename T, uint32_t NumAlphas>
__host__
void lineSearchAndUpdateBatched()

*/

template <typename T, uint32_t BatchSize, uint32_t NumAlphas>
__global__
void lineSearchAndUpdateBatchedKernel(
    T *d_xu_traj_batch,
    T *d_dz_batch,
    T *d_merit_batch,
    T *d_merit_initial_batch,
    T *d_step_size_batch
) {
    // launched with batch_size blocks
    const uint32_t solve_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    __shared__ T s_merit[NumAlphas];
    __shared__ uint32_t s_step_idx[NumAlphas];
    
    // Initialize for parallel min reduction
    T local_min_merit = static_cast<T>(1e38); // max float
    uint32_t local_step_idx = 0;
    
    // Each thread handles multiple alphas if needed
    for (uint32_t i = tid; i < NumAlphas; i += blockDim.x) {
        T merit = d_merit_batch[solve_idx * NumAlphas + i];
        //printf("alpha: %d, merit: %4f  ", tid, merit);
        d_merit_batch[solve_idx * NumAlphas + i] = 0; // reset merit to 0
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
    
    // line search failed
    if (s_merit[0] >= d_merit_initial_batch[solve_idx]) {
        if (tid == 0) {
            d_step_size_batch[solve_idx] = -1;
        }
        return;
    }

    // update only on success
    T step_size =  1.0 / (T)(1 << s_step_idx[0]);
    T *d_xu_traj = getOffsetTraj<T, BatchSize>(d_xu_traj_batch, solve_idx, 0);
    T *d_dz = getOffsetTraj<T, BatchSize>(d_dz_batch, solve_idx, 0);
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < TRAJ_SIZE; i += blockDim.x) {
        d_xu_traj[i] += step_size * d_dz[i];
    }

    if (tid == 0) {
        d_merit_initial_batch[solve_idx] = s_merit[0];
        d_step_size_batch[solve_idx] = step_size;
    }
}

template <typename T, uint32_t BatchSize, uint32_t NumAlphas>
__host__
void lineSearchAndUpdateBatched(
    T *d_xu_traj_batch,
    T *d_dz_batch,
    T *d_merit_batch,
    T *d_merit_initial_batch,
    T *d_step_size_batch
) {
    dim3 grid(BatchSize);
    dim3 thread_block(LINE_SEARCH_THREADS);
    size_t s_mem_size = sizeof(T) * NumAlphas + sizeof(uint32_t) * NumAlphas;

    lineSearchAndUpdateBatchedKernel<T, BatchSize, NumAlphas><<<grid, thread_block, s_mem_size>>>(
        d_xu_traj_batch,
        d_dz_batch,
        d_merit_batch,
        d_merit_initial_batch,
        d_step_size_batch
    );
}