#pragma once

#include <cublas_v2.h>

#include "gato.cuh"
#include "utils/utils.cuh"

/**
 * @brief AtomicMin for floating-point types.
 */
template <typename T>
__device__ T atomicMinFloat(T* address, T val)
{
    T old = *address, assumed;
    do {
        assumed = old;
        old = __int_as_float(atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(val)));
    } while (assumed != old && val < old);
    return old;
}

/**
 * @brief Kernel to find alpha that minimizes merit function.
 * @tparam T Data type
 * @param solve_count Number of solves
 * @param num_alphas Number of alphas
 * @param d_merit_news Merit results
 * @param d_merit_initial Previous merit values
 * @param d_min_merit Minimum merit values
 * @param d_line_search_step Line search step
 */
template <typename T>
__global__ 
void find_alpha_kernel_n(uint32_t solve_count, uint32_t num_alphas,
                            const T* d_merit_news, const T* d_merit_initial, T* d_min_merit, 
                            uint32_t* d_line_search_step) 
{   //x: solve_count, y: num_alphas

    const uint32_t solve_id = blockIdx.x;

    if (solve_id >= solve_count) return;

    __shared__ T s_min_merit;
    __shared__ uint32_t s_best_alpha;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        s_min_merit = d_merit_initial[solve_id];
        s_best_alpha = 0;
    }                            
    __syncthreads();

    // Find alpha that minimizes merit function
    for (uint32_t alpha_id = blockIdx.y; alpha_id < num_alphas; alpha_id += gridDim.y) {
        T current_merit = d_merit_news[solve_id * num_alphas + alpha_id];
        if (current_merit < s_min_merit) {
            T old_min = atomicMinFloat(&s_min_merit, current_merit);
            if (s_min_merit == current_merit) {
                atomicExch(&s_best_alpha, alpha_id);
            }
        }
    }
    __syncthreads();

    // Update global memory with best alpha and merit
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        d_min_merit[solve_id] = s_min_merit;
        d_line_search_step[solve_id] = s_best_alpha;
    }
    __syncthreads();
}

/**
 * @brief Launch find_alpha kernel.
 * @tparam T Data type
 * @param solve_count Number of solves
 * @param num_alphas Number of alphas
 * @param d_merit_news Merit results
 * @param d_merit_initial Previous merit values
 * @param d_min_merit Minimum merit values
 * @param d_line_search_step Line search step
 */
template <typename T>
void find_alpha_n(uint32_t solve_count, uint32_t num_alphas,
    const T* d_merit_news, const T* d_merit_initial, T* d_min_merit, uint32_t* d_line_search_step)
{
    dim3 grid(solve_count, num_alphas);
    dim3 block(MERIT_THREADS);
    find_alpha_kernel_n<T><<<grid, block>>>(
        solve_count, num_alphas,
        d_merit_news, d_merit_initial, d_min_merit, d_line_search_step
    );
}
