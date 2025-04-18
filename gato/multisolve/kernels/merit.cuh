#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "settings.h"
#include "constants.h"
#include "utils/cuda_utils.cuh"
#include "utils/linalg.cuh"
#include "utils/integrator.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

/*

template <typename T, uint32_t TimeStep, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void computeMeritKernelBatched()

template <typename T>
__host__
size_t getComputeMeritBatchedSMemSize()

template <typename T, uint32_t NumAlphas>
__host__
void computeMeritBatched()

*/

template <typename T, uint32_t BatchSize, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__global__
void computeMeritBatchedKernel1(
    T *d_merit_batch_temp,
    T *d_dz_batch,
    T *d_xu_traj_batch, 
    T *d_x_initial_batch, 
    T *d_reference_traj_batch, 
    void *d_GRiD_mem,
    T mu,
    T *d_f_ext_batch,
    T timestep
) {
    grid::robotModel<T> *d_robot_model = (grid::robotModel<T> *)d_GRiD_mem;

    // TODO: change to for loops if we can't launch enough blocks for all solves
    // launched with 3D grid (KNOT_POINTS, batch_size, num_alphas)
    const uint32_t solve_idx = blockIdx.y;
    const uint32_t knot_idx = blockIdx.x;
    const uint32_t alpha_idx = blockIdx.z;
    T alpha = 1.0 / (1 << alpha_idx);
    
    T cost_k, constraint_k, merit_k; // cost function, constraint error, per-point merit

    extern __shared__ T s_mem[];
    T *s_xux_k = s_mem; // current state, control, and next state
    T *s_reference_traj_k = s_xux_k + STATE_S_CONTROL + STATE_SIZE;
    T *s_temp = s_reference_traj_k + grid::EE_POS_SIZE;

    
    T *d_xu_k = getOffsetTraj<T, BatchSize>(d_xu_traj_batch, solve_idx, knot_idx);
    T *d_dz_k = getOffsetTraj<T, BatchSize>(d_dz_batch, solve_idx, knot_idx);
    T *d_x_initial_k = d_x_initial_batch + solve_idx * STATE_SIZE;
    T *d_f_ext = getOffsetWrench<T, BatchSize>(d_f_ext_batch, solve_idx);

    if (knot_idx == KNOT_POINTS - 1) {
        #pragma unroll
        for(int i = threadIdx.x; i < STATE_SIZE; i += blockDim.x){
            s_xux_k[i] = d_xu_k[i] + alpha * d_dz_k[i];
        }
    } else {
        #pragma unroll
        for(int i = threadIdx.x; i < STATE_SIZE + STATE_S_CONTROL; i += blockDim.x){
            s_xux_k[i] = d_xu_k[i] + alpha * d_dz_k[i];
        }
    }
    
    T *d_reference_traj_k = getOffsetReferenceTraj<T, BatchSize>(d_reference_traj_batch, solve_idx, knot_idx);
    block::copy<T, grid::EE_POS_SIZE>(s_reference_traj_k, d_reference_traj_k);
    __syncthreads();

        // cost function
    cost_k = gato::plant::trackingcost<T>(
        STATE_SIZE, 
        CONTROL_SIZE, 
        KNOT_POINTS, 
        s_xux_k, 
        s_reference_traj_k, 
        s_temp, 
        d_robot_model
    );
    __syncthreads();

        // constraint error
    if (knot_idx < KNOT_POINTS - 1) { // not last knot
        constraint_k = integratorError<T>(
            STATE_SIZE, 
            s_xux_k, 
            &s_xux_k[STATE_SIZE + CONTROL_SIZE], 
            s_temp, 
            d_robot_model, 
            timestep,
            cooperative_groups::this_thread_block(),
            d_f_ext
        );
    } else {
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
            s_temp[i] =  abs(d_xu_k[i] + alpha * d_dz_k[i] - d_x_initial_k[i]);  //initial state constraint error
        }
        __syncthreads();
        block::reduce<T>(STATE_SIZE, s_temp); //TODO: use warp reduce instead
        __syncthreads();
        constraint_k = s_temp[0];
    }
    __syncthreads();

    // compute merit
    if (threadIdx.x == 0) {
        d_merit_batch_temp[solve_idx * gridDim.z * gridDim.x + alpha_idx * gridDim.x + knot_idx] = cost_k + mu * constraint_k;
    }
}

template <typename T, uint32_t NumAlphas>
__global__
void computeMeritBatchedKernel2(
    T *d_merit_batch,
    T *d_merit_batch_temp
) {
    const uint32_t solve_idx = blockIdx.x;
    const uint32_t alpha_idx = blockIdx.y;

    extern __shared__ T s_mem[];

    block::copy<T, KNOT_POINTS>(s_mem, d_merit_batch_temp + solve_idx * NumAlphas * KNOT_POINTS + alpha_idx * KNOT_POINTS);
    __syncthreads();

    block::reduce<T>(KNOT_POINTS, s_mem);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_merit_batch[solve_idx * NumAlphas + alpha_idx] = s_mem[0];
    }
}

template <typename T>
__host__
size_t getComputeMeritBatchedSMemSize() {
    size_t size = sizeof(T) * 2 * (
        2 * STATE_SIZE + CONTROL_SIZE + // xux_k
        grid::EE_POS_SIZE + // reference_traj_k
        grid::EE_POS_SHARED_MEM_COUNT +
        2 * STATE_SIZE + gato::plant::forwardDynamics_TempMemSize_Shared()
    ); //TODO: verify this
    return size;
}

template <typename T, uint32_t BatchSize, uint32_t NumAlphas>
__host__
void computeMeritBatched(
    T *d_merit_batch,
    T *d_merit_batch_temp,
    T *d_dz_batch,
    T *d_xu_traj_batch,
    T *d_f_ext_batch,
    ProblemInputs<T, BatchSize> inputs
) {
    dim3 grid1(KNOT_POINTS, BatchSize, NumAlphas);
    dim3 grid2(BatchSize, NumAlphas);
    dim3 thread_block(MERIT_THREADS);
    size_t s_mem_size = getComputeMeritBatchedSMemSize<T>();

    computeMeritBatchedKernel1<T, BatchSize><<<grid1, thread_block, s_mem_size>>>(
        d_merit_batch_temp,
        d_dz_batch,
        d_xu_traj_batch, 
        inputs.d_x_s_batch,
        inputs.d_reference_traj_batch,
        inputs.d_GRiD_mem,
        static_cast<T>(10.0), //TODO: tweak this
        d_f_ext_batch,
        inputs.timestep
    );

    computeMeritBatchedKernel2<T, NumAlphas><<<grid2, thread_block, KNOT_POINTS * sizeof(T)>>>(
        d_merit_batch,
        d_merit_batch_temp
    );
}