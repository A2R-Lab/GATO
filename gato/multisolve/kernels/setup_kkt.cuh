#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "settings.h"
#include "constants.h"
#include "utils/linalg.cuh"
#include "utils/integrator.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

/*

template <typename T, uint32_t INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void setupKKTSystemBatchedKernel()

template <typename T>
__host__
size_t getSetupKKTSystemBatchedSMemSize()

template <typename T>
__host__
void setupKKTSystemBatched()

*/

template <typename T, uint32_t BatchSize, uint32_t INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__global__
void setupKKTSystemBatchedKernel(
    T *d_Q_batch,
    T *d_R_batch,
    T *d_q_batch,
    T *d_r_batch,
    T *d_A_batch,
    T *d_B_batch,
    T *d_c_batch,
    T *d_xu_traj_batch, // X, U trajectories
    void *d_GRiD_mem, // dynamics constraint TODO: can be const?
    T *d_x_s_batch, // initial state
    T *d_reference_traj_batch, // end effector position trajectory
    T *d_f_ext_batch,
    T timestep
) {
    // kernel launched with 2D grid: (knot_idx, solve_idx)
    const uint32_t solve_idx = blockIdx.y;

    extern __shared__ T s_mem[];
    T *s_xux_k = s_mem; // x_k, u_k, x_k+1
    T *s_reference_traj_k = s_xux_k + 2 * STATE_SIZE + CONTROL_SIZE;
    T *s_Q_k = s_reference_traj_k + 2 * grid::EE_POS_SIZE;
    T *s_R_k = s_Q_k + STATE_SIZE_SQ;
    T *s_q_k = s_R_k + CONTROL_SIZE_SQ;
    T *s_r_k = s_q_k + STATE_SIZE;
    T *s_A_k = s_r_k + CONTROL_SIZE;
    T *s_B_k = s_A_k + STATE_SIZE_SQ;
    T *s_c_k = s_B_k + STATE_P_CONTROL; // integrator error
    T *s_temp = s_c_k + STATE_SIZE;


    for (uint32_t knot_idx = blockIdx.x; knot_idx < KNOT_POINTS - 1; knot_idx += gridDim.x) {

        // Input pointers
        T *d_xu_traj_k = getOffsetTraj<T, BatchSize>(d_xu_traj_batch, solve_idx, knot_idx);
        T *d_reference_traj_k = getOffsetReferenceTraj<T, BatchSize>(d_reference_traj_batch, solve_idx, knot_idx);
        T *d_f_ext = getOffsetWrench<T, BatchSize>(d_f_ext_batch, solve_idx);

        // Output pointers
        T *d_Q_k = getOffsetStateSq<T, BatchSize>(d_Q_batch, solve_idx, knot_idx);
        T *d_R_k = getOffsetControlSq<T, BatchSize>(d_R_batch, solve_idx, knot_idx);
        T *d_q_k = getOffsetState<T, BatchSize>(d_q_batch, solve_idx, knot_idx);
        T *d_r_k = getOffsetControl<T, BatchSize>(d_r_batch, solve_idx, knot_idx);
        T *d_A_k = getOffsetStateSq<T, BatchSize>(d_A_batch, solve_idx, knot_idx);
        T *d_B_k = getOffsetStatePControl<T, BatchSize>(d_B_batch, solve_idx, knot_idx);
        T *d_c_k = getOffsetState<T, BatchSize>(d_c_batch, solve_idx, knot_idx + 1); // c_k+1 = e_k

        block::copy<T, STATE_S_CONTROL + STATE_SIZE>(s_xux_k, d_xu_traj_k);
        block::copy<T, 2 * grid::EE_POS_SIZE>(s_reference_traj_k, d_reference_traj_k); //TODO: is this correct?
        __syncthreads();

        integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
            s_xux_k,
            s_A_k, s_B_k, s_c_k,
            s_temp,
            d_GRiD_mem,
            timestep,
            d_f_ext
        );
        __syncthreads();

        block::copy<T, STATE_SIZE_SQ>(d_A_k, s_A_k);
        block::copy<T, STATE_P_CONTROL>(d_B_k, s_B_k);
        block::copy<T, STATE_SIZE>(d_c_k, s_c_k); // c_k+1 = e_k

        if (knot_idx < KNOT_POINTS - 2) {

            gato::plant::trackingCostGradientAndHessian<T>(
                STATE_SIZE, CONTROL_SIZE,
                s_xux_k, s_reference_traj_k,
                s_Q_k, s_q_k, s_R_k, s_r_k,
                s_temp,
                d_GRiD_mem
            );
            __syncthreads();

        } else { // compute Q_last, q_last, and c_0 as well for the last knot point

            T *s_Q_last = s_c_k + STATE_SIZE;
            T *s_q_last = s_Q_last + STATE_SIZE_SQ;
            s_temp = s_q_last + STATE_SIZE;

            // compute Q_k, q_k, R_k, r_k, Q_last, q_last
            gato::plant::trackingCostGradientAndHessian_lastblock<T>(
                STATE_SIZE, CONTROL_SIZE,
                s_xux_k, s_reference_traj_k,
                s_Q_k, s_q_k, s_R_k, s_r_k,
                s_Q_last, s_q_last,
                s_temp,
                d_GRiD_mem
            );
            
            // c_0 = x_0 - x_s
            T *d_c_0 = getOffsetState<T, BatchSize>(d_c_batch, solve_idx, 0);
            T *d_xu_0 = getOffsetTraj<T, BatchSize>(d_xu_traj_batch, solve_idx, 0);
            block::vecSub<T, STATE_SIZE>(
                d_c_0, 
                d_xu_0,
                d_x_s_batch + solve_idx * STATE_SIZE
            );
            __syncthreads();

            block::copy<T, STATE_SIZE_SQ>(d_Q_k + STATE_SIZE_SQ, s_Q_last);
            block::copy<T, STATE_SIZE>(d_q_k + STATE_SIZE, s_q_last);
        }

        block::copy<T, STATE_SIZE_SQ>(d_Q_k, s_Q_k);
        block::copy<T, CONTROL_SIZE_SQ>(d_R_k, s_R_k);
        block::copy<T, STATE_SIZE>(d_q_k, s_q_k);
        block::copy<T, CONTROL_SIZE>(d_r_k, s_r_k);
    }
}

template <typename T>
__host__
size_t getSetupKKTSystemBatchedSMemSize() {
    size_t size = sizeof(T) * (
        STATE_S_CONTROL + STATE_SIZE + // xux_k
        2 * grid::EE_POS_SIZE + // reference_traj_k
        STATE_SIZE_SQ + // Q_k
        CONTROL_SIZE_SQ + // R_k
        STATE_SIZE + // q_k
        CONTROL_SIZE + // r_k
        STATE_SIZE_SQ + // A_k
        STATE_P_CONTROL + // B_k
        STATE_SIZE + // c_k
        STATE_SIZE_SQ + // Q_last
        STATE_SIZE + // q_last
        max(grid::EE_POS_SHARED_MEM_COUNT, grid::DEE_POS_SHARED_MEM_COUNT) + 
        max((STATE_SIZE/2)*(STATE_S_CONTROL + 1) + gato::plant::forwardDynamicsAndGradient_TempMemSize_Shared(), 3 + (STATE_SIZE/2)*6)
    );
    return size;
}

template <typename T, uint32_t BatchSize>
__host__
void setupKKTSystemBatched(
    KKTSystem<T, BatchSize> kkt,
    ProblemInputs<T, BatchSize> inputs,
    T *d_xu_traj_batch,
    T *d_f_ext_batch
) {
    dim3 grid(KNOT_POINTS, BatchSize);
    dim3 block(KKT_THREADS);
    size_t s_mem_size = getSetupKKTSystemBatchedSMemSize<T>(); //TODO: why is MPCGPU launched with 2 * s_mem_size ?

    setupKKTSystemBatchedKernel<T, BatchSize><<<grid, block, s_mem_size>>>(
        kkt.d_Q_batch,
        kkt.d_R_batch,
        kkt.d_q_batch,
        kkt.d_r_batch,
        kkt.d_A_batch,
        kkt.d_B_batch,
        kkt.d_c_batch,
        d_xu_traj_batch,
        inputs.d_GRiD_mem,
        inputs.d_x_s_batch,
        inputs.d_reference_traj_batch,
        d_f_ext_batch,
        inputs.timestep
    );
}