#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "dynamics/integrator.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;


template<typename T, uint32_t BatchSize, unsigned INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false>
__global__ void computeMeritBatchedKernel1(T*    __restrict__ d_merit_batch,  // accumulated per-(solve, alpha) merit
                                           T*    __restrict__ d_dz_batch,
                                           T*    __restrict__ d_xu_traj_batch,
                                           T*    __restrict__ d_x_initial_batch,
                                           T*    __restrict__ d_reference_traj_batch,
                                           void*             d_GRiD_mem,
                                           const T* __restrict__ d_mu_batch,
                                           T*    __restrict__ d_f_ext_batch,
                                           T                 timestep,
                                           T                 q_cost,
                                           T                 qd_cost,
                                           T                 u_cost,
                                           T                 N_cost,
                                           T                 q_lim_cost,
                                           T                 vel_lim_cost,
                                           T                 ctrl_lim_cost)
{
        // launched with 3D grid (KNOT_POINTS, batch_size, num_alphas)

        grid::robotModel<T>* d_robot_model = (grid::robotModel<T>*)d_GRiD_mem;
        const uint32_t       solve_idx = blockIdx.y;
        const uint32_t       knot_idx = blockIdx.x;
        const uint32_t       alpha_idx = blockIdx.z;
        T                    alpha = 1.0 / (1 << alpha_idx);

        T cost_k, constraint_k;  // cost function, constraint error, per-point merit
        T mu = d_mu_batch ? d_mu_batch[solve_idx] : static_cast<T>(1.0);
        if (!(mu > static_cast<T>(-1e30) && mu < static_cast<T>(1e30))) { mu = static_cast<T>(1.0); }

        extern __shared__ T s_mem[];
        T*                  s_xux_k = s_mem;  // current state, control, and next state
        T*                  s_reference_traj_k = s_xux_k + STATE_S_CONTROL + STATE_SIZE;
        T*                  s_temp = s_reference_traj_k + grid::EE_POS_SIZE;


        T* d_xu_k = getOffsetTraj<T, BatchSize>(d_xu_traj_batch, solve_idx, knot_idx);
        T* d_dz_k = getOffsetTraj<T, BatchSize>(d_dz_batch, solve_idx, knot_idx);
        T* d_x_initial_k = d_x_initial_batch + solve_idx * STATE_SIZE;
        T* d_f_ext = getOffsetWrench<T, BatchSize>(d_f_ext_batch, solve_idx);

        if (knot_idx == KNOT_POINTS - 1) {
                for (int i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) { s_xux_k[i] = d_xu_k[i] + alpha * d_dz_k[i]; }
        } else {
                for (int i = threadIdx.x; i < STATE_SIZE + STATE_S_CONTROL; i += blockDim.x) { s_xux_k[i] = d_xu_k[i] + alpha * d_dz_k[i]; }
        }

        T* d_reference_traj_k = getOffsetReferenceTraj<T, BatchSize>(d_reference_traj_batch, solve_idx, knot_idx);
        block::copy<T, grid::EE_POS_SIZE>(s_reference_traj_k, d_reference_traj_k);
        __syncthreads();

        // cost function
        cost_k =
            plant::trackingcost<T>(STATE_SIZE, CONTROL_SIZE, KNOT_POINTS, s_xux_k, s_reference_traj_k, s_temp, d_robot_model, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
        __syncthreads();

        // constraint error
        if (knot_idx < KNOT_POINTS - 1) {  // not last knot
                constraint_k = compute_integrator_error<T, INTEGRATOR_TYPE, ANGLE_WRAP>(s_xux_k, s_xux_k + STATE_SIZE + CONTROL_SIZE, s_temp, d_robot_model, timestep, d_f_ext);
        } else {
                d_xu_k = getOffsetTraj<T, BatchSize>(d_xu_traj_batch, solve_idx, 0);
                d_dz_k = getOffsetTraj<T, BatchSize>(d_dz_batch, solve_idx, 0);
                for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
                        s_temp[i] = abs(d_xu_k[i] + alpha * d_dz_k[i] - d_x_initial_k[i]);  // initial state constraint error
                }
                __syncthreads();
                block::reduce<T>(STATE_SIZE, s_temp);  // TODO: use warp reduce instead
                __syncthreads();
                constraint_k = s_temp[0];
        }
        __syncthreads();

        // Defensive guards against NaN/Inf propagating
        if (!(cost_k > static_cast<T>(-1e30) && cost_k < static_cast<T>(1e30))) { cost_k = static_cast<T>(1e9); }
        if (!(constraint_k > static_cast<T>(-1e30) && constraint_k < static_cast<T>(1e30))) { constraint_k = static_cast<T>(1e9); }

        // accumulate merit directly to global output (fused reduction)
        if (threadIdx.x == 0) { atomicAdd(&d_merit_batch[solve_idx * gridDim.z + alpha_idx], cost_k + mu * constraint_k); }
}

// Note: previous Kernel2 reduction is no longer needed; we now atomically
// accumulate per-knot merit inside Kernel1 to avoid extra global memory traffic.

template<typename T>
__host__ size_t getComputeMeritBatchedSMemSize()
{
        size_t size = sizeof(T)
                      * (2 * STATE_SIZE + CONTROL_SIZE + grid::EE_POS_SIZE +                                                                                          // reference_traj_k
                         max(gato::plant::trackingcost_TempMemCt_Shared(STATE_SIZE, CONTROL_SIZE, KNOT_POINTS), gato::plant::forwardDynamics_TempMemSize_Shared()));  // TODO: verify this
        return size;
}

template<typename T, uint32_t BatchSize, uint32_t NumAlphas>
__host__ void computeMeritBatched(T*                          d_merit_batch,
                                  T*                          d_merit_batch_temp,
                                  T*                          d_dz_batch,
                                  T*                          d_xu_traj_batch,
                                  T*                          d_f_ext_batch,
                                  ProblemInputs<T, BatchSize> inputs,
                                  const T*                    d_mu_batch,
                                  void*                       d_GRiD_mem,
                                  T                           q_cost,
                                  T                           qd_cost,
                                  T                           u_cost,
                                  T                           N_cost,
                                  T                           q_lim_cost,
                                  T                           vel_lim_cost,
                                  T                           ctrl_lim_cost)
{
        dim3   grid1(KNOT_POINTS, BatchSize, NumAlphas);
        // Use GRiD's suggested thread count to speed up inner device calls
        dim3   thread_block1(grid::SUGGESTED_THREADS);
        size_t s_mem_size = getComputeMeritBatchedSMemSize<T>();

        // Zero the output buffer before accumulation
        gpuErrchk(cudaMemset(d_merit_batch, 0, BatchSize * NumAlphas * sizeof(T)));

        computeMeritBatchedKernel1<T, BatchSize><<<grid1, thread_block1, s_mem_size>>>(d_merit_batch,
                                                                                       d_dz_batch,
                                                                                       d_xu_traj_batch,
                                                                                       inputs.d_x_s_batch,
                                                                                       inputs.d_reference_traj_batch,
                                                                                       d_GRiD_mem,
                                                                                       d_mu_batch,
                                                                                       d_f_ext_batch,
                                                                                       inputs.timestep,
                                                                                       q_cost,
                                                                                       qd_cost,
                                                                                       u_cost,
                                                                                       N_cost,
                                                                                       q_lim_cost,
                                                                                       vel_lim_cost,
                                                                                       ctrl_lim_cost);
}
