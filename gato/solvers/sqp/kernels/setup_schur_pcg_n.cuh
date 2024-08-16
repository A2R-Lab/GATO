#pragma once
#include <cstdint>
#include "GBD-PCG/include/gpuassert.cuh"
#include "glass.cuh"
#include "utils/matrix.cuh"
#include "setup_schur_pcg.cuh"


/**
 * @brief Kernel to form Schur system for a batch of trajectories.
 */
template <typename T>
__global__ void form_schur_system_kernel_n(uint32_t state_size,
                                        uint32_t control_size,
                                        uint32_t knot_points,
                                        T *d_G_dense,
                                        T *d_C_dense,
                                        T *d_g,
                                        T *d_c,
                                        T *d_S,
                                        T *d_Pinv,
                                        T *d_gamma,
                                        T *d_rhos) {

    extern __shared__ T s_temp[];

    int traj_id = blockIdx.x;
    int knot_id = blockIdx.y;

    // Calculate offsets for this trajectory
    uint32_t G_size = (state_size * state_size + control_size * control_size) * knot_points - control_size * control_size;
    uint32_t C_size = (state_size * state_size + state_size * control_size) * (knot_points - 1);
    uint32_t g_size = (state_size + control_size) * knot_points - control_size;
    uint32_t c_size = state_size * knot_points;
    uint32_t S_size = 3 * state_size * state_size * knot_points;
    uint32_t Pinv_size = 3 * state_size * state_size * knot_points;
    uint32_t gamma_size = state_size * knot_points;

    T *traj_G = d_G_dense + traj_id * G_size;
    T *traj_C = d_C_dense + traj_id * C_size;
    T *traj_g = d_g + traj_id * g_size;
    T *traj_c = d_c + traj_id * c_size;
    T *traj_S = d_S + traj_id * S_size;
    T *traj_Pinv = d_Pinv + traj_id * Pinv_size;
    T *traj_gamma = d_gamma + traj_id * gamma_size;

    T rho = d_rhos[traj_id];

    form_S_gamma_and_jacobi_Pinv_blockrow<T>(
        state_size, control_size, knot_points,
        traj_G, traj_C, traj_g, traj_c,
        traj_S, traj_Pinv, traj_gamma,
        rho, s_temp, knot_id
    );

    __syncthreads();

    complete_SS_Pinv_blockrow<T>(
        state_size, knot_points,
        traj_S, traj_Pinv, traj_gamma,
        s_temp, knot_id
    );
}


/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/


/**
 * @brief Form Schur system for a batch of trajectories.
 */ 
template <typename T>
void form_schur_system_n(uint32_t solve_count,
                        uint32_t state_size,
                        uint32_t control_size,
                        uint32_t knot_points,
                        T *d_G_dense,
                        T *d_C_dense,
                        T *d_g,
                        T *d_c,
                        T *d_S,
                        T *d_Pinv,
                        T *d_gamma,
                        T *d_rhos) {
    
    const uint32_t s_temp_size = sizeof(T) * (8 * state_size * state_size +
                                            7 * state_size +
                                            state_size * control_size +
                                            3 * control_size +
                                            2 * control_size * control_size +
                                            3);

    dim3 blockDim(SCHUR_THREADS);
    dim3 gridDim(solve_count, knot_points);

    form_schur_system_kernel_n<<<gridDim, blockDim, s_temp_size>>>(
        state_size, control_size, knot_points,
        d_G_dense, d_C_dense, d_g, d_c,
        d_S, d_Pinv, d_gamma, d_rhos
    );
}