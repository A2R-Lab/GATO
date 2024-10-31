#pragma once

#include "gato.cuh"
#include "utils/utils.cuh"
#include "setup_kkt.cuh"

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void setup_kkt_kernel_n(int solve_count, int knot_points,
                                      uint32_t state_size, 
                                      uint32_t control_size, 
                                      T *d_G_dense, 
                                      T *d_C_dense, 
                                      T *d_g, 
                                      T *d_c,
                                      void *d_dynMem_const, 
                                      T timestep,
                                      T *d_eePos_traj, 
                                      T *d_xs, 
                                      T *d_xu)
{
    const cgrps::thread_block block = cgrps::this_thread_block();
    int traj_id = blockIdx.y;
    int num_blocks = gridDim.x * gridDim.y;
    
    const uint32_t states_sq = state_size * state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t traj_size = states_s_controls * knot_points - control_size;
    const uint32_t G_size = (states_sq + controls_sq) * knot_points - controls_sq;
    const uint32_t C_size = (states_sq + states_p_controls) * (knot_points - 1);

    extern __shared__ T s_temp[];

    T *s_xux = s_temp;
    T *s_eePos_traj = s_xux + 2*state_size + control_size;
    T *s_Qk = s_eePos_traj + grid::EE_POS_SIZE;
    T *s_Rk = s_Qk + states_sq;
    T *s_qk = s_Rk + controls_sq;
    T *s_rk = s_qk + state_size;
    T *s_end = s_rk + control_size;

    // Offset pointers for this trajectory
    d_G_dense += traj_id * G_size;
    d_C_dense += traj_id * C_size;
    d_g += traj_id * traj_size;
    d_c += traj_id * state_size * knot_points;
    d_eePos_traj += traj_id * grid::EE_POS_SIZE * knot_points;
    d_xs += traj_id * state_size;
    d_xu += traj_id * traj_size;

    for (unsigned knot_id = blockIdx.x + blockIdx.y * gridDim.x; knot_id < knot_points-1; knot_id+=num_blocks) {
        
        glass::copy<T>(2 * state_size + control_size, &d_xu[knot_id * states_s_controls], s_xux);
        glass::copy<T>(2 * grid::EE_POS_SIZE, &d_eePos_traj[knot_id * grid::EE_POS_SIZE], s_eePos_traj);

        block.sync();

        if (knot_id == knot_points - 2) {
            // Last knot point
            T* s_Ak = s_end;
            T* s_Bk = s_Ak + states_sq;
            T* s_Qkp1 = s_Bk + states_p_controls;
            T* s_qkp1 = s_Qkp1 + states_sq;
            T* s_integrator_error = s_qkp1 + state_size;
            T* s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            block.sync();
            
            gato::plant::trackingCostGradientAndHessian_lastblock<T>(
                state_size,
                control_size,
                s_xux,
                s_eePos_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_Qkp1,
                s_qkp1,
                s_extra_temp,
                d_dynMem_const
            );
            block.sync();

            for (int i = threadIdx.x; i < state_size; i += blockDim.x) {
                d_c[i] = d_xu[i] - d_xs[i];
            }
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq + controls_sq) * knot_id]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq + controls_sq) * knot_id + states_sq]);
            glass::copy<T>(states_sq, s_Qkp1, &d_G_dense[(states_sq + controls_sq) * (knot_id + 1)]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls * knot_id]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls * knot_id + state_size]);
            glass::copy<T>(state_size, s_qkp1, &d_g[states_s_controls * (knot_id + 1)]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq + states_p_controls) * knot_id]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq + states_p_controls) * knot_id + states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size * (knot_id + 1)]);
        }
        else {
            // Not last knot point
            T* s_Ak = s_end;
            T* s_Bk = s_Ak + states_sq;
            T* s_integrator_error = s_Bk + states_p_controls;
            T* s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            block.sync();
            
            gato::plant::trackingCostGradientAndHessian<T>(
                state_size,
                control_size,
                s_xux,
                s_eePos_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_extra_temp,
                d_dynMem_const
            );
            block.sync();


            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq + controls_sq) * knot_id]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq + controls_sq) * knot_id + states_sq]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls * knot_id]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls * knot_id + state_size]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq + states_p_controls) * knot_id]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq + states_p_controls) * knot_id + states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size * (knot_id + 1)]);
        }
    }
}

/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/

template <typename T>
void setup_kkt_n(
    int solve_count,
    int knot_points,
    uint32_t state_size,
    uint32_t control_size,
    T *d_G_dense,
    T *d_C_dense,
    T *d_g,
    T *d_c,
    void *d_dynMem_const,
    T timestep,
    T *d_eePos_traj,
    T *d_xs,
    T *d_xu
){
    const uint32_t kkt_smem_size = 2 * get_kkt_kernel_smem_size<T>();
    dim3 block(KKT_THREADS);
    dim3 grid(knot_points, solve_count, 1);

    void *kernel = (void*)setup_kkt_kernel_n<T>;
    void *args[] = {
        &solve_count,
        &knot_points,
        &state_size,
        &control_size,
        &d_G_dense,
        &d_C_dense,
        &d_g,
        &d_c,
        &d_dynMem_const,
        &timestep,
        &d_eePos_traj,
        &d_xs,
        &d_xu
    };

    gpuErrchk(cudaLaunchKernel(kernel, grid, block, args, kkt_smem_size));
}


