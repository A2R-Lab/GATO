#pragma once

#include "gato.cuh"
#include "utils/utils.cuh"

// size of smem for kkt kernel
template <typename T>
size_t get_kkt_kernel_smem_size() {

    size_t smem_size = sizeof(T) * (3*gato::STATES_SQ + 
                                    gato::CONTROLS_SQ + 
                                    7 * gato::STATE_SIZE + 
                                    3 * gato::CONTROL_SIZE + 
                                    gato::STATES_P_CONTROLS + 
                                    max(grid::EE_POS_SHARED_MEM_COUNT, grid::DEE_POS_SHARED_MEM_COUNT) + 
                                    max((gato::STATE_SIZE/2)*(gato::STATES_S_CONTROLS + 1) + gato::plant::forwardDynamicsAndGradient_TempMemSize_Shared(), 3 + (gato::STATE_SIZE/2)*6));

    return smem_size;
}


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void setup_kkt_kernel(T *d_G_dense, 
                    T *d_C_dense, 
                    T *d_g, 
                    T *d_c,
                    void *d_dynMem_const,
                    T *d_eePos_traj, 
                    T *d_xs, 
                    T *d_xu){

    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    //shared memory pointers
    extern __shared__ T s_temp[];
    T *s_xux = s_temp;
    T *s_eePos_traj = s_xux + 2*gato::STATE_SIZE + gato::CONTROL_SIZE;
    T *s_Qk = s_eePos_traj + 6;
    T *s_Rk = s_Qk + gato::STATES_SQ;
    T *s_qk = s_Rk + gato::CONTROLS_SQ;
    T *s_rk = s_qk + gato::STATE_SIZE;
    T *s_end = s_rk + gato::CONTROL_SIZE;

    for(unsigned k = block_id; k < gato::KNOT_POINTS-1; k += num_blocks) {

        glass::copy<T>(2*gato::STATE_SIZE + gato::CONTROL_SIZE, &d_xu[k*gato::STATES_S_CONTROLS], s_xux);
        glass::copy<T>(2 * 6, &d_eePos_traj[k*6], s_eePos_traj);
        
        __syncthreads();    

        if(k==gato::KNOT_POINTS-2) {          // last block

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + gato::STATES_SQ;
            T *s_Qkp1 = s_Bk + gato::STATES_P_CONTROLS;
            T *s_qkp1 = s_Qkp1 + gato::STATES_SQ;
            T *s_integrator_error = s_qkp1 + gato::STATE_SIZE;
            T *s_extra_temp = s_integrator_error + gato::STATE_SIZE;
            
            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(gato::STATE_SIZE, gato::CONTROL_SIZE,
                                                                        s_xux,
                                                                        s_Ak,
                                                                        s_Bk,
                                                                        s_integrator_error,
                                                                        s_extra_temp,
                                                                        d_dynMem_const,
                                                                        gato::TIMESTEP,
                                                                        block);
            __syncthreads();
            
            gato::plant::trackingCostGradientAndHessian_lastblock<T>(gato::STATE_SIZE, gato::CONTROL_SIZE,
                                                                    s_xux,
                                                                    s_eePos_traj,
                                                                    s_Qk,
                                                                    s_qk,
                                                                    s_Rk,
                                                                    s_rk,
                                                                    s_Qkp1,
                                                                    s_qkp1,
                                                                    s_extra_temp,
                                                                    d_dynMem_const);
            __syncthreads();

            for(int i = thread_id; i < gato::STATE_SIZE; i+=num_threads){
                d_c[i] = d_xu[i] - d_xs[i];
            }
            glass::copy<T>(gato::STATES_SQ, s_Qk, &d_G_dense[(gato::STATES_SQ+gato::CONTROLS_SQ)*k]);
            glass::copy<T>(gato::CONTROLS_SQ, s_Rk, &d_G_dense[(gato::STATES_SQ+gato::CONTROLS_SQ)*k+gato::STATES_SQ]);
            glass::copy<T>(gato::STATES_SQ, s_Qkp1, &d_G_dense[(gato::STATES_SQ+gato::CONTROLS_SQ)*(k+1)]);
            
            glass::copy<T>(gato::STATE_SIZE, s_qk, &d_g[gato::STATES_S_CONTROLS*k]);
            glass::copy<T>(gato::CONTROL_SIZE, s_rk, &d_g[gato::STATES_S_CONTROLS*k+gato::STATE_SIZE]);
            glass::copy<T>(gato::STATE_SIZE, s_qkp1, &d_g[gato::STATES_S_CONTROLS*(k+1)]);
            
            glass::copy<T>(gato::STATES_SQ, static_cast<T>(-1), s_Ak, &d_C_dense[(gato::STATES_SQ+gato::STATES_P_CONTROLS)*k]);
            glass::copy<T>(gato::STATES_P_CONTROLS, static_cast<T>(-1), s_Bk, &d_C_dense[(gato::STATES_SQ+gato::STATES_P_CONTROLS)*k+gato::STATES_SQ]);
            
            glass::copy<T>(gato::STATE_SIZE, s_integrator_error, &d_c[gato::STATE_SIZE*(k+1)]);

        }
        else{ // not last knot

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + gato::STATES_SQ;
            T *s_integrator_error = s_Bk + gato::STATES_P_CONTROLS;
            T *s_extra_temp = s_integrator_error + gato::STATE_SIZE;

            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(gato::STATE_SIZE, gato::CONTROL_SIZE,
                                                                        s_xux,
                                                                        s_Ak,
                                                                        s_Bk,
                                                                        s_integrator_error,
                                                                        s_extra_temp,
                                                                        d_dynMem_const,
                                                                        gato::TIMESTEP,
                                                                        block);
            __syncthreads();
           
            gato::plant::trackingCostGradientAndHessian<T>(gato::STATE_SIZE, gato::CONTROL_SIZE,
                                                        s_xux,
                                                        s_eePos_traj,
                                                        s_Qk,
                                                        s_qk,
                                                        s_Rk,
                                                        s_rk,
                                                        s_extra_temp,
                                                        d_dynMem_const);
            __syncthreads();
 
            glass::copy<T>(gato::STATES_SQ, s_Qk, &d_G_dense[(gato::STATES_SQ+gato::CONTROLS_SQ)*k]);
            glass::copy<T>(gato::CONTROLS_SQ, s_Rk, &d_G_dense[(gato::STATES_SQ+gato::CONTROLS_SQ)*k+gato::STATES_SQ]);
            
            glass::copy<T>(gato::STATE_SIZE, s_qk, &d_g[gato::STATES_S_CONTROLS*k]);
            glass::copy<T>(gato::CONTROL_SIZE, s_rk, &d_g[gato::STATES_S_CONTROLS*k+gato::STATE_SIZE]);
            
            glass::copy<T>(gato::STATES_SQ, static_cast<T>(-1), s_Ak, &d_C_dense[(gato::STATES_SQ+gato::STATES_P_CONTROLS)*k]);
            glass::copy<T>(gato::STATES_P_CONTROLS, static_cast<T>(-1), s_Bk, &d_C_dense[(gato::STATES_SQ+gato::STATES_P_CONTROLS)*k+gato::STATES_SQ]);
            
            glass::copy<T>(gato::STATE_SIZE, s_integrator_error, &d_c[gato::STATE_SIZE*(k+1)]);
        }
    }
}
