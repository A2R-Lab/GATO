#pragma once

#include "gato.cuh"

/**
 * @brief Kernel for integrating directly from host.
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of state vector
 * @param control_size Size of control vector
 * @param d_xkp1 Output next state
 * @param d_xuk Input state and control
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_xkp1, T *d_xuk, void *d_dynMem_const, T dt){
    extern __shared__ T s_smem[];
    T *s_xkp1 = s_smem;
    T *s_xuk = s_xkp1 + state_size; 
    T *s_temp = s_xuk + state_size + control_size;
    cgrps::thread_block block = cgrps::this_thread_block();	  
    cgrps::grid_group grid = cgrps::this_grid();
    for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
        s_xuk[ind] = d_xuk[ind];
    }

    block.sync();
    integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
    block.sync();

    for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
        d_xkp1[ind] = s_xkp1[ind];
    }
}
 
 
 /**
  * @brief Host function for integrator.
  *
  * take start state from h_xs and control input from h_xu -> and update h_xs
  *
  * @tparam T Data type
  * @param state_size Size of the state vector
  * @param control_size Size of the control vector
  * @param d_xs State vector
  * @param d_xu State and control vector
  * @param d_dynMem_const Dynamics memory
  * @param dt Time step
  */
template <typename T>
void integrator_host(uint32_t state_size, uint32_t control_size, T *d_xs, T *d_xu, void *d_dynMem_const, T dt){

    const size_t integrator_kernel_smem_size = sizeof(T)*(2*state_size + control_size + state_size/2 + gato::plant::forwardDynamics_TempMemSize_Shared());
    //TODO: one block one thread? Why?
    integrator_kernel<T><<<1,1, integrator_kernel_smem_size>>>(state_size, control_size, d_xs, d_xu, d_dynMem_const, dt);

    //TODO: needs sync?
}
