#pragma once

#include <algorithm>
#include <cmath>

#include "gato.cuh"
#include "utils/integrator.cuh"


/**
 * @brief Shifts vector by one step
 * @tparam T Data type
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param d_xu State and control vector
 */
 template <typename T>
 void just_shift(uint32_t state_size, uint32_t control_size, T *d_xu){ //state_size and control_size are params so we can shift vectors of different sizes
     for (uint32_t knot = 0; knot < gato::KNOT_POINTS-1; knot++){
         uint32_t stepsize = (state_size+(knot<gato::KNOT_POINTS-2)*control_size);
         gpuErrchk(cudaMemcpy(&d_xu[knot*(state_size+control_size)], &d_xu[(knot+1)*(state_size+control_size)], stepsize*sizeof(T), cudaMemcpyDeviceToDevice));
     }
 }
 
/**
 * @brief Kernel to integrate system based on x_k+1 = f(x_k,u_k,dt)
 *
 * shared memory size: sizeof(T)*(2*state_size + control_size + state_size/2 + gato::plant::forwardDynamics_TempMemSize_Shared())
 * x_k+1 (state_size) , xu_k (state_size + control_size), joint accelerations (state_size/2), 
 *
 * @tparam T Data type
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param d_x State vector
 * @param d_u Control vector
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 */
 template <typename T>
 __global__
 void simple_integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_x, T *d_u, void *d_dynMem_const, T dt){
 
     extern __shared__ T s_mem[];
     T *s_xkp1 = s_mem;
     T *s_xuk = s_xkp1 + state_size; 
     T *s_temp = s_xuk + state_size + control_size;
     cgrps::thread_block block = cgrps::this_thread_block();	  
     cgrps::grid_group grid = cgrps::this_grid();
     for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
         if(ind < state_size){
             s_xuk[ind] = d_x[ind];
         }
         else{
             s_xuk[ind] = d_u[ind-state_size];
         }
     }
     
     block.sync();
     integrator<T,0,0>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
     block.sync();
 
     for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
         d_x[ind] = s_xkp1[ind];
     }
 }

/**
 * @brief Simulates system based on x_k+1 = f(x_k,u_k,timestep)
 *
 * @tparam T Data type
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param knot_points Number of knot points
 * @param d_xs State vector
 * @param d_xu State and control vector
 * @param d_dynMem_const Dynamics memory
 * @param timestep Time step (s)
 * @param time_offset_us Time offset from start of trajectory (us)
 * @param sim_time_us Simulation duration (us)
 */
template <typename T>
void simple_simulate(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xs, T *d_xu, void *d_dynMem_const, double timestep, double time_offset_us, double sim_time_us, unsigned long long = 123456){
    // convert to seconds
    double time_offset = time_offset_us * 1e-6;
    double sim_time = sim_time_us * 1e-6;

    // initialize
    const T sim_step_time = 2e-4;
    const size_t simple_integrator_kernel_smem_size = sizeof(T)*(2*state_size + control_size + state_size/2 + gato::plant::forwardDynamicsAndGradient_TempMemSize_Shared());
    const uint32_t states_s_controls = state_size + control_size;
    uint32_t control_offset = static_cast<uint32_t>((time_offset) / timestep);
    T *control = &d_xu[control_offset * states_s_controls + state_size];
    
    // simulate through steps needed
    uint32_t sim_steps_needed = static_cast<uint32_t>(sim_time / sim_step_time);
    for(uint32_t step = 0; step < sim_steps_needed; step++){
        control_offset = static_cast<uint32_t>((time_offset + step * sim_step_time) / timestep);
        control = &d_xu[control_offset * states_s_controls + state_size];

        simple_integrator_kernel<T><<<1,32,simple_integrator_kernel_smem_size>>>(state_size, control_size, d_xs, control, d_dynMem_const, sim_step_time);
    }

    // simulate the remaining time
    T half_sim_step_time = fmod(sim_time, sim_step_time);
    simple_integrator_kernel<T><<<1,32,simple_integrator_kernel_smem_size>>>(state_size, control_size, d_xs, control, d_dynMem_const, half_sim_step_time);
}