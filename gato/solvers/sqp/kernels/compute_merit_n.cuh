#pragma once

#include "gato.cuh"
#include "utils/utils.cuh"
#include "compute_merit.cuh"


/**
 * @brief Initial merit function kernel.
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void initial_merit_kernel_n(uint32_t solve_count,
                           uint32_t state_size,
                           uint32_t control_size,
                           uint32_t knot_points,
                           T *d_xu,
                           T *d_eePos_traj,
                           T mu,
                           T dt,
                           void *d_dynMem_const,
                           T *d_merit_out)
{
    grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *)d_dynMem_const;
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t traj_len = knot_points * (state_size + control_size) - control_size;


    const uint32_t states_s_controls = state_size + control_size;
    extern __shared__ T s_mem[];

    T *s_xux_k = s_mem;
    T *s_eePos_k_traj = s_xux_k + 2 * state_size + control_size;
    T *s_temp = s_eePos_k_traj + grid::EE_POS_SIZE;

    T Jk, ck, pointmerit;
    for(unsigned solve = blockIdx.y; solve < solve_count; solve += gridDim.y){
        for(unsigned knot = blockIdx.x; knot < knot_points - 1; knot += gridDim.x){
            // Load data into shared memory
            for(int i = thread_id; i < state_size+(knot < knot_points-1)*(states_s_controls); i+=num_threads){
                s_xux_k[i] = d_xu[solve*traj_len + knot*states_s_controls + i];
                if (i < grid::EE_POS_SIZE){
                    s_eePos_k_traj[i] = d_eePos_traj[solve*knot_points*grid::EE_POS_SIZE + knot*grid::EE_POS_SIZE + i];                          
                }
            }
        
            block.sync();
            Jk = gato::plant::trackingCost<T>(state_size, control_size, knot_points, s_xux_k, s_eePos_k_traj, s_temp, d_robotModel);

            block.sync();
            if(knot < knot_points-1){
                ck = integratorError<T>(state_size, s_xux_k, &s_xux_k[states_s_controls], s_temp, d_robotModel, dt, block);
            }
            else{
                ck = 0;
            }
            block.sync();

            if(thread_id == 0){
                pointmerit = Jk + mu*ck;
                atomicAdd(d_merit_out, pointmerit);
            }
        }
    }
}


/**
 * @brief Batched line search merit function kernel.
 */
template <typename T>
__global__
void compute_ls_merit_kernel_n(uint32_t solve_count,
                                uint32_t state_size,
                                uint32_t control_size,
                                uint32_t knot_points,
                                T *d_xs,
                                T *d_xu, 
                                T *d_eePos_traj, 
                                T mu, 
                                T dt, 
                                void *d_dynMem_const, 
                                T *d_dz,
                                uint32_t num_alphas,
                                T *d_merits_out, 
                                T *d_merit_temp){
    // x: knot_points, y: solve_count * num_alphas
    grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *)d_dynMem_const;
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t knot_id = blockIdx.x;
    const uint32_t alpha_id = blockIdx.y % num_alphas;
    
    //const uint32_t traj_id = blockIdx.y;
    const uint32_t states_s_controls = state_size + control_size;

    extern __shared__ T s_mem[];
    T *s_xux_k = s_mem;
    T *s_eePos_k_traj = s_xux_k + 2*state_size+control_size;
    T *s_temp = s_eePos_k_traj + grid::EE_POS_SIZE;
    T *s_alpha_merits = s_temp + max(2 * state_size + control_size, state_size + gato::plant::forwardDynamics_TempMemSize_Shared());

    // Load data into shared memory
    for(int i = thread_id; i < state_size+(knot_id < knot_points-1)*(states_s_controls); i+=num_threads){
        s_xux_k[i] = d_xu[knot_id*states_s_controls+i];
        if (i < grid::EE_POS_SIZE){
            s_eePos_k_traj[i] = d_eePos_traj[knot_id*grid::EE_POS_SIZE+i];                            
        }
    }
    block.sync();

    // Compute merit for each alpha
    for (uint32_t alpha_idx = alpha_id; alpha_idx < num_alphas; alpha_idx += num_threads) {
        T alpha = -1.0 / (1 << alpha_idx);
        T Jk, ck, pointmerit;

        // Apply alpha * dz
        for(int i = thread_id; i < state_size+(knot_id < knot_points-1)*(states_s_controls); i+= num_threads){
            s_xux_k[i] += alpha * d_dz[knot_id*states_s_controls+i];
        }
        block.sync();
        
        Jk = gato::plant::trackingCost<T>(state_size, control_size, knot_points, s_xux_k, s_eePos_k_traj, s_temp, d_robotModel);
        
        block.sync();
        if(knot_id < knot_points-1){
            ck = integratorError<T>(state_size, s_xux_k, &s_xux_k[states_s_controls], s_temp, d_robotModel, dt, block);
        }
        else{
            // diff xs vs xs_traj
            for(int i = thread_id; i < state_size; i+=num_threads){
                s_temp[i] = abs(s_xux_k[i] - d_xs[i]);
            }
            block.sync();
            glass::reduce<T>(state_size, s_temp);
            block.sync();
            ck = s_temp[0];
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            d_merit_temp[alpha_idx*knot_points+knot_id] = pointmerit;
            s_alpha_merits[alpha_idx] = pointmerit;
        }
        
        // Revert s_xux_k for next alpha
        for(int i = thread_id; i < state_size+(knot_id < knot_points-1)*(states_s_controls); i+=num_threads){
            s_xux_k[i] -= alpha * d_dz[knot_id*states_s_controls+i];
        }
        block.sync();
    }

    // Reduce merits across knot points
    if (knot_id == 0) {
        for (uint32_t alpha_idx = alpha_id; alpha_idx < num_alphas; alpha_idx += num_threads) {
            T total_merit = 0;
            for (uint32_t k = thread_id; k < knot_points; k += num_threads) {
                total_merit += d_merit_temp[alpha_idx*knot_points + k];
            }
            s_temp[thread_id] = total_merit;
            block.sync();
            
            for (uint32_t s = num_threads/2; s > 0; s >>= 1) {
                if (thread_id < s) {
                    s_temp[thread_id] += s_temp[thread_id + s];
                }
                block.sync();
            }
            
            if (thread_id == 0) {
                d_merits_out[alpha_idx] = s_temp[0];
            }
        }
    }
}


/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/

/**
 * @brief Launch compute initial merit kernel.
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
void initial_merit_n(
    uint32_t solve_count,
    uint32_t state_size,
    uint32_t control_size,
    uint32_t knot_points,
    T *d_xu,
    T *d_eePos_traj,
    T mu,
    T dt,
    void *d_dynMem_const,
    T *d_merit_out)
{
    dim3 block(MERIT_THREADS);
    dim3 grid(knot_points, solve_count);
    const uint32_t merit_smem_size = get_merit_kernel_smem_size<T>();

    initial_merit_kernel_n<T, INTEGRATOR_TYPE, ANGLE_WRAP><<<grid, block, merit_smem_size>>>(
            solve_count, state_size, control_size, knot_points,
            d_xu, d_eePos_traj, mu, dt, d_dynMem_const, d_merit_out);
}


/**
 * @brief Launch compute line search merit kernel.
 */
template <typename T>
void compute_ls_merit_n(uint32_t solve_count,
                            uint32_t state_size,
                            uint32_t control_size,
                            uint32_t knot_points,
                            T *d_xs,
                            T *d_xu, 
                            T *d_eePos_traj, 
                            T mu, 
                            T dt, 
                            void *d_dynMem_const, 
                            T *d_dz,
                            uint32_t num_alphas,
                            T *d_merits_out, 
                            T *d_merit_temp) {

    dim3 block(MERIT_THREADS);
    dim3 grid(knot_points, solve_count);  // Parallelize across knot_points and solve_count
    const uint32_t merit_smem_size = get_merit_kernel_smem_size<T>() + sizeof(T);  // Only need space for one alpha at a time
    
    compute_ls_merit_kernel_n<T><<<grid, block, merit_smem_size>>>(
            solve_count, state_size, control_size, knot_points, 
            d_xs, d_xu, d_eePos_traj, 
            mu, dt, d_dynMem_const, 
            d_dz, num_alphas, 
            d_merits_out,
            d_merit_temp);
}