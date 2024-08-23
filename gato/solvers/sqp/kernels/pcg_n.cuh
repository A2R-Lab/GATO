#pragma once

#include <stdint.h>

#include "gato.cuh"
#include "GBD-PCG/include/pcg.cuh"

/**
 * @brief Batched preconditioned conjugate gradient solver.
 */
 template <typename T, uint32_t state_size, uint32_t knot_points>
 __global__
 void pcg_kernel_n(
     uint32_t solve_count,
     T *d_S, 
     T *d_Pinv, 
     T *d_gamma,  				
     T *d_lambda, 
     T *d_r, 
     T *d_p, 
     T *d_v_temp, 
     T *d_eta_new_temp,
     uint32_t *d_iters, 
     bool *d_max_iter_exit,
     uint32_t max_iter, 
     T exit_tol)
 {   
     const uint32_t traj_id_base = blockIdx.y * (solve_count / gridDim.y);
     const uint32_t traj_id_offset = threadIdx.y;
     const uint32_t traj_id = traj_id_base + traj_id_offset;
     const uint32_t knot_id = blockIdx.x;
     
     if (traj_id >= solve_count) return;
 
     const uint32_t thread_id = threadIdx.x;
     const uint32_t block_dim = blockDim.x;
     const uint32_t states_sq = state_size * state_size;
 
     // Offset pointers for this trajectory and knot
     const uint32_t traj_offset = traj_id * knot_points * state_size;
     const uint32_t knot_offset = knot_id * state_size;
     const uint32_t matrix_offset = traj_id * 3 * knot_points * states_sq + knot_id * 3 * states_sq;
     T *traj_S = d_S + matrix_offset;
     T *traj_Pinv = d_Pinv + matrix_offset;
     T *traj_gamma = d_gamma + traj_offset + knot_offset;
     T *traj_lambda = d_lambda + traj_offset + knot_offset;
     T *traj_r = d_r + traj_offset + knot_offset;
     T *traj_p = d_p + traj_offset + knot_offset;
     T *traj_v_temp = d_v_temp + traj_id * (knot_points + 1) + knot_id;
     T *traj_eta_new_temp = d_eta_new_temp + traj_id * (knot_points + 1) + knot_id;
     uint32_t *traj_iters = d_iters + traj_id;
     bool *traj_max_iter_exit = d_max_iter_exit + traj_id;
 
     extern __shared__ T s_temp[];
 
     T *s_S = s_temp;
     T *s_Pinv = s_S + 3*states_sq;
     T *s_gamma = s_Pinv + 3*states_sq;
     T *s_lambda = s_gamma + state_size;
     T *s_r = s_lambda + state_size;
     T *s_p = s_r + state_size;
     T *s_r_tilde = s_p + state_size;
     T *s_upsilon = s_r_tilde + state_size;
     T *s_v = s_upsilon + state_size;
 
     uint32_t iter;
     T alpha, beta, eta, eta_new;
     bool max_iter_exit = true;
 
     // populate shared memory
     for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim) {
         s_S[ind] = traj_S[ind];
         s_Pinv[ind] = traj_Pinv[ind];
     }
     if (thread_id < state_size) {
         s_gamma[thread_id] = traj_gamma[thread_id];
         s_lambda[thread_id] = traj_lambda[thread_id];
     }
 
     __syncthreads();
 
     // PCG algorithm
     // r = gamma - S * lambda
     bdmv<T>(s_r, s_S, s_lambda, state_size, knot_points, knot_id);
     for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
         s_r[ind] = s_gamma[ind] - s_r[ind];
         traj_r[ind] = s_r[ind]; 
     }
     
     __syncthreads();
 
     // r_tilde = Pinv * r
     bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points, knot_id);
     
     // p = r_tilde
     for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
         s_p[ind] = s_r_tilde[ind];
         traj_p[ind] = s_p[ind]; 
     }
 
     // eta = r * r_tilde
     T local_eta = 0;
     for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
         local_eta += s_r[ind] * s_r_tilde[ind];
     }
     atomicAdd(&traj_eta_new_temp[knot_points], local_eta);
     __syncthreads();
     
     if (knot_id == 0 && thread_id == 0) {
         eta = traj_eta_new_temp[knot_points];
         traj_eta_new_temp[knot_points] = 0;  // Reset for next iteration
     }
     __syncthreads();
     
     // MAIN PCG LOOP
     for(iter = 0; iter < max_iter; iter++) {
         // upsilon = S * p
         bdmv<T>(s_upsilon, s_S, s_p, state_size, knot_points, knot_id);
 
         // alpha = eta / (p * upsilon)
         T local_v = 0;
         for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
             local_v += s_p[ind] * s_upsilon[ind];
         }
         atomicAdd(&traj_v_temp[knot_points], local_v);
         __syncthreads();
         
         if (knot_id == 0 && thread_id == 0) {
             alpha = eta / traj_v_temp[knot_points];
             traj_v_temp[knot_points] = 0;  // Reset for next iteration
         }
         __syncthreads();
 
         // lambda = lambda + alpha * p
         // r = r - alpha * upsilon
         for(uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
             s_lambda[ind] += alpha * s_p[ind];
             s_r[ind] -= alpha * s_upsilon[ind];
             traj_lambda[ind] = s_lambda[ind];
             traj_r[ind] = s_r[ind];
         }
 
         __syncthreads();
 
         // r_tilde = Pinv * r
         bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points, knot_id);
 
         // eta_new = r * r_tilde
         local_eta = 0;
         for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
             local_eta += s_r[ind] * s_r_tilde[ind];
         }
         atomicAdd(&traj_eta_new_temp[knot_points], local_eta);
         __syncthreads();
         
         if (knot_id == 0 && thread_id == 0) {
             eta_new = traj_eta_new_temp[knot_points];
             traj_eta_new_temp[knot_points] = 0;  // Reset for next iteration
 
             if(abs(eta_new) < exit_tol) { 
                 iter++;
                 max_iter_exit = false;
                 traj_iters[0] = iter;
                 traj_max_iter_exit[0] = max_iter_exit;
                 eta = 0;  // Signal to exit
             } else {
                 // beta = eta_new / eta
                 // eta = eta_new
                 beta = eta_new / eta;
                 eta = eta_new;
                 traj_v_temp[knot_points] = beta;  // Use traj_v_temp to store beta
             }
         }
         __syncthreads();
 
         if (eta == 0) break;  // Exit condition
 
         beta = traj_v_temp[knot_points];  // Retrieve beta
 
         // p = r_tilde + beta*p
         for(uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
             s_p[ind] = s_r_tilde[ind] + beta * s_p[ind];
             traj_p[ind] = s_p[ind];
         }
         __syncthreads();
     }
 
     // save output
     if(knot_id == 0 && thread_id == 0) { 
         traj_iters[0] = iter; 
         traj_max_iter_exit[0] = max_iter_exit; 
     }
 }

/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/

template <typename T>
void pcg_n(
    const uint32_t solve_count,
    const uint32_t state_size,
    const uint32_t knot_points,
    T *d_S,
    T *d_Pinv,
    T *d_gamma,
    T *d_lambda,
    T *d_r,
    T *d_p,
    T *d_v_temp,
    T *d_eta_new_temp,
    uint32_t *d_pcg_iters,
    bool *d_pcg_exit,
    struct pcg_config<T> *config)
{
    dim3 grid(knot_points, min(solve_count,64u));
    dim3 block(PCG_NUM_THREADS, (solve_count + 64u - 1)/ 64u);

    size_t pcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);

    pcg_kernel_n<T, gato::STATE_SIZE, gato::KNOT_POINTS><<<grid, block, pcg_kernel_smem_size>>>(
        solve_count,
        d_S,
        d_Pinv,
        d_gamma, 
        d_lambda,
        d_r,
        d_p,
        d_v_temp,
        d_eta_new_temp,
        d_pcg_iters,
        d_pcg_exit,
        config->pcg_max_iter,
        config->pcg_exit_tol
    );
}