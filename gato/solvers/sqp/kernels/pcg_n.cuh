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
    const uint32_t traj_id = blockIdx.y;
    const uint32_t block_id = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    // Offset pointers for this trajectory
    const uint32_t traj_offset = traj_id * knot_points * state_size;
    const uint32_t matrix_offset = traj_id * 3 * knot_points * states_sq;
    d_S += matrix_offset;
    d_Pinv += matrix_offset;
    d_gamma += traj_offset;
    d_lambda += traj_offset;
    d_r += traj_offset;
    d_p += traj_offset;
    d_v_temp += traj_id * (knot_points + 1);  // +1 for global reduction
    d_eta_new_temp += traj_id * (knot_points + 1);  // +1 for global reduction
    d_iters += traj_id;
    d_max_iter_exit += traj_id;

    extern __shared__ T s_temp[];

    T *s_S = s_temp;
    T *s_Pinv = s_S + 3*states_sq;
    T *s_gamma = s_Pinv + 3*states_sq;
    T *s_scratch = s_gamma + state_size;
    T *s_lambda = s_scratch;
    T *s_r_tilde = s_lambda + 3*state_size;
    T *s_upsilon = s_r_tilde + state_size;
    T *s_v_b = s_upsilon + state_size;
    T *s_eta_new_b = s_v_b + max(knot_points, state_size);
    T *s_r = s_eta_new_b + max(knot_points, state_size);
    T *s_p = s_r + 3*state_size;
    T *s_r_b = s_r + state_size;
    T *s_p_b = s_p + state_size;
    T *s_lambda_b = s_lambda + state_size;

    uint32_t iter;
    T alpha, beta, eta, eta_new;
    bool max_iter_exit = true;

    // populate shared memory
    for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim) {
        if(block_id == 0 && ind < states_sq) { continue; }
        if(block_id == knot_points-1 && ind >= 2*states_sq) { continue; }

        s_S[ind] = d_S[block_id*states_sq*3 + ind];
        s_Pinv[ind] = d_Pinv[block_id*states_sq*3 + ind];
    }
    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);

    __syncthreads();

    // PCG algorithm
    // r = gamma - S * lambda
    loadbdVec<T, state_size, knot_points-1>(s_lambda, block_id, &d_lambda[block_x_statesize]);
    __syncthreads();
    bdmv<T>(s_r_b, s_S, s_lambda, state_size, knot_points-1, block_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind]; 
    }
    
    __syncthreads();

    // r_tilde = Pinv * r
    loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
    __syncthreads();
    bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
    __syncthreads();
    
    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind]; 
    }

    // eta = r * r_tilde
    glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_tilde);
    if(thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    __syncthreads();
    
    if (block_id == 0 && thread_id == 0) {
        T sum_eta = 0;
        for (int i = 0; i < knot_points; ++i) {
            sum_eta += d_eta_new_temp[i];
        }
        d_eta_new_temp[knot_points] = sum_eta;
        eta = sum_eta;
    }
    __syncthreads();
    
    // MAIN PCG LOOP
    for(iter = 0; iter < max_iter; iter++) {
        // upsilon = S * p
        loadbdVec<T, state_size, knot_points-1>(s_p, block_id, &d_p[block_x_statesize]);
        __syncthreads();
        bdmv<T>(s_upsilon, s_S, s_p, state_size, knot_points-1, block_id);
        __syncthreads();

        // alpha = eta / (p * upsilon)
        glass::dot<T, state_size>(s_v_b, s_p_b, s_upsilon);
        __syncthreads();
        if(thread_id == 0) { d_v_temp[block_id] = s_v_b[0]; }
        __syncthreads();
        
        if (block_id == 0 && thread_id == 0) {
            T sum_v = 0;
            for (int i = 0; i < knot_points; ++i) {
                sum_v += d_v_temp[i];
            }
            d_v_temp[knot_points] = sum_v;
            alpha = eta / sum_v;
        }
        __syncthreads();

        // lambda = lambda + alpha * p
        // r = r - alpha * upsilon
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[block_x_statesize + ind] = s_r_b[ind];
        }

        __syncthreads();

        // r_tilde = Pinv * r
        loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
        __syncthreads();
        bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
        __syncthreads();

        // eta_new = r * r_tilde
        glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_tilde);
        __syncthreads();
        if(thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        __syncthreads();
        
        if (block_id == 0 && thread_id == 0) {
            T sum_eta_new = 0;
            for (int i = 0; i < knot_points; ++i) {
                sum_eta_new += d_eta_new_temp[i];
            }
            d_eta_new_temp[knot_points] = sum_eta_new;
            eta_new = sum_eta_new;

            if(abs(eta_new) < exit_tol) { 
                iter++;
                max_iter_exit = false;
                d_iters[0] = iter;
                d_max_iter_exit[0] = max_iter_exit;
                eta = 0;  // Signal to exit
            } else {
                // beta = eta_new / eta
                // eta = eta_new
                T beta = eta_new / eta;
                eta = eta_new;
                d_v_temp[knot_points] = beta;  // Use d_v_temp to store beta
            }
        }
        __syncthreads();

        if (eta == 0) break;  // Exit condition

        beta = d_v_temp[knot_points];  // Retrieve beta

        // p = r_tilde + beta*p
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
            d_p[block_x_statesize + ind] = s_p_b[ind];
        }
        __syncthreads();
    }

    // save output
    if(block_id == 0 && thread_id == 0) { 
        d_iters[0] = iter; 
        d_max_iter_exit[0] = max_iter_exit; 
    }
    
    __syncthreads();
    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);
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
    dim3 grid(knot_points, solve_count);
    dim3 block(PCG_NUM_THREADS);

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