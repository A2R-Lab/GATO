#pragma once

#include <stdint.h>

#include "gato.cuh"
#include "GBD-PCG/include/pcg.cuh"

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
    const uint32_t traj_id = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t knot_id = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t states_sq = state_size * state_size;

    if (traj_id >= solve_count) return;

    extern __shared__ T s_temp[];
    
    T *s_S = s_temp;
    T *s_Pinv = s_S + 3*states_sq;
    T *s_gamma = s_Pinv + 3*states_sq;
    T *s_scratch = s_gamma + state_size;
    T *s_lambda = s_scratch;
    T *s_r_tilde = s_lambda + 3*state_size;
    T *s_upsilon = s_r_tilde + state_size;
    T *s_v = s_upsilon + state_size;
    T *s_eta_new = s_v + max(knot_points, state_size);
    T *s_r = s_eta_new + max(knot_points, state_size);
    T *s_p = s_r + 3*state_size;

    uint32_t iter;
    T alpha, beta, eta, eta_new;

    bool max_iter_exit = true;

    // Offset pointers for this trajectory
    const uint32_t traj_offset = traj_id * knot_points * state_size;
    const uint32_t matrix_offset = traj_id * 3 * knot_points * states_sq + knot_id * 3 * states_sq;
    T *traj_S = d_S + matrix_offset;
    T *traj_Pinv = d_Pinv + matrix_offset;
    T *traj_gamma = d_gamma + traj_offset + knot_id * state_size;
    T *traj_lambda = d_lambda + traj_offset + knot_id * state_size;
    T *traj_r = d_r + traj_offset + knot_id * state_size;
    T *traj_p = d_p + traj_offset + knot_id * state_size;
    T *traj_v_temp = d_v_temp + traj_id * (knot_points + 1);
    T *traj_eta_new_temp = d_eta_new_temp + traj_id * (knot_points + 1);

    // populate shared memory
    for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim) {
        if(knot_id == 0 && ind < states_sq) { continue; }
        if(knot_id == knot_points-1 && ind >= 2*states_sq) { continue; }

        s_S[ind] = traj_S[ind];
        s_Pinv[ind] = traj_Pinv[ind];
    }
    glass::copy<T>(state_size, traj_gamma, s_gamma);
    __syncthreads();


    // r = gamma - S * lambda
    loadbdVec<T, state_size, knot_points-1>(s_lambda, knot_id, traj_lambda);
    __syncthreads();
    bdmv<T>(s_r, s_S, s_lambda, state_size, knot_points-1, knot_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_r[ind] = s_gamma[ind] - s_r[ind];
        traj_r[ind] = s_r[ind]; 
    }

    // r_tilde = Pinv * r
    loadbdVec<T, state_size, knot_points-1>(s_r, knot_id, traj_r);
    __syncthreads();
    bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, knot_id);
    __syncthreads();
    
    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_p[ind] = s_r_tilde[ind];
        traj_p[ind] = s_p[ind]; 
    }

    // eta = r * r_tilde
    glass::dot<T, state_size>(s_eta_new, s_r, s_r_tilde);
    if(thread_id == 0) { traj_eta_new_temp[knot_id] = s_eta_new[0]; }
    __syncthreads();
    if (knot_id == 0 && thread_id == 0) {
        eta = 0;
        for (int i = 0; i < knot_points; i++) {
            eta += traj_eta_new_temp[i];
        }
    }
    __syncthreads();

    // MAIN PCG LOOP
    for(iter = 0; iter < max_iter; iter++) {
        // upsilon = S * p
        loadbdVec<T, state_size, knot_points-1>(s_p, knot_id, traj_p);
        __syncthreads();
        bdmv<T>(s_upsilon, s_S, s_p, state_size, knot_points-1, knot_id);
        __syncthreads();

        // alpha = eta / p * upsilon
        glass::dot<T, state_size>(s_v, s_p, s_upsilon);
        __syncthreads();
        if(thread_id == 0) { traj_v_temp[knot_id] = s_v[0]; }
        __syncthreads();
        if (knot_id == 0 && thread_id == 0) {
            T v_sum = 0;
            for (int i = 0; i < knot_points; i++) {
                v_sum += traj_v_temp[i];
            }
            alpha = eta / v_sum;
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

        // r_tilde = Pinv * r
        loadbdVec<T, state_size, knot_points-1>(s_r, knot_id, traj_r);
        __syncthreads();
        bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, knot_id);
        __syncthreads();

        // eta = r * r_tilde
        glass::dot<T, state_size>(s_eta_new, s_r, s_r_tilde);
        if(thread_id == 0) { traj_eta_new_temp[knot_id] = s_eta_new[0]; }
        __syncthreads();
        if (knot_id == 0 && thread_id == 0) {
            eta_new = 0;
            for (int i = 0; i < knot_points; i++) {
                eta_new += traj_eta_new_temp[i];
            }

            if(abs(eta_new) < exit_tol) { 
                iter++; 
                max_iter_exit = false; 
                d_iters[traj_id] = iter;
                d_max_iter_exit[traj_id] = max_iter_exit;
                break; 
            }
            beta = eta_new / eta;
            eta = eta_new;
            traj_v_temp[knot_points] = beta;  // Use traj_v_temp to store beta
        }
        __syncthreads();

        beta = traj_v_temp[knot_points];  // Retrieve beta

        // p = r_tilde + beta*p
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_p[ind] = s_r_tilde[ind] + beta * s_p[ind];
            traj_p[ind] = s_p[ind];
        }
    }

    // save output
    if(knot_id == 0 && thread_id == 0) { 
        d_iters[traj_id] = iter;
        d_max_iter_exit[traj_id] = max_iter_exit;
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