#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#include "gato.cuh"
#include "GBD-PCG/include/pcg.cuh"


template <typename T>
size_t pcgNSharedMemSize(uint32_t state_size, uint32_t knot_points) {
    return sizeof(T) * (
        3 * state_size * state_size * 3 + // s_S, s_Pinv
        state_size +                      // s_gamma
        3 * state_size +                  // s_lambda
        state_size +                      // s_r_tilde
        state_size +                      // s_upsilon
        max(knot_points, state_size) +    // s_v_b
        max(knot_points, state_size) +    // s_eta_new_b
        3 * state_size +                  // s_r
        3 * state_size +                  // s_p
        state_size +                      // s_r_b
        state_size +                      // s_p_b
        state_size                        // s_lambda_b
    );
}

template <typename T>
__global__
void pcg(
         T *d_S, 
         T *d_Pinv, 
         T *d_gamma,  				
         T *d_lambda, 
         T  *d_r, 
         T  *d_p, 
         T *d_v_temp, 
         T *d_eta_new_temp,
         uint32_t *d_iters, 
         bool *d_max_iter_exit,
         uint32_t max_iter, 
         T exit_tol,
         uint32_t state_size,
         uint32_t knot_points)
{   
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    extern __shared__ T s_temp[];
    
    T  *s_S = s_temp;
    T  *s_Pinv = s_S + 3 * states_sq;
    T  *s_gamma = s_Pinv + 3 * states_sq;
    T  *s_scratch = s_gamma + state_size;
    T *s_lambda = s_scratch;
    T *s_r_tilde = s_lambda + 3 * state_size;
    T  *s_upsilon = s_r_tilde + state_size;
    T  *s_v_b = s_upsilon + state_size;
    T  *s_eta_new_b = s_v_b + max(knot_points, state_size);
    T  *s_r = s_eta_new_b + max(knot_points, state_size);
    T  *s_p = s_r + 3 * state_size;
    T  *s_r_b = s_r + state_size;
    T  *s_p_b = s_p + state_size;
    T *s_lambda_b = s_lambda + state_size;

    uint32_t iter;
    T alpha, beta, eta, eta_new;

    bool max_iter_exit = true;

    for (unsigned ind = thread_id; ind < 3 * states_sq; ind += block_dim){
        if(block_id == 0 && ind < states_sq){ continue; }
        if(block_id == knot_points-1 && ind >= 2 * states_sq){ continue; }

        s_S[ind] = d_S[block_id * states_sq * 3 + ind];
        s_Pinv[ind] = d_Pinv[block_id * states_sq * 3 + ind];
    }
    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);

    __syncthreads();

    loadbdVecDynamic<T>(s_lambda, block_id, &d_lambda[block_x_statesize], state_size, knot_points-1);
    __syncthreads();
    bdmv<T>(s_r_b, s_S, s_lambda, state_size, knot_points-1, block_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind]; 
    }
    
    __syncthreads();

    loadbdVecDynamic<T>(s_r, block_id, &d_r[block_x_statesize], state_size, knot_points-1);
    __syncthreads();
    bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
    __syncthreads();
    
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind]; 
    }

    __syncthreads();

    glass::dotDynamic<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde);
    if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    __syncthreads();
    glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
    __syncthreads();
    eta = s_eta_new_b[0];

    for(iter = 0; iter < max_iter; iter++){

        loadbdVecDynamic<T>(s_p, block_id, &d_p[block_x_statesize], state_size, knot_points-1);
        __syncthreads();
        bdmv<T>(s_upsilon, s_S, s_p, state_size, knot_points-1, block_id);
        __syncthreads();

        glass::dotDynamic<T>(s_v_b, state_size, s_p_b, s_upsilon);
        __syncthreads();
        if(thread_id == 0){ d_v_temp[block_id] = s_v_b[0]; }
        __syncthreads();
        glass::reduce<T>(s_v_b, knot_points, d_v_temp);
        __syncthreads();
        alpha = eta / s_v_b[0];

        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[block_x_statesize + ind] = s_r_b[ind];
        }

        __syncthreads();

        loadbdVecDynamic<T>(s_r, block_id, &d_r[block_x_statesize], state_size, knot_points-1);
        __syncthreads();
        bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
        __syncthreads();

        glass::dotDynamic<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde);
        __syncthreads();
        if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        __syncthreads();
        glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
        __syncthreads();
        eta_new = s_eta_new_b[0];

        if(abs(eta_new) < exit_tol){ 
            iter++; 
            max_iter_exit = false; 
            break; 
        }

        beta = eta_new / eta;
        eta = eta_new;

        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
            s_p_b[ind] = s_r_tilde[ind] + beta * s_p_b[ind];
            d_p[block_x_statesize + ind] = s_p_b[ind];
        }
        __syncthreads();
    }

    if(thread_id == 0){ 
        d_iters[block_id] = iter; 
        d_max_iter_exit[block_id] = max_iter_exit; 
    }
    
    __syncthreads();
    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);

    __syncthreads();
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
    dim3 grid(solve_count);
    dim3 block(PCG_NUM_THREADS);

    size_t pcg_kernel_smem_size = pcgNSharedMemSize<T>(state_size, knot_points);

    cudaDeviceProp deviceProp;
    int dev = 0;
    cudaGetDeviceProperties(&deviceProp, dev);

    if(pcg_kernel_smem_size > deviceProp.sharedMemPerBlock){
        printf("[Error] Required shared memory size (%zu bytes) exceeds device limit (%zu bytes).\n", 
               pcg_kernel_smem_size, deviceProp.sharedMemPerBlock);
        exit(EXIT_FAILURE);
    }

    pcg<<<grid, block, pcg_kernel_smem_size>>>(
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
        config->pcg_exit_tol,
        state_size,
        knot_points
    );

    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaDeviceSynchronize());
}
// #pragma once

// #include <stdint.h>

// #include "gato.cuh"
// #include "GBD-PCG/include/pcg.cuh"

// template <typename T>
// size_t pcgNSharedMemSize(uint32_t state_size, uint32_t knot_points) {
//     return sizeof(T) * (
//         3 * state_size * state_size * 3 + // s_S, s_Pinv
//         state_size + // s_gamma
//         3 * state_size + // s_lambda
//         state_size + // s_r_tilde
//         state_size + // s_upsilon
//         max(knot_points, state_size) + // s_v_b
//         max(knot_points, state_size) + // s_eta_new_b
//         3 * state_size + // s_r
//         3 * state_size + // s_p
//         state_size + // s_r_b
//         state_size + // s_p_b
//         state_size // s_lambda_b
//     );
// }

// template <typename T>
// __global__
// void pcg(
//          T *d_S, 
//          T *d_Pinv, 
//          T *d_gamma,  				
//          T *d_lambda, 
//          T  *d_r, 
//          T  *d_p, 
//          T *d_v_temp, 
//          T *d_eta_new_temp,
//          uint32_t *d_iters, 
//          bool *d_max_iter_exit,
//          uint32_t max_iter, 
//          T exit_tol,
//          uint32_t state_size,
//          uint32_t knot_points)
// {   
//     const cgrps::thread_block block = cgrps::this_thread_block();	 
//     const cgrps::grid_group grid = cgrps::this_grid();
//     const uint32_t block_id = blockIdx.x;
//     const uint32_t block_dim = blockDim.x;
//     const uint32_t thread_id = threadIdx.x;
//     const uint32_t block_x_statesize = block_id * state_size;
//     const uint32_t states_sq = state_size * state_size;

//     extern __shared__ T s_temp[];
    
//     T  *s_S = s_temp;
//     T  *s_Pinv = s_S +3*states_sq;
//     T  *s_gamma = s_Pinv + 3*states_sq;
//     T  *s_scratch = s_gamma + state_size;
//     T *s_lambda = s_scratch;
//     T *s_r_tilde = s_lambda + 3*state_size;
//     T  *s_upsilon = s_r_tilde + state_size;
//     T  *s_v_b = s_upsilon + max(knot_points, state_size);
//     T  *s_eta_new_b = s_v_b + max(knot_points, state_size);
//     T  *s_r = s_eta_new_b + max(knot_points, state_size);
//     T  *s_p = s_r + 3*state_size;
//     T  *s_r_b = s_r + state_size;
//     T  *s_p_b = s_p + state_size;
//     T *s_lambda_b = s_lambda + state_size;

//     uint32_t iter;
//     T alpha, beta, eta, eta_new;

//     bool max_iter_exit = true;

//     // populate shared memory
//     for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim){
//         if(block_id == 0 && ind < states_sq){ continue; }
//         if(block_id == knot_points-1 && ind >= 2*states_sq){ continue; }

//         s_S[ind] = d_S[block_id*states_sq*3 + ind];
//         s_Pinv[ind] = d_Pinv[block_id*states_sq*3 + ind];
//     }
//     glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);

//     //
//     // PCG
//     //

//     // r = gamma - S * lambda
//     loadbdVecDynamic<T>(s_lambda, block_id, &d_lambda[block_x_statesize], state_size, knot_points-1);
//     __syncthreads();
//     bdmv<T>(s_r_b, s_S, s_lambda, state_size, knot_points-1,  block_id);
//     __syncthreads();
//     for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
//         s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
//         d_r[block_x_statesize + ind] = s_r_b[ind]; 
//     }
    
//     grid.sync(); //-------------------------------------

//     // r_tilde = Pinv * r
//     loadbdVecDynamic<T>(s_r, block_id, &d_r[block_x_statesize], state_size, knot_points-1);
//     __syncthreads();
//     bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
//     __syncthreads();
    
//     // p = r_tilde
//     for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
//         s_p_b[ind] = s_r_tilde[ind];
//         d_p[block_x_statesize + ind] = s_p_b[ind]; 
//     }

//     // eta = r * r_tilde
//     glass::dotDynamic<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde);
//     if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
//     grid.sync(); //-------------------------------------
//     glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
//     __syncthreads();
//     eta = s_eta_new_b[0];
    

//     // MAIN PCG LOOP

//     for(iter = 0; iter < max_iter; iter++){

//         // upsilon = S * p
//         loadbdVecDynamic<T>(s_p, block_id, &d_p[block_x_statesize], state_size, knot_points-1);
//         __syncthreads();
//         bdmv<T>(s_upsilon,  s_S, s_p,state_size, knot_points-1, block_id);
//         __syncthreads();

//         // alpha = eta / p * upsilon
//         glass::dotDynamic<T>(s_v_b, state_size, s_p_b, s_upsilon);
//         __syncthreads();
//         if(thread_id == 0){ d_v_temp[block_id] = s_v_b[0]; }
//         grid.sync(); //-------------------------------------
//         glass::reduce<T>(s_v_b, knot_points, d_v_temp);
//         __syncthreads();
//         alpha = eta / s_v_b[0];
//         // lambda = lambda + alpha * p
//         // r = r - alpha * upsilon
//         for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
//             s_lambda_b[ind] += alpha * s_p_b[ind];
//             s_r_b[ind] -= alpha * s_upsilon[ind];
//             d_r[block_x_statesize + ind] = s_r_b[ind];
//         }

//         grid.sync(); //-------------------------------------

//         // r_tilde = Pinv * r
//         loadbdVecDynamic<T>(s_r, block_id, &d_r[block_x_statesize], state_size, knot_points-1);
//         __syncthreads();
//         bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
//         __syncthreads();

//         // eta = r * r_tilde
//         glass::dotDynamic<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde);
//         __syncthreads();
//         if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
//         grid.sync(); //-------------------------------------
//         glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
//         __syncthreads();
//         eta_new = s_eta_new_b[0];

//         if(abs(eta_new) < exit_tol){ iter++; max_iter_exit = false; break; }

//         // beta = eta_new / eta
//         // eta = eta_new
//         beta = eta_new / eta;
//         eta = eta_new;

//         // p = r_tilde + beta*p
//         for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
//             s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
//             d_p[block_x_statesize + ind] = s_p_b[ind];
//         }
//         grid.sync(); //-------------------------------------
//     }


//     // save output
//     if(block_id == 0 && thread_id == 0){ d_iters[0] = iter; d_max_iter_exit[0] = max_iter_exit; }
    
//     __syncthreads();
//     glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);

//     grid.sync();
// }


// /*******************************************************************************
//  *                           Interface Functions                                *
//  *******************************************************************************/

// template <typename T>
// void pcg_n(
//     const uint32_t solve_count,
//     const uint32_t state_size,
//     const uint32_t knot_points,
//     T *d_S,
//     T *d_Pinv,
//     T *d_gamma,
//     T *d_lambda,
//     T *d_r,
//     T *d_p,
//     T *d_v_temp,
//     T *d_eta_new_temp,
//     uint32_t *d_pcg_iters,
//     bool *d_pcg_exit,
//     struct pcg_config<T> *config)
// {
//     // Calculate the grid and block dimensions
//     dim3 grid(knot_points, solve_count);
//     dim3 block(PCG_NUM_THREADS); // max 1024 threads per block
    

//     // Calculate the shared memory size using the new function
//     size_t pcg_kernel_smem_size = pcgNSharedMemSize<T>(state_size, knot_points);


//     //TODO: there's some issue with the shared memory size, need to look into it

//     // Check if the device supports cooperative launch
//     int dev = 0;
//     cudaDeviceProp deviceProp;
//     cudaGetDeviceProperties(&deviceProp, dev);
//     int supportsCoopLaunch = 0;
//     cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
//     if (!supportsCoopLaunch) {
//         printf("[Error] Device does not support Cooperative Threads\n");
//         exit(5);
//     }

//     // Launch the PCG kernel cooperatively
//     void *kernelArgs[] = {
//         (void *)&d_S,
//         (void *)&d_Pinv,
//         (void *)&d_gamma,
//         (void *)&d_lambda,
//         (void *)&d_r,
//         (void *)&d_p,
//         (void *)&d_v_temp,
//         (void *)&d_eta_new_temp,
//         (void *)&d_pcg_iters,
//         (void *)&d_pcg_exit,
//         (void *)&config->pcg_max_iter,
//         (void *)&config->pcg_exit_tol,
//         (void *)&state_size,
//         (void *)&knot_points
//     };
//     cudaLaunchCooperativeKernel((void*)pcg<T>, grid, block, kernelArgs, pcg_kernel_smem_size);

//     // Synchronize the device
//     gpuErrchk(cudaDeviceSynchronize());
// }