#pragma once

#include <cstdint>
#include <cooperative_groups.h>

#include "settings.h"
#include "constants.h"
#include "utils/cuda_utils.cuh"
#include "utils/linalg.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

/*

template <typename T>
__global__
void formSchurSystemBatchedKernel()

template <typename T>
__host__
size_t getFormSchurSystemBatchedSMemSize()

template <typename T>
__host__
void formSchurSystemBatched()

--------------------------------------------------

template <typename T>
__global__
void computeDzBatchedKernel()

template <typename T>
__host__
size_t getComputeDzBatchedSMemSize()

template <typename T>
__host__
void computeDzBatched()

*/

template <typename T, uint32_t BatchSize>
__global__
void formSchurSystemBatchedKernel1(
    T *d_S_batch,
    T *d_P_inv_batch,
    T *d_gamma_batch,
    T *d_Q_batch, // modified in-place to become Q_inv
    T *d_R_batch, // modified in-place to become R_inv
    T *d_q_batch,
    T *d_r_batch,
    T *d_A_batch,
    T *d_B_batch,
    T *d_c_batch,
    T *d_rho_penalty_batch
) {
    // launched with grid of (KNOT_POINTS, solve_idx)
    uint32_t knot_idx = blockIdx.x;
    uint32_t solve_idx = blockIdx.y;

    extern __shared__ T s_mem[]; 

    T *s_Q_k = s_mem;
    T *s_Q_k_inv = s_Q_k + STATE_SIZE_SQ;
    T *s_Q_kp1 = s_Q_k_inv + STATE_SIZE_SQ;
    T *s_Q_kp1_inv = s_Q_kp1 + STATE_SIZE_SQ;
    T *s_R_k = s_Q_kp1_inv + STATE_SIZE_SQ;
    T *s_R_k_inv = s_R_k + CONTROL_SIZE_SQ;

    T *s_q_k = s_R_k_inv + CONTROL_SIZE_SQ;
    T *s_q_kp1 = s_q_k + STATE_SIZE;
    T *s_r_k = s_q_kp1 + STATE_SIZE;
    
    T *s_A_k = s_r_k + CONTROL_SIZE;
    T *s_B_k = s_A_k + STATE_SIZE_SQ;

    T *s_A_Q_inv = s_B_k + STATE_P_CONTROL;
    T *s_B_R_inv = s_A_Q_inv + STATE_SIZE_SQ;

    T *s_theta_k = s_B_R_inv + STATE_P_CONTROL;
    T *s_theta_k_inv = s_theta_k + STATE_SIZE_SQ;
    T *s_gamma_k = s_theta_k_inv + STATE_SIZE_SQ;
    T *s_scratch = s_gamma_k + STATE_SIZE;

    if (knot_idx < KNOT_POINTS - 1) { // all except last knot

        // ----- Populate shared memory -----

        T *d_Q_k = getOffsetStateSq<T, BatchSize>(d_Q_batch, solve_idx, knot_idx);
        T *d_Q_kp1 = getOffsetStateSq<T, BatchSize>(d_Q_batch, solve_idx, knot_idx + 1);
        T *d_R_k = getOffsetControlSq<T, BatchSize>(d_R_batch, solve_idx, knot_idx);
        block::copy<T, STATE_SIZE_SQ>(s_Q_k, d_Q_k);
        block::copy<T, STATE_SIZE_SQ>(s_Q_kp1, d_Q_kp1);
        block::copy<T, CONTROL_SIZE_SQ>(s_R_k, d_R_k);
        block::loadIdentity<T, STATE_SIZE>(s_Q_k_inv);
        block::loadIdentity<T, STATE_SIZE>(s_Q_kp1_inv);
        block::loadIdentity<T, CONTROL_SIZE>(s_R_k_inv);


        T *d_q_k = getOffsetState<T, BatchSize>(d_q_batch, solve_idx, knot_idx);
        T *d_q_kp1 = getOffsetState<T, BatchSize>(d_q_batch, solve_idx, knot_idx + 1);
        T *d_r_k = getOffsetControl<T, BatchSize>(d_r_batch, solve_idx, knot_idx);
        block::copy<T, STATE_SIZE>(s_q_k, d_q_k);
        block::copy<T, STATE_SIZE>(s_q_kp1, d_q_kp1);
        block::copy<T, CONTROL_SIZE>(s_r_k, d_r_k);

        T *d_A_k = getOffsetStateSq<T, BatchSize>(d_A_batch, solve_idx, knot_idx);
        T *d_B_k = getOffsetStatePControl<T, BatchSize>(d_B_batch, solve_idx, knot_idx);
        T *d_c_k = getOffsetState<T, BatchSize>(d_c_batch, solve_idx, knot_idx + 1);
        block::copy<T, STATE_SIZE_SQ>(s_A_k, d_A_k);
        block::copy<T, STATE_P_CONTROL>(s_B_k, d_B_k);
        block::copy<T, STATE_SIZE>(s_gamma_k, d_c_k, static_cast<T>(-1));
        __syncthreads();

        // ----- Compute theta_k, phi_k, and gamma_k -----
        // theta_k = - ( (A_k * Q_k_inv * A_k^T) + (B_k * R_k_inv * B_k^T) + (Q_kp1_inv) )
        // phi_k = A_k * Q_k_inv
        // gamma_k = c - (- (A_k * Q_k_inv * q_k) - (B_k * R_k_inv * r_k) + (Q_kp1_inv * q_kp1))

        // Q_k_inv and R_k_inv
        // add scaled identity with rho to penalize constraint violations
        T rho_penalty = d_rho_penalty_batch[solve_idx];
        block::addScaledIdentity<T, STATE_SIZE>(s_Q_k, rho_penalty);
        block::addScaledIdentity<T, STATE_SIZE>(s_Q_kp1, rho_penalty);
        block::addScaledIdentity<T, CONTROL_SIZE>(s_R_k, rho_penalty);
        __syncthreads();

        // TODO: cholesky factorization inverse (because symmetric positive definite)
        block::invertMatrix<T>(STATE_SIZE, STATE_SIZE, CONTROL_SIZE, STATE_SIZE, s_Q_k, s_Q_kp1, s_R_k, s_scratch);
        __syncthreads();

        // save Q_k_inv and R_k_inv into d_Q_batch and d_R_batch for computing dz
        block::copy<T, STATE_SIZE_SQ>(d_Q_k, s_Q_k_inv);
        block::copy<T, CONTROL_SIZE_SQ>(d_R_k, s_R_k_inv);
        if (knot_idx == KNOT_POINTS - 2) { // last knot doesn't compute Q_k_inv, so use second last knot's Q_kp1_inv
            block::copy<T, STATE_SIZE_SQ>(d_Q_kp1, s_Q_kp1_inv);
        }

        // copy Q_kp1_inv into theta_k to save a sum operation
        block::copy<T, STATE_SIZE_SQ>(s_theta_k, s_Q_kp1_inv);
        __syncthreads();

        // A_k * Q_k_inv (phi) and B_k * R_k_inv 
        block::matMul<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(s_A_Q_inv, s_A_k, s_Q_k_inv);
        __syncthreads();
        block::matMul<T, STATE_SIZE, CONTROL_SIZE, CONTROL_SIZE>(s_B_R_inv, s_B_k, s_R_k_inv);
        __syncthreads();

        // theta_k = (A_k * Q_k_inv * A_k^T) + (B_k * R_k_inv * B_k^T) + (Q_kp1_inv)
        block::matMulTransposeSum<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(s_theta_k, s_A_Q_inv, s_A_k);
        __syncthreads();

        block::matMulTransposeSum<T, STATE_SIZE, CONTROL_SIZE, STATE_SIZE>(s_theta_k, s_B_R_inv, s_B_k);
        __syncthreads();

        // gamma_k
        block::matMulSum<T, STATE_SIZE, STATE_SIZE, 1>(s_gamma_k, s_Q_kp1_inv, s_q_kp1);
        __syncthreads(); 
        block::matMulSum<T, STATE_SIZE, STATE_SIZE, 1>(s_gamma_k, s_A_Q_inv, s_q_k, true);
        __syncthreads();
        block::matMulSum<T, STATE_SIZE, CONTROL_SIZE, 1>(s_gamma_k, s_B_R_inv, s_r_k, true);
        __syncthreads();
        T *d_gamma_k = getOffsetStatePadded<T, BatchSize>(d_gamma_batch, solve_idx, knot_idx + 1);
        block::copy<T, STATE_SIZE>(d_gamma_k, s_gamma_k, static_cast<T>(-1));


        // ----- save theta_k, phi_k, and gamma_k in S and gamma -----

        // S_k (right diag: phi_k^T, left diag: phi_k, next main diag: theta_k)
        // S_k is stored in row-major order
        // k refers to knot, not block row
        T *d_S_k_right = getOffsetBlockRowPadded<T, BatchSize>(d_S_batch, solve_idx, knot_idx) + 2 * STATE_SIZE;
        T *d_S_k_left = getOffsetBlockRowPadded<T, BatchSize>(d_S_batch, solve_idx, knot_idx + 1);
        T *d_S_kp1_main = d_S_k_left + STATE_SIZE;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
            uint32_t x = i % STATE_SIZE;
            uint32_t y = i / STATE_SIZE;
            uint32_t block_matrix_offset = y * BLOCK_ROW_R_DIM + x;
            d_S_k_right[block_matrix_offset] = s_A_Q_inv[i]; // phi_k^T
            d_S_k_left[block_matrix_offset] = s_A_Q_inv[x * STATE_SIZE + y]; // phi_k
            d_S_kp1_main[block_matrix_offset] = -s_theta_k[x * STATE_SIZE + y]; // theta_k
        }
        __syncthreads(); 

        // ----- Compute theta_k_inv and save in P_inv -----
        block::loadIdentity<T, STATE_SIZE>(s_theta_k_inv);
        __syncthreads();
        block::invertMatrix<T>(STATE_SIZE, s_theta_k, s_scratch);
        __syncthreads();

        // main diag: theta_k_inv (offset by STATE_SIZE)
        T *d_P_inv_k = getOffsetBlockRowPadded<T, BatchSize>(d_P_inv_batch, solve_idx, knot_idx + 1) + STATE_SIZE;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
            uint32_t x = i % STATE_SIZE;
            uint32_t y = i / STATE_SIZE;
            d_P_inv_k[y * BLOCK_ROW_R_DIM + x] = -s_theta_k_inv[x * STATE_SIZE + y];
        }

    } else { // last knot deals with Q_0 computations

        T *d_Q_0 = getOffsetStateSq<T, BatchSize>(d_Q_batch, solve_idx, 0);
        T *d_q_0 = getOffsetState<T, BatchSize>(d_q_batch, solve_idx, 0);
        T *d_c_0 = getOffsetState<T, BatchSize>(d_c_batch, solve_idx, 0);
        block::copy<T, STATE_SIZE_SQ>(s_Q_k, d_Q_0);
        block::copy<T, STATE_SIZE>(s_q_k, d_q_0);
        block::copy<T, STATE_SIZE>(s_gamma_k, d_c_0);
        block::loadIdentity<T, STATE_SIZE>(s_Q_k_inv);
        __syncthreads();

        T rho_penalty = d_rho_penalty_batch[solve_idx];
        block::addScaledIdentity<T, STATE_SIZE>(s_Q_k, rho_penalty);
        __syncthreads();

        //store -Q_0 in P_inv
        T *d_P_inv_0 = getOffsetBlockRowPadded<T, BatchSize>(d_P_inv_batch, solve_idx, 0) + STATE_SIZE;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
            uint32_t x = i % STATE_SIZE;
            uint32_t y = i / STATE_SIZE;
            d_P_inv_0[y * BLOCK_ROW_R_DIM + x] = -s_Q_k[x * STATE_SIZE + y];
        }
        __syncthreads();

        block::invertMatrix<T>(STATE_SIZE, s_Q_k, s_scratch);
        __syncthreads();

        // save Q_0_inv to S (S is row-major)
        T *d_S_0 = getOffsetBlockRowPadded<T, BatchSize>(d_S_batch, solve_idx, 0) + STATE_SIZE;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
            uint32_t x = i % STATE_SIZE;
            uint32_t y = i / STATE_SIZE;
            d_S_0[y * BLOCK_ROW_R_DIM + x] = -s_Q_k_inv[x * STATE_SIZE + y];
        }

        // gamma_0 = - Q_0_inv * q_0 (c_0 is already in s_gamma_0)
        block::matMulSum<T, STATE_SIZE, STATE_SIZE, 1>(s_gamma_k, s_Q_k_inv, s_q_k, true);
        __syncthreads();

        // save gamma_0
        T *d_gamma_k = getOffsetStatePadded<T, BatchSize>(d_gamma_batch, solve_idx, 0);
        block::copy<T, STATE_SIZE>(d_gamma_k, s_gamma_k);
    }
}


template <typename T, uint32_t BatchSize>
__global__
void formSchurSystemBatchedKernel2(
    T *d_S_batch,
    T *d_P_inv_batch
) {
    // launched with grid of (KNOT_POINTS - 1, solve_idx)
    uint32_t knot_idx = blockIdx.x;
    uint32_t solve_idx = blockIdx.y;

    extern __shared__ T s_mem[];

    T *s_theta_k_inv = s_mem;
    T *s_theta_km1_inv = s_theta_k_inv + STATE_SIZE_SQ;
    T *s_phi_k = s_theta_km1_inv + STATE_SIZE_SQ;
    T *s_scratch = s_phi_k + STATE_SIZE_SQ;

    // load theta_k_inv, theta_km1_inv from P_inv, phi_k from S
    T *d_P_inv_k_main = getOffsetBlockRowPadded<T, BatchSize>(d_P_inv_batch, solve_idx, knot_idx + 1) + STATE_SIZE;
    T *d_P_inv_km1_main = getOffsetBlockRowPadded<T, BatchSize>(d_P_inv_batch, solve_idx, knot_idx) + STATE_SIZE;
    T *d_S_k_left = getOffsetBlockRowPadded<T, BatchSize>(d_S_batch, solve_idx, knot_idx + 1);
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
        uint32_t x = i % STATE_SIZE;
        uint32_t y = i / STATE_SIZE;
        uint32_t matrix_offset = x * STATE_SIZE + y;
        uint32_t block_matrix_offset = y * BLOCK_ROW_R_DIM + x;
        s_theta_k_inv[matrix_offset] = d_P_inv_k_main[block_matrix_offset];
        s_theta_km1_inv[matrix_offset] = d_P_inv_km1_main[block_matrix_offset];
        s_phi_k[matrix_offset] = d_S_k_left[block_matrix_offset];
    }
    __syncthreads();

    // left diag = - theta_k_inv * phi_k * theta_km1_inv
    block::matMul<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(s_scratch, s_phi_k, s_theta_km1_inv);
    __syncthreads();
    block::matMul<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(s_theta_km1_inv, s_theta_k_inv, s_scratch);
    __syncthreads();

    // Save left and right diagonals into P_inv (row-major)
    T *d_P_inv_k_right = d_P_inv_km1_main + STATE_SIZE;
    T *d_P_inv_k_left = d_P_inv_k_main - STATE_SIZE;
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
        uint32_t x = i % STATE_SIZE;
        uint32_t y = i / STATE_SIZE; 
        uint32_t block_matrix_offset = y * BLOCK_ROW_R_DIM + x;
        d_P_inv_k_right[block_matrix_offset] = -s_theta_km1_inv[i]; // right_diag = left_diag^T
        d_P_inv_k_left[block_matrix_offset] = -s_theta_km1_inv[x * STATE_SIZE + y]; // left_diag
    }
}

template <typename T>
__host__
size_t getFormSchurSystemBatched1SMemSize() {
    size_t size = sizeof(T) * (
        STATE_SIZE_SQ + // Q_k
        STATE_SIZE_SQ + // Q_k_inv
        STATE_SIZE_SQ + // Q_kp1
        STATE_SIZE_SQ + // Q_kp1_inv
        CONTROL_SIZE_SQ + // R_k
        CONTROL_SIZE_SQ + // R_k_inv
        STATE_SIZE + // q_k
        STATE_SIZE + // q_kp1
        CONTROL_SIZE + // r_k
        STATE_SIZE_SQ + // A_k
        STATE_P_CONTROL + // B_k
        STATE_SIZE_SQ + // A_Q_inv
        STATE_P_CONTROL + // B_R_inv
        STATE_SIZE_SQ + // theta_k
        STATE_SIZE_SQ + // theta_k_inv
        STATE_SIZE + // gamma_k
        (2 * (2 * STATE_SIZE + 1)) + (2 * CONTROL_SIZE + 1) // max scratch needed for invertMatrix
    ); // total = 8*STATE_SIZE_SQ + 2*CONTROL_SIZE_SQ + 2*STATE_P_CONTROL + 7*STATE_SIZE + 3*CONTROL_SIZE + 3

    return size;
}

template <typename T>
__host__
size_t getFormSchurSystemBatched2SMemSize() {
    size_t size = sizeof(T) * (4 * STATE_SIZE_SQ);
    return size;
}

template <typename T, uint32_t BatchSize>
__host__
void formSchurSystemBatched(
    SchurSystem<T, BatchSize> schur,
    KKTSystem<T, BatchSize> kkt,
    T *d_rho_penalty_batch
) {
    dim3 grid1(KNOT_POINTS, BatchSize);
    dim3 grid2(KNOT_POINTS - 1, BatchSize);
    dim3 thread_block(SCHUR_THREADS);
    const uint32_t s_mem_size1 = getFormSchurSystemBatched1SMemSize<T>();
    const uint32_t s_mem_size2 = getFormSchurSystemBatched2SMemSize<T>();

    formSchurSystemBatchedKernel1<T, BatchSize><<<grid1, thread_block, s_mem_size1>>>(
        schur.d_S_batch,
        schur.d_P_inv_batch,
        schur.d_gamma_batch,
        kkt.d_Q_batch,
        kkt.d_R_batch,
        kkt.d_q_batch,
        kkt.d_r_batch,
        kkt.d_A_batch,
        kkt.d_B_batch,
        kkt.d_c_batch,
        d_rho_penalty_batch
    );

    formSchurSystemBatchedKernel2<T, BatchSize><<<grid2, thread_block, s_mem_size2>>>(
        schur.d_S_batch,
        schur.d_P_inv_batch
    );
}

// --------------------------------------------------

// dz = G_inv * (g - C^T * lambda)
// dz_state_k = Q_k_inv * (q_k - (A_k^T * lambda_kp1 + lambda_k))
// dz_control_k = R_k_inv * (r_k - (B_k^T * lambda_kp1))
template <typename T, uint32_t BatchSize>
__global__
void computeDzBatchedKernel(
    T *d_dz_batch,
    T *d_lambda_batch,
    T *d_Q_inv_batch,
    T *d_R_inv_batch,
    T *d_q_batch,
    T *d_r_batch,
    T *d_A_batch,
    T *d_B_batch
) {
    // launched with grid of size (KNOT_POINTS, batch_size, 2)
    const uint32_t knot_idx = blockIdx.x;
    const uint32_t solve_idx = blockIdx.y;

    extern __shared__ T s_mem[];

    if (blockIdx.z == 0){ // state row (Q_inv_k, A_k, q_k)

        T *s_Q_k_inv = s_mem;
        T *s_A_k = s_Q_k_inv + STATE_SIZE_SQ;
        T *s_scratch = s_A_k + STATE_SIZE_SQ;

        T *d_Q_k_inv = getOffsetStateSq<T, BatchSize>(d_Q_inv_batch, solve_idx, knot_idx);
        block::copy<T, STATE_SIZE_SQ>(s_Q_k_inv, d_Q_k_inv);

        // -A_k^T * lambda_kp1
        if (knot_idx < KNOT_POINTS - 1){
            // load A_k
            T *d_A_k = getOffsetStateSq<T, BatchSize>(d_A_batch, solve_idx, knot_idx);
            block::copy<T, STATE_SIZE_SQ>(s_A_k, d_A_k);
            __syncthreads();

            // A_k^T * lambda_next (x^T * A is equivalent to A^T * x)
            T *d_lambda_kp1 = getOffsetStatePadded<T, BatchSize>(d_lambda_batch, solve_idx, knot_idx + 1); 
            __syncthreads();
            
            T sum;
            #pragma unroll
            for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
                sum = static_cast<T>(0);
                for (uint32_t j = 0; j < STATE_SIZE; j++) {
                    sum += -s_A_k[i * STATE_SIZE + j] * d_lambda_kp1[j];
                }
                s_scratch[i] = sum;
            }
            
            //block::matMul<T, 1, STATE_SIZE, STATE_SIZE>(s_scratch, d_lambda_kp1, s_A_k, true);

        } else { // last knot
            // no lambda_next, set scratch to 0
            #pragma unroll
            for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
                s_scratch[i] = static_cast<T>(0.0);
            }
        }
        __syncthreads();

        // scratch += lambda_k
        T *d_lambda_k = getOffsetStatePadded<T, BatchSize>(d_lambda_batch, solve_idx, knot_idx);
        block::vecSum<T, STATE_SIZE>(s_scratch, d_lambda_k);
        __syncthreads();

        // q_k - (lambda_k - A_k^T * lambda_kp1)
        T *d_q_k = getOffsetState<T, BatchSize>(d_q_batch, solve_idx, knot_idx);
        block::vecSub<T, STATE_SIZE>(s_A_k, d_q_k, s_scratch);
        __syncthreads();

        // Q_inv_k * (q_k - (lambda_k - A_k^T * lambda_kp1))
        T sum;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
            sum = static_cast<T>(0);
            for (uint32_t j = 0; j < STATE_SIZE; j++) {
                sum += s_Q_k_inv[i + STATE_SIZE * j] * s_A_k[j];
            }
            s_scratch[i] = sum;
        }
        //block::matMul<T, STATE_SIZE, STATE_SIZE, 1>(s_scratch, s_Q_k_inv, s_A_k);
        __syncthreads();

        // store to dz
        T *d_dz_k = getOffsetTraj<T, BatchSize>(d_dz_batch, solve_idx, knot_idx);
        block::copy<T, STATE_SIZE>(d_dz_k, s_scratch, static_cast<T>(-1));

    } else { // control row (R_inv_k, B_k, r_k)

        if (knot_idx == KNOT_POINTS - 1) { return; } // no computation for last knot

        T *s_R_k_inv = s_mem;
        T *s_B_k = s_R_k_inv + CONTROL_SIZE_SQ;
        T *s_scratch = s_B_k + STATE_P_CONTROL;

        T *d_R_k_inv = getOffsetControlSq<T, BatchSize>(d_R_inv_batch, solve_idx, knot_idx);
        T *d_B_k = getOffsetStatePControl<T, BatchSize>(d_B_batch, solve_idx, knot_idx);
        block::copy<T, CONTROL_SIZE_SQ>(s_R_k_inv, d_R_k_inv);
        block::copy<T, STATE_P_CONTROL>(s_B_k, d_B_k);
        __syncthreads();

        // r_k - (- B_k^T * lambda_next) (x^T * A is equivalent to A^T * x)
        T *d_lambda_kp1 = getOffsetStatePadded<T, BatchSize>(d_lambda_batch, solve_idx, knot_idx + 1); 
        
        T sum;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < CONTROL_SIZE; i += blockDim.x) {
            sum = static_cast<T>(0);
            for (uint32_t j = 0; j < STATE_SIZE; j++) {
                sum += -s_B_k[i * STATE_SIZE + j] * d_lambda_kp1[j]; //TODO: used shared mem
            }
            s_scratch[i] = sum;
        }
        //block::matMulSum<T, 1, STATE_SIZE, CONTROL_SIZE>(s_scratch, d_lambda_kp1, s_B_k);
        __syncthreads();

        T *d_r_k = getOffsetControl<T, BatchSize>(d_r_batch, solve_idx, knot_idx);
        block::vecSub<T, CONTROL_SIZE>(s_scratch, d_r_k, s_scratch);
        __syncthreads();

        // R_inv_k * scratch, store in s_B_k
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < CONTROL_SIZE; i += blockDim.x) {
            sum = static_cast<T>(0);
            for (uint32_t j = 0; j < CONTROL_SIZE; j++) {
                sum += s_R_k_inv[i + CONTROL_SIZE * j] * s_scratch[j];
            }
            s_B_k[i] = sum;
        }
        //block::matMul<T, CONTROL_SIZE, CONTROL_SIZE, 1>(s_B_k, s_R_k_inv, s_scratch);
        __syncthreads();

        // store to dz
        T *d_dz_k = getOffsetTraj<T, BatchSize>(d_dz_batch, solve_idx, knot_idx) + STATE_SIZE;
        block::copy<T, CONTROL_SIZE>(d_dz_k, s_B_k, static_cast<T>(-1));
    }
}

template <typename T>
__host__
size_t getComputeDzBatchedSMemSize() {
    size_t size = sizeof(T) * (
        STATE_SIZE_SQ + // Q_k_inv or R_k_inv
        STATE_SIZE_SQ + // A_k or B_k
        STATE_SIZE // scratch
    );

    return size;
}

template <typename T, uint32_t BatchSize>
__host__
void computeDzBatched(
    T *d_dz_batch,
    T *d_lambda_batch,
    KKTSystem<T, BatchSize> kkt
) {
    dim3 grid(KNOT_POINTS, BatchSize, 2);
    dim3 thread_block(DZ_THREADS);
    const uint32_t s_mem_size = getComputeDzBatchedSMemSize<T>();

    computeDzBatchedKernel<T, BatchSize><<<grid, thread_block, s_mem_size>>>(
        d_dz_batch,
        d_lambda_batch,
        kkt.d_Q_batch,
        kkt.d_R_batch,
        kkt.d_q_batch,
        kkt.d_r_batch,
        kkt.d_A_batch,
        kkt.d_B_batch
    );
}