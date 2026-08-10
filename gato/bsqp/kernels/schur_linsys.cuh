#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)

using namespace sqp;
using namespace gato;
using namespace gato::constants;

template<typename T>
__global__ __launch_bounds__(SCHUR_THREADS) void formSchurSystemBatchedKernel1(T* __restrict__       d_S_batch,
                                                                              T* __restrict__       d_P_inv_batch,
                                                                              T* __restrict__       d_gamma_batch,
                                                                              T* __restrict__       d_Q_batch,
                                                                              T* __restrict__       d_R_batch,
                                                                              const T* __restrict__ d_q_batch,
                                                                              const T* __restrict__ d_r_batch,
                                                                              const T* __restrict__ d_A_batch,
                                                                              const T* __restrict__ d_B_batch,
                                                                              const T* __restrict__ d_c_batch,
                                                                              const T* __restrict__ d_rho_penalty_batch,
                                                                              const int32_t* __restrict__ d_kkt_converged_batch)
{
        // launched with grid of (KNOT_POINTS, solve_idx)
        uint32_t knot_idx = blockIdx.x;
        uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch[solve_idx]) return;  // converged solve: skip

        extern __shared__ T s_mem[];

        T* s_Q_k = s_mem;
        T* s_Q_k_inv = s_Q_k + STATE_SIZE_SQ;
        T* s_Q_kp1 = s_Q_k_inv + STATE_SIZE_SQ;
        T* s_Q_kp1_inv = s_Q_kp1 + STATE_SIZE_SQ;
        T* s_R_k = s_Q_kp1_inv + STATE_SIZE_SQ;
        T* s_R_k_inv = s_R_k + CONTROL_SIZE_SQ;

        T* s_q_k = s_R_k_inv + CONTROL_SIZE_SQ;
        T* s_q_kp1 = s_q_k + STATE_SIZE;
        T* s_r_k = s_q_kp1 + STATE_SIZE;

        T* s_A_k = s_r_k + CONTROL_SIZE;
        T* s_B_k = s_A_k + STATE_SIZE_SQ;

        T* s_A_Q_inv = s_B_k + STATE_P_CONTROL;
        T* s_B_R_inv = s_A_Q_inv + STATE_SIZE_SQ;

        T* s_theta_k = s_B_R_inv + STATE_P_CONTROL;
        T* s_theta_k_inv = s_theta_k + STATE_SIZE_SQ;
        T* s_gamma_k = s_theta_k_inv + STATE_SIZE_SQ;
        T* s_scratch = s_gamma_k + STATE_SIZE;

        if (knot_idx < KNOT_POINTS - 1) {  // all except last knot

                // ----- Populate shared memory -----

                T* d_Q_k = getOffsetStateSq<T>(d_Q_batch, solve_idx, knot_idx);
                T* d_Q_kp1 = getOffsetStateSq<T>(d_Q_batch, solve_idx, knot_idx + 1);
                T* d_R_k = getOffsetControlSq<T>(d_R_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE_SQ>(d_Q_k, s_Q_k);
                glass::copy<T, STATE_SIZE_SQ>(d_Q_kp1, s_Q_kp1);
                glass::copy<T, CONTROL_SIZE_SQ>(d_R_k, s_R_k);
                glass::set_identity<T, STATE_SIZE>(s_Q_k_inv);    // augmented [A|I] right-half for glass::inv
                glass::set_identity<T, STATE_SIZE>(s_Q_kp1_inv);
                glass::set_identity<T, CONTROL_SIZE>(s_R_k_inv);


                const T* d_q_k = getOffsetState<T>(d_q_batch, solve_idx, knot_idx);
                const T* d_q_kp1 = getOffsetState<T>(d_q_batch, solve_idx, knot_idx + 1);
                const T* d_r_k = getOffsetControl<T>(d_r_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_k), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_kp1), s_q_kp1);
                glass::copy<T, CONTROL_SIZE>(const_cast<T*>(d_r_k), s_r_k);

                const T* d_A_k = getOffsetStateSq<T>(d_A_batch, solve_idx, knot_idx);
                const T* d_B_k = getOffsetStatePControl<T>(d_B_batch, solve_idx, knot_idx);
                const T* d_c_k = getOffsetState<T>(d_c_batch, solve_idx, knot_idx + 1);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_A_k), s_A_k);
                glass::copy<T, STATE_P_CONTROL>(const_cast<T*>(d_B_k), s_B_k);
                glass::copy<T, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(d_c_k), s_gamma_k);
                __syncthreads();

                // ----- Compute theta_k, phi_k, and gamma_k -----
                // theta_k = - ( (A_k * Q_k_inv * A_k^T) + (B_k * R_k_inv * B_k^T) + (Q_kp1_inv) )
                // phi_k = A_k * Q_k_inv
                // gamma_k = c - (- (A_k * Q_k_inv * q_k) - (B_k * R_k_inv * r_k) + (Q_kp1_inv * q_kp1))

                // // Q_k_inv and R_k_inv
                // // add scaled identity with rho to penalize constraint violations
                T rho_penalty = d_rho_penalty_batch[solve_idx];
                glass::add_identity_partial<T, STATE_SIZE, STATE_SIZE / 2>(s_Q_k, rho_penalty);
                glass::add_identity_partial<T, STATE_SIZE, STATE_SIZE / 2>(s_Q_kp1, rho_penalty);
                __syncthreads();

                glass::inv<T>(STATE_SIZE, STATE_SIZE, CONTROL_SIZE, STATE_SIZE, s_Q_k, s_Q_kp1, s_R_k, s_scratch);  // fused 3-matrix invert (glass::, P4.3)
                __syncthreads();

                // save Q_k_inv and R_k_inv into d_Q_batch and d_R_batch for computing dz
                glass::copy<T, STATE_SIZE_SQ>(s_Q_k_inv, d_Q_k);
                glass::copy<T, CONTROL_SIZE_SQ>(s_R_k_inv, d_R_k);
                if (knot_idx == KNOT_POINTS - 2) {  // last knot doesn't compute Q_k_inv, so use second last knot's Q_kp1_inv
                        glass::copy<T, STATE_SIZE_SQ>(s_Q_kp1_inv, d_Q_kp1);
                }

                // copy Q_kp1_inv into theta_k to save a sum operation
                glass::copy<T, STATE_SIZE_SQ>(s_Q_kp1_inv, s_theta_k);
                __syncthreads();

                // A_k * Q_k_inv (phi) and B_k * R_k_inv
                glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_A_k, s_Q_k_inv, s_A_Q_inv);
                glass::gemm<T, STATE_SIZE, CONTROL_SIZE, CONTROL_SIZE>(static_cast<T>(1), s_B_k, s_R_k_inv, s_B_R_inv);
                __syncthreads();

                // theta_k = (A_k * Q_k_inv * A_k^T) + (B_k * R_k_inv * B_k^T) + (Q_kp1_inv)
                glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE, /*TA=*/false, /*TB=*/true>(static_cast<T>(1), s_A_Q_inv, s_A_k, static_cast<T>(1), s_theta_k);
                // B_R_inv (S x C) * B_k^T: new glass::gemm handles rectangular TRANSPOSE_B natively (gemm_ex removed)
                glass::gemm<T, /*TA=*/false, /*TB=*/true, /*ROW_MAJOR_C=*/false>(STATE_SIZE, STATE_SIZE, CONTROL_SIZE, static_cast<T>(1), s_B_R_inv, s_B_k, static_cast<T>(1), s_theta_k);
                // __syncthreads();

                // gamma_k
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(1), s_Q_kp1_inv, s_q_kp1, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(-1), s_A_Q_inv, s_q_k, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                glass::gemm<T, STATE_SIZE, 1, CONTROL_SIZE>(static_cast<T>(-1), s_B_R_inv, s_r_k, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                T* d_gamma_k = getOffsetStatePadded<T>(d_gamma_batch, solve_idx, knot_idx + 1);
                glass::copy<T, STATE_SIZE>(static_cast<T>(-1), s_gamma_k, d_gamma_k);


                // ----- save theta_k, phi_k, and gamma_k in S and gamma -----

                // S_k (right diag: phi_k^T, left diag: phi_k, next main diag: theta_k)
                // S_k is stored in row-major order
                // k refers to knot, not block row
                T* d_S_k_right = getOffsetBlockRowPadded<T>(d_S_batch, solve_idx, knot_idx) + 2 * STATE_SIZE;
                T* d_S_k_left = getOffsetBlockRowPadded<T>(d_S_batch, solve_idx, knot_idx + 1);
                T* d_S_kp1_main = d_S_k_left + STATE_SIZE;
#pragma unroll
                for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
                        uint32_t x = i % STATE_SIZE;
                        uint32_t y = i / STATE_SIZE;
                        uint32_t block_matrix_offset = y * BLOCK_ROW_R_DIM + x;
                        d_S_k_right[block_matrix_offset] = s_A_Q_inv[i];                     // phi_k^T
                        d_S_k_left[block_matrix_offset] = s_A_Q_inv[x * STATE_SIZE + y];     // phi_k
                        d_S_kp1_main[block_matrix_offset] = -s_theta_k[x * STATE_SIZE + y];  // theta_k
                }
                __syncthreads();

                // ----- Compute theta_k_inv and save in P_inv -----
                glass::set_identity<T, STATE_SIZE>(s_theta_k_inv);  // augmented [A|I] right-half for glass::inv
                glass::add_identity_partial<T, STATE_SIZE, STATE_SIZE / 2>(s_theta_k, rho_penalty);
                __syncthreads();
                glass::inv<T>(STATE_SIZE, s_theta_k, s_scratch);  // single augmented invert (glass::)
                __syncthreads();

                // main diag: theta_k_inv (offset by STATE_SIZE)
                T* d_P_inv_k = getOffsetBlockRowPadded<T>(d_P_inv_batch, solve_idx, knot_idx + 1) + STATE_SIZE;
#pragma unroll
                for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
                        uint32_t x = i % STATE_SIZE;
                        uint32_t y = i / STATE_SIZE;
                        d_P_inv_k[y * BLOCK_ROW_R_DIM + x] = -s_theta_k_inv[x * STATE_SIZE + y];
                }

        } else {  // last knot deals with Q_0 computations

                T*       d_Q_0 = getOffsetStateSq<T>(d_Q_batch, solve_idx, 0);
                const T* d_q_0 = getOffsetState<T>(d_q_batch, solve_idx, 0);
                const T* d_c_0 = getOffsetState<T>(d_c_batch, solve_idx, 0);
                glass::copy<T, STATE_SIZE_SQ>(d_Q_0, s_Q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_0), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_c_0), s_gamma_k);
                glass::set_identity<T, STATE_SIZE>(s_Q_k_inv);  // augmented [A|I] right-half for glass::inv
                __syncthreads();

                T rho_penalty = d_rho_penalty_batch[solve_idx];
                glass::add_identity_partial<T, STATE_SIZE, STATE_SIZE / 2>(s_Q_k, rho_penalty);
                __syncthreads();

                // store -Q_0 in P_inv
                T* d_P_inv_0 = getOffsetBlockRowPadded<T>(d_P_inv_batch, solve_idx, 0) + STATE_SIZE;
#pragma unroll
                for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
                        uint32_t x = i % STATE_SIZE;
                        uint32_t y = i / STATE_SIZE;
                        d_P_inv_0[y * BLOCK_ROW_R_DIM + x] = -s_Q_k[x * STATE_SIZE + y];
                }
                __syncthreads();

                glass::inv<T>(STATE_SIZE, s_Q_k, s_scratch);  // single augmented invert (glass::)
                __syncthreads();

                // save Q_0_inv to S (S is row-major)
                T* d_S_0 = getOffsetBlockRowPadded<T>(d_S_batch, solve_idx, 0) + STATE_SIZE;
#pragma unroll
                for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
                        uint32_t x = i % STATE_SIZE;
                        uint32_t y = i / STATE_SIZE;
                        d_S_0[y * BLOCK_ROW_R_DIM + x] = -s_Q_k_inv[x * STATE_SIZE + y];
                }

                // gamma_0 = - Q_0_inv * q_0 (c_0 is already in s_gamma_0)
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(-1), s_Q_k_inv, s_q_k, static_cast<T>(1), s_gamma_k);
                __syncthreads();

                // save gamma_0
                T* d_gamma_k = getOffsetStatePadded<T>(d_gamma_batch, solve_idx, 0);
                glass::copy<T, STATE_SIZE>(s_gamma_k, d_gamma_k);
        }
}

template<typename T>
__global__ __launch_bounds__(SCHUR_THREADS) void formSchurSystemBatchedKernel2(T* __restrict__ d_S_batch, T* __restrict__ d_P_inv_batch, const int32_t* __restrict__ d_kkt_converged_batch)
{
        // launched with grid of (KNOT_POINTS - 1, solve_idx)
        uint32_t knot_idx = blockIdx.x;
        uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch[solve_idx]) return;  // converged solve: skip

        extern __shared__ T s_mem[];

        T* s_theta_k_inv = s_mem;
        T* s_theta_km1_inv = s_theta_k_inv + STATE_SIZE_SQ;
        T* s_phi_k = s_theta_km1_inv + STATE_SIZE_SQ;
        T* s_scratch = s_phi_k + STATE_SIZE_SQ;

        // load theta_k_inv, theta_km1_inv from P_inv, phi_k from S
        T* d_P_inv_k_main = getOffsetBlockRowPadded<T>(d_P_inv_batch, solve_idx, knot_idx + 1) + STATE_SIZE;
        T* d_P_inv_km1_main = getOffsetBlockRowPadded<T>(d_P_inv_batch, solve_idx, knot_idx) + STATE_SIZE;
        T* d_S_k_left = getOffsetBlockRowPadded<T>(d_S_batch, solve_idx, knot_idx + 1);
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
        glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_phi_k, s_theta_km1_inv, s_scratch);
        __syncthreads();
        glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_theta_k_inv, s_scratch, s_theta_km1_inv);
        __syncthreads();

        // Save left and right diagonals into P_inv (row-major)
        T* d_P_inv_k_right = d_P_inv_km1_main + STATE_SIZE;
        T* d_P_inv_k_left = d_P_inv_k_main - STATE_SIZE;
#pragma unroll
        for (uint32_t i = threadIdx.x; i < STATE_SIZE_SQ; i += blockDim.x) {
                uint32_t x = i % STATE_SIZE;
                uint32_t y = i / STATE_SIZE;
                uint32_t block_matrix_offset = y * BLOCK_ROW_R_DIM + x;
                d_P_inv_k_right[block_matrix_offset] = -s_theta_km1_inv[i];                  // right_diag = left_diag^T
                d_P_inv_k_left[block_matrix_offset] = -s_theta_km1_inv[x * STATE_SIZE + y];  // left_diag
        }
}

template<typename T>
__host__ size_t getFormSchurSystemBatched1SMemSize()
{
        size_t size = sizeof(T)
                      * (STATE_SIZE_SQ +                                      // Q_k
                         STATE_SIZE_SQ +                                      // Q_k_inv
                         STATE_SIZE_SQ +                                      // Q_kp1
                         STATE_SIZE_SQ +                                      // Q_kp1_inv
                         CONTROL_SIZE_SQ +                                    // R_k
                         CONTROL_SIZE_SQ +                                    // R_k_inv
                         STATE_SIZE +                                         // q_k
                         STATE_SIZE +                                         // q_kp1
                         CONTROL_SIZE +                                       // r_k
                         STATE_SIZE_SQ +                                      // A_k
                         STATE_P_CONTROL +                                    // B_k
                         STATE_SIZE_SQ +                                      // A_Q_inv
                         STATE_P_CONTROL +                                    // B_R_inv
                         STATE_SIZE_SQ +                                      // theta_k
                         STATE_SIZE_SQ +                                      // theta_k_inv
                         STATE_SIZE +                                         // gamma_k
                         (2 * (2 * STATE_SIZE + 1)) + (2 * CONTROL_SIZE + 1)  // max scratch needed for inv
                      );                                                      // total = 8*STATE_SIZE_SQ + 2*CONTROL_SIZE_SQ + 2*STATE_P_CONTROL + 7*STATE_SIZE + 3*CONTROL_SIZE + 3

        return size;
}

template<typename T>
__host__ size_t getFormSchurSystemBatched2SMemSize()
{
        size_t size = sizeof(T) * (4 * STATE_SIZE_SQ);
        return size;
}

template<typename T>
__host__ void formSchurSystemBatched(uint32_t batch_size, SchurSystem<T> schur, KKTSystem<T> kkt, T* d_rho_penalty_batch, const int32_t* d_kkt_converged_batch)
{
        dim3           grid1(KNOT_POINTS, batch_size);
        dim3           grid2(KNOT_POINTS - 1, batch_size);
        dim3           thread_block(SCHUR_THREADS);
        const uint32_t s_mem_size1 = getFormSchurSystemBatched1SMemSize<T>();
        const uint32_t s_mem_size2 = getFormSchurSystemBatched2SMemSize<T>();

        formSchurSystemBatchedKernel1<T><<<grid1, thread_block, s_mem_size1>>>(
            schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch, kkt.d_A_batch, kkt.d_B_batch, kkt.d_c_batch, d_rho_penalty_batch, d_kkt_converged_batch);

        formSchurSystemBatchedKernel2<T><<<grid2, thread_block, s_mem_size2>>>(schur.d_S_batch, schur.d_P_inv_batch, d_kkt_converged_batch);
}

// --------------------------------------------------
// gamma-only recompute (constraint-layer arc CL-1): the ADMM inner loop
// re-solves the SAME factored Schur system with a new RHS each iteration —
// only q/r change (dual/projection terms), so S, Pinv, and the stored
// Q^-1/R^-1 blocks are all still valid. This kernel rebuilds ONLY gamma from
// the inverses formSchurSystemBatchedKernel1 left in d_Q_batch/d_R_batch,
// mirroring its gamma op sequence exactly (same glass calls, same order) so
// the parity gate is BITWISE (test/cuda/gamma_parity.cu). knot 0's gamma uses
// the STORED (Q_0 + rho)^-1 — formSchur's last-knot block re-inverts fresh via
// a different (single vs fused) glass::inv, so gamma_0 parity is near-ulp,
// not bitwise (the harness gates it at 1e-6 rel).

template<typename T>
__global__ __launch_bounds__(SCHUR_THREADS) void computeGammaBatchedKernel(T* __restrict__       d_gamma_batch,
                                                                           const T* __restrict__ d_Q_inv_batch,
                                                                           const T* __restrict__ d_R_inv_batch,
                                                                           const T* __restrict__ d_q_batch,
                                                                           const T* __restrict__ d_r_batch,
                                                                           const T* __restrict__ d_A_batch,
                                                                           const T* __restrict__ d_B_batch,
                                                                           const T* __restrict__ d_c_batch,
                                                                           const int32_t* __restrict__ d_kkt_converged_batch)
{
        // launched with grid of (KNOT_POINTS, batch_size)
        uint32_t knot_idx = blockIdx.x;
        uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch && d_kkt_converged_batch[solve_idx]) return;

        extern __shared__ T s_mem[];
        T* s_Q_k_inv = s_mem;
        T* s_Q_kp1_inv = s_Q_k_inv + STATE_SIZE_SQ;
        T* s_R_k_inv = s_Q_kp1_inv + STATE_SIZE_SQ;
        T* s_q_k = s_R_k_inv + CONTROL_SIZE_SQ;
        T* s_q_kp1 = s_q_k + STATE_SIZE;
        T* s_r_k = s_q_kp1 + STATE_SIZE;
        T* s_A_k = s_r_k + CONTROL_SIZE;
        T* s_B_k = s_A_k + STATE_SIZE_SQ;
        T* s_A_Q_inv = s_B_k + STATE_P_CONTROL;
        T* s_B_R_inv = s_A_Q_inv + STATE_SIZE_SQ;
        T* s_gamma_k = s_B_R_inv + STATE_P_CONTROL;

        if (knot_idx < KNOT_POINTS - 1) {
                const T* d_Q_k_inv = getOffsetStateSq<T>(d_Q_inv_batch, solve_idx, knot_idx);
                const T* d_Q_kp1_inv = getOffsetStateSq<T>(d_Q_inv_batch, solve_idx, knot_idx + 1);
                const T* d_R_k_inv = getOffsetControlSq<T>(d_R_inv_batch, solve_idx, knot_idx);
                const T* d_q_k = getOffsetState<T>(d_q_batch, solve_idx, knot_idx);
                const T* d_q_kp1 = getOffsetState<T>(d_q_batch, solve_idx, knot_idx + 1);
                const T* d_r_k = getOffsetControl<T>(d_r_batch, solve_idx, knot_idx);
                const T* d_A_k = getOffsetStateSq<T>(d_A_batch, solve_idx, knot_idx);
                const T* d_B_k = getOffsetStatePControl<T>(d_B_batch, solve_idx, knot_idx);
                const T* d_c_k = getOffsetState<T>(d_c_batch, solve_idx, knot_idx + 1);

                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_Q_k_inv), s_Q_k_inv);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_Q_kp1_inv), s_Q_kp1_inv);
                glass::copy<T, CONTROL_SIZE_SQ>(const_cast<T*>(d_R_k_inv), s_R_k_inv);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_k), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_kp1), s_q_kp1);
                glass::copy<T, CONTROL_SIZE>(const_cast<T*>(d_r_k), s_r_k);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_A_k), s_A_k);
                glass::copy<T, STATE_P_CONTROL>(const_cast<T*>(d_B_k), s_B_k);
                glass::copy<T, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(d_c_k), s_gamma_k);
                __syncthreads();

                // same op sequence as formSchur: A*Qinv and B*Rinv as gemms, then the
                // three gamma gemvs in the same order (bitwise parity for knots > 0)
                glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_A_k, s_Q_k_inv, s_A_Q_inv);
                glass::gemm<T, STATE_SIZE, CONTROL_SIZE, CONTROL_SIZE>(static_cast<T>(1), s_B_k, s_R_k_inv, s_B_R_inv);
                __syncthreads();
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(1), s_Q_kp1_inv, s_q_kp1, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(-1), s_A_Q_inv, s_q_k, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                glass::gemm<T, STATE_SIZE, 1, CONTROL_SIZE>(static_cast<T>(-1), s_B_R_inv, s_r_k, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                T* d_gamma_k = getOffsetStatePadded<T>(d_gamma_batch, solve_idx, knot_idx + 1);
                glass::copy<T, STATE_SIZE>(static_cast<T>(-1), s_gamma_k, d_gamma_k);

        } else {  // gamma_0 = c_0 - Q_0^-1 q_0, with the STORED inverse
                const T* d_Q_0_inv = getOffsetStateSq<T>(d_Q_inv_batch, solve_idx, 0);
                const T* d_q_0 = getOffsetState<T>(d_q_batch, solve_idx, 0);
                const T* d_c_0 = getOffsetState<T>(d_c_batch, solve_idx, 0);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_Q_0_inv), s_Q_k_inv);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_0), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_c_0), s_gamma_k);
                __syncthreads();

                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(-1), s_Q_k_inv, s_q_k, static_cast<T>(1), s_gamma_k);
                __syncthreads();
                T* d_gamma_k = getOffsetStatePadded<T>(d_gamma_batch, solve_idx, 0);
                glass::copy<T, STATE_SIZE>(s_gamma_k, d_gamma_k);
        }
}

template<typename T>
__host__ size_t getComputeGammaBatchedSMemSize()
{
        return sizeof(T)
               * (2 * STATE_SIZE_SQ + CONTROL_SIZE_SQ            // Q_k_inv, Q_kp1_inv, R_k_inv
                  + 2 * STATE_SIZE + CONTROL_SIZE                // q_k, q_kp1, r_k
                  + STATE_SIZE_SQ + STATE_P_CONTROL              // A_k, B_k
                  + STATE_SIZE_SQ + STATE_P_CONTROL              // A_Q_inv, B_R_inv
                  + STATE_SIZE);                                 // gamma
}

template<typename T>
__host__ void computeGammaBatched(uint32_t batch_size, SchurSystem<T> schur, KKTSystem<T> kkt, const int32_t* d_kkt_converged_batch)
{
        dim3 grid(KNOT_POINTS, batch_size);
        dim3 thread_block(SCHUR_THREADS);
        computeGammaBatchedKernel<T><<<grid, thread_block, getComputeGammaBatchedSMemSize<T>()>>>(
            schur.d_gamma_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch, kkt.d_A_batch, kkt.d_B_batch, kkt.d_c_batch, d_kkt_converged_batch);
}

// --------------------------------------------------

// dz = G_inv * (g - C^T * lambda)
// dz_state_k = Q_k_inv * (q_k - (A_k^T * lambda_kp1 + lambda_k))
// dz_control_k = R_k_inv * (r_k - (B_k^T * lambda_kp1))
template<typename T>
__global__ __launch_bounds__(DZ_THREADS) void computeDzBatchedKernel(T* __restrict__       d_dz_batch,
                                                                    const T* __restrict__ d_lambda_batch,
                                                                    const T* __restrict__ d_Q_inv_batch,
                                                                    const T* __restrict__ d_R_inv_batch,
                                                                    T* __restrict__       d_q_batch,
                                                                    T* __restrict__       d_r_batch,
                                                                    const T* __restrict__ d_A_batch,
                                                                    const T* __restrict__ d_B_batch,
                                                                    const int32_t* __restrict__ d_kkt_converged_batch)
{
        // launched with grid of size (KNOT_POINTS, batch_size, 2)
        const uint32_t knot_idx = blockIdx.x;
        const uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch[solve_idx]) return;  // converged solve: freeze dz/residuals

        extern __shared__ T s_mem[];

        if (blockIdx.z == 0) {  // state row (Q_inv_k, A_k, q_k)

                T* s_Q_k_inv = s_mem;
                T* s_A_k = s_Q_k_inv + STATE_SIZE_SQ;
                T* s_scratch = s_A_k + STATE_SIZE_SQ;

                const T* d_Q_k_inv = getOffsetStateSq<T>(d_Q_inv_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_Q_k_inv), s_Q_k_inv);

                // -A_k^T * lambda_kp1
                if (knot_idx < KNOT_POINTS - 1) {
                        // load A_k
                        const T* d_A_k = getOffsetStateSq<T>(d_A_batch, solve_idx, knot_idx);
                        glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_A_k), s_A_k);
                        __syncthreads();

                        // A_k^T * lambda_next (x^T * A is equivalent to A^T * x)
                        const T* d_lambda_kp1 = getOffsetStatePadded<T>(d_lambda_batch, solve_idx, knot_idx + 1);
                        __syncthreads();

                        glass::gemm<T, 1, STATE_SIZE, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(d_lambda_kp1), s_A_k, s_scratch);

                } else {  // last knot
// no lambda_next, set scratch to 0
                        glass::set_const<T, STATE_SIZE>(static_cast<T>(0), s_scratch);
                }
                const T* d_lambda_k = getOffsetStatePadded<T>(d_lambda_batch, solve_idx, knot_idx);
                __syncthreads();

                // scratch += lambda_k
                glass::axpy<T, STATE_SIZE>(static_cast<T>(1), const_cast<T*>(d_lambda_k), s_scratch);
                __syncthreads();

                // q_k - (lambda_k - A_k^T * lambda_kp1)
                T* d_q_k = getOffsetState<T>(d_q_batch, solve_idx, knot_idx);
                glass::axpby<T, STATE_SIZE>(static_cast<T>(1), d_q_k, static_cast<T>(-1), s_scratch, s_A_k);
                __syncthreads();

                // Q_inv_k * (q_k - (lambda_k - A_k^T * lambda_kp1))
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(1), s_Q_k_inv, s_A_k, s_scratch);
                __syncthreads();

                // store to dz
                T* d_dz_k = getOffsetDz<T>(d_dz_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE>(static_cast<T>(-1), s_scratch, d_dz_k);
                // store KKT residual for state row: q_k - (lambda_k - A_k^T * lambda_kp1)
                glass::copy<T, STATE_SIZE>(s_A_k, d_q_k);

        } else {  // control row (R_inv_k, B_k, r_k)

                if (knot_idx == KNOT_POINTS - 1) {

                        T* d_r_k = getOffsetControl<T>(d_r_batch, solve_idx, knot_idx);
                        glass::set_const<T, CONTROL_SIZE>(static_cast<T>(0), d_r_k);

                        return;
                }  // Regular case

                T* s_R_k_inv = s_mem;
                T* s_B_k = s_R_k_inv + CONTROL_SIZE_SQ;
                T* s_scratch = s_B_k + STATE_P_CONTROL;

                const T* d_R_k_inv = getOffsetControlSq<T>(d_R_inv_batch, solve_idx, knot_idx);
                const T* d_B_k = getOffsetStatePControl<T>(d_B_batch, solve_idx, knot_idx);
                glass::copy<T, CONTROL_SIZE_SQ>(const_cast<T*>(d_R_k_inv), s_R_k_inv);
                glass::copy<T, STATE_P_CONTROL>(const_cast<T*>(d_B_k), s_B_k);
                __syncthreads();

                // r_k - (- B_k^T * lambda_next) (x^T * A is equivalent to A^T * x)
                const T* d_lambda_kp1 = getOffsetStatePadded<T>(d_lambda_batch, solve_idx, knot_idx + 1);

                // s_scratch = -(B_k^T * lambda_kp1)
                glass::gemm<T, 1, CONTROL_SIZE, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(d_lambda_kp1), s_B_k, s_scratch);
                __syncthreads();

                T* d_r_k = getOffsetControl<T>(d_r_batch, solve_idx, knot_idx);
                glass::axpby<T, CONTROL_SIZE>(static_cast<T>(1), d_r_k, static_cast<T>(-1), s_scratch, s_scratch);
                __syncthreads();

                // s_B_k = R_inv_k * s_scratch
                glass::gemm<T, CONTROL_SIZE, 1, CONTROL_SIZE>(static_cast<T>(1), s_R_k_inv, s_scratch, s_B_k);
                __syncthreads();

                // store to dz
                T* d_dz_k = getOffsetDz<T>(d_dz_batch, solve_idx, knot_idx) + STATE_SIZE;
                glass::copy<T, CONTROL_SIZE>(static_cast<T>(-1), s_B_k, d_dz_k);
                // store KKT residual for control row: r_k - ( -B_k^T * lambda_kp1 )
                glass::copy<T, CONTROL_SIZE>(s_scratch, d_r_k);
        }
}

template<typename T>
__host__ size_t getComputeDzBatchedSMemSize()
{
        size_t size = sizeof(T)
                      * (STATE_SIZE_SQ +  // Q_k_inv or R_k_inv
                         STATE_SIZE_SQ +  // A_k or B_k
                         STATE_SIZE       // scratch
                      );

        return size;
}

template<typename T>
__host__ void computeDzBatched(uint32_t batch_size, T* d_dz_batch, T* d_lambda_batch, KKTSystem<T> kkt, const int32_t* d_kkt_converged_batch)
{
        dim3           grid(KNOT_POINTS, batch_size, 2);
        dim3           thread_block(DZ_THREADS);
        const uint32_t s_mem_size = getComputeDzBatchedSMemSize<T>();

        computeDzBatchedKernel<T><<<grid, thread_block, s_mem_size>>>(d_dz_batch, d_lambda_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch, kkt.d_A_batch, kkt.d_B_batch, d_kkt_converged_batch);
}
