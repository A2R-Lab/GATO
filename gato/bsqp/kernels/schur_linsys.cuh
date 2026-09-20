#pragma once
// Schur-complement assembly for the batched SQP step (one block per (knot, solve)):
//   kernel 1  invert Q_k / Q_{k+1} / R_k (augmented [A|I], glass::inv), form
//             theta_k / phi_k / gamma_k, scatter them into the block-tridiagonal
//             [L|D|R] strips S and P_inv (glass::store_block) — S is stored NEGATED;
//   kernel 2  finish the block-Jacobi preconditioner's off-diagonals;
//   gamma     rebuild gamma alone (ADMM inner loop: S is constant, q/r move);
//   dz        recover the primal step from the multipliers lambda.
// Every kernel's shared-memory layout is ONE table (the *Smem structs below) that
// the host sizer and the device carve both read — they cannot drift.

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)

using namespace sqp;
using namespace gato;
using namespace gato::constants;

// ---- shared-memory layouts (element offsets; host sizer = kernel carve) ----
template<typename T>
struct Schur1Smem {
        static constexpr uint32_t Q_k = 0;
        static constexpr uint32_t Q_k_inv = Q_k + STATE_SIZE_SQ;
        static constexpr uint32_t Q_kp1 = Q_k_inv + STATE_SIZE_SQ;
        static constexpr uint32_t Q_kp1_inv = Q_kp1 + STATE_SIZE_SQ;
        static constexpr uint32_t R_k = Q_kp1_inv + STATE_SIZE_SQ;
        static constexpr uint32_t R_k_inv = R_k + CONTROL_SIZE_SQ;
        static constexpr uint32_t q_k = R_k_inv + CONTROL_SIZE_SQ;
        static constexpr uint32_t q_kp1 = q_k + STATE_SIZE;
        static constexpr uint32_t r_k = q_kp1 + STATE_SIZE;
        static constexpr uint32_t A_k = r_k + CONTROL_SIZE;
        static constexpr uint32_t B_k = A_k + STATE_SIZE_SQ;
        static constexpr uint32_t A_Q_inv = B_k + STATE_P_CONTROL;
        static constexpr uint32_t B_R_inv = A_Q_inv + STATE_SIZE_SQ;
        static constexpr uint32_t theta_k = B_R_inv + STATE_P_CONTROL;
        static constexpr uint32_t theta_k_inv = theta_k + STATE_SIZE_SQ;
        static constexpr uint32_t gamma_k = theta_k_inv + STATE_SIZE_SQ;
        static constexpr uint32_t scratch = gamma_k + STATE_SIZE;
        static constexpr uint32_t scratch_ct = (2 * (2 * STATE_SIZE + 1)) + (2 * CONTROL_SIZE + 1);  // max glass::inv scratch
        static constexpr uint32_t total = scratch + scratch_ct;
        static constexpr size_t   bytes() { return total * sizeof(T); }
};

template<typename T>
struct Schur2Smem {
        static constexpr uint32_t theta_k_inv = 0;
        static constexpr uint32_t theta_km1_inv = theta_k_inv + STATE_SIZE_SQ;
        static constexpr uint32_t phi_k = theta_km1_inv + STATE_SIZE_SQ;
        static constexpr uint32_t scratch = phi_k + STATE_SIZE_SQ;
        static constexpr uint32_t total = scratch + STATE_SIZE_SQ;
        static constexpr size_t   bytes() { return total * sizeof(T); }
};

template<typename T>
struct GammaSmem {
        static constexpr uint32_t Q_k_inv = 0;
        static constexpr uint32_t Q_kp1_inv = Q_k_inv + STATE_SIZE_SQ;
        static constexpr uint32_t R_k_inv = Q_kp1_inv + STATE_SIZE_SQ;
        static constexpr uint32_t q_k = R_k_inv + CONTROL_SIZE_SQ;
        static constexpr uint32_t q_kp1 = q_k + STATE_SIZE;
        static constexpr uint32_t r_k = q_kp1 + STATE_SIZE;
        static constexpr uint32_t A_k = r_k + CONTROL_SIZE;
        static constexpr uint32_t B_k = A_k + STATE_SIZE_SQ;
        static constexpr uint32_t A_Q_inv = B_k + STATE_P_CONTROL;
        static constexpr uint32_t B_R_inv = A_Q_inv + STATE_SIZE_SQ;
        static constexpr uint32_t gamma_k = B_R_inv + STATE_P_CONTROL;
        static constexpr uint32_t total = gamma_k + STATE_SIZE;
        static constexpr size_t   bytes() { return total * sizeof(T); }
};

// dz has two block roles (state rows / control rows) sharing one launch size
template<typename T>
struct DzSmem {
        static constexpr uint32_t x_Q_k_inv = 0;
        static constexpr uint32_t x_A_k = x_Q_k_inv + STATE_SIZE_SQ;
        static constexpr uint32_t x_scratch = x_A_k + STATE_SIZE_SQ;
        static constexpr uint32_t x_total = x_scratch + STATE_SIZE;
        static constexpr uint32_t u_R_k_inv = 0;
        static constexpr uint32_t u_B_k = u_R_k_inv + CONTROL_SIZE_SQ;
        static constexpr uint32_t u_scratch = u_B_k + STATE_P_CONTROL;
        static constexpr uint32_t u_total = u_scratch + CONTROL_SIZE;
        static constexpr uint32_t total = x_total > u_total ? x_total : u_total;
        static constexpr size_t   bytes() { return total * sizeof(T); }
};

// gamma_k for knot k >= 0 (k < K-1), the SAME op order in both producers
// (formSchur kernel 1 and the ADMM-loop computeGamma) so they stay bitwise:
//   gamma_k = -( Q_kp1_inv q_kp1 - (A Q_inv) q_k - (B R_inv) r_k + c_k )
// s_gamma_k must hold -c_k on entry; result stored (negated) into the padded
// gamma vector. All threads; ends on a barrier.
template<typename T>
__device__ __forceinline__ void schur_gamma_k(const T* s_Q_kp1_inv, const T* s_q_kp1, const T* s_A_Q_inv, const T* s_q_k,
                                              const T* s_B_R_inv, const T* s_r_k, T* s_gamma_k, T* d_gamma_k)
{
        glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(1), const_cast<T*>(s_Q_kp1_inv), const_cast<T*>(s_q_kp1), static_cast<T>(1), s_gamma_k);
        __syncthreads();
        glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(s_A_Q_inv), const_cast<T*>(s_q_k), static_cast<T>(1), s_gamma_k);
        __syncthreads();
        glass::gemm<T, STATE_SIZE, 1, CONTROL_SIZE>(static_cast<T>(-1), const_cast<T*>(s_B_R_inv), const_cast<T*>(s_r_k), static_cast<T>(1), s_gamma_k);
        __syncthreads();
        glass::copy<T, STATE_SIZE>(static_cast<T>(-1), s_gamma_k, d_gamma_k);
}

// gamma_0 = c_0 - Q_0_inv q_0 (s_gamma_k holds c_0 on entry). Ends on a barrier.
template<typename T>
__device__ __forceinline__ void schur_gamma_0(const T* s_Q_0_inv, const T* s_q_0, T* s_gamma_k, T* d_gamma_0)
{
        glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(s_Q_0_inv), const_cast<T*>(s_q_0), static_cast<T>(1), s_gamma_k);
        __syncthreads();
        glass::copy<T, STATE_SIZE>(s_gamma_k, d_gamma_0);
}

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
        using L = Schur1Smem<T>;
        T* s_Q_k = s_mem + L::Q_k;
        T* s_Q_k_inv = s_mem + L::Q_k_inv;
        T* s_Q_kp1 = s_mem + L::Q_kp1;
        T* s_Q_kp1_inv = s_mem + L::Q_kp1_inv;
        T* s_R_k = s_mem + L::R_k;
        T* s_R_k_inv = s_mem + L::R_k_inv;
        T* s_q_k = s_mem + L::q_k;
        T* s_q_kp1 = s_mem + L::q_kp1;
        T* s_r_k = s_mem + L::r_k;
        T* s_A_k = s_mem + L::A_k;
        T* s_B_k = s_mem + L::B_k;
        T* s_A_Q_inv = s_mem + L::A_Q_inv;
        T* s_B_R_inv = s_mem + L::B_R_inv;
        T* s_theta_k = s_mem + L::theta_k;
        T* s_theta_k_inv = s_mem + L::theta_k_inv;
        T* s_gamma_k = s_mem + L::gamma_k;
        T* s_scratch = s_mem + L::scratch;

        if (knot_idx < KNOT_POINTS - 1) {  // all except last knot

                // ----- Populate shared memory -----

                T* d_Q_k = get_offset_state_sq<T>(d_Q_batch, solve_idx, knot_idx);
                T* d_Q_kp1 = get_offset_state_sq<T>(d_Q_batch, solve_idx, knot_idx + 1);
                T* d_R_k = get_offset_control_sq<T>(d_R_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE_SQ>(d_Q_k, s_Q_k);
                glass::copy<T, STATE_SIZE_SQ>(d_Q_kp1, s_Q_kp1);
                glass::copy<T, CONTROL_SIZE_SQ>(d_R_k, s_R_k);
                glass::set_identity<T, STATE_SIZE>(s_Q_k_inv);    // augmented [A|I] right-half for glass::inv
                glass::set_identity<T, STATE_SIZE>(s_Q_kp1_inv);
                glass::set_identity<T, CONTROL_SIZE>(s_R_k_inv);


                const T* d_q_k = get_offset_state<T>(d_q_batch, solve_idx, knot_idx);
                const T* d_q_kp1 = get_offset_state<T>(d_q_batch, solve_idx, knot_idx + 1);
                const T* d_r_k = get_offset_control<T>(d_r_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_k), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_kp1), s_q_kp1);
                glass::copy<T, CONTROL_SIZE>(const_cast<T*>(d_r_k), s_r_k);

                const T* d_A_k = get_offset_state_sq<T>(d_A_batch, solve_idx, knot_idx);
                const T* d_B_k = get_offset_state_p_control<T>(d_B_batch, solve_idx, knot_idx);
                const T* d_c_k = get_offset_state<T>(d_c_batch, solve_idx, knot_idx + 1);
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
                // no barrier between the two beta=1 accumulations: glass::gemm assigns
                // each C element to one owner thread, so the second gemm reads back
                // only its own writes (racecheck-clean on the full solve, 2026-09-20)

                // gamma_k (shared helper: same op order as computeGamma -> bitwise)
                schur_gamma_k<T>(s_Q_kp1_inv, s_q_kp1, s_A_Q_inv, s_q_k, s_B_R_inv, s_r_k, s_gamma_k,
                                 get_offset_state_padded<T>(d_gamma_batch, solve_idx, knot_idx + 1));


                // ----- save theta_k, phi_k, and gamma_k in S and gamma -----

                // S_k (right diag: phi_k^T, left diag: phi_k, next main diag: theta_k)
                // S_k is stored in row-major order
                // k refers to knot, not block row
                // GLASS owns the [L|D|R] strip layout: dense block -> slot movers
                // (block_access.cuh), transpose/negate folded in. Same element
                // mapping as the old hand loops (bitwise), one writer per element.
                T* d_S_row_k = get_offset_block_row_padded<T>(d_S_batch, solve_idx, knot_idx);
                T* d_S_row_kp1 = get_offset_block_row_padded<T>(d_S_batch, solve_idx, knot_idx + 1);
                glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/false, /*SYNC=*/false>(d_S_row_k, glass::BandSlot::RIGHT, s_A_Q_inv);                              // phi_k^T
                glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(d_S_row_kp1, glass::BandSlot::LEFT, s_A_Q_inv);                              // phi_k
                glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(d_S_row_kp1, glass::BandSlot::MAIN, s_theta_k, static_cast<T>(-1));        // -theta_k
                __syncthreads();

                // ----- Compute theta_k_inv and save in P_inv -----
                glass::set_identity<T, STATE_SIZE>(s_theta_k_inv);  // augmented [A|I] right-half for glass::inv
                glass::add_identity_partial<T, STATE_SIZE, STATE_SIZE / 2>(s_theta_k, rho_penalty);
                __syncthreads();
                glass::inv<T>(STATE_SIZE, s_theta_k, s_scratch);  // single augmented invert (glass::)
                __syncthreads();

                // main diag: theta_k_inv (offset by STATE_SIZE)
                glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(
                    get_offset_block_row_padded<T>(d_P_inv_batch, solve_idx, knot_idx + 1), glass::BandSlot::MAIN, s_theta_k_inv, static_cast<T>(-1));

        } else {  // last knot deals with Q_0 computations

                T*       d_Q_0 = get_offset_state_sq<T>(d_Q_batch, solve_idx, 0);
                const T* d_q_0 = get_offset_state<T>(d_q_batch, solve_idx, 0);
                const T* d_c_0 = get_offset_state<T>(d_c_batch, solve_idx, 0);
                glass::copy<T, STATE_SIZE_SQ>(d_Q_0, s_Q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_0), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_c_0), s_gamma_k);
                glass::set_identity<T, STATE_SIZE>(s_Q_k_inv);  // augmented [A|I] right-half for glass::inv
                __syncthreads();

                T rho_penalty = d_rho_penalty_batch[solve_idx];
                glass::add_identity_partial<T, STATE_SIZE, STATE_SIZE / 2>(s_Q_k, rho_penalty);
                __syncthreads();

                // store -Q_0 in P_inv
                glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true>(
                    get_offset_block_row_padded<T>(d_P_inv_batch, solve_idx, 0), glass::BandSlot::MAIN, s_Q_k, static_cast<T>(-1));

                glass::inv<T>(STATE_SIZE, s_Q_k, s_scratch);  // single augmented invert (glass::)
                __syncthreads();

                // save Q_0_inv to S (S is row-major)
                glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(
                    get_offset_block_row_padded<T>(d_S_batch, solve_idx, 0), glass::BandSlot::MAIN, s_Q_k_inv, static_cast<T>(-1));

                // gamma_0 = c_0 - Q_0_inv q_0 (c_0 is already in s_gamma_k)
                schur_gamma_0<T>(s_Q_k_inv, s_q_k, s_gamma_k, get_offset_state_padded<T>(d_gamma_batch, solve_idx, 0));
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
        using L = Schur2Smem<T>;
        T* s_theta_k_inv = s_mem + L::theta_k_inv;
        T* s_theta_km1_inv = s_mem + L::theta_km1_inv;
        T* s_phi_k = s_mem + L::phi_k;
        T* s_scratch = s_mem + L::scratch;

        // load theta_k_inv, theta_km1_inv from P_inv, phi_k from S
        T* d_P_inv_row_kp1 = get_offset_block_row_padded<T>(d_P_inv_batch, solve_idx, knot_idx + 1);
        T* d_P_inv_row_k = get_offset_block_row_padded<T>(d_P_inv_batch, solve_idx, knot_idx);
        glass::load_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(s_theta_k_inv, d_P_inv_row_kp1, glass::BandSlot::MAIN);
        glass::load_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(s_theta_km1_inv, d_P_inv_row_k, glass::BandSlot::MAIN);
        glass::load_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(s_phi_k, get_offset_block_row_padded<T>(d_S_batch, solve_idx, knot_idx + 1), glass::BandSlot::LEFT);
        __syncthreads();

        // left diag = - theta_k_inv * phi_k * theta_km1_inv
        glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_phi_k, s_theta_km1_inv, s_scratch);
        __syncthreads();
        glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_theta_k_inv, s_scratch, s_theta_km1_inv);
        __syncthreads();

        // Save left and right diagonals into P_inv (row-major)
        glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/false, /*SYNC=*/false>(d_P_inv_row_k, glass::BandSlot::RIGHT, s_theta_km1_inv, static_cast<T>(-1));   // right_diag = left_diag^T
        glass::store_block<T, STATE_SIZE, BLOCK_ROW_R_DIM, /*TRANSPOSE=*/true, /*SYNC=*/false>(d_P_inv_row_kp1, glass::BandSlot::LEFT, s_theta_km1_inv, static_cast<T>(-1));   // left_diag
}

template<typename T>
__host__ size_t get_form_schur_system_batched1_smem_size()
{
        return Schur1Smem<T>::bytes();
}

template<typename T>
__host__ size_t get_form_schur_system_batched2_smem_size()
{
        return Schur2Smem<T>::bytes();
}

template<typename T>
__host__ void form_schur_system_batched(uint32_t batch_size, SchurSystem<T> schur, KKTSystem<T> kkt, T* d_rho_penalty_batch, const int32_t* d_kkt_converged_batch)
{
        dim3           grid1(KNOT_POINTS, batch_size);
        dim3           grid2(KNOT_POINTS - 1, batch_size);
        dim3           thread_block(SCHUR_THREADS);
        const uint32_t s_mem_size1 = get_form_schur_system_batched1_smem_size<T>();
        const uint32_t s_mem_size2 = get_form_schur_system_batched2_smem_size<T>();

        formSchurSystemBatchedKernel1<T><<<grid1, thread_block, s_mem_size1>>>(
            schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch, kkt.d_A_batch, kkt.d_B_batch, kkt.d_c_batch, d_rho_penalty_batch, d_kkt_converged_batch);
        gpuErrchk(cudaGetLastError());  // launch-config failures must not pass silently

        formSchurSystemBatchedKernel2<T><<<grid2, thread_block, s_mem_size2>>>(schur.d_S_batch, schur.d_P_inv_batch, d_kkt_converged_batch);
        gpuErrchk(cudaGetLastError());  // launch-config failures must not pass silently
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
        using L = GammaSmem<T>;
        T* s_Q_k_inv = s_mem + L::Q_k_inv;
        T* s_Q_kp1_inv = s_mem + L::Q_kp1_inv;
        T* s_R_k_inv = s_mem + L::R_k_inv;
        T* s_q_k = s_mem + L::q_k;
        T* s_q_kp1 = s_mem + L::q_kp1;
        T* s_r_k = s_mem + L::r_k;
        T* s_A_k = s_mem + L::A_k;
        T* s_B_k = s_mem + L::B_k;
        T* s_A_Q_inv = s_mem + L::A_Q_inv;
        T* s_B_R_inv = s_mem + L::B_R_inv;
        T* s_gamma_k = s_mem + L::gamma_k;

        if (knot_idx < KNOT_POINTS - 1) {
                const T* d_Q_k_inv = get_offset_state_sq<T>(d_Q_inv_batch, solve_idx, knot_idx);
                const T* d_Q_kp1_inv = get_offset_state_sq<T>(d_Q_inv_batch, solve_idx, knot_idx + 1);
                const T* d_R_k_inv = get_offset_control_sq<T>(d_R_inv_batch, solve_idx, knot_idx);
                const T* d_q_k = get_offset_state<T>(d_q_batch, solve_idx, knot_idx);
                const T* d_q_kp1 = get_offset_state<T>(d_q_batch, solve_idx, knot_idx + 1);
                const T* d_r_k = get_offset_control<T>(d_r_batch, solve_idx, knot_idx);
                const T* d_A_k = get_offset_state_sq<T>(d_A_batch, solve_idx, knot_idx);
                const T* d_B_k = get_offset_state_p_control<T>(d_B_batch, solve_idx, knot_idx);
                const T* d_c_k = get_offset_state<T>(d_c_batch, solve_idx, knot_idx + 1);

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
                // shared gamma helper (bitwise parity with formSchur's gamma)
                glass::gemm<T, STATE_SIZE, STATE_SIZE, STATE_SIZE>(static_cast<T>(1), s_A_k, s_Q_k_inv, s_A_Q_inv);
                glass::gemm<T, STATE_SIZE, CONTROL_SIZE, CONTROL_SIZE>(static_cast<T>(1), s_B_k, s_R_k_inv, s_B_R_inv);
                __syncthreads();
                schur_gamma_k<T>(s_Q_kp1_inv, s_q_kp1, s_A_Q_inv, s_q_k, s_B_R_inv, s_r_k, s_gamma_k,
                                 get_offset_state_padded<T>(d_gamma_batch, solve_idx, knot_idx + 1));

        } else {  // gamma_0 = c_0 - Q_0^-1 q_0, with the STORED inverse
                const T* d_Q_0_inv = get_offset_state_sq<T>(d_Q_inv_batch, solve_idx, 0);
                const T* d_q_0 = get_offset_state<T>(d_q_batch, solve_idx, 0);
                const T* d_c_0 = get_offset_state<T>(d_c_batch, solve_idx, 0);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_Q_0_inv), s_Q_k_inv);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_q_0), s_q_k);
                glass::copy<T, STATE_SIZE>(const_cast<T*>(d_c_0), s_gamma_k);
                __syncthreads();

                schur_gamma_0<T>(s_Q_k_inv, s_q_k, s_gamma_k, get_offset_state_padded<T>(d_gamma_batch, solve_idx, 0));
        }
}

template<typename T>
__host__ size_t get_compute_gamma_batched_smem_size()
{
        return GammaSmem<T>::bytes();
}

template<typename T>
__host__ void compute_gamma_batched(uint32_t batch_size, SchurSystem<T> schur, KKTSystem<T> kkt, const int32_t* d_kkt_converged_batch)
{
        dim3 grid(KNOT_POINTS, batch_size);
        dim3 thread_block(SCHUR_THREADS);
        computeGammaBatchedKernel<T><<<grid, thread_block, get_compute_gamma_batched_smem_size<T>()>>>(
            schur.d_gamma_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch, kkt.d_A_batch, kkt.d_B_batch, kkt.d_c_batch, d_kkt_converged_batch);
        gpuErrchk(cudaGetLastError());  // launch-config failures must not pass silently
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

                T* s_Q_k_inv = s_mem + DzSmem<T>::x_Q_k_inv;
                T* s_A_k = s_mem + DzSmem<T>::x_A_k;
                T* s_scratch = s_mem + DzSmem<T>::x_scratch;

                const T* d_Q_k_inv = get_offset_state_sq<T>(d_Q_inv_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_Q_k_inv), s_Q_k_inv);

                // -A_k^T * lambda_kp1
                if (knot_idx < KNOT_POINTS - 1) {
                        // load A_k
                        const T* d_A_k = get_offset_state_sq<T>(d_A_batch, solve_idx, knot_idx);
                        glass::copy<T, STATE_SIZE_SQ>(const_cast<T*>(d_A_k), s_A_k);
                        __syncthreads();

                        // A_k^T * lambda_next (x^T * A is equivalent to A^T * x)
                        const T* d_lambda_kp1 = get_offset_state_padded<T>(d_lambda_batch, solve_idx, knot_idx + 1);
                        __syncthreads();

                        glass::gemm<T, 1, STATE_SIZE, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(d_lambda_kp1), s_A_k, s_scratch);

                } else {  // last knot
// no lambda_next, set scratch to 0
                        glass::set_const<T, STATE_SIZE>(static_cast<T>(0), s_scratch);
                }
                const T* d_lambda_k = get_offset_state_padded<T>(d_lambda_batch, solve_idx, knot_idx);
                __syncthreads();

                // scratch += lambda_k
                glass::axpy<T, STATE_SIZE>(static_cast<T>(1), const_cast<T*>(d_lambda_k), s_scratch);
                __syncthreads();

                // q_k - (lambda_k - A_k^T * lambda_kp1)
                T* d_q_k = get_offset_state<T>(d_q_batch, solve_idx, knot_idx);
                glass::axpby<T, STATE_SIZE>(static_cast<T>(1), d_q_k, static_cast<T>(-1), s_scratch, s_A_k);
                __syncthreads();

                // Q_inv_k * (q_k - (lambda_k - A_k^T * lambda_kp1))
                glass::gemm<T, STATE_SIZE, 1, STATE_SIZE>(static_cast<T>(1), s_Q_k_inv, s_A_k, s_scratch);
                __syncthreads();

                // store to dz
                T* d_dz_k = get_offset_dz<T>(d_dz_batch, solve_idx, knot_idx);
                glass::copy<T, STATE_SIZE>(static_cast<T>(-1), s_scratch, d_dz_k);

        } else {  // control row (R_inv_k, B_k, r_k)

                if (knot_idx == KNOT_POINTS - 1) { return; }  // no control at the terminal knot

                T* s_R_k_inv = s_mem + DzSmem<T>::u_R_k_inv;
                T* s_B_k = s_mem + DzSmem<T>::u_B_k;
                T* s_scratch = s_mem + DzSmem<T>::u_scratch;

                const T* d_R_k_inv = get_offset_control_sq<T>(d_R_inv_batch, solve_idx, knot_idx);
                const T* d_B_k = get_offset_state_p_control<T>(d_B_batch, solve_idx, knot_idx);
                glass::copy<T, CONTROL_SIZE_SQ>(const_cast<T*>(d_R_k_inv), s_R_k_inv);
                glass::copy<T, STATE_P_CONTROL>(const_cast<T*>(d_B_k), s_B_k);
                __syncthreads();

                // r_k - (- B_k^T * lambda_next) (x^T * A is equivalent to A^T * x)
                const T* d_lambda_kp1 = get_offset_state_padded<T>(d_lambda_batch, solve_idx, knot_idx + 1);

                // s_scratch = -(B_k^T * lambda_kp1)
                glass::gemm<T, 1, CONTROL_SIZE, STATE_SIZE>(static_cast<T>(-1), const_cast<T*>(d_lambda_kp1), s_B_k, s_scratch);
                __syncthreads();

                T* d_r_k = get_offset_control<T>(d_r_batch, solve_idx, knot_idx);
                glass::axpby<T, CONTROL_SIZE>(static_cast<T>(1), d_r_k, static_cast<T>(-1), s_scratch, s_scratch);
                __syncthreads();

                // s_B_k = R_inv_k * s_scratch
                glass::gemm<T, CONTROL_SIZE, 1, CONTROL_SIZE>(static_cast<T>(1), s_R_k_inv, s_scratch, s_B_k);
                __syncthreads();

                // store to dz
                T* d_dz_k = get_offset_dz<T>(d_dz_batch, solve_idx, knot_idx) + STATE_SIZE;
                glass::copy<T, CONTROL_SIZE>(static_cast<T>(-1), s_B_k, d_dz_k);
        }
}

template<typename T>
__host__ size_t get_compute_dz_batched_smem_size()
{
        return DzSmem<T>::bytes();
}

template<typename T>
__host__ void compute_dz_batched(uint32_t batch_size, T* d_dz_batch, T* d_lambda_batch, KKTSystem<T> kkt, const int32_t* d_kkt_converged_batch)
{
        dim3           grid(KNOT_POINTS, batch_size, 2);
        dim3           thread_block(DZ_THREADS);
        const uint32_t s_mem_size = get_compute_dz_batched_smem_size<T>();

        computeDzBatchedKernel<T><<<grid, thread_block, s_mem_size>>>(d_dz_batch, d_lambda_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch, kkt.d_A_batch, kkt.d_B_batch, d_kkt_converged_batch);
        gpuErrchk(cudaGetLastError());  // launch-config failures must not pass silently
}
