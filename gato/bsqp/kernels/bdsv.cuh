#pragma once

#include <cstdint>
#include <algorithm>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)

using namespace sqp;
using namespace gato;
using namespace gato::constants;

// Direct linear solve for S·λ = γ via glass::bdsv (block-Cholesky), the exact
// alternative to solvePCGBatchedKernel on the SAME buffers. GATO stores the
// NEGATED Schur complement (−S is SPD), so this kernel solves (−S)λ = (−γ):
// it negates the [L|D|R] strips in place (the factor destroys S anyway — the
// next SQP iteration's formSchur rewrites every touched slot) and seeds λ with
// −γ before the in-place bdsv_solve.
//
// Semantics match the PCG kernel where the SQP loop depends on them:
//   - d_kkt_converged skip → iterations = 0.
//   - converged-start guard: r = γ − S·λ_warm (on the UN-negated S), z = Pinv·r;
//     |rᵀz| < abs_tol → iterations = 0, λ untouched (feeds the `pcg_iters == 0`
//     convergence signal in bsqp.cuh).
//   - successful direct solve → iterations = 1.
//   - non-PD pivot → iterations = 2, λ UNTOUCHED (see below).
// Optional per-problem mask (stage-2 hybrid): a block whose mask entry is 0
// returns immediately WITHOUT writing iterations (the PCG launch owns that
// problem). Pass nullptr for "all problems take the direct path".
//
// The Cholesky ALWAYS runs with the non-PD pivot CHECK, and λ is only
// overwritten AFTER a successful factor. −S is SPD analytically, but in f32
// it can fail numerically when the cost is barely regularized (measured:
// indy7 at interface-default rho=0, u_cost=1e-6 → R⁻¹ blocks ~1e6 → the
// factor's cancellations push pivots negative; PCG tolerates the same system).
// On failure the update is skipped for this problem — λ keeps its warm start
// (computeDz then reproduces the previous iterate's step; the line search
// re-evaluates it and rho adaptation improves the next linearization) — and
// iterations = 2 makes the skip visible in stats (≠ 0, so never mistaken for
// convergence). NaN must never leave this kernel: the vendored grid::glass in
// grid.cuh predates the GLASS beta==0 write-only fix, so NaN left in shared
// memory poisons later cost/dynamics kernels on the same SM (0*NaN reads).
template<typename T>
__global__ __launch_bounds__(BDSV_THREADS) void solveBDSVBatchedKernel(uint32_t* __restrict__      d_iterations,
                                                                       T* __restrict__             d_x_batch,
                                                                       T* __restrict__             d_A_batch,
                                                                       const T* __restrict__       d_M_inv_batch,
                                                                       const T* __restrict__       d_b_batch,
                                                                       const int32_t* __restrict__ d_kkt_converged_batch,
                                                                       const int32_t* __restrict__ d_use_bdsv_mask)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        constexpr uint32_t VEC = VEC_SIZE_PADDED;

        const T abs_tol = 1e-6;  // matches the PCG kernel's converged-start tolerance

        if (d_use_bdsv_mask && !d_use_bdsv_mask[solve_idx]) { return; }

        // skip solve if rho_max_reached (mirrors the PCG kernel)
        if (d_kkt_converged_batch[solve_idx]) {
                if (rank == 0) { d_iterations[solve_idx] = 0; }
                return;
        }

        // s_mem: [s_r | s_z | warp-dot scratch] for the converged-start guard; after the
        // guard r/z are dead and the front of s_mem is reused as bdsv's 2·SS² staging
        // scratch (getSolveBDSVBatchedSMemSize takes the max of the two footprints).
        extern __shared__ T s_mem[];
        T* s_r   = s_mem;
        T* s_z   = s_r + VEC;
        T* s_scr = s_z + VEC;
        __shared__ T s_rho;

        T*       d_S_matrix = getOffsetBlockRowPadded<T>(d_A_batch, solve_idx, 0);
        const T* d_M_inv_matrix = getOffsetBlockRowPadded<T>(d_M_inv_batch, solve_idx, 0);
        // getOffsetStatePadded points to the start of data; back up one block to the padding start.
        const T* d_b_vector = getOffsetStatePadded<T>(d_b_batch, solve_idx, 0) - STATE_SIZE;
        T*       d_x_vector = getOffsetStatePadded<T>(d_x_batch, solve_idx, 0) - STATE_SIZE;

        // ---- converged-start guard (must run BEFORE the destructive negate/factor) ----
        glass::set_const<T, 2 * VEC>(static_cast<T>(0), s_r);  // zero r,z incl. pads
        __syncthreads();
        glass::bdmv<T, KNOT_POINTS, STATE_SIZE>(s_r, d_S_matrix, d_x_vector);  // r = S·λ_warm
        __syncthreads();
        glass::axpby<T, VEC>(static_cast<T>(1), const_cast<T*>(d_b_vector), static_cast<T>(-1), s_r, s_r);  // r = γ − S·λ
        __syncthreads();
        glass::bdmv<T, KNOT_POINTS, STATE_SIZE>(s_z, d_M_inv_matrix, s_r);  // z = Pinv·r
        __syncthreads();
        glass::dot_fast<T, VEC>(s_r, s_z, &s_rho, s_scr);  // rho = rᵀz
        __syncthreads();

        T arho = (s_rho < static_cast<T>(0)) ? -s_rho : s_rho;
        if (arho < abs_tol) {
                if (rank == 0) { d_iterations[solve_idx] = 0; }
                return;
        }

        // ---- negate the strips in place: (−S) is SPD (stored S is the negated Schur) ----
        for (uint32_t i = rank; i < BLOCK_ROW_SIZE * KNOT_POINTS; i += size) { d_S_matrix[i] = -d_S_matrix[i]; }
        __syncthreads();

        // ---- factor FIRST (checked), touch λ only on success ----
        // r/z are dead after the guard → reuse s_mem as bdsv's 2·SS² staging scratch.
        __shared__ int s_fail;
        if (rank == 0) { s_fail = 0; }
        __syncthreads();
        glass::bdsv_factor<T, KNOT_POINTS, STATE_SIZE, /*CHECK=*/true>(d_S_matrix, s_mem, &s_fail);
        __syncthreads();
        if (s_fail) {  // skip update: λ untouched; strips are rewritten next iteration
                // scrub the factor's smem staging — potrf ran to completion and left NaNs
                // there, and shared memory persists on the SM (see the header note on the
                // vendored-glass beta==0 hazard)
                glass::set_const<T>(2 * STATE_SIZE * STATE_SIZE, static_cast<T>(0), s_mem);
                __syncthreads();
                if (rank == 0) { d_iterations[solve_idx] = 2; }
                return;
        }

        for (uint32_t i = rank; i < VEC; i += size) { d_x_vector[i] = -d_b_vector[i]; }  // λ ← −γ (pads stay 0)
        __syncthreads();
        glass::bdsv_solve<T, KNOT_POINTS, STATE_SIZE>(d_S_matrix, d_x_vector, s_mem);
        __syncthreads();

        if (rank == 0) { d_iterations[solve_idx] = 1; }
}

template<typename T>
__host__ size_t getSolveBDSVBatchedSMemSize()
{
        // guard phase: 2 padded vectors + warp-dot scratch; factor/solve phase: 2·SS² staging
        const size_t guard_bytes = (2 * (size_t)VEC_SIZE_PADDED + (BDSV_THREADS + 31) / 32) * sizeof(T);
        return std::max(guard_bytes, glass::bdsv_scratch_bytes<T, STATE_SIZE>());
}

template<typename T>
__host__ void solveBDSVBatched(uint32_t batch_size, T* d_lambda_batch, SchurSystem<T> schur, int32_t* d_kkt_converged_batch, uint32_t* d_iterations, const int32_t* d_use_bdsv_mask = nullptr)
{
        dim3           grid(batch_size);
        dim3           thread_block(BDSV_THREADS);
        const uint32_t s_mem_size = getSolveBDSVBatchedSMemSize<T>();

        solveBDSVBatchedKernel<T>
            <<<grid, thread_block, s_mem_size>>>(d_iterations, d_lambda_batch, schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch, d_kkt_converged_batch, d_use_bdsv_mask);
}
