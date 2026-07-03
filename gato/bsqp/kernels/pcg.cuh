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
__global__ __launch_bounds__(PCG_THREADS) void solvePCGBatchedKernel(uint32_t* __restrict__       d_iterations,
                                                                    T* __restrict__              d_x_batch,
                                                                    const T* __restrict__        d_A_batch,
                                                                    const T* __restrict__        d_M_inv_batch,
                                                                    const T* __restrict__        d_b_batch,
                                                                    const T* __restrict__        d_epsilon_batch,
                                                                    uint32_t                     max_pcg_iters,
                                                                    int32_t* __restrict__        d_kkt_converged_batch)
{
        const uint32_t solve_idx = blockIdx.x;
        const T        epsilon = d_epsilon_batch[solve_idx];

        const T abs_tol = 1e-6;

        // skip solve if rho_max_reached
        if (d_kkt_converged_batch[solve_idx]) {
                if (threadIdx.x == 0) { d_iterations[solve_idx] = 0; }
                return;
        }

        // glass::pcg manages its own shared layout within s_mem (5 padded vectors +
        // warp-dot scratch + 5 static scalars). getSolvePCGBatchedSMemSize() stays >=
        // glass::pcg_scratch_bytes (now returns bytes; GATO sizes its own smem, so unaffected).
        extern __shared__ T s_mem[];

        // get S, P_inv, b, x pointers for this batch element (padded vectors).
        const T* d_A_matrix = getOffsetBlockRowPadded<T>(d_A_batch, solve_idx, 0);
        const T* d_M_inv_matrix = getOffsetBlockRowPadded<T>(d_M_inv_batch, solve_idx, 0);
        // getOffsetStatePadded points to the start of data; back up one block to the padding start.
        const T* d_b_vector = getOffsetStatePadded<T>(d_b_batch, solve_idx, 0) - STATE_SIZE;
        T* d_x_vector = getOffsetStatePadded<T>(d_x_batch, solve_idx, 0) - STATE_SIZE;

        // Block-wide preconditioned CG (GLASS). S (=d_A) / P_inv (=d_M_inv) are the same
        // [L|D|R] row-major block-tridiagonal strips that glass::bdmv consumes internally;
        // x/b are the same (KNOT_POINTS+2)*STATE_SIZE padded vectors. glass::pcg seeds from x,
        // iterates, and writes the solution back to x. Convergence on the preconditioned residual
        // |rho| < abs_tol + rel_tol*|rho_init| matches the old loop. Read-only S/P_inv/b are
        // const-cast (glass::pcg does not write them).
        glass::pcg<T, STATE_SIZE, KNOT_POINTS>(
            d_x_vector, const_cast<T*>(d_A_matrix), const_cast<T*>(d_M_inv_matrix), const_cast<T*>(d_b_vector),
            s_mem, max_pcg_iters, /*rel_tol=*/epsilon, abs_tol, &d_iterations[solve_idx]);
}

template<typename T>
__host__ size_t getSolvePCGBatchedSMemSize()
{
        size_t size = sizeof(T) * (5 * VEC_SIZE_PADDED + 32 + 5 + PCG_THREADS);
        return size;
}

template<typename T>
__host__ void solvePCGBatched(uint32_t batch_size, T* d_lambda_batch, SchurSystem<T> schur, T* d_epsilon_batch, uint32_t max_pcg_iters, int32_t* d_kkt_converged_batch, uint32_t* d_iterations)
{
        dim3           grid(batch_size);
        dim3           thread_block(PCG_THREADS);
        const uint32_t s_mem_size = getSolvePCGBatchedSMemSize<T>();

        solvePCGBatchedKernel<T>
            <<<grid, thread_block, s_mem_size>>>(d_iterations, d_lambda_batch, schur.d_S_batch, schur.d_P_inv_batch, schur.d_gamma_batch, d_epsilon_batch, max_pcg_iters, d_kkt_converged_batch);
}
