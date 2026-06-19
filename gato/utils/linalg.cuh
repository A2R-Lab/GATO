#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"

using namespace sqp;
using namespace gato::constants;

namespace block {  // utils for block-level operations

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t FULL_MASK = 0xffffffff;

template<typename T>
__device__ __forceinline__ void reduce(const uint32_t n, T* x)
{
        uint32_t idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
        unsigned size_left = n;
        bool     odd_flag;

        // loop until only a few values left
        while (size_left > 3) {
                // determine if odd_adjust needed and update size
                odd_flag = size_left % 2;
                size_left = (size_left - odd_flag) / 2;
                // reduce in half
                for (uint32_t i = idx; i < size_left; i += stride) { x[i] += x[i + size_left]; }
                // add the odd size adjust if needed
                if (idx == 0 && odd_flag) { x[0] += x[2 * size_left]; }
                // sync and repeat
                __syncthreads();
        }
        // when we get really small sum up what is left
        if (idx == 0) {
                for (uint32_t i = 1; i < size_left; i++) { x[0] += x[i]; }
        }
}



// print m x n matrix (column-major)
template<typename T, uint32_t m, uint32_t n>
__host__ __device__ void printMatrix(T* A)
{
        for (uint32_t y = 0; y < m; y++) {
                for (uint32_t x = 0; x < n; x++) { printf("%.4f ", A[x * m + y]); }
                printf("\n");
        }
}

// print m x n matrix (row-major)
template<typename T, uint32_t m, uint32_t n>
__host__ __device__ void printMatrixRowMajor(T* A)
{
        for (uint32_t y = 0; y < m; y++) {
                for (uint32_t x = 0; x < n; x++) { printf("%.4f ", A[y * n + x]); }
                printf("\n");
        }
}


}  // namespace block


namespace gato {

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetWrench(T* batch, uint32_t solve_idx)
{
        return batch + solve_idx * 6 * grid::NUM_BODIES;   // [K3] body-major f_ext: 6*NUM_BODIES per solve
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetWrench(const T* batch, uint32_t solve_idx)
{
        return batch + solve_idx * 6 * grid::NUM_BODIES;   // [K3] body-major f_ext: 6*NUM_BODIES per solve
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetState(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetState(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

// compute pointer to a (CONTROL_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetControl(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetControl(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

// compute pointer to a (STATE_SIZE x STATE_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetStateSq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetStateSq(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

// compute pointer to a (CONTROL_SIZE x CONTROL_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetControlSq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetControlSq(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetStatePControl(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetStatePControl(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

// compute offset for accessing a knot point of a batch of trajectories (BATCH_SIZE X ((STATE_SIZE + CONTROL_SIZE) X KNOT_POINTS - CONTROL_SIZE))
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetTraj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * STATE_S_CONTROL;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetTraj(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * STATE_S_CONTROL;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetReferenceTraj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetReferenceTraj(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X (KNOT_POINTS + 2))
// each solve batch is padded to (KNOT_POINTS + 2) * STATE_SIZE for the PCG solver
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetStatePadded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetStatePadded(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetBlockRowPadded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ const T* getOffsetBlockRowPadded(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

}  // namespace gato
