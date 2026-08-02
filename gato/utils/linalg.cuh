#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"

using namespace sqp;
using namespace gato::constants;

namespace gato {

template<typename T>
__device__ __forceinline__ T* getOffsetWrench(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        // body-major f_ext: 6*NUM_BODIES per knot, KNOT_POINTS knots per solve
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetWrench(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* getOffsetState(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetState(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

// compute pointer to a (CONTROL_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* getOffsetControl(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetControl(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

// compute pointer to a (STATE_SIZE x STATE_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* getOffsetStateSq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetStateSq(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

// compute pointer to a (CONTROL_SIZE x CONTROL_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* getOffsetControlSq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetControlSq(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T>
__device__ __forceinline__ T* getOffsetStatePControl(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetStatePControl(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

// compute offset for accessing a knot point of a batch of trajectories (BATCH_SIZE X ((STATE_SIZE + CONTROL_SIZE) X KNOT_POINTS - CONTROL_SIZE))
template<typename T>
__device__ __forceinline__ T* getOffsetTraj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * STATE_S_CONTROL;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetTraj(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * STATE_S_CONTROL;
}

template<typename T>
__device__ __forceinline__ T* getOffsetReferenceTraj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetReferenceTraj(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X (KNOT_POINTS + 2))
// each solve batch is padded to (KNOT_POINTS + 2) * STATE_SIZE for the PCG solver
template<typename T>
__device__ __forceinline__ T* getOffsetStatePadded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetStatePadded(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

template<typename T>
__device__ __forceinline__ T* getOffsetBlockRowPadded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetBlockRowPadded(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

}  // namespace gato
