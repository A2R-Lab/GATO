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
        // body-major f_ext: 6*NUM_BODIES per knot, KNOT_POINTS knots per solve.
        // Null propagates: the host passes nullptr when the whole band is zero
        // (f_ext_ptr()), and the generated dynamics null-check every d_f_ext use —
        // the skip is the zero-wrench fast path.
        if (batch == nullptr) return nullptr;
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetWrench(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        if (batch == nullptr) return nullptr;
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

// Knot offsets into the two per-solve trajectory layouts (identical on fixed
// base, distinct on floating base — CL-3):
//   xu (stored):  [q(NQ); qd(NV); u(NU)] per knot, stride XU_KNOT_STRIDE
//   dz (tangent): [dq(NV); dqd(NV); du(NU)] per knot, stride DZ_KNOT_STRIDE
// There is deliberately NO shared accessor: every call site must pick one.
template<typename T>
__device__ __forceinline__ T* getOffsetXU(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * XU_TRAJ_SIZE + knot_idx * XU_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetXU(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * XU_TRAJ_SIZE + knot_idx * XU_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ T* getOffsetDz(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * DZ_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ const T* getOffsetDz(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * DZ_KNOT_STRIDE;
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
