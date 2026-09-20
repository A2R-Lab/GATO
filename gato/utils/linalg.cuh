#pragma once
// Batch-layout accessors: pointer arithmetic into the per-solve / per-knot device
// buffers (KKT blocks, trajectories, padded PCG vectors, [L|D|R] strips). GATO's
// only hand-written "linalg"; every numeric kernel is glass::.

#include <cstdint>
#include "settings.h"
#include "constants.h"

using namespace sqp;
using namespace gato::constants;

namespace gato {

template<typename T>
__device__ __forceinline__ T* get_offset_wrench(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        // body-major f_ext: 6*NUM_BODIES per knot, KNOT_POINTS knots per solve.
        // Null propagates: the host passes nullptr when the whole band is zero
        // (f_ext_ptr()), and the generated dynamics null-check every d_f_ext use —
        // the skip is the zero-wrench fast path.
        if (batch == nullptr) return nullptr;
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_wrench(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        if (batch == nullptr) return nullptr;
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* get_offset_state(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_state(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

// compute pointer to a (CONTROL_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* get_offset_control(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_control(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

// compute pointer to a (STATE_SIZE x STATE_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* get_offset_state_sq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_state_sq(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

// compute pointer to a (CONTROL_SIZE x CONTROL_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T>
__device__ __forceinline__ T* get_offset_control_sq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_control_sq(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T>
__device__ __forceinline__ T* get_offset_state_p_control(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_state_p_control(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

// Knot offsets into the two per-solve trajectory layouts (identical on fixed
// base, distinct on floating base — CL-3):
//   xu (stored):  [q(NQ); qd(NV); u(NU)] per knot, stride XU_KNOT_STRIDE
//   dz (tangent): [dq(NV); dqd(NV); du(NU)] per knot, stride DZ_KNOT_STRIDE
// There is deliberately NO shared accessor: every call site must pick one.
template<typename T>
__device__ __forceinline__ T* get_offset_xu(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * XU_TRAJ_SIZE + knot_idx * XU_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_xu(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * XU_TRAJ_SIZE + knot_idx * XU_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ T* get_offset_dz(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * DZ_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_dz(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * DZ_KNOT_STRIDE;
}

template<typename T>
__device__ __forceinline__ T* get_offset_reference_traj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_reference_traj(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X (KNOT_POINTS + 2))
// each solve batch is padded to (KNOT_POINTS + 2) * STATE_SIZE for the PCG solver
template<typename T>
__device__ __forceinline__ T* get_offset_state_padded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_state_padded(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

// the whole padded (KNOT_POINTS + 2) * STATE_SIZE vector of one solve, pads included
// (what glass::pcg / bdsv consume) — replaces the "get_offset_state_padded(..., 0) - STATE_SIZE" idiom
template<typename T>
__device__ __forceinline__ T* get_padded_vector(T* batch, uint32_t solve_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED;
}

template<typename T>
__device__ __forceinline__ const T* get_padded_vector(const T* batch, uint32_t solve_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED;
}

template<typename T>
__device__ __forceinline__ T* get_offset_block_row_padded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

template<typename T>
__device__ __forceinline__ const T* get_offset_block_row_padded(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

}  // namespace gato
