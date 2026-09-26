#pragma once
// Batch-layout accessors: pointer arithmetic into the per-solve / per-knot device
// buffers (KKT blocks, trajectories, padded PCG vectors, [L|D|R] strips). GATO's
// only hand-written "linalg"; every numeric kernel is glass::.
//
// Each accessor exists as a const and a non-const overload with the same offset
// expression (callers name the element type explicitly, get_offset_x<T>(...), so
// the pointer constness cannot be deduced from a single template) — the two
// overloads are generated from one offset expression below.

#include <cstdint>
#include "settings.h"
#include "constants.h"

using namespace sqp;
using namespace gato::constants;

namespace gato {

// (solve_idx, knot_idx) accessors: OFFSET is an expression in solve_idx / knot_idx
#define GATO_BATCH_ACCESSOR_2(NAME, OFFSET)                                                                          \
        template<typename T>                                                                                          \
        __device__ __forceinline__ const T* NAME(const T* batch, uint32_t solve_idx, uint32_t knot_idx)               \
        {                                                                                                             \
                return batch + (OFFSET);                                                                              \
        }                                                                                                             \
        template<typename T>                                                                                          \
        __device__ __forceinline__ T* NAME(T* batch, uint32_t solve_idx, uint32_t knot_idx)                           \
        {                                                                                                             \
                return batch + (OFFSET);                                                                              \
        }

// per-solve accessors: OFFSET is an expression in solve_idx
#define GATO_BATCH_ACCESSOR_1(NAME, OFFSET)                                                                          \
        template<typename T>                                                                                          \
        __device__ __forceinline__ const T* NAME(const T* batch, uint32_t solve_idx)                                  \
        {                                                                                                             \
                return batch + (OFFSET);                                                                              \
        }                                                                                                             \
        template<typename T>                                                                                          \
        __device__ __forceinline__ T* NAME(T* batch, uint32_t solve_idx)                                              \
        {                                                                                                             \
                return batch + (OFFSET);                                                                              \
        }

// body-major f_ext: 6*NUM_BODIES per knot, KNOT_POINTS knots per solve. Null
// propagates: the host passes nullptr when the whole band is zero (f_ext_ptr()),
// and the generated dynamics null-check every d_f_ext use — the skip is the
// zero-wrench fast path.
template<typename T>
__device__ __forceinline__ const T* get_offset_wrench(const T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        if (batch == nullptr) return nullptr;
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}
template<typename T>
__device__ __forceinline__ T* get_offset_wrench(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        if (batch == nullptr) return nullptr;
        return batch + (solve_idx * KNOT_POINTS + knot_idx) * 6 * grid::NUM_BODIES;
}

// (STATE_SIZE) vector / (STATE_SIZE x STATE_SIZE) / (CONTROL_SIZE) / (CONTROL_SIZE^2) /
// (STATE_SIZE x CONTROL_SIZE) blocks of a (BATCH_SIZE x KNOT_POINTS) batch
GATO_BATCH_ACCESSOR_2(get_offset_state, solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE)
GATO_BATCH_ACCESSOR_2(get_offset_control, solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE)
GATO_BATCH_ACCESSOR_2(get_offset_state_sq, solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ)
GATO_BATCH_ACCESSOR_2(get_offset_control_sq, solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ)
GATO_BATCH_ACCESSOR_2(get_offset_state_p_control, solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL)
// stored trajectory xu (XU_KNOT_STRIDE per knot) and the tangent step dz (DZ_KNOT_STRIDE)
GATO_BATCH_ACCESSOR_2(get_offset_xu, solve_idx * XU_TRAJ_SIZE + knot_idx * XU_KNOT_STRIDE)
GATO_BATCH_ACCESSOR_2(get_offset_dz, solve_idx * TRAJ_SIZE + knot_idx * DZ_KNOT_STRIDE)
// EE reference (EE_POS_SIZE per knot)
GATO_BATCH_ACCESSOR_2(get_offset_reference_traj, solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * EE_POS_SIZE)
// padded PCG vectors: one STATE_SIZE pad block precedes knot 0
GATO_BATCH_ACCESSOR_2(get_offset_state_padded, solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE)
GATO_BATCH_ACCESSOR_1(get_padded_vector, solve_idx * VEC_SIZE_PADDED)
// [L|D|R] block rows of the padded Schur strips
GATO_BATCH_ACCESSOR_2(get_offset_block_row_padded, solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE)

#undef GATO_BATCH_ACCESSOR_2
#undef GATO_BATCH_ACCESSOR_1

}  // namespace gato
