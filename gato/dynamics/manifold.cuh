#pragma once
#include <cstdint>
#include "glass.cuh"
#include "constants.h"

// ─── Floating-base state manifold ops (CL-3) ─────────────────────────────────
//
// Stored state x = [q(NQ); qd(NV)] with the free-flyer prefix q[0:7] =
// [p(3); quat xyzw(4)] (pin layout). Tangent dx = [dq(NV); dqd(NV)] with the
// base block ordered [v_lin(3); omega(3)] — LINEAR-FIRST, the pinocchio
// integrate/difference convention shared by grid_integrate_floating_q and
// glass::se3_retract/se3_difference (NOT the Featherstone [omega; v] order).
//
// These are the ⊞ / ⊟ the SQP needs: trial points and accepted steps
// (x ⊞ alpha*dz), the dynamics defect and initial-state gap (x_a ⊟ x_b).
// FIXED BASE: both collapse to plain vector ops — but fixed-base kernel call
// sites deliberately KEEP their historic glass::axpby/subtract code (bitwise
// discipline); these helpers are called on the FLOATING_BASE branch only.
// Block-cooperative and thread-count invariant: the 7-slot base pose math is
// serial on rank 0 (glass::thread tier), everything else strided. No trailing
// sync — callers own the barrier (matching the kernel style around them).

// grid_difference_floating_q (GRiD ASK 4, @61a83fb) is EMITTED ONLY in
// floating-base headers, so fixed-base TUs have no such symbol — and a
// qualified name inside `if constexpr` is still looked up at definition time.
// Forward-declare the exact emitted signature: the discarded fixed-base branch
// never instantiates it (no definition needed), and on floating TUs grid.cuh's
// definition matches — any upstream signature drift fails loud at go2 compile.
namespace grid {
template <typename T, int NUM_POS>
__device__ inline void grid_difference_floating_q(const T* q_from, const T* q_to, T* dv);
}

namespace gato::plant {

using gato::constants::FLOATING_BASE;

// x_out = x ⊞ alpha*dx (state only: NQ+NV in, 2*NV tangent in).
// s_x_out MAY alias s_x (in-place update): the base-pose math computes into a
// register tmp before writing (glass::thread::se3_retract), and the strided
// slots are read/written by the same thread.
template<typename T>
__device__ __forceinline__ void state_retract(T* s_x_out, const T* s_x, const T* s_dx, T alpha)
{
        constexpr int NQi = (int)gato::constants::NQ;
        constexpr int NVi = (int)gato::constants::NV;
        const int rank = (int)threadIdx.x;
        const int nth  = (int)blockDim.x;
        if constexpr (FLOATING_BASE) {
                if (rank == 0) {
                        T rho[3] = {alpha * s_dx[0], alpha * s_dx[1], alpha * s_dx[2]};
                        T phi[3] = {alpha * s_dx[3], alpha * s_dx[4], alpha * s_dx[5]};
                        glass::thread::se3_retract<T>(s_x, rho, phi, s_x_out);
                }
                for (int i = rank; i < NQi - 7; i += nth)
                        s_x_out[7 + i] = s_x[7 + i] + alpha * s_dx[6 + i];
                for (int i = rank; i < NVi; i += nth)
                        s_x_out[NQi + i] = s_x[NQi + i] + alpha * s_dx[NVi + i];
        } else {
                for (int i = rank; i < NQi + NVi; i += nth)
                        s_x_out[i] = s_x[i] + alpha * s_dx[i];
        }
}

// s_out(2*NV) = x_to ⊟ x_from: the tangent with x_from ⊞ s_out == x_to
// (pinocchio difference(model, q_from, q_to) on the free-flyer block). The q
// block goes through grid::grid_difference_floating_q — GRiD's emitted boxminus
// twin of the retract (ASK 4, GRiD @61a83fb; same glass se3_difference math,
// upstream-gated vs pin.difference at 1e-10 — NOT glass::pose_error, which is
// deliberately the DECOUPLED R³×SO(3) error). Also locked in test_manifold.py.
template<typename T>
__device__ __forceinline__ void state_difference(T* s_out, const T* s_x_from, const T* s_x_to)
{
        constexpr int NQi = (int)gato::constants::NQ;
        constexpr int NVi = (int)gato::constants::NV;
        const int rank = (int)threadIdx.x;
        const int nth  = (int)blockDim.x;
        if constexpr (FLOATING_BASE) {
                if (rank == 0)
                        grid::grid_difference_floating_q<T, NQi>(s_x_from, s_x_to, s_out);
                for (int i = rank; i < NVi; i += nth)
                        s_out[NVi + i] = s_x_to[NQi + i] - s_x_from[NQi + i];
        } else {
                for (int i = rank; i < NQi + NVi; i += nth)
                        s_out[i] = s_x_to[i] - s_x_from[i];
        }
}

}  // namespace gato::plant
