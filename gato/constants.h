#pragma once

#include <cstdint>
#include "settings.h"

using namespace sqp;

namespace gato {
namespace constants {
    // EE pose size (xyz + orientation). The generated grid.cuh no longer exposes grid::EE_POS_SIZE
    // (it now uses NUM_EES); the pose dimension is 6 per end-effector.
    constexpr uint32_t EE_POS_SIZE = 6;
    constexpr uint32_t REFERENCE_TRAJ_SIZE = EE_POS_SIZE * KNOT_POINTS;
    // Configuration vs tangent dimensions (CL-3 floating base). Fixed base:
    // NUM_POS == NUM_VEL == NUM_JOINTS and every pair below collapses to the
    // historic value, so fixed-base modules are preprocessor-identical.
    // Floating base (quaternion free-flyer): NUM_POS = NUM_VEL + 1.
    constexpr uint32_t NQ = grid::NUM_POS;  // configuration size (q, with quaternion)
    constexpr uint32_t NV = grid::NUM_VEL;  // tangent / velocity size
    constexpr bool FLOATING_BASE = (NQ != NV);
    static_assert(!FLOATING_BASE || NQ == NV + 1,
                  "floating base must be a single quaternion free-flyer (nq = nv + 1)");
    // STATE_SIZE is the TANGENT state size (2*nv): the KKT/Schur/PCG blocks,
    // A|B, Q/q, dz, gamma all live in tangent space. The stored trajectory
    // state is XU_STATE_SIZE (nq + nv) — identical on fixed base.
    constexpr uint32_t STATE_SIZE = NV * 2;
    constexpr uint32_t XU_STATE_SIZE = NQ + NV;
    // Actuation: fixed base = every joint; floating base = everything but the
    // 6 base dofs (grid's plant surfaces take a full NV-sized generalized
    // force; GATO scatters the actuated slots and zeros the base rows).
    constexpr uint32_t ACTUATED_SIZE = FLOATING_BASE ? (NV - 6) : grid::NUM_JOINTS;
#if GATO_CONTACT_FORCES
    #ifndef GRID_HAS_CONTACT_FRAMES
        #error "GATO_CONTACT_FORCES=1 needs a grid.cuh generated with contact_frames (2b.1+)"
    #endif
    // per-knot world-aligned wrench [n; f] per contact frame, appended to u
    constexpr uint32_t FC_SIZE = 6 * grid::NUM_CONTACT_FRAMES;
#else
    constexpr uint32_t FC_SIZE = 0;
#endif
    constexpr uint32_t CONTROL_SIZE = ACTUATED_SIZE + FC_SIZE; // torques [+ contact wrench]
    constexpr uint32_t STATE_SIZE_SQ = STATE_SIZE * STATE_SIZE;
    constexpr uint32_t CONTROL_SIZE_SQ = CONTROL_SIZE * CONTROL_SIZE;
    constexpr uint32_t STATE_P_CONTROL = STATE_SIZE * CONTROL_SIZE;
    constexpr uint32_t STATE_S_CONTROL = STATE_SIZE + CONTROL_SIZE;
    constexpr uint32_t STATE_SQ_P_KNOTS = STATE_SIZE * STATE_SIZE * KNOT_POINTS; // Q, A
    constexpr uint32_t CONTROL_SQ_P_KNOTS = CONTROL_SIZE * CONTROL_SIZE * KNOT_POINTS; // R
    constexpr uint32_t STATE_P_KNOTS = STATE_SIZE * KNOT_POINTS; // q, c
    constexpr uint32_t CONTROL_P_KNOTS = CONTROL_SIZE * KNOT_POINTS; // r
    constexpr uint32_t STATE_P_CONTROL_P_KNOTS = STATE_SIZE * CONTROL_SIZE * KNOT_POINTS; // B
    // Per-knot strides: the stored trajectory xu packs [q(NQ); qd(NV); u(NU)],
    // the KKT step dz packs tangent [dq(NV); dqd(NV); du(NU)]. Fixed base:
    // XU_KNOT_STRIDE == DZ_KNOT_STRIDE and XU_TRAJ_SIZE == TRAJ_SIZE.
    constexpr uint32_t XU_KNOT_STRIDE = XU_STATE_SIZE + CONTROL_SIZE;
    constexpr uint32_t DZ_KNOT_STRIDE = STATE_SIZE + CONTROL_SIZE;
    constexpr uint32_t TRAJ_SIZE = (STATE_SIZE + CONTROL_SIZE) * KNOT_POINTS - CONTROL_SIZE; // dz (tangent)
    constexpr uint32_t XU_TRAJ_SIZE = XU_KNOT_STRIDE * KNOT_POINTS - CONTROL_SIZE; // xu (stored)
    constexpr uint32_t VEC_SIZE_PADDED = (KNOT_POINTS + 2) * STATE_SIZE; // gamma
    constexpr uint32_t BLOCK_ROW_R_DIM = 3 * STATE_SIZE;
    constexpr uint32_t BLOCK_ROW_SIZE = 3 * STATE_SIZE * STATE_SIZE;
    constexpr uint32_t B3D_MATRIX_SIZE_PADDED = 3 * STATE_SIZE * STATE_SIZE * KNOT_POINTS; // S, P_inv
} // namespace constants
} // namespace gato