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
    constexpr uint32_t STATE_SIZE = grid::NUM_JOINTS * 2; // positions, velocities
    constexpr uint32_t ACTUATED_SIZE = grid::NUM_JOINTS; // torques (the applied control)
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
    constexpr uint32_t TRAJ_SIZE = (STATE_SIZE + CONTROL_SIZE) * KNOT_POINTS - CONTROL_SIZE; // xu, dz
    constexpr uint32_t VEC_SIZE_PADDED = (KNOT_POINTS + 2) * STATE_SIZE; // gamma
    constexpr uint32_t BLOCK_ROW_R_DIM = 3 * STATE_SIZE;
    constexpr uint32_t BLOCK_ROW_SIZE = 3 * STATE_SIZE * STATE_SIZE;
    constexpr uint32_t B3D_MATRIX_SIZE_PADDED = 3 * STATE_SIZE * STATE_SIZE * KNOT_POINTS; // S, P_inv
} // namespace constants
} // namespace gato