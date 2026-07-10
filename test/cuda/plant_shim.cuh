#pragma once
// Minimal GATO_PLANT_HEADER stand-in for standalone kernel harnesses that need
// gato/constants.h dimensions WITHOUT pulling the generated grid.cuh (~MB of
// device code). Provides exactly what constants.h reads from the plant:
// grid::NUM_JOINTS (+ NUM_BODIES for utils/linalg.cuh's wrench offset helper).
// 7 = iiwa14/indy7 (STATE_SIZE 14, the shipped shape).
#include <cstdint>
namespace grid {
constexpr uint32_t NUM_JOINTS = 7;
constexpr uint32_t NUM_BODIES = 7;
}
