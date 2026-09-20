#pragma once
// Minimal GATO_PLANT_HEADER stand-in for standalone kernel harnesses that need
// gato/constants.h dimensions WITHOUT pulling the generated grid.cuh (~MB of
// device code). Provides exactly what constants.h / linalg.cuh read from the
// plant: NUM_JOINTS, NUM_POS/NUM_VEL (fixed base: == NUM_JOINTS), NUM_BODIES
// (wrench offsets), EE_POS_SIZE (reference rows), NUM_CONTACT_FRAMES (fc slots).
// 7 = iiwa14/indy7 (STATE_SIZE 14, the shipped shape).
#include <cstdint>
namespace grid {
constexpr uint32_t NUM_JOINTS = 7;
constexpr uint32_t NUM_POS = 7;
constexpr uint32_t NUM_VEL = 7;
constexpr uint32_t NUM_BODIES = 7;
constexpr uint32_t EE_POS_SIZE = 6;
constexpr uint32_t NUM_CONTACT_FRAMES = 1;
}
