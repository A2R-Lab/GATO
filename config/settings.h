#pragma once

#include <cstdint>

namespace sqp {

// uncomment to remove debug and error checking
// #define NDEBUG

// float precision: float(32)  double(64)
// half supported by CUDA but not C++ https://www.reddit.com/r/gcc/comments/1dv1l8e/support_for_half_precision_data_types_fp16_and/
using T = float;

constexpr uint32_t KNOT_POINTS = 32;
//constexpr T TIMESTEP = 0.01; // 1/64 s

constexpr uint32_t SQP_MAX_ITER = 20;
constexpr uint32_t PCG_MAX_ITER = 160;

constexpr T PCG_TOLERANCE = static_cast<T>(1e-3); // relative tolerance

constexpr uint32_t NUM_ALPHAS = 16;

// TODO: SQP max time (const frequency)

// ----- Cost -----
constexpr float CONTROL_COST = 1e-8;
constexpr float VELOCITY_COST = 1e-3;
constexpr float TERMINAL_COST = 10.0;

constexpr float RHO_INIT = 1e-2;
constexpr float RHO_FACTOR = 1.2;
constexpr float RHO_MAX = 10.0;
constexpr float RHO_MIN = 1e-8;

// ----- Kernels -----

constexpr uint32_t KKT_THREADS = 128;
constexpr uint32_t SCHUR_THREADS = 128;
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 128;
constexpr uint32_t LINE_SEARCH_THREADS = 128;

} // namespace sqp

// ----- Plant -----
//#include "dynamics/iiwa14/iiwa14_plant.cuh"
#include "dynamics/indy7/indy7_plant.cuh"
// TODO: add other plants
