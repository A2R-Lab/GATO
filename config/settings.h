#pragma once

#include <cstdint>

namespace sqp {

// uncomment to remove debug and error checking
// #define NDEBUG

// float precision: float(32)  double(64)
// half supported by CUDA but not C++ https://www.reddit.com/r/gcc/comments/1dv1l8e/support_for_half_precision_data_types_fp16_and/
using T = float;

constexpr uint32_t KNOT_POINTS = 32;
constexpr T TIMESTEP = 0.015625; // 1/64 s

constexpr uint32_t SQP_MAX_ITER = 1;
constexpr uint32_t PCG_MAX_ITER = 100;

constexpr T PCG_TOLERANCE = static_cast<T>(1e-4);

constexpr uint32_t NUM_ALPHAS = 8;


// TODO: SQP max time (const frequency)

// ----- Cost -----

constexpr float CONTROL_COST = 0.0001;
constexpr float VELOCITY_COST = 0.0001;

constexpr float RHO_INIT = 1e-3;
constexpr float RHO_FACTOR = 1.2;
constexpr float RHO_MAX = 10.0;
constexpr float RHO_MIN = 1e-10;

// ----- Kernels -----

constexpr uint32_t KKT_THREADS = 128;
constexpr uint32_t SCHUR_THREADS = 128;
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 96;
constexpr uint32_t LINE_SEARCH_THREADS = 128;

} // namespace sqp

// ----- Plant -----
#include "dynamics/iiwa14/iiwa14_plant.cuh"
// TODO: add other plants
