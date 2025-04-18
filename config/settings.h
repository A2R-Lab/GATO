#pragma once

#include <cstdint>

namespace sqp {

// uncomment to remove debug and error checking
// #define NDEBUG

// float precision: float(32)  double(64)
// half supported by CUDA but not C++ https://www.reddit.com/r/gcc/comments/1dv1l8e/support_for_half_precision_data_types_fp16_and/
using T = float;

constexpr uint32_t KNOT_POINTS = 32;
constexpr uint32_t INTEGRATOR_TYPE = 2; // 0: euler, 1: semi-implicit euler, 2: trapezoidal

constexpr uint32_t SQP_MAX_ITER = 2;
constexpr uint32_t PCG_MAX_ITER = 200;

constexpr T PCG_TOLERANCE = static_cast<T>(5e-5); // relative tolerance

constexpr uint32_t NUM_ALPHAS = 8;

constexpr uint32_t F_EXT_KNOTS = 32;

// TODO: SQP max time (const frequency)

// ----- Cost -----
constexpr float CONTROL_COST = 1e-7;
constexpr float VELOCITY_COST = 1e-2;
constexpr float TERMINAL_COST = 100.0;
constexpr float BARRIER_COST = 0.05;

constexpr float RHO_INIT = 1e-4;
constexpr float RHO_FACTOR = 1.25;
constexpr float RHO_MAX = 10.0;
constexpr float RHO_MIN = 1e-8;

// ----- Kernels -----

constexpr uint32_t KKT_THREADS = 128;
constexpr uint32_t SCHUR_THREADS = 128;
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 128;
constexpr uint32_t LINE_SEARCH_THREADS = 128;
constexpr uint32_t SIM_FORWARD_THREADS = 128;
} // namespace sqp

// ----- Plant -----
//#include "dynamics/iiwa14/iiwa14_plant.cuh"
#include "dynamics/indy7/indy7_plant.cuh"
// TODO: add other plants
