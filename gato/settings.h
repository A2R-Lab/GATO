#pragma once

#include <cstdint>

namespace sqp {

using T = float;
constexpr uint32_t INTEGRATOR_TYPE = 2;  // 0: euler, 1: semi-implicit euler, 2: verlet (?)

// constexpr uint32_t SQP_MAX_ITER = 5;
// constexpr T KKT_TOL = static_cast<T>(1e-4);
// constexpr T SOLVED_RATIO = 1.0;


// constexpr uint32_t PCG_MAX_ITER = 100;
// constexpr uint32_t PCG_MIN_ITER = 1;
// constexpr T PCG_TOL = static_cast<T>(1e-4); // relative tol

// constexpr T MU = 1.0;  // scaling factor for constraint violation in merit function

// -——————————————————compile time settings——————————————————

constexpr uint32_t NUM_ALPHAS = 8;
constexpr float    q_COST = 2.0;
// constexpr float q_reg_COST = 1e-5;
constexpr float dq_COST = 5e-3;
constexpr float u_COST = 1e-6;
constexpr float N_COST = 20.0;
constexpr float q_lim_COST = 0.1;

constexpr float RHO = 1e-8;
// TODO: add u_lim


constexpr uint32_t KKT_THREADS = 128;
constexpr uint32_t SCHUR_THREADS = 128;
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 128;
constexpr uint32_t LINE_SEARCH_THREADS = 1024;
constexpr uint32_t SIM_FORWARD_THREADS = 512;

}  // namespace sqp

// ----- Plant -----
#include "dynamics/indy7/indy7_plant.cuh"
// #include "dynamics/iiwa14/iiwa14_plant.cuh"
//  TODO: add other plants
