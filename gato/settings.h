#pragma once

#include <cstdint>

namespace sqp {

#ifdef USE_DOUBLES
typedef double T;
#else
typedef float T;
#endif

constexpr uint32_t INTEGRATOR_TYPE = 2;  // 0: euler, 1: semi-implicit euler, 2: verlet (?)

// -——————————————————compile time settings——————————————————

constexpr uint32_t NUM_ALPHAS = 8;
constexpr float    q_COST = 2.0;
constexpr float    dq_COST = 5e-3;
constexpr float    u_COST = 1e-6;
constexpr float    N_COST = 20.0;
constexpr float    q_lim_COST = 0.1;

constexpr float RHO = 1e-8;

constexpr uint32_t KKT_THREADS = 64;
constexpr uint32_t SCHUR_THREADS = 128;
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 64;
constexpr uint32_t LINE_SEARCH_THREADS = 1024;
constexpr uint32_t SIM_FORWARD_THREADS = 1024;

}  // namespace sqp

// ----- Plant -----
#include "dynamics/indy7/indy7_plant.cuh"
// #include "dynamics/iiwa14/iiwa14_plant.cuh"
