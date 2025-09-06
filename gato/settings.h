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
constexpr float    settings_q_COST = 2.0;
constexpr float    settings_dq_COST = 5e-3;
constexpr float    settings_u_COST = 1e-6;
constexpr float    settings_N_COST = 20.0;
constexpr float    settings_q_lim_COST = 0.0;
constexpr float    settings_vel_lim_COST = 0.0;
constexpr float    settings_ctrl_lim_COST = 0.0;

// constexpr float RHO = 1e-8;
constexpr float RHO_INIT = 1e-3;
constexpr float RHO_FACTOR = 1.5;
constexpr float RHO_MIN = 1e-4;
constexpr float RHO_MAX = 100;

constexpr uint32_t KKT_THREADS = 128;
constexpr uint32_t SCHUR_THREADS = 256;
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 128;
constexpr uint32_t LINE_SEARCH_THREADS = 512;
constexpr uint32_t SIM_FORWARD_THREADS = 128;

}  // namespace sqp

// ----- Plant -----
// #include "dynamics/indy7/indy7_plant.cuh"
#include "dynamics/iiwa14/iiwa14_plant.cuh"
