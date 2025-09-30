#pragma once

#include <cstdint>

namespace sqp {

#ifdef USE_DOUBLES
typedef double T;
#else
typedef float T;
#endif

// -——————————————————compile time settings——————————————————

constexpr uint32_t NUM_ALPHAS = 8;

// constexpr float RHO = 1e-8;
constexpr float RHO_INIT = 1e-3;
constexpr float RHO_FACTOR = 1.2;
constexpr float RHO_MIN = 1e-8;
constexpr float RHO_MAX = 10;

constexpr uint32_t KKT_THREADS = 128;
constexpr uint32_t SCHUR_THREADS = 128;
constexpr uint32_t PCG_THREADS = 512;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t MERIT_THREADS = 128;
constexpr uint32_t LINE_SEARCH_THREADS = 512;
constexpr uint32_t SIM_FORWARD_THREADS = 128;

}  // namespace sqp

// ----- Plant Selection -----
// Plant type is defined at compile time via CMake
#if defined(PLANT_INDY7)
    #include "dynamics/indy7/indy7_plant.cuh"
#elif defined(PLANT_IIWA14)
    #include "dynamics/iiwa14/iiwa14_plant.cuh"
#else
    #error "Plant type must be defined: PLANT_INDY7 or PLANT_IIWA14"
#endif
