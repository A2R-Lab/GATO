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
constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t DZ_THREADS = 128;
constexpr uint32_t LINE_SEARCH_THREADS = 512;
constexpr uint32_t SIM_FORWARD_THREADS = 128;

}  // namespace sqp

// ----- Plant Selection -----
// The plant adapter header is injected at compile time (CMake sets
// -DGATO_PLANT_HEADER="dynamics/<name>/<name>_plant.cuh"), so adding a robot
// needs no edits here.
#ifndef GATO_PLANT_HEADER
    #error "GATO_PLANT_HEADER must be defined (path to the plant adapter header)"
#endif
#include GATO_PLANT_HEADER
