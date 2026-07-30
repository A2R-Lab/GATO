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
// direct-solve kernel thread count (serial knot chain over 14x14 blocks).
// TUNED 2026-07-10 (bdsv_timing_session, RTX 5090): 512 fastest at every
// (N, B) swept — N32 297µs, N64 631-641µs, batch-invariant; 256 was 3-7%
// slower, 128 clearly worse (data/bdsv_timing/BDSV_TIMING_RESULTS.md).
// -D override kept for the thread-invariance gate + retuning on new arches.
#ifndef GATO_BDSV_THREADS
    #define GATO_BDSV_THREADS 512
#endif
constexpr uint32_t BDSV_THREADS = GATO_BDSV_THREADS;
constexpr uint32_t DZ_THREADS = 128;

// Exact-Hessian (SO-SQP) build flag: 0 (default) compiles the path OUT, so the
// device tree is preprocessor-identical to the GN-only solver (bitwise parity
// gate holds by construction). Build with cmake -DGATO_EXACT_HESSIAN=ON
// (-DUSE_EXACT_HESSIAN=1) and enable per solver via set_exact_hessian(true):
// per-TASK toggle — the projection wins on EE-terminal tasks, is neutral-to-
// worse on full-rank joint-terminal ones (so_sqp_prototype/RESULTS_2026-07-17).
#ifndef USE_EXACT_HESSIAN
    #define USE_EXACT_HESSIAN 0
#endif
constexpr uint32_t LINE_SEARCH_THREADS = 512;
constexpr uint32_t SIM_FORWARD_THREADS = 128;

}  // namespace sqp

// ----- Plant Selection -----
// The plant adapter header is injected at compile time (CMake sets
// -DGATO_PLANT_HEADER="dynamics/plant.cuh" and puts gato/dynamics/<name>/ on
// the include path for grid.cuh + limits.cuh), so adding a robot needs no
// edits here.
#ifndef GATO_PLANT_HEADER
    #error "GATO_PLANT_HEADER must be defined (path to the plant adapter header)"
#endif
#include GATO_PLANT_HEADER
