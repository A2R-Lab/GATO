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

// -D override kept for the thread-invariance gate (same convention as
// GATO_BDSV_THREADS below); production value 128.
#ifndef GATO_KKT_THREADS
    #define GATO_KKT_THREADS 128
#endif
constexpr uint32_t KKT_THREADS = GATO_KKT_THREADS;
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

// CL-3a contact-force flag: 0 (default) keeps CONTROL_SIZE = actuated torques and
// the device tree preprocessor-identical to the baseline solver (bitwise parity
// by construction). 1 appends a per-knot world-aligned contact WRENCH decision
// variable (6 per contact frame) to every control slot — f_c extends u; the KKT/
// Schur/merit/batch machinery sees one wider control and is structurally
// unchanged. Requires a grid.cuh generated with contact_frames (2b.1+). Build
// with cmake -DGATO_CONTACT_FORCES=ON into build_fc/ (module-ABI change: .so
// variants, the build_eh pattern). SSOT: docs/open-tasks/
// cl3a_contact_forces_2026-08-02.md.
#ifndef GATO_CONTACT_FORCES
    #define GATO_CONTACT_FORCES 0
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
