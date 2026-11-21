#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// grid.cuh (always pulled in via settings.h -> dynamics/plant.cuh) defines gpuAssert + gpuErrchk
// unconditionally; guard on the macro so we defer to it when present (avoids a same-TU
// redefinition / ODR clash) and only provide our own when cuda.cuh is used standalone.
#ifndef gpuErrchk
#ifndef NDEBUG //disable gpuAssert in No Debug mode
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#else
#define gpuErrchk(ans) ans
#endif
#endif // gpuErrchk

// The most block shared memory (static plus dynamic) a kernel may opt in to on
// the current device: 101,376 bytes on sm_120, more on the Jetson AGX Orin.
inline size_t max_block_smem_bytes()
{
        int device = 0, limit = 0;
        gpuErrchk(cudaGetDevice(&device));
        gpuErrchk(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        return static_cast<size_t>(limit > 0 ? limit : 0);
}

// Opt a kernel in past the 48 KB default dynamic-smem ceiling. Most kernels
// size their carve from per-module constants, a few from runtime flags
// (setup_kkt/merit: exact-Hessian / collision carves), so the caller keeps a
// per-site latch and the attribute is re-set whenever the request GROWS (a
// once-only latch launched a bigger carve over the old ceiling — 2026-09-20
// audit). A request past the device maximum is REFUSED here with a
// std::runtime_error that names the stage (`what`), the request and the limit:
// the attribute set would fail with "invalid argument" and the process exit
// (issue #19 sweep on the RTX 5090: PCG at N512, and the ADMM residual table,
// both of which grow with the horizon). An over-ceiling request used to fail
// SILENTLY at launch, leaving the outputs unwritten while the solve "ran"
// (2026-08-11 collision-carve bug). Pair every launch with
// gpuErrchk(cudaGetLastError()).
template<typename Kernel>
inline void opt_in_dynamic_smem(Kernel kernel, size_t bytes, size_t& latch, const char* what = "a solver kernel")
{
        if (bytes > 48 * 1024 && bytes > latch) {
                const size_t limit = max_block_smem_bytes();
                if (bytes > limit) {
                        throw std::runtime_error(std::string(what) + " cannot launch for this module on this GPU: it needs " + std::to_string(bytes)
                                                 + " bytes of block shared memory and the device allows " + std::to_string(limit)
                                                 + ". The request grows with the horizon; use a shorter one.");
                }
                gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
                latch = bytes;
        }
}
