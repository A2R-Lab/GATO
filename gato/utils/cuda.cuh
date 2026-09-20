#pragma once

#include <cstdint>
#include <cstdio>

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

// Opt a kernel in past the 48 KB default dynamic-smem ceiling. Most kernels
// size their carve from per-module constants, a few from runtime flags
// (setup_kkt/merit: exact-Hessian / collision carves), so the caller keeps a
// per-site latch and the attribute is re-set whenever the request GROWS (a
// once-only latch launched a bigger carve over the old ceiling — 2026-09-20
// audit). FAIL LOUD on the attribute set: an over-ceiling request used to
// fail SILENTLY at launch, leaving the outputs unwritten while the solve
// "ran" (2026-08-11 collision-carve bug); past the device opt-in maximum
// (~99 KB on sm_120) the attribute set itself fails — also loud. Pair every
// launch with gpuErrchk(cudaGetLastError()).
template<typename Kernel>
inline void opt_in_dynamic_smem(Kernel kernel, size_t bytes, size_t& latch)
{
        if (bytes > 48 * 1024 && bytes > latch) {
                gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
                latch = bytes;
        }
}
