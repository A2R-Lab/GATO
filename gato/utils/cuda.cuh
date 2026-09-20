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


