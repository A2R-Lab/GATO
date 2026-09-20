#pragma once

#include <cstdint>
#include <cstdio>
#include <iostream>

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

void printDeviceInfo() {
   int deviceCount = 0;
   cudaError_t err = cudaGetDeviceCount(&deviceCount);
   if (err != cudaSuccess || deviceCount == 0) {
      std::cerr << "Error: No CUDA devices found. Exiting." << std::endl;
      exit(EXIT_FAILURE);
   }
   cudaDeviceProp prop;
   cudaError_t propErr = cudaGetDeviceProperties(&prop, 0); // device 0
   if (propErr != cudaSuccess) {
      std::cerr << "Error: Unable to get CUDA device properties. Exiting." << std::endl;
      exit(EXIT_FAILURE);
   }
   std::cout << "Device name: " << prop.name << std::endl;
   // bus width, peak memory bandwidth (memoryClockRate was removed in CUDA 13)
#if CUDART_VERSION < 13000
   std::cout << "Memory clock rate: " << prop.memoryClockRate << " kHz" << std::endl;
   std::cout << "Peak memory bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
#endif
   std::cout << "Bus width: " << prop.memoryBusWidth << " bits" << std::endl;
   // total global memory, shared memory, concurrent kernels
   std::cout << "Total global memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
   std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " kB" << std::endl;
   std::cout << "Concurrent kernels: " << prop.concurrentKernels << std::endl;
}

