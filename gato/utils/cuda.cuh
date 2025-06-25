#pragma once

#include <cstdint>
#include <cstdio>
#include <iostream>

#ifndef NDEBUG //disable gpuAssert in No Debug mode
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
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
   // memory clock rate, bus width, peak memory bandwidth
   std::cout << "Memory clock rate: " << prop.memoryClockRate << " kHz" << std::endl;
   std::cout << "Bus width: " << prop.memoryBusWidth << " bits" << std::endl;
   std::cout << "Peak memory bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
   // total global memory, shared memory, concurrent kernels
   std::cout << "Total global memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
   std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " kB" << std::endl;
   std::cout << "Concurrent kernels: " << prop.concurrentKernels << std::endl;
}

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management
// - Repeated accesses to data region in the global memory are considered to be persisting.
// Allocate a fraction of the L2 cache for persisting accesses to global memory
void setL2PersistingAccess(float fraction) {
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0); // device 0
   size_t l2_kb = prop.l2CacheSize / 1024;
   size_t persisting_l2_max_kb = prop.persistingL2CacheMaxSize / 1024;
   size_t size = std::min(static_cast<size_t>(prop.l2CacheSize * fraction), static_cast<size_t>(prop.persistingL2CacheMaxSize));
   size_t size_kb = size / 1024;
   std::cout << "Total L2 cache size: " << l2_kb << " kB" << std::endl;
   std::cout << "Setting persisting L2 size to: " << size_kb << " / " << persisting_l2_max_kb << " kB" << std::endl;
   cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
}

void resetL2PersistingAccess() {
   gpuErrchk(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
   gpuErrchk(cudaCtxResetPersistingL2Cache()); //reset all persisting L2 cache lines to normal
}