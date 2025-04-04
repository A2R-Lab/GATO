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

void setL2PersistingAccess(float fraction) {
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0); // device 0
   std::cout << "L2 cache size: " << prop.l2CacheSize << std::endl;
   std::cout << "Persisting L2 cache max size: " << prop.persistingL2CacheMaxSize << std::endl;
   std::cout << "--------------------------------" << std::endl;
   size_t size = min(int(prop.l2CacheSize * fraction), prop.persistingL2CacheMaxSize);
   cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
}

void resetL2PersistingAccess() {
   gpuErrchk(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
   gpuErrchk(cudaCtxResetPersistingL2Cache());
} // use cudaCtxResetPersistingL2Cache() to reset all persisting L2 cache lines to normal
