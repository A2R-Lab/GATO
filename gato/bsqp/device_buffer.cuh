#pragma once
// A device array that grows to the largest upload it has seen; the host
// classes of the cost menu and the centre-of-mass term keep their per-field
// arrays this way.
#include <cstddef>
#include "utils/cuda.cuh"

namespace gato {
namespace device_buffer {

template<typename U>
inline void ensure(U*& d, size_t& cap, size_t count)
{
        if (cap < count) {
                if (d) { gpuErrchk(cudaFree(d)); d = nullptr; }
                gpuErrchk(cudaMalloc(&d, count * sizeof(U)));
                cap = count;
        }
}

template<typename U>
inline void upload(U*& d, size_t& cap, const U* h, size_t count)
{
        ensure(d, cap, count);
        gpuErrchk(cudaMemcpy(d, h, count * sizeof(U), cudaMemcpyHostToDevice));
}

}  // namespace device_buffer
}  // namespace gato
