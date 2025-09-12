#pragma once

#include <cstdint>
#include "constants.h"

using namespace gato::constants;

namespace gato {
namespace batch {

// Replicate a single vector to all batches: dst has shape (BatchSize, len)
template <typename T, uint32_t BatchSize>
__global__ void replicateVectorKernel(T* __restrict__ dst_batch,
                                       const T* __restrict__ src,
                                       uint32_t len)
{
    const uint32_t b = blockIdx.y;
    T* dst = dst_batch + static_cast<size_t>(b) * len;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < len; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

template <typename T, uint32_t BatchSize>
__host__ inline void replicateVectorToBatch(T* dst_batch, const T* src, uint32_t len)
{
    dim3 grid( (len + 255) / 256, BatchSize );
    dim3 block(256);
    replicateVectorKernel<T, BatchSize><<<grid, block>>>(dst_batch, src, len);
}

// Compute per-batch squared L2 error between predicted x (BatchSize x STATE_SIZE) and x_curr (STATE_SIZE)
template <typename T, uint32_t BatchSize>
__global__ void squaredErrorKernel(const T* __restrict__ xkp1_batch,
                                   const T* __restrict__ x_curr,
                                   T* __restrict__ errors)
{
    const uint32_t b = blockIdx.x; // one block per batch element
    __shared__ T ssum[256];
    T sum = static_cast<T>(0);
    const T* xpred = xkp1_batch + static_cast<size_t>(b) * STATE_SIZE;
    for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
        T d = xpred[i] - x_curr[i];
        sum += d * d;
    }
    ssum[threadIdx.x] = sum;
    __syncthreads();

    // reduction
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) ssum[threadIdx.x] += ssum[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) errors[b] = ssum[0];
}

template <typename T, uint32_t BatchSize>
__host__ inline void computeSquaredErrorsBatched(const T* xkp1_batch,
                                                 const T* x_curr,
                                                 T* errors)
{
    dim3 grid(BatchSize);
    dim3 block(256);
    squaredErrorKernel<T, BatchSize><<<grid, block>>>(xkp1_batch, x_curr, errors);
}

// Copy selected trajectory k to all batches
template <typename T, uint32_t BatchSize>
__global__ void copyTrajectoryToAllKernel(T* __restrict__ xu_traj_batch,
                                          const T* __restrict__ xu_best,
                                          uint32_t len)
{
    const uint32_t b = blockIdx.y;
    T* dst = xu_traj_batch + static_cast<size_t>(b) * len;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < len; i += blockDim.x * gridDim.x) {
        dst[i] = xu_best[i];
    }
}

template <typename T, uint32_t BatchSize>
__host__ inline void copyBestTrajectoryToAll(T* xu_traj_batch,
                                             const T* xu_best,
                                             uint32_t len)
{
    dim3 grid( (len + 255) / 256, BatchSize );
    dim3 block(256);
    copyTrajectoryToAllKernel<T, BatchSize><<<grid, block>>>(xu_traj_batch, xu_best, len);
}

} // namespace batch
} // namespace gato

