#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/linalg.cuh"
#include "dynamics/integrator.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;
using namespace gato::plant;

template <typename T, uint32_t BatchSize, uint32_t INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false>
__global__
void simForwardBatchedKernel(
    T *d_xkp1_batch,
    T *d_xk,
    T *d_uk,
    void *d_GRiD_mem,
    T *d_f_ext_batch,
    T dt
) {
    const uint32_t solve_idx = blockIdx.y;
    T *d_xkp1 = d_xkp1_batch + solve_idx * STATE_SIZE;
    T *d_f_ext = getOffsetWrench<T, BatchSize>(d_f_ext_batch, solve_idx);

    extern __shared__ T s_mem[];
    T *s_xkp1 = s_mem;
    T *s_xk = s_xkp1 + STATE_SIZE;
    T *s_uk = s_xk + STATE_SIZE;
    T *s_temp = s_uk + CONTROL_SIZE;

    block::copy<T, STATE_SIZE>(s_xk, d_xk);
    block::copy<T, CONTROL_SIZE>(s_uk, d_uk);

    sim_step<T, INTEGRATOR_TYPE, ANGLE_WRAP>(
        s_xkp1,
        s_xk,
        s_uk,
        s_temp,
        d_GRiD_mem,
        dt,
        d_f_ext
    );
    __syncthreads();

    block::copy<T, STATE_SIZE>(d_xkp1, s_xkp1);
}

template <typename T>
__host__
size_t getSimForwardBatchedKernelSMemSize() {
    size_t size = sizeof(T) * 2 * (
        STATE_SIZE + // xkp1
        STATE_SIZE + // xk
        CONTROL_SIZE + // uk
        2 * STATE_SIZE + // temp
        gato::plant::forwardDynamics_TempMemSize_Shared()
    );
    return size;
}

template <typename T, uint32_t BatchSize>
__host__
void simForwardBatched(
    T *d_xkp1_batch,
    T *d_xk,
    T *d_uk,
    void *d_GRiD_mem,
    T *d_f_ext_batch,
    T dt
) {
    dim3 grid(1, BatchSize);
    dim3 block(SIM_FORWARD_THREADS);
    size_t s_mem_size = getSimForwardBatchedKernelSMemSize<T>();

    simForwardBatchedKernel<T, BatchSize><<<grid, block, s_mem_size>>>(
        d_xkp1_batch,
        d_xk,
        d_uk,
        d_GRiD_mem,
        d_f_ext_batch,
        dt
    );
}


