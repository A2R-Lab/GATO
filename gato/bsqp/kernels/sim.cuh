#pragma once
// Batched plant step on the DEVICE integrator (sim_forward): x_{k+1} = f(x_k, u_k)
// under knot 0's external wrench — the same integrator the KKT linearization uses,
// so a simulated step and the solver's model agree bitwise.

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)
#include "dynamics/integrator.cuh"
#include "dynamics/grid_plant_step.cuh"  // floating twins + GATO_FLOATING_STEP

using namespace sqp;
using namespace gato;
using namespace gato::constants;
// no file-scope `using namespace gato::plant` (it leaks across kernel headers in the shared
// TU); qualify plant:: calls explicitly.

// shared layout (kernel carve == host sizer)
template <typename T>
struct SimSmem {
    static constexpr size_t xkp1 = 0;
    static constexpr size_t xk = xkp1 + XU_STATE_SIZE;
    static constexpr size_t uk = xk + XU_STATE_SIZE;
    static constexpr size_t temp = uk + CONTROL_SIZE;
#if GATO_FLOATING_STEP
    static constexpr size_t temp_ct = gato::plant::stepValueFloating_TempMemCt<T>();
#else
    static constexpr size_t temp_ct = 2 * STATE_SIZE + gato::plant::forwardDynamics_TempMemSize_Shared();
#endif
    static constexpr size_t total = temp + temp_ct;
    static constexpr size_t bytes() { return total * sizeof(T); }
};

template <typename T, uint32_t INTEGRATOR_TYPE = gato::constants::INTEGRATOR_TYPE_DEFAULT, bool ANGLE_WRAP = false>
__global__ __launch_bounds__(SIM_FORWARD_THREADS)
void sim_forward_batched_kernel(
    T *d_xkp1_batch,
    T *d_xk,
    T *d_uk,
    void *d_GRiD_mem,
    T *d_f_ext_batch,
    T dt
) {
    const uint32_t solve_idx = blockIdx.y;
    // states are STORED format (XU_STATE_SIZE == STATE_SIZE on fixed base)
    T *d_xkp1 = d_xkp1_batch + solve_idx * XU_STATE_SIZE;
    // sim_forward rolls the CURRENT step: knot 0's wrench
    T *d_f_ext = get_offset_wrench<T>(d_f_ext_batch, solve_idx, 0);

    extern __shared__ T s_mem[];
    T *s_xkp1 = s_mem + SimSmem<T>::xkp1;
    T *s_xk = s_mem + SimSmem<T>::xk;
    T *s_uk = s_mem + SimSmem<T>::uk;
    T *s_temp = s_mem + SimSmem<T>::temp;

    glass::copy<T, XU_STATE_SIZE>(d_xk, s_xk);
    glass::copy<T, CONTROL_SIZE>(d_uk, s_uk);

#if GATO_FLOATING_STEP
    __syncthreads();  // the twin reads s_xk/s_uk from a different thread mapping
    gato::plant::sim_step_floating<T, INTEGRATOR_TYPE>(s_xkp1, s_xk, s_uk, s_temp, d_GRiD_mem, dt, d_f_ext);
#else
    gato::plant::sim_step<T, INTEGRATOR_TYPE, ANGLE_WRAP>(
        s_xkp1,
        s_xk,
        s_uk,
        s_temp,
        d_GRiD_mem,
        dt,
        d_f_ext
    );
#endif
    __syncthreads();

    glass::copy<T, XU_STATE_SIZE>(s_xkp1, d_xkp1);
}

template <typename T>
__host__
size_t get_sim_forward_batched_kernel_smem_size() {
    return SimSmem<T>::bytes();
}

template <typename T>
__host__
void sim_forward_batched(
    uint32_t batch_size,
    T *d_xkp1_batch,
    T *d_xk,
    T *d_uk,
    void *d_GRiD_mem,
    T *d_f_ext_batch,
    T dt
) {
    dim3 grid(1, batch_size);
    dim3 block(SIM_FORWARD_THREADS);
    size_t s_mem_size = get_sim_forward_batched_kernel_smem_size<T>();

    sim_forward_batched_kernel<T><<<grid, block, s_mem_size>>>(
        d_xkp1_batch,
        d_xk,
        d_uk,
        d_GRiD_mem,
        d_f_ext_batch,
        dt
    );
        gpuErrchk(cudaGetLastError());  // launch-config failures must not pass silently
}


