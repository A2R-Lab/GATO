#pragma once
// Centre-of-mass term (0.1.5): 0.5 sum_r W[r] (c_r(q) - c_des_r)^2 over the
// three CoM coordinates, added to whichever objective is on (the tracking
// cost or the cost menu) on modules whose header bakes the centroidal
// families (gato.build(..., com=True); the vendored Go2). The solver host
// (bsqp/com_term_host.cuh) owns the device target array and fills this view;
// the setup-KKT and merit kernels read it through the plant adapters
// (dynamics/plant.cuh, com_term_value / com_term_grad_hess) in their COM
// instantiations. The target is per knot, one copy shared across the batch
// (batch stride 0) or one copy per solve (batch stride 3 KNOT_POINTS).
#include <cstdint>
#include <cstddef>

namespace gato {
namespace com {

template<typename T>
struct Buffers {
        const T* target;    // KNOT_POINTS x 3 (shared) or batch x KNOT_POINTS x 3 (per solve)
        uint32_t bstride;   // elements per solve (0 = shared across the batch)
        T        w[3];      // per-coordinate weight
        int      on;

        __host__ __device__ __forceinline__ const T* at(uint32_t solve_idx, uint32_t knot_idx) const
        {
                return target + static_cast<size_t>(solve_idx) * bstride + static_cast<size_t>(knot_idx) * 3u;
        }
};

template<typename T>
__host__ inline Buffers<T> empty_buffers()
{
        Buffers<T> c;
        c.target = nullptr; c.bstride = 0u; c.w[0] = c.w[1] = c.w[2] = static_cast<T>(0); c.on = 0;
        return c;
}

}  // namespace com
}  // namespace gato
