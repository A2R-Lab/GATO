#pragma once
// Host side of the centre-of-mass term (dynamics/com_term.cuh): owns the
// device target array, validates the inputs and hands the kernels their view.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include "constants.h"
#include "utils/cuda.cuh"
#include "dynamics/integrator.cuh"   // the plant adapter: GATO_COM_AVAILABLE
#include "dynamics/com_term.cuh"

namespace gato {
namespace com {

template<typename T>
class Host {
    public:
        Host() = default;
        ~Host() { release(); }
        Host(const Host&) = delete;
        Host& operator=(const Host&) = delete;
        void init(uint32_t batch_size) { batch_size_ = batch_size; view_ = empty_buffers<T>(); }
        static constexpr bool available() { return GATO_COM_AVAILABLE != 0; }

        // count = KNOT_POINTS * 3 (shared across the batch) or batch_size * KNOT_POINTS * 3 (per solve)
        void set_target(const T* h, size_t count, bool per_solve)
        {
                require_available();
                const size_t expected = (per_solve ? (size_t)batch_size_ : (size_t)1) * (size_t)KNOT_POINTS * 3u;
                if (count != expected) {
                        throw std::invalid_argument("CoM target: expected " + std::to_string(expected) + " entries ("
                                                    + (per_solve ? "per solve" : "shared across the batch") + "), got " + std::to_string(count));
                }
                for (size_t i = 0; i < count; ++i) { if (!std::isfinite((double)h[i])) { throw std::invalid_argument("CoM target: entry " + std::to_string(i) + " is not finite"); } }
                if (cap_ < count) {
                        if (d_) { gpuErrchk(cudaFree(d_)); d_ = nullptr; }
                        gpuErrchk(cudaMalloc(&d_, count * sizeof(T)));
                        cap_ = count;
                }
                gpuErrchk(cudaMemcpy(d_, h, count * sizeof(T), cudaMemcpyHostToDevice));
                view_.target = d_;
                view_.bstride = per_solve ? static_cast<uint32_t>((size_t)KNOT_POINTS * 3u) : 0u;
                target_set_ = true;
        }
        void set_weight(const T* w, size_t count)
        {
                require_available();
                if (count != 3) { throw std::invalid_argument("CoM weight: expected 3 entries, got " + std::to_string(count)); }
                for (int i = 0; i < 3; ++i) {
                        if (!(w[i] >= static_cast<T>(0)) || !std::isfinite((double)w[i])) { throw std::invalid_argument("CoM weight: entries must be finite and nonnegative"); }
                        view_.w[i] = w[i];
                }
                weight_set_ = true;
        }
        void set_on(bool on)
        {
                if (on) {
                        require_available();
                        if (!target_set_ || !weight_set_) { throw std::runtime_error("CoM term: set the target and the weight before turning it on"); }
                }
                view_.on = on ? 1 : 0;
        }
        bool on() const { return view_.on != 0; }
        const Buffers<T>* view_ptr() const { return view_.on ? &view_ : nullptr; }
        void clear() { view_ = empty_buffers<T>(); target_set_ = weight_set_ = false; }

    private:
        void require_available() const
        {
                if (!available()) { throw std::runtime_error("CoM term: this module's header does not bake the centroidal families (gato.build(..., com=True))"); }
                if (batch_size_ == 0) { throw std::runtime_error("CoM term used before the solver was constructed"); }
        }
        void release()
        {
                if (d_) { cudaFree(d_); d_ = nullptr; }
                cap_ = 0;
        }
        uint32_t   batch_size_ = 0;
        T*         d_ = nullptr;
        size_t     cap_ = 0;
        bool       target_set_ = false, weight_set_ = false;
        Buffers<T> view_ = empty_buffers<T>();
};

}  // namespace com
}  // namespace gato
