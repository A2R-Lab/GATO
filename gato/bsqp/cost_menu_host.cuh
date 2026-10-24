#pragma once
// Host side of the cost menu (grid_plant::tracking_cost_menu*, grid-rbd 0.5.5):
// owns the per-knot device arrays, validates what a knot activates against
// what is set, and hands the kernels a gato::menu::Buffers view. Fields are
// addressed by name (gato::menu::FIELD_NAMES); each is one copy shared across
// the batch or one copy per solve. Buffers are allocated on first use and
// grow in place, so repeated updates of the same shapes do not allocate.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include "constants.h"
#include "utils/cuda.cuh"
#include "dynamics/integrator.cuh"   // the plant adapter: GATO_MENU_AVAILABLE, gato::plant::{NQ, NX, NU, MENU_CF2, GRAVITY}
#include "dynamics/cost_menu.cuh"
#include "utils/linalg.cuh"   // get_offset_xu
#include "device_buffer.cuh"

namespace gato {
namespace menu {

#if GATO_MENU_AVAILABLE
// u_des[k] = g(q_ref[k]): the generalized gravity at the reference configuration
// (grid_plant::gravity_torque, RNEA at zero velocity and acceleration), one block
// per knot. Dynamic shared memory: s_q (NQ) | s_g (NU) | the gravity_torque arena (16-byte aligned).
template<typename T>
__global__ void gravity_target_kernel(T* __restrict__ d_u_des, const T* __restrict__ d_q_ref, const grid::robotModel<T>* d_robotModel, T gravity)
{
        constexpr int NQ = gato::plant::NQ;
        constexpr int NU = gato::plant::NU;
        extern __shared__ T s_mem[];
        T* s_q = s_mem;
        T* s_g = s_q + NQ;
        const uintptr_t a = reinterpret_cast<uintptr_t>(s_g + NU);
        T* s_scratch = reinterpret_cast<T*>((a + 15u) & ~static_cast<uintptr_t>(15u));
        const uint32_t k = blockIdx.x;
        for (int i = threadIdx.x; i < NQ; i += blockDim.x) { s_q[i] = d_q_ref[(size_t)k * NQ + i]; }
        __syncthreads();
        grid_plant::gravity_torque<T>(s_g, s_q, s_scratch, d_robotModel, gravity);
        __syncthreads();
        for (int i = threadIdx.x; i < NU; i += blockDim.x) { d_u_des[(size_t)k * NU + i] = s_g[i]; }
}

// per-term values of the menu along a trajectory, each term evaluated alone:
// d_out[(solve * KNOT_POINTS + knot) * NUM_TERMS + term]. The joint-limit barriers are
// not menu terms and are left out (mu = 0); the terminal knot reads a zero control.
template<typename T>
__global__ void menu_term_values_kernel(T* __restrict__ d_out, const T* __restrict__ d_xu_traj_batch, const Buffers<T> m, const grid::robotModel<T>* d_robotModel)
{
        constexpr int NX = gato::plant::NX;
        constexpr int NU = gato::plant::NU;
        extern __shared__ T s_mem[];
        T* s_xu = s_mem;             // x_k (NX) | u_k (NU)
        T* s_temp = s_xu + NX + NU;
        __shared__ int s_mask[NUM_TERMS];
        const uint32_t k = blockIdx.x, b = blockIdx.y;
        const bool terminal = (k == KNOT_POINTS - 1);
        const T* d_xu_k = get_offset_xu<T>(d_xu_traj_batch, b, k);
        for (int i = threadIdx.x; i < NX + NU; i += blockDim.x) { s_xu[i] = (terminal && i >= NX) ? static_cast<T>(0) : d_xu_k[i]; }
        __syncthreads();
        for (int t = 0; t < NUM_TERMS; ++t) {
                for (int j = threadIdx.x; j < NUM_TERMS; j += blockDim.x) { s_mask[j] = (j == t) ? m.activation[k * NUM_TERMS + t] : -1; }
                __syncthreads();
                const T v = gato::plant::menu_cost_value<T>(s_xu, s_xu + NX, m, b, k, terminal, s_temp, d_robotModel,
                                                           static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), s_mask);
                if (threadIdx.x == 0) { d_out[((size_t)b * KNOT_POINTS + k) * NUM_TERMS + t] = v; }
                __syncthreads();
        }
}
#endif

template<typename T>
class Host {
public:
        Host() = default;
        ~Host() { release(); }
        Host(const Host&) = delete;
        Host& operator=(const Host&) = delete;

        void init(uint32_t batch_size) { batch_size_ = batch_size; view_ = empty_buffers<T>(); }

        static constexpr bool available() { return GATO_MENU_AVAILABLE != 0; }
        static constexpr bool tangent()
        {
#if GATO_MENU_AVAILABLE
                return gato::plant::MENU_TANGENT;
#else
                return false;
#endif
        }
        static constexpr Dims dims()
        {
#if GATO_MENU_AVAILABLE
                return gato::plant::menu_dims();
#else
                return Dims{0u, 0u, 0u, 0u};
#endif
        }
        static int field_index(const std::string& name)
        {
                for (int f = 0; f < NUM_FIELDS; ++f) { if (name == FIELD_NAMES[f]) return f; }
                throw std::invalid_argument("unknown cost-menu field '" + name + "'");
        }
        static uint32_t per_knot_size(int f) { return field_size(f, dims()); }

        // one field: count = KNOT_POINTS * size (shared across the batch) or batch_size * KNOT_POINTS * size (per solve)
        void set_field(const std::string& name, const T* h, size_t count, bool per_solve)
        {
                require_available();
                const int f = field_index(name);
                const size_t size = per_knot_size(f);
                const size_t expected = (per_solve ? (size_t)batch_size_ : (size_t)1) * (size_t)KNOT_POINTS * size;
                if (count != expected) {
                        throw std::invalid_argument("cost-menu field '" + name + "': expected " + std::to_string(expected) + " entries ("
                                                    + (per_solve ? "per solve" : "shared across the batch") + "), got " + std::to_string(count));
                }
                upload(d_[f], cap_[f], h, count);
                view_.ptr[f] = d_[f];
                view_.bstride[f] = per_solve ? static_cast<uint32_t>((size_t)KNOT_POINTS * size) : 0u;
                set_[f] = true;
        }
        void clear_field(const std::string& name)
        {
                const int f = field_index(name);
                view_.ptr[f] = nullptr; view_.bstride[f] = 0u; set_[f] = false;
                if (view_.on) { validate(); }
        }
        // KNOT_POINTS x NUM_TERMS activation ids (GRID_PLANT_ACTIVATION_*), shared across the batch
        void set_activation(const int* h, size_t count)
        {
                require_available();
                const size_t expected = (size_t)KNOT_POINTS * NUM_TERMS;
                if (count != expected) { throw std::invalid_argument("cost-menu activation: expected " + std::to_string(expected) + " entries, got " + std::to_string(count)); }
                for (size_t i = 0; i < count; ++i) {
                        if (h[i] < -1 || h[i] > 4) { throw std::invalid_argument("cost-menu activation id out of range at entry " + std::to_string(i) + " (NONE -1, QUADRATIC 0, PSEUDO_HUBER 1, RELAXED_LOG_BARRIER 2, QUADRATIC_BARRIER 3, SMOOTH_HINGE 4)"); }
                }
                h_act_.assign(h, h + count);
                upload(d_act_, cap_act_, h, count);
                view_.activation = d_act_;
        }
        void set_on(bool on)
        {
                if (on) { require_available(); validate(); }
                view_.on = on ? 1 : 0;
        }
        bool on() const { return view_.on != 0; }
        // the kernels' view: nullptr while the menu is off (the tracking objective runs)
        const Buffers<T>* view_ptr() const { return view_.on ? &view_ : nullptr; }
        void clear()
        {
                view_ = empty_buffers<T>();
                for (int f = 0; f < NUM_FIELDS; ++f) { set_[f] = false; }
                h_act_.clear();
        }

        // u_des = g(q_ref) per knot (shared across the batch): count = KNOT_POINTS * NQ. Fixed base only:
        // on a floating base the actuated part of g(q) is not a contact-consistent equilibrium torque.
        void gravity_target(const T* h_q_ref, size_t count, const grid::robotModel<T>* d_robotModel)
        {
                require_available();
                if (tangent()) { throw std::invalid_argument("cost menu: the gravity target is defined for fixed-base plants only"); }
#if GATO_MENU_AVAILABLE
                constexpr int NQ = gato::plant::NQ;
                constexpr int NU = gato::plant::NU;
                if (count != (size_t)KNOT_POINTS * NQ) { throw std::invalid_argument("cost-menu gravity target: expected " + std::to_string((size_t)KNOT_POINTS * NQ) + " entries (KNOT_POINTS x NQ), got " + std::to_string(count)); }
                upload(d_q_ref_, cap_q_ref_, h_q_ref, count);
                ensure(d_[U_DES], cap_[U_DES], (size_t)KNOT_POINTS * NU);
                const size_t smem = grid_plant::GRAVITY_TORQUE_DYNAMIC_SHARED_MEM_BYTES<T>() + (size_t)(NQ + NU) * sizeof(T) + 16;
                static size_t latch = 0;
                opt_in_dynamic_smem(gravity_target_kernel<T>, smem, latch);
                gravity_target_kernel<T><<<KNOT_POINTS, 128, smem>>>(d_[U_DES], d_q_ref_, d_robotModel, gato::plant::GRAVITY<T>());
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
                view_.ptr[U_DES] = d_[U_DES]; view_.bstride[U_DES] = 0u; set_[U_DES] = true;
#else
                (void)h_q_ref; (void)count; (void)d_robotModel;
#endif
        }

        // per-term values along the device trajectory (opt-in readout, never in the solve):
        // h_out is batch x KNOT_POINTS x NUM_TERMS
        void term_values(const T* d_xu_traj_batch, const grid::robotModel<T>* d_robotModel, T* h_out)
        {
                require_available();
                if (!view_.on) { throw std::runtime_error("cost menu: no menu is on (set_cost_menu first)"); }
#if GATO_MENU_AVAILABLE
                const size_t n = (size_t)batch_size_ * KNOT_POINTS * NUM_TERMS;
                ensure(d_terms_, cap_terms_, n);
                const size_t smem = sizeof(T) * ((size_t)gato::plant::NX + gato::plant::NU + gato::plant::menu_cost_smem_ct<T>());
                static size_t latch = 0;
                opt_in_dynamic_smem(menu_term_values_kernel<T>, smem, latch);
                menu_term_values_kernel<T><<<dim3(KNOT_POINTS, batch_size_), 128, smem>>>(d_terms_, d_xu_traj_batch, view_, d_robotModel);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaMemcpy(h_out, d_terms_, n * sizeof(T), cudaMemcpyDeviceToHost));
#else
                (void)d_xu_traj_batch; (void)d_robotModel; (void)h_out;
#endif
        }

        // every term a knot activates has its buffers set (the kernels dereference them without a check)
        void validate() const
        {
                if (h_act_.empty()) { throw std::invalid_argument("cost menu: set the activation table before turning the menu on"); }
                if (!set_[EPS]) { throw std::invalid_argument("cost menu: 'eps' (the activation parameter per term per knot) must be set"); }
                static const int need[NUM_TERMS][4] = {{EE_DES, W, -1, -1}, {X_DES, Q, -1, -1}, {U_DES, R, -1, -1}, {AX, BX, TX, WX},
                                                       {AU, BU, TU, WU}, {R_DES, WO, -1, -1}, {V_DES, WV, -1, -1}, {CF_DES, WCF, -1, -1}};
                static const int bounds[NUM_TERMS][2] = {{-1, -1}, {X_LOWER, X_UPPER}, {UB_LOWER, UB_UPPER}, {LX_LOWER, LX_UPPER},
                                                         {LU_LOWER, LU_UPPER}, {-1, -1}, {V_LOWER, V_UPPER}, {CF_LOWER, CF_UPPER}};
                static const char* term_names[NUM_TERMS] = {"frame_pos", "state", "input", "linear_x", "linear_u", "orientation", "velocity", "frame_pos_2"};
                for (uint32_t k = 0; k < KNOT_POINTS; ++k) {
                        for (int t = 0; t < NUM_TERMS; ++t) {
                                const int a = h_act_[(size_t)k * NUM_TERMS + t];
                                if (a < 0) continue;
                                for (int j = 0; j < 4; ++j) {
                                        const int f = need[t][j];
                                        if (f >= 0 && !set_[f]) { throw std::invalid_argument(std::string("cost menu: term '") + term_names[t] + "' is active at knot " + std::to_string(k) + " but field '" + FIELD_NAMES[f] + "' is not set"); }
                                }
                                if (a >= 2 && bounds[t][0] >= 0 && (!set_[bounds[t][0]] || !set_[bounds[t][1]])) {
                                        throw std::invalid_argument(std::string("cost menu: term '") + term_names[t] + "' uses a barrier activation at knot " + std::to_string(k) + " but its bounds ('" + FIELD_NAMES[bounds[t][0]] + "', '" + FIELD_NAMES[bounds[t][1]] + "') are not set");
                                }
#if GATO_MENU_AVAILABLE
                                if (t == NUM_TERMS - 1 && gato::plant::MENU_CF2 < 0) { throw std::invalid_argument("cost menu: 'frame_pos_2' needs a second baked cost frame (build the plant with cost_frames=[...])"); }
#endif
                        }
                }
        }

private:
        void require_available() const
        {
                if (!available()) { throw std::runtime_error("the cost menu is not available in this module: grid-rbd 0.5.5 emits it for fixed-base plants and free-flyer plants with single-axis joints"); }
                if (batch_size_ == 0) { throw std::runtime_error("cost menu used before the solver was constructed"); }
        }
        template<typename U>
        static void ensure(U*& d, size_t& cap, size_t count) { gato::device_buffer::ensure(d, cap, count); }
        template<typename U>
        static void upload(U*& d, size_t& cap, const U* h, size_t count) { gato::device_buffer::upload(d, cap, h, count); }
        void release()
        {
                for (int f = 0; f < NUM_FIELDS; ++f) { if (d_[f]) { cudaFree(d_[f]); d_[f] = nullptr; } cap_[f] = 0; }
                if (d_act_) { cudaFree(d_act_); d_act_ = nullptr; } cap_act_ = 0;
                if (d_q_ref_) { cudaFree(d_q_ref_); d_q_ref_ = nullptr; } cap_q_ref_ = 0;
                if (d_terms_) { cudaFree(d_terms_); d_terms_ = nullptr; } cap_terms_ = 0;
        }

        uint32_t         batch_size_ = 0;
        T*               d_[NUM_FIELDS] = {};
        size_t           cap_[NUM_FIELDS] = {};
        bool             set_[NUM_FIELDS] = {};
        int*             d_act_ = nullptr;
        size_t           cap_act_ = 0;
        T*               d_q_ref_ = nullptr;
        size_t           cap_q_ref_ = 0;
        T*               d_terms_ = nullptr;
        size_t           cap_terms_ = 0;
        std::vector<int> h_act_;
        Buffers<T>       view_ = empty_buffers<T>();
};

}  // namespace menu
}  // namespace gato
