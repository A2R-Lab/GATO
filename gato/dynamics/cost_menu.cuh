#pragma once
// Cost-menu buffers: the per-knot data of grid_plant::tracking_cost_menu
// (grid-rbd 0.5.5). The solver host (bsqp/cost_menu_host.cuh) owns the device
// arrays and fills this view; the setup-KKT and merit kernels read it through
// the plant adapters (dynamics/plant.cuh, menu_cost_value / menu_cost_grad_hess)
// when the menu is on. Every field is a per-knot array, either one copy shared
// across the batch (batch stride 0) or one copy per solve (batch stride
// KNOT_POINTS times the field's per-knot size).
#include <cstdint>
#include <cstddef>

namespace gato {
namespace menu {

enum Field : int {
        X_DES = 0, Q, X_LOWER, X_UPPER,                 // state term: target, weight, barrier bounds
        U_DES, R, UB_LOWER, UB_UPPER,                   // input term
        EE_DES, W,                                      // frame position on cost frame 0
        AX, BX, TX, WX, LX_LOWER, LX_UPPER,             // linear feature on x: A (NX x NX column-major), b, target, weight, bounds
        AU, BU, TU, WU, LU_LOWER, LU_UPPER,             // linear feature on u (NU x NU)
        R_DES, WO,                                      // frame orientation on cost frame 0: R_des (9, column-major), weight (3)
        V_DES, WV, V_LOWER, V_UPPER,                    // frame twist on cost frame 0 (6: linear, angular)
        CF_DES, WCF, CF_LOWER, CF_UPPER,                // second frame position on cost frame 1
        EPS,                                            // activation parameter per term (NUM_TERMS)
        NUM_FIELDS
};
constexpr int NUM_TERMS = 8;   // GRID_PLANT_MENU_TERMS, order GRID_PLANT_MENU_{FRAME_POS, STATE, INPUT, LINEAR_X, LINEAR_U, ORIENTATION, VELOCITY, FRAME_POS_2}

// field names in Field order (the host and Python layers address fields by name)
inline const char* const FIELD_NAMES[NUM_FIELDS] = {
        "x_des", "Q", "x_lower", "x_upper", "u_des", "R", "ub_lower", "ub_upper", "ee_des", "W",
        "Ax", "bx", "tx", "Wx", "lx_lower", "lx_upper", "Au", "bu", "tu", "Wu", "lu_lower", "lu_upper",
        "R_des", "Wo", "v_des", "Wv", "v_lower", "v_upper", "cf_des", "Wcf", "cf_lower", "cf_upper", "eps"};

// the per-knot sizes of a plant's menu: x_stored is the stored state width (NQ + NV),
// x_tangent the state block width the weights and bounds index (2 NV on a floating
// base, equal to x_stored on a fixed base), nu the preset's actuated input width, nlx
// the variable width of the linear x feature (x on a fixed base, [q[7:]; qd] on a
// floating base). gato::plant::menu_dims() fills it from the generated constants.
struct Dims { uint32_t x_stored; uint32_t x_tangent; uint32_t nu; uint32_t nlx; };

// per-knot element count of a field; the linear features are full-rank (MX = nlx rows, MU = nu rows)
__host__ __device__ constexpr uint32_t field_size(int f, Dims d)
{
        return (f == X_DES) ? d.x_stored
             : (f == Q || f == X_LOWER || f == X_UPPER) ? d.x_tangent
             : (f == BX || f == TX || f == WX || f == LX_LOWER || f == LX_UPPER) ? d.nlx
             : (f == U_DES || f == R || f == UB_LOWER || f == UB_UPPER || f == BU || f == TU || f == WU || f == LU_LOWER || f == LU_UPPER) ? d.nu
             : (f == AX) ? d.nlx * d.nlx
             : (f == AU) ? d.nu * d.nu
             : (f == R_DES) ? 9u
             : (f == V_DES || f == WV || f == V_LOWER || f == V_UPPER) ? 6u
             : (f == EPS) ? static_cast<uint32_t>(NUM_TERMS)
             : 3u;
}

template<typename T>
struct Buffers {
        const T*   ptr[NUM_FIELDS];      // nullptr = the field is not set
        uint32_t   bstride[NUM_FIELDS];  // elements per solve (0 = one copy shared across the batch)
        const int* activation;           // KNOT_POINTS x NUM_TERMS: GRID_PLANT_ACTIVATION_* per term per knot (shared across the batch)
        int        on;                   // 1 = the menu replaces the tracking objective

        __host__ __device__ __forceinline__ const T* at(int f, uint32_t solve_idx, uint32_t knot_idx, uint32_t size) const
        {
                const T* p = ptr[f];
                return p ? p + static_cast<size_t>(solve_idx) * bstride[f] + static_cast<size_t>(knot_idx) * size : nullptr;
        }
};

template<typename T>
__host__ inline Buffers<T> empty_buffers()
{
        Buffers<T> m;
        for (int f = 0; f < NUM_FIELDS; ++f) { m.ptr[f] = nullptr; m.bstride[f] = 0u; }
        m.activation = nullptr;
        m.on = 0;
        return m;
}

}  // namespace menu
}  // namespace gato
