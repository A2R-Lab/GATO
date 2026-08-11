#pragma once
#include <cstdint>
#include "glass.cuh"
#include "constants.h"
#include "settings.h"
#include "dynamics/manifold.cuh"

// ─── Floating-base step/linearization twins over grid_plant:: (CL-3 W3.3) ────
//
// The hand-rolled integrator.cuh is vector-space only; floating base consumes
// the generated grid_plant:: surfaces instead (SE(3) retract step, tangent
// A|B via the dIntegrate blocks). These twins mirror the fixed-base entry
// points' call shape so the kernels branch with one #if.
//
// Scratch: the grid inner surfaces are caller-scratch, but their per-piece
// arena layout is only emitted INSIDE the generated kernels — gato.builder
// parses those recipes into a per-robot gato_abi.cuh (floating robots only;
// ASK 6 in asks_from_gato_2026-08-09.md asks GRiD to emit carve helpers so
// the parser can die). Its presence doubles as the preprocessor-level
// floating fact: fixed-base TUs compile this header to (almost) nothing and
// keep the integrator.cuh path preprocessor-identical.
//
// ⚠ plant_step (value) wraps grid::integrator_device, which is AUTO-ALLOCATING
// (extern __shared__ at arena offset 0) — calling it from a GATO kernel would
// alias our smem carves. The value twins therefore compose the caller-scratch
// grid::load_update_XImats_helpers + grid::integrator_inner directly (the
// ee_pos_cost pattern); only the gradient path goes through the grid_plant
// wrapper (plant_step_gradient_and_value is caller-scratch all the way down).

#if __has_include("gato_abi.cuh")
#include "gato_abi.cuh"
#define GATO_FLOATING_STEP 1
#else
#define GATO_FLOATING_STEP 0
#endif

#if GATO_FLOATING_STEP

static_assert(gato::constants::FLOATING_BASE,
              "gato_abi.cuh present but grid constants say fixed base — stale headers, regen");
static_assert(gato::constants::FC_SIZE == 0,
              "contact-force controls on floating base are not wired yet (CL-3 later wave): "
              "the step twins scatter ACTUATED_SIZE torques only");
#if USE_EXACT_HESSIAN
#error "exact-Hessian (SO-SQP) is fixed-base only for now: fdsva_so's stage-block contraction is not manifold-aware (CL-3 later wave)"
#endif

namespace gato::plant {

// GATO settings.h integrator id -> grid enum. The grid floating gradient path
// supports EULER / SEMI_IMPLICIT_EULER only (GATO's default trapezoidal is
// fixed-base-only); the go2 module build must set INTEGRATOR_TYPE 0 or 1.
template<unsigned INTEGRATOR_TYPE>
__host__ __device__ constexpr grid::IntegratorType grid_integrator()
{
        static_assert(INTEGRATOR_TYPE <= 1,
                      "floating base supports INTEGRATOR_TYPE 0 (euler) / 1 (semi-implicit) only");
        return INTEGRATOR_TYPE == 0 ? grid::IntegratorType::EULER
                                    : grid::IntegratorType::SEMI_IMPLICIT_EULER;
}

// element counts for the two arenas: the parsed T region + the topology int
// region (carved at the T tail; alignment holds for f32/f64) + the constexpr
// linalg helper region (16B-aligned pad; 0 on non-mathdx builds — note the
// twins do not WIRE a linalg pointer, matching the nullptr plant.cuh passes).
template<typename T>
__host__ __device__ constexpr uint32_t stepGradFloating_TempMemCt()
{
        constexpr uint32_t ints_as_T =
            ((uint32_t)gato_abi::SG_TOPOLOGY_INT_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T);
        constexpr uint32_t linalg_as_T = grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>() > 0
            ? (uint32_t)((grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T)) : 0u;
        return (uint32_t)gato_abi::SG_T_COUNT + ints_as_T + linalg_as_T;
}

template<typename T>
__host__ __device__ constexpr uint32_t stepValueFloating_TempMemCt()
{
        constexpr uint32_t ints_as_T =
            ((uint32_t)gato_abi::ST_TOPOLOGY_INT_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T);
        constexpr uint32_t linalg_as_T = grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>() > 0
            ? (uint32_t)((grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T)) : 0u;
        return (uint32_t)gato_abi::ST_T_COUNT + ints_as_T + linalg_as_T;
}

// build [x(NQ+NV); pad; u_full(NV)] in the arena's q_qd_u slot: grid takes a
// FULL generalized force (tangent order [base 6; joints]); GATO's control is
// the ACTUATED tail, so the 6 base slots are zeroed. u slot at 2*NQ mirrors
// the generated kernels' q/qd/u packing. No trailing sync — callers barrier.
template<typename T>
__device__ __forceinline__ void loadStateAndFullControl(T* s_q_qd_u, const T* s_x, const T* s_u_act)
{
        constexpr int NXi = (int)gato::constants::XU_STATE_SIZE;  // NQ+NV
        constexpr int NUi = (int)gato::constants::ACTUATED_SIZE;
        T* s_u_full = s_q_qd_u + 2 * grid::NUM_POS;
        for (int i = (int)threadIdx.x; i < NXi + 6 + NUi; i += (int)blockDim.x) {
                if (i < NXi)          s_q_qd_u[i] = s_x[i];
                else if (i < NXi + 6) s_u_full[i - NXi] = static_cast<T>(0);
                else                  s_u_full[i - NXi] = s_u_act[i - NXi - 6];
        }
}

// Linearization twin (fixed-base compute_linearized_dynamics call shape):
// s_xux = STORED [x_k(NQ+NV); u_k(NU); x_{k+1}(NQ+NV)]; outputs the TANGENT
// A (2NV x 2NV), B (2NV x NU, actuated columns of grid's full-force B) and
// the defect c = x_{k+1}^traj ⊟ x_{k+1}^pred (signed tangent, the manifold
// analog of integrator_error's traj - pred). Ends UNSYNCED like the fixed
// twin — the caller barriers before reading.
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ __forceinline__ void compute_linearized_dynamics_floating(
    const T* s_xux, T* s_Ak, T* s_Bk, T* s_ck, T* s_temp, void* d_dynMem_const, T dt, T* d_f_ext = nullptr)
{
        using gato::constants::STATE_SIZE;      // 2*NV (tangent)
        using gato::constants::XU_STATE_SIZE;   // NQ+NV (stored)
        using gato::constants::XU_KNOT_STRIDE;
        constexpr int NUi = (int)gato::constants::ACTUATED_SIZE;

        T*   a      = s_temp;
        T*   s_x    = a + gato_abi::SG_Q_QD_U;   // [q; qd] contiguous; u_full at +2*NQ
        T*   s_dAB  = a + gato_abi::SG_DAB;      // [A(2NV x 2NV) | B_full(2NV x NV)] col-major
        T*   s_xkp1 = a + gato_abi::SG_X_KP1;    // predicted next state (stored, NQ+NV)
        int* s_topo = reinterpret_cast<int*>(a + gato_abi::SG_T_COUNT);

        loadStateAndFullControl<T>(s_x, s_xux, s_xux + XU_STATE_SIZE);
        __syncthreads();

        grid_plant::plant_step_gradient_and_value<T, grid_integrator<INTEGRATOR_TYPE>(),
                                                  /*SCRATCH_IN_SMEM=*/true, /*USE_DA_DF_SPILL=*/false,
                                                  /*MUJOCO_OUTPUT=*/false>(
            s_dAB, s_xkp1, s_x, s_x + 2 * grid::NUM_POS,
            a + gato_abi::SG_DF_DU, a + gato_abi::SG_DC_DU, a + gato_abi::SG_VAF,
            a + gato_abi::SG_MINV, a + gato_abi::SG_QDD, a + gato_abi::SG_Q_ORIG,
            a + gato_abi::SG_QD_ORIG, a + gato_abi::SG_STAGE_GRAD_QDD, a + gato_abi::SG_D_QDD_STAGE,
            a + gato_abi::SG_DINT_Q_6X6, a + gato_abi::SG_DINT_V_6X6, a + gato_abi::SG_XIMATS,
            s_topo, a + gato_abi::SG_TEMP, /*d_workspace=*/nullptr, /*d_temp_spill=*/nullptr,
            (const grid::robotModel<T>*)d_dynMem_const, GRAVITY<T>(), dt, d_f_ext);
        __syncthreads();

        // A = the first 2NV columns (contiguous col-major); B = the ACTUATED
        // columns 6..6+NU-1 of grid's full-force B block.
        for (int i = (int)threadIdx.x; i < STATE_SIZE * STATE_SIZE; i += (int)blockDim.x)
                s_Ak[i] = s_dAB[i];
        for (int i = (int)threadIdx.x; i < STATE_SIZE * NUi; i += (int)blockDim.x) {
                const int r = i % STATE_SIZE, j = i / STATE_SIZE;
                s_Bk[i] = s_dAB[(STATE_SIZE + 6 + j) * STATE_SIZE + r];
        }
        state_difference<T>(s_ck, /*from=*/s_xkp1, /*to=*/s_xux + XU_KNOT_STRIDE);
}

// Value step twin: x_{k+1} = grid integrator(x, u_act, dt), stored format out.
// Caller-scratch composition (load XImats + integrator_inner); ends UNSYNCED.
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ __forceinline__ void sim_step_floating(
    T* s_xkp1, const T* s_x, const T* s_u_act, T* s_temp, void* d_dynMem_const, T dt, T* d_f_ext = nullptr)
{
        T*   a      = s_temp;
        T*   s_qqdu = a + gato_abi::ST_Q_QD_U;
        int* s_topo = reinterpret_cast<int*>(a + gato_abi::ST_T_COUNT);
        const grid::robotModel<T>* d_robotModel = (const grid::robotModel<T>*)d_dynMem_const;

        loadStateAndFullControl<T>(s_qqdu, s_x, s_u_act);
        __syncthreads();
        grid::load_update_XImats_helpers<T>(a + gato_abi::ST_XIMATS, s_qqdu, s_topo,
                                            (grid::robotModel<T>*)d_robotModel, a + gato_abi::ST_TEMP);
        __syncthreads();
        grid::integrator_inner<T, grid_integrator<INTEGRATOR_TYPE>(), /*MINV_F_IN_SMEM=*/true>(
            s_xkp1, s_qqdu, s_qqdu + grid::NUM_POS, s_qqdu + 2 * grid::NUM_POS,
            a + gato_abi::ST_QDD, a + gato_abi::ST_STAGE_QDD, a + gato_abi::ST_STAGE_POINT,
            a + gato_abi::ST_XIMATS, s_topo, d_robotModel, a + gato_abi::ST_TEMP,
            /*d_workspace=*/nullptr, d_f_ext, GRAVITY<T>(), dt);
}

// Merit integrator-error twin: || x_{k+1}^traj ⊟ integrator(x_k, u_k) ||_1
// over the tangent. s_xuk = STORED [x_k; u_k; ...]; s_xkp1_traj = stored next
// state. Returns the block-reduced scalar (ends on a barrier like the fixed
// twin's reduce).
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ T compute_integrator_error_floating(
    const T* s_xuk, const T* s_xkp1_traj, T* s_temp, void* d_dynMem_const, T dt, T* d_f_ext = nullptr)
{
        using gato::constants::STATE_SIZE;
        using gato::constants::XU_STATE_SIZE;
        T* a       = s_temp;
        T* s_xkp1  = a + gato_abi::ST_X_KP1;
        T* s_err   = a + gato_abi::ST_STAGE_POINT;  // stages are dead post-step; 111 >= 2*NV

        sim_step_floating<T, INTEGRATOR_TYPE>(s_xkp1, s_xuk, s_xuk + XU_STATE_SIZE,
                                              s_temp, d_dynMem_const, dt, d_f_ext);
        __syncthreads();
        state_difference<T>(s_err, /*from=*/s_xkp1, /*to=*/s_xkp1_traj);
        __syncthreads();
        for (int i = (int)threadIdx.x; i < STATE_SIZE; i += (int)blockDim.x)
                s_err[i] = abs(s_err[i]);
        __syncthreads();
        ::glass::reduce<T>(STATE_SIZE, s_err);
        __syncthreads();
        return s_err[0];
}

}  // namespace gato::plant

#endif  // GATO_FLOATING_STEP
