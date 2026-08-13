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
// Scratch: the grid inner surfaces are caller-scratch, and since 2026-08-13
// the per-piece arena layout comes from the EMITTED carve structs
// (grid::integrator_arena / grid::integrator_du_arena — GRiD ASK 6; generated
// from the same buffer lists the kernels use, so drift is impossible by
// construction; the old gato.builder._write_gato_abi comment-regex parser is
// deleted). ⚠ The du struct mirrors the robot's TIER_SHARED SPILL RUNG: on a
// big-enough robot s_D_qdd_stage / s_dAB leave the struct for workspace bands
// and this header stops compiling — loud, by design (wire the band pointers
// then). The floating fact is the emitted GRID_PLANT_HAS_TANGENT_STATE_COST
// marker (floating-base, non-spherical robots only — grid.cuh is included via
// settings.h before any kernel header, so it is visible here): fixed-base TUs
// compile this header to (almost) nothing and keep the integrator.cuh path
// preprocessor-identical.
//
// ⚠ plant_step (value) wraps grid::integrator_device, which is AUTO-ALLOCATING
// (extern __shared__ at arena offset 0) — calling it from a GATO kernel would
// alias our smem carves. The value twins therefore compose the caller-scratch
// grid::load_update_XImats_helpers + grid::integrator_inner directly (the
// ee_pos_cost pattern); only the gradient path goes through the grid_plant
// wrapper (plant_step_gradient_and_value is caller-scratch all the way down).

#ifdef GRID_PLANT_HAS_TANGENT_STATE_COST
#define GATO_FLOATING_STEP 1
#else
#define GATO_FLOATING_STEP 0
#endif

#if GATO_FLOATING_STEP

static_assert(gato::constants::FLOATING_BASE,
              "GRID_PLANT_HAS_TANGENT_STATE_COST emitted but grid constants say fixed base — stale headers, regen");
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

// element counts for the two arenas: the emitted TIER_SHARED totals
// (T region + topology ints + linalg pad, f32-conservative — the same legacy
// _COUNT constants the auto-allocating device fns size extern __shared__
// with; the byte sizers are not constexpr, these const ints are). The carve
// structs' GRID_CUDA_DEBUG_LAYOUT asserts pin them against the layouts.
template<typename T>
__host__ __device__ constexpr uint32_t stepGradFloating_TempMemCt()
{
        return (uint32_t)grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_COUNT;
}

template<typename T>
__host__ __device__ constexpr uint32_t stepValueFloating_TempMemCt()
{
        return (uint32_t)grid::INTEGRATOR_DYNAMIC_SHARED_MEM_COUNT;
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

        // emitted carve (mirrors the integrator_with_gradient kernel's
        // TIER_SHARED layout; s_q_qd_u packs [q; qd] with u_full at +2*NQ)
        const auto a = grid::integrator_du_arena<T>::carve(s_temp);

        loadStateAndFullControl<T>(a.s_q_qd_u, s_xux, s_xux + XU_STATE_SIZE);
        __syncthreads();

        grid_plant::plant_step_gradient_and_value<T, grid_integrator<INTEGRATOR_TYPE>(),
                                                  /*SCRATCH_IN_SMEM=*/true, /*USE_DA_DF_SPILL=*/false,
                                                  /*MUJOCO_OUTPUT=*/false>(
            a.s_dAB, a.s_x_kp1, a.s_q_qd_u, a.s_q_qd_u + 2 * grid::NUM_POS,
            a.s_df_du, a.s_dc_du, a.s_vaf,
            a.s_Minv, a.s_qdd, a.s_q_orig,
            a.s_qd_orig, a.s_stage_grad_qdd, a.s_D_qdd_stage,
            a.s_dInt_q_6x6, a.s_dInt_v_6x6, a.s_XImats,
            a.s_topology_helpers, a.s_temp, /*d_workspace=*/nullptr, /*d_temp_spill=*/nullptr,
            (const grid::robotModel<T>*)d_dynMem_const, GRAVITY<T>(), dt, d_f_ext);
        __syncthreads();

        // A = the first 2NV columns (contiguous col-major); B = the ACTUATED
        // columns 6..6+NU-1 of grid's full-force B block.
        for (int i = (int)threadIdx.x; i < STATE_SIZE * STATE_SIZE; i += (int)blockDim.x)
                s_Ak[i] = a.s_dAB[i];
        for (int i = (int)threadIdx.x; i < STATE_SIZE * NUi; i += (int)blockDim.x) {
                const int r = i % STATE_SIZE, j = i / STATE_SIZE;
                s_Bk[i] = a.s_dAB[(STATE_SIZE + 6 + j) * STATE_SIZE + r];
        }
        state_difference<T>(s_ck, /*from=*/a.s_x_kp1, /*to=*/s_xux + XU_KNOT_STRIDE);
}

// Value step twin: x_{k+1} = grid integrator(x, u_act, dt), stored format out.
// Caller-scratch composition (load XImats + integrator_inner); ends UNSYNCED.
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ __forceinline__ void sim_step_floating(
    T* s_xkp1, const T* s_x, const T* s_u_act, T* s_temp, void* d_dynMem_const, T dt, T* d_f_ext = nullptr)
{
        const auto a = grid::integrator_arena<T>::carve(s_temp);
        const grid::robotModel<T>* d_robotModel = (const grid::robotModel<T>*)d_dynMem_const;

        loadStateAndFullControl<T>(a.s_q_qd_u, s_x, s_u_act);
        __syncthreads();
        grid::load_update_XImats_helpers<T>(a.s_XImats, a.s_q_qd_u, a.s_topology_helpers,
                                            (grid::robotModel<T>*)d_robotModel, a.s_temp);
        __syncthreads();
        grid::integrator_inner<T, grid_integrator<INTEGRATOR_TYPE>(), /*MINV_F_IN_SMEM=*/true>(
            s_xkp1, a.s_q_qd_u, a.s_q_qd_u + grid::NUM_POS, a.s_q_qd_u + 2 * grid::NUM_POS,
            a.s_qdd, a.s_stage_qdd, a.s_stage_point,
            a.s_XImats, a.s_topology_helpers, d_robotModel, a.s_temp,
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
        const auto a = grid::integrator_arena<T>::carve(s_temp);
        T* s_xkp1  = a.s_x_kp1;
        T* s_err   = a.s_stage_point;  // stages are dead post-step; 111 >= 2*NV

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
