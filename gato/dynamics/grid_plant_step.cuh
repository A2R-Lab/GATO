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
//
// ─── Contact-force controls on the floating base (Wave F, 2026-09-20) ─────
//
// On GATO_CONTACT_FORCES builds the control slice is [tau(ACTUATED); fc(FC)]
// with fc = one world-aligned wrench [n; f] per baked contact frame (the go2
// header bakes the four *_foot_joint frames). The twins consume it exactly
// like the arms' CL-3a adapter, with the composition moved to the OUTPUT
// side of the grid step (the grid step's own internals stay untouched):
//   value:    f_ext = grid::f_ext_body(q, fc) (+ the known P4.6 band) is
//             handed to the integrator as d_f_ext — the mapped wrench is the
//             same array the arms pass.
//   gradient: the grid step returns the tangent A (2NV x 2NV) and the FULL
//             generalized-force B (2NV x NV, B_full = chain ∘ Minv). The
//             integrator chain is linear in qdd, so
//               B_fc        = B_full · (dqdd/dtau)^-1 · dqdd/dfc
//                           = B_full · (-dtau/dfext · dfext/dfc)        (2NV x FC)
//               A[:, dq]   += B_full · (-dtau/dfext · dfext/dq)         (the CL-3a
//                             W2 chain term: the world wrench rotates with the
//                             body; linear in fc ⇒ identically 0 at fc = 0)
//             using the emitted f_ext_gradient_jacobianT (-J^T, at the XImats
//             the step loaded for q_k: single-stage SI-Euler never mutates
//             s_q) and the f_ext_body_jacobian_{dfc,dq} inners.
// Scratch budget (the device opt-in smem ceiling is ~99 KB and the go2
// setup_kkt carve already sits at 88 KB): only the mapped wrench (6*NB) is
// APPENDED after the emitted arena — it must survive the step, which
// overwrites every arena piece. Everything else lives in arena pieces that
// are DEAD at the time: XmatsHom for the wrench map goes in the arena's temp
// pool before the step; the post-step Jacobian composition re-carves that
// same pool (rebuilding XmatsHom — cheap). The pool's extent is only known
// to the emitted carve, so the twin checks it at runtime and traps loudly
// (never silently overruns): go2 has 9612 floats against 7044 needed. The
// jacobian inners' own scratch is DYNAMICS_XI_T_COUNT wide (the emitted
// contract on every generated robot: 432/504/936). fc = 0 on an fc build is
// bitwise the default module's linearization (x - 0 == x).

#ifdef GRID_PLANT_HAS_TANGENT_STATE_COST
#define GATO_FLOATING_STEP 1
#else
#define GATO_FLOATING_STEP 0
#endif

#if GATO_FLOATING_STEP

static_assert(gato::constants::FLOATING_BASE,
              "GRID_PLANT_HAS_TANGENT_STATE_COST emitted but grid constants say fixed base — stale headers, regen");
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

// ─── fc scratch layout (appended after the emitted arenas) ──────────────────
namespace fc_floating {
        constexpr int NVi   = grid::NUM_VEL;
        constexpr int FEXT  = 6 * grid::NUM_BODIES;                        // per-body wrench array
        constexpr int FC    = (int)gato::constants::FC_SIZE;               // wrench slots (0 on default builds)
        constexpr int XHOM  = grid::XHOM_T_COUNT;
        constexpr int JTEMP = grid::DYNAMICS_XI_T_COUNT;                   // jacobianT / f_ext_body inner scratch
        static_assert(16 * grid::NUM_JOINTS <= JTEMP,
                      "f_ext_body inners keep s_Xworld (16*NUM_JOINTS) in a DYNAMICS_XI_T_COUNT-wide scratch");
        // appended after the emitted arena (both twins): the mapped wrench only
        constexpr int VALUE_COUNT = FC > 0 ? FEXT : 0;
        constexpr int GRAD_COUNT  = FC > 0 ? FEXT : 0;
        // pre-step use of the arena temp pool: XmatsHom | inner scratch (s_Xworld)
        constexpr int PRE_POOL_COUNT = XHOM + JTEMP;
        // post-step re-carve of the arena temp pool (gradient twin):
        //   XmatsHom | dtau_dfext (NV x 6NB) | dfext_dfc (6NB x FC)
        //   | G = dtau_dfext·dfext_dfc (NV x FC) | dfext_dq (6NB x NV)
        //   | dtau_dq_corr (NV x NV) | inner scratch
        constexpr int POST_POOL_COUNT = XHOM + NVi * FEXT + FEXT * FC + NVi * FC + FEXT * NVi + NVi * NVi + JTEMP;
        constexpr int POOL_COUNT = PRE_POOL_COUNT > POST_POOL_COUNT ? PRE_POOL_COUNT : POST_POOL_COUNT;
}  // namespace fc_floating

#if GATO_CONTACT_FORCES
// The emitted carve is the only place that knows the temp pool's extent —
// verify it once per call site, loudly (a silent overrun would corrupt the
// topology ints that follow the pool).
template<typename T>
__device__ __forceinline__ void check_fc_pool(const T* s_pool, const int* s_after_pool, int needed)
{
        const long avail = (long)(((const char*)s_after_pool - (const char*)s_pool) / (long)sizeof(T));
        if (avail < needed) {
                if (threadIdx.x == 0) {
                        printf("GATO fc-floating: grid arena temp pool too small (%ld < %d elements) — the fc "
                               "jacobian re-carve does not fit this robot's rung; grow the pool upstream\n", avail, needed);
                }
                __trap();
        }
}
#endif

// element counts for the two arenas: the emitted TIER_SHARED totals
// (T region + topology ints + linalg pad, f32-conservative — the same legacy
// _COUNT constants the auto-allocating device fns size extern __shared__
// with; the byte sizers are not constexpr, these const ints are). The carve
// structs' GRID_CUDA_DEBUG_LAYOUT asserts pin them against the layouts. fc
// builds append their scratch after the arena (fc_floating above).
template<typename T>
__host__ __device__ constexpr uint32_t stepGradFloating_TempMemCt()
{
        return (uint32_t)grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_COUNT + (uint32_t)fc_floating::GRAD_COUNT;
}

template<typename T>
__host__ __device__ constexpr uint32_t stepValueFloating_TempMemCt()
{
        return (uint32_t)grid::INTEGRATOR_DYNAMIC_SHARED_MEM_COUNT + (uint32_t)fc_floating::VALUE_COUNT;
}

// build [x(NQ+NV); pad; u_full(NV)] in the arena's q_qd_u slot: grid takes a
// FULL generalized force (tangent order [base 6; joints]); GATO's control is
// the ACTUATED tail, so the 6 base slots are zeroed. u slot at 2*NQ mirrors
// the generated kernels' q/qd/u packing. No trailing sync — callers barrier.
// s_u is the CONTROL_SIZE-wide control slice; only its actuated head is read
// here (the fc tail goes through f_ext, below).
template<typename T>
__device__ __forceinline__ void load_state_and_full_control(T* s_q_qd_u, const T* s_x, const T* s_u)
{
        constexpr int NXi = (int)gato::constants::XU_STATE_SIZE;  // NQ+NV
        constexpr int NUi = (int)gato::constants::ACTUATED_SIZE;
        T* s_u_full = s_q_qd_u + 2 * grid::NUM_POS;
        for (int i = (int)threadIdx.x; i < NXi + 6 + NUi; i += (int)blockDim.x) {
                if (i < NXi)          s_q_qd_u[i] = s_x[i];
                else if (i < NXi + 6) s_u_full[i - NXi] = static_cast<T>(0);
                else                  s_u_full[i - NXi] = s_u[i - NXi - 6];
        }
}

#if GATO_CONTACT_FORCES
// fc tail of the control -> joint-local per-body wrenches (Featherstone
// [angular; linear] about each joint origin), ADDING the known external band
// (P4.6 d_f_ext) if present. XmatsHom is built here (and kept: the gradient
// twin's jacobian inners reuse it after the step). s_temp = an arena scratch
// that is dead at call time (>= 38 for the sincos + 16*NUM_JOINTS for
// s_Xworld). Ends on a barrier.
template<typename T>
__device__ void build_contact_fext_floating(T* s_fext, const T* s_fc, const T* s_q, T* s_XmatsHom,
                                            int* s_topology_helpers, T* s_temp,
                                            const grid::robotModel<T>* d_robotModel, const T* d_f_ext_band)
{
        grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
        __syncthreads();
        grid::f_ext_body_inner<T>(s_fext, s_fc, s_q, s_XmatsHom, s_topology_helpers, s_temp,
                                  /*d_workspace*/nullptr, /*s_linalg_smem*/nullptr);
        __syncthreads();
        if (d_f_ext_band != nullptr) {
                for (int i = (int)threadIdx.x; i < fc_floating::FEXT; i += (int)blockDim.x) {
                        s_fext[i] += d_f_ext_band[i];
                }
                __syncthreads();
        }
}
#endif

// Linearization twin (fixed-base compute_linearized_dynamics call shape):
// s_xux = STORED [x_k(NQ+NV); u_k(NU); x_{k+1}(NQ+NV)]; outputs the TANGENT
// A (2NV x 2NV), B (2NV x CONTROL_SIZE: actuated columns of grid's full-force
// B, then the fc columns on fc builds) and the defect
// c = x_{k+1}^traj ⊟ x_{k+1}^pred (signed tangent, the manifold analog of
// integrator_error's traj - pred). Ends UNSYNCED like the fixed twin — the
// caller barriers before reading.
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ __forceinline__ void compute_linearized_dynamics_floating(
    const T* s_xux, T* s_Ak, T* s_Bk, T* s_ck, T* s_temp, void* d_dynMem_const, T dt, T* d_f_ext = nullptr)
{
        using gato::constants::STATE_SIZE;      // 2*NV (tangent)
        using gato::constants::XU_STATE_SIZE;   // NQ+NV (stored)
        using gato::constants::XU_KNOT_STRIDE;
        constexpr int NUi = (int)gato::constants::ACTUATED_SIZE;
        constexpr int TS  = (int)STATE_SIZE;
        const grid::robotModel<T>* d_robotModel = (const grid::robotModel<T>*)d_dynMem_const;

        // emitted carve (mirrors the integrator_with_gradient kernel's
        // TIER_SHARED layout; s_q_qd_u packs [q; qd] with u_full at +2*NQ)
        const auto a = grid::integrator_du_arena<T>::carve(s_temp);

        load_state_and_full_control<T>(a.s_q_qd_u, s_xux, s_xux + XU_STATE_SIZE);
        __syncthreads();

#if GATO_CONTACT_FORCES
        namespace F = fc_floating;   // (plant.cuh has its own gato::plant::FC — keep these qualified)
        constexpr int NVi = F::NVi, FEXT = F::FEXT, FC = F::FC;
        check_fc_pool<T>(a.s_temp, a.s_topology_helpers, F::POOL_COUNT);
        T* s_fext     = s_temp + grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_COUNT;  // appended: survives the step
        const T* s_fc = s_xux + XU_STATE_SIZE + NUi;                           // the control's fc tail
        // pre-step: XmatsHom + the wrench map's scratch in the (dead) temp pool
        build_contact_fext_floating<T>(s_fext, s_fc, s_xux, /*s_XmatsHom=*/a.s_temp, a.s_topology_helpers,
                                       a.s_temp + F::XHOM, d_robotModel, d_f_ext);
        d_f_ext = s_fext;
#endif

        grid_plant::plant_step_gradient_and_value<T, grid_integrator<INTEGRATOR_TYPE>(),
                                                  /*SCRATCH_IN_SMEM=*/true, /*USE_DA_DF_SPILL=*/false,
                                                  /*MUJOCO_OUTPUT=*/false>(
            a.s_dAB, a.s_x_kp1, a.s_q_qd_u, a.s_q_qd_u + 2 * grid::NUM_POS,
            a.s_df_du, a.s_dc_du, a.s_vaf,
            a.s_Minv, a.s_qdd, a.s_q_orig,
            a.s_qd_orig, a.s_stage_grad_qdd, a.s_D_qdd_stage,
            a.s_dInt_q_6x6, a.s_dInt_v_6x6, a.s_XImats,
            a.s_topology_helpers, a.s_temp, /*d_workspace=*/nullptr, /*d_temp_spill=*/nullptr,
            d_robotModel, GRAVITY<T>(), dt, d_f_ext);
        __syncthreads();

        // A = the first 2NV columns (contiguous col-major); B = the ACTUATED
        // columns 6..6+NU-1 of grid's full-force B block.
        for (int i = (int)threadIdx.x; i < TS * TS; i += (int)blockDim.x)
                s_Ak[i] = a.s_dAB[i];
        for (int i = (int)threadIdx.x; i < TS * NUi; i += (int)blockDim.x) {
                const int r = i % TS, j = i / TS;
                s_Bk[i] = a.s_dAB[(TS + 6 + j) * TS + r];
        }

#if GATO_CONTACT_FORCES
        // post-step re-carve of the (now dead) temp pool
        T* s_XmatsHom   = a.s_temp;
        T* s_dtau_dfext = s_XmatsHom + F::XHOM;          // NV x 6NB, [v + NV*(6i+r)]
        T* s_dfext_dfc  = s_dtau_dfext + NVi * FEXT;     // 6NB x FC, [row + 6NB*col]
        T* s_G          = s_dfext_dfc + FEXT * FC;       // NV x FC,  [k + NV*c]  = dtau_dfext·dfext_dfc
        T* s_dfext_dq   = s_G + NVi * FC;                // 6NB x NV, [row + 6NB*v]
        T* s_dtau_dq    = s_dfext_dq + FEXT * NVi;       // NV x NV,  [k + NV*j]  = dtau_dfext·dfext_dq
        T* s_fc_temp    = s_dtau_dq + NVi * NVi;         // F::JTEMP (sincos / s_Xworld / jacobianT slab)
        __syncthreads();   // the A/B copies above read s_dAB only; the pool writes below need the step done
        // XmatsHom again (the step reused the pool), then dtau/dfext = -J^T
        // (columns = the local body Jacobians) at the XImats the step loaded
        // for q_k; the dq inner needs it too, so it is computed before either
        // jacobian.
        grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, a.s_topology_helpers, s_xux, d_robotModel, s_fc_temp);
        __syncthreads();
        grid::f_ext_gradient_jacobianT_inner<T>(s_dtau_dfext, s_xux, a.s_XImats, a.s_topology_helpers, s_fc_temp);
        __syncthreads();
        grid::f_ext_body_jacobian_dfc_inner<T>(s_dfext_dfc, s_xux, s_XmatsHom, a.s_topology_helpers, s_fc_temp,
                                               /*d_workspace*/nullptr, /*s_linalg_smem*/nullptr);
        __syncthreads();
        for (int ind = (int)threadIdx.x; ind < NVi * FC; ind += (int)blockDim.x) {   // G = dtau_dfext · dfext_dfc
                const int k = ind % NVi, c = ind / NVi;
                T acc = static_cast<T>(0);
                for (int i = 0; i < FEXT; i++) { acc += s_dtau_dfext[k + NVi * i] * s_dfext_dfc[i + FEXT * c]; }
                s_G[ind] = acc;
        }
        __syncthreads();
        // dfext/dq at fixed fc (linear in fc: identically 0 at fc = 0)
        grid::f_ext_body_jacobian_dq_inner<T>(s_dfext_dq, s_fc, s_dtau_dfext, s_xux, s_XmatsHom, a.s_topology_helpers,
                                              s_fc_temp, /*d_workspace*/nullptr, /*s_linalg_smem*/nullptr);
        __syncthreads();
        for (int ind = (int)threadIdx.x; ind < NVi * NVi; ind += (int)blockDim.x) {  // dtau_dq_corr = dtau_dfext · dfext_dq
                const int k = ind % NVi, j = ind / NVi;
                T acc = static_cast<T>(0);
                for (int i = 0; i < FEXT; i++) { acc += s_dtau_dfext[k + NVi * i] * s_dfext_dq[i + FEXT * j]; }
                s_dtau_dq[ind] = acc;
        }
        __syncthreads();
        // B_fc = -B_full · G  (B_full[r, k] = dAB[(TS + k)*TS + r], the NV full-force columns)
        for (int ind = (int)threadIdx.x; ind < TS * FC; ind += (int)blockDim.x) {
                const int r = ind % TS, c = ind / TS;
                T acc = static_cast<T>(0);
                for (int k = 0; k < NVi; k++) { acc += a.s_dAB[(TS + k) * TS + r] * s_G[k + NVi * c]; }
                s_Bk[(NUi + c) * TS + r] = -acc;
        }
        // A[:, dq] -= B_full · dtau_dq_corr   (the W2 chain term)
        for (int ind = (int)threadIdx.x; ind < TS * NVi; ind += (int)blockDim.x) {
                const int r = ind % TS, j = ind / TS;
                T acc = static_cast<T>(0);
                for (int k = 0; k < NVi; k++) { acc += a.s_dAB[(TS + k) * TS + r] * s_dtau_dq[k + NVi * j]; }
                s_Ak[j * TS + r] -= acc;
        }
#endif
        state_difference<T>(s_ck, /*from=*/a.s_x_kp1, /*to=*/s_xux + XU_KNOT_STRIDE);
}

// Value step twin: x_{k+1} = grid integrator(x, u, dt), stored format out.
// s_u is the CONTROL_SIZE-wide control slice ([tau; fc] on fc builds).
// Caller-scratch composition (load XImats + integrator_inner); ends UNSYNCED.
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ __forceinline__ void sim_step_floating(
    T* s_xkp1, const T* s_x, const T* s_u, T* s_temp, void* d_dynMem_const, T dt, T* d_f_ext = nullptr)
{
        const auto a = grid::integrator_arena<T>::carve(s_temp);
        const grid::robotModel<T>* d_robotModel = (const grid::robotModel<T>*)d_dynMem_const;

        load_state_and_full_control<T>(a.s_q_qd_u, s_x, s_u);
        __syncthreads();
#if GATO_CONTACT_FORCES
        // the wrench map runs in the (dead) temp pool; only the mapped wrench
        // is appended — it must survive the integrator, which reuses the pool
        check_fc_pool<T>(a.s_temp, a.s_topology_helpers, fc_floating::PRE_POOL_COUNT);
        T* s_fext = s_temp + grid::INTEGRATOR_DYNAMIC_SHARED_MEM_COUNT;
        build_contact_fext_floating<T>(s_fext, s_u + gato::constants::ACTUATED_SIZE, s_x, /*s_XmatsHom=*/a.s_temp,
                                       a.s_topology_helpers, a.s_temp + fc_floating::XHOM, d_robotModel, d_f_ext);
        d_f_ext = s_fext;
#endif
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
