#pragma once

#include <stdio.h>
#include "grid.cuh"
#include "glass.cuh"
#include "settings.h"
#include "utils/linalg.cuh"

// Per-robot ABI aliases (generated for FLOATING robots only — see
// builder._write_gato_abi). The fixed-target EE-pose inners are suffixed with
// the EE frame name; the vendored arms all use "EE", which the fallback keeps.
#if __has_include("gato_abi.cuh")
#include "gato_abi.cuh"
#endif
#ifndef GATO_GRID_EE_POSE_INNER
#define GATO_GRID_EE_POSE_INNER end_effector_pose_inner_EE
#define GATO_GRID_EE_POSE_GRAD_INNER end_effector_pose_gradient_inner_EE
#endif

using namespace sqp;

// Generic plant adapter: everything robot-specific comes from the generated
// grid.cuh (dimensions, dynamics, grid_plant costs) and the generated
// limits.cuh (URDF <limit> tables). Both are resolved through the per-robot
// include dir that CMake adds for each (plant, N) module — this file is
// robot-agnostic and shared by all plants.

namespace gato {
namespace plant {

        // Dimension aliases (the generated grid.cuh exposes NUM_JOINTS / NUM_POS /
        // NUM_VEL / NUM_EES). ⚠ grid::NUM_JOINTS == NUM_POS (= nq), NOT nv —
        // on floating base (CL-3) nq = nv + 1 and the two must not be mixed:
        //   NQ = configuration size (q, with quaternion) — q-vector shapes
        //   NV = tangent/velocity size — ALL derivative shapes (dqdd, Minv,
        //        dtau_dq, A|B columns) are NV-based, even where this file
        //        historically wrote NQ (equal on fixed base; the floating
        //        relabel of the interior uses is CL-3 W3.3/W3.4 work).
        inline constexpr int NQ          = grid::NUM_POS;          // configuration size
        inline constexpr int NV          = grid::NUM_VEL;          // tangent / velocity size
        inline constexpr int NX          = grid::NUM_POS + grid::NUM_VEL;  // stored state [q; qd]
        // NU = the ACTUATED torque width (cost/control code is NU-wide).
        // Floating base: grid's plant surfaces take a full NV generalized
        // force; GATO scatters the NU actuated slots and zeros the 6 base rows.
        inline constexpr int NU          = (grid::NUM_POS == grid::NUM_VEL)
                                               ? grid::NUM_JOINTS : (grid::NUM_VEL - 6);
        inline constexpr int NEE         = grid::NUM_EES;          // end-effector count
        inline constexpr int EE_POS_SIZE = 6;                      // pose size (xyz + orientation)

        // Shared-memory extents of the grid:: inner buffers this adapter carves by hand.
        // XImats is per-BODY (72*NUM_BODIES == 72*NUM_JOINTS on fixed base, where the
        // hand-carved paths live; floating NUM_JOINTS is nq and must not size it) —
        // use the emitted count.
        inline constexpr int XIMATS_COUNT = grid::DYNAMICS_XI_T_COUNT;  // X + I (6x6 each) per body
        inline constexpr int VAF_COUNT    = 18 * grid::NUM_BODIES;      // v, a, f (6 each) per body

#if GATO_CONTACT_FORCES
    #if USE_EXACT_HESSIAN
        #error "GATO_CONTACT_FORCES x USE_EXACT_HESSIAN unsupported (fdsva_so is not fc-aware; W2+ item)"
    #endif
        // fc controls on floating base: not wired (the fc scratch below is NQ-sized
        // generalized-dims math and the step twins scatter actuated torques only)
        static_assert(grid::NUM_POS == grid::NUM_VEL,
                      "GATO_CONTACT_FORCES is fixed-base only for now (CL-3 later wave)");
        // CL-3a: NU stays the ACTUATED width (the generated grid_plant cost code is
        // NU-wide); gato::constants::CONTROL_SIZE = NU + FC is the solver-facing width.
        inline constexpr int FC         = 6 * grid::NUM_CONTACT_FRAMES; // wrench slots appended to u
        inline constexpr int FEXT_COUNT = 6 * grid::NUM_BODIES;         // per-body wrench array
        // Persistent fc scratch APPENDED after the FD/FD_DU arenas (inner-call scratch
        // reuses the existing temp region — ID-gradient's need is the high-water mark):
        // XmatsHom + fext + dtau_dfext + dfext_dfc + dtau_dfc  (+ W2: dfext_dq + dtau_dq).
        inline constexpr int FC_PERSIST_COUNT =
            grid::XHOM_T_COUNT + FEXT_COUNT + NQ * FEXT_COUNT + FEXT_COUNT * FC + NQ * FC
            + FEXT_COUNT * NQ + NQ * NQ;
#else
        inline constexpr int FC = 0;
        inline constexpr int FC_PERSIST_COUNT = 0;
#endif

        template<class T>
        __host__ __device__ constexpr T PI()
        {
                return static_cast<T>(3.14159);
        }
        template<class T>
        __host__ __device__ constexpr T GRAVITY()
        {
                // -9.81: grid.cuh applies +g as upward base accel, so a positive scalar
                // yields gravity pointing UP (= -pinocchio). The MPC sim integrates with
                // physical downward gravity (pinocchio -9.81); the solver must match it,
                // else the controller's gravity compensation is inverted (limit cycle).
                return static_cast<T>(-9.81);
        }

        template<class T>
        __host__ __device__ constexpr T JOINT_LIMIT_MARGIN()
        {
                return static_cast<T>(-0.1);
        }

}  // namespace plant
}  // namespace gato

// Per-robot limit tables ({JOINT,VEL,CTRL}_LIMITS_DATA<T>[NQ][2]), generated
// from the URDF <limit> tags. They reference JOINT_LIMIT_MARGIN above.
#include "limits.cuh"

namespace gato {
namespace plant {

        // Limit tables cover the ACTUATED joints only (NU rows): the floating
        // base free-joint has no URDF <limit> tags and its dofs are unbounded.
        // Fixed base: NU == NQ, tables unchanged.
        static_assert(sizeof(JOINT_LIMITS_DATA<float>) == sizeof(float) * 2 * NU,
                      "limits.cuh joint table does not match the actuated joint count");
        static_assert(sizeof(VEL_LIMITS_DATA<float>) == sizeof(float) * 2 * NU,
                      "limits.cuh velocity table does not match the actuated joint count");
        static_assert(sizeof(CTRL_LIMITS_DATA<float>) == sizeof(float) * 2 * NU,
                      "limits.cuh effort table does not match the actuated joint count");

        template<class T>
        __host__ __device__ constexpr const T (&JOINT_LIMITS())[NU][2]
        {
                return JOINT_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&VEL_LIMITS())[NU][2]
        {
                return VEL_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&CTRL_LIMITS())[NU][2]
        {
                return CTRL_LIMITS_DATA<T>;
        }

        template<typename T>
        void* initializeDynamicsConstMem()
        {
                grid::robotModel<T>* d_robotModel = grid::init_robotModel<T>();
                return (void*)d_robotModel;
        }

        template<typename T>
        void freeDynamicsConstMem(void* d_dynMem_const)
        {
                // grid::free_robotModel was removed (folded into close_grid); free the model directly.
                cudaFree((grid::robotModel<T>*)d_dynMem_const);
        }

#if GATO_CONTACT_FORCES
        // CL-3a: map the per-knot contact wrench decision variables (the fc tail of the
        // control slice, world-aligned [n; f] per contact frame) to the body-major
        // per-body wrench array the dynamics inners consume, ADDING the known external
        // band (P4.6 d_f_ext) if present. s_fc_scratch layout: XmatsHom | fext | temp
        // (temp = the caller's ordinary scratch region, reused sequentially — safe:
        // strictly-ordered inner calls, same pattern as contact_debug.cuh).
        template<typename T>
        __device__ void buildContactFext(T* s_fext, const T* s_fc, const T* s_q,
                                         T* s_XmatsHom, T* s_temp,
                                         grid::robotModel<T>* d_robotModel, const T* d_f_ext_band)
        {
                int* s_topology_helpers = nullptr;
                grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                __syncthreads();
                grid::f_ext_body_inner<T>(s_fext, s_fc, s_q, s_XmatsHom, s_topology_helpers, s_temp,
                                          /*d_workspace*/nullptr, /*s_linalg_smem*/nullptr);
                __syncthreads();
                if (d_f_ext_band != nullptr) {
                        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < FEXT_COUNT; i += blockDim.x * blockDim.y) {
                                s_fext[i] += d_f_ext_band[i];
                        }
                        __syncthreads();
                }
        }

        // CL-3a W2: the chain-rule term the A-block drops when the applied wrench is
        // treated as q-independent. f_c is WORLD-aligned, so f_ext = f_ext_body(q, f_c)
        // rotates with the arm and dqdd/dq gains
        //     dqdd_dfext . dfext_dq = -Minv . (dtau_dfext . dfext_dq).
        // At a 10 N wrench this term is the same size as the whole first-order gradient
        // (oracle measurement, test/test_f_ext.py) — dropping it costs SQP convergence
        // rate, not fixed points (defects use exact rollouts). dfext_dq is LINEAR in f_c
        // ⇒ identically zero at f_c = 0, so zero-wrench trajectories are unchanged.
        // __noinline__: cicc-cliff guard (this body lands in the setup_kkt TU).
        template<typename T>
        __device__ __noinline__ void addContactChainCorrection(T* s_df_du, T* s_dfext_dq, T* s_dtau_dq,
                                                               const T* s_fc, const T* s_dtau_dfext,
                                                               const T* s_Minv, const T* s_q, const T* s_XmatsHom,
                                                               int* s_topology_helpers, T* s_temp)
        {
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                // dfext/dq at fixed f_c (needs the LOCAL body Jacobians == dtau_dfext's columns)
                grid::f_ext_body_jacobian_dq_inner<T>(s_dfext_dq, s_fc, s_dtau_dfext, s_q, s_XmatsHom,
                                                      s_topology_helpers, s_temp,
                                                      /*d_workspace*/nullptr, /*s_linalg_smem*/nullptr);
                __syncthreads();
                // dtau_dq_corr = dtau_dfext @ dfext_dq   (NQ x NQ, col-major [j*NQ + k])
                for (int ind = tid; ind < NQ * NQ; ind += nth) {
                        const int k = ind % NQ, j = ind / NQ;
                        T acc = static_cast<T>(0);
                        for (int i = 0; i < FEXT_COUNT; i++) {
                                acc += s_dtau_dfext[k + NQ * i] * s_dfext_dq[i + FEXT_COUNT * j];
                        }
                        s_dtau_dq[ind] = acc;
                }
                __syncthreads();
                // dqdd/dq += -Minv (SYMMETRIC_UPPER) @ dtau_dq_corr
                for (int ind = tid; ind < NQ * NQ; ind += nth) {
                        const int r = ind % NQ, c = ind / NQ;
                        T val = static_cast<T>(0);
#pragma unroll
                        for (int col = 0; col < NQ; col++) {
                                const int index = (r <= col) * (col * NQ + r) + (r > col) * (r * NQ + col);
                                val += s_Minv[index] * s_dtau_dq[c * NQ + col];
                        }
                        s_df_du[ind] -= val;
                }
                __syncthreads();
        }
#endif

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext = nullptr)
        {
                // TOPOLOGY_HELPERS_COUNT == 0 for fixed serial chains; the inners never
                // dereference a zero-count helper buffer, so a null pointer is safe.
                int* s_topology_helpers = nullptr;
                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[XIMATS_COUNT];
#if GATO_CONTACT_FORCES
                // fc persistent block = the appended region AFTER the grid FD arena, so
                // s_temp keeps its usual extent in front of it (TempMemSize grew by
                // FC_PERSIST_COUNT below).
                T* s_XmatsHom = &s_XITemp[grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT];
                T* s_fext = s_XmatsHom + grid::XHOM_T_COUNT;
                buildContactFext<T>(s_fext, &s_u[NU], s_q, s_XmatsHom, s_temp,
                                    (grid::robotModel<T>*)d_dynMem_const, d_f_ext);
                d_f_ext = s_fext;
#endif
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, (grid::robotModel<T>*)d_dynMem_const, s_temp);
                __syncthreads();

                // d_workspace == nullptr: at the in-smem placement default the inner keeps all
                // scratch in shared memory (INNER_WORKSPACE_BYTES == 0). d_f_ext is body-major
                // 6*NUM_BODIES (or nullptr); gravity is now the last argument.
                grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr, d_f_ext, gato::plant::GRAVITY<T>());
        }

        __host__ __device__ constexpr unsigned forwardDynamics_TempMemSize_Shared()
        {
                return grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT + FC_PERSIST_COUNT;
        }

        template<typename T, bool INCLUDE_DU = true>
        __device__ void forwardDynamicsAndGradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const, T* d_f_ext = nullptr)
        {
                T*                   s_XITemp = s_temp_in;
                grid::robotModel<T>* d_robotModel = (grid::robotModel<T>*)d_dynMem_const;

                int* s_topology_helpers = nullptr;   // TOPOLOGY_HELPERS_COUNT == 0 (fixed chain)
                T* s_XImats = s_XITemp;
                T* s_vaf = &s_XITemp[XIMATS_COUNT];
                T* s_dc_du = &s_vaf[VAF_COUNT];
                T* s_Minv = &s_dc_du[2 * NQ * NQ];
                T* s_temp = &s_Minv[NQ * NQ];
#if GATO_CONTACT_FORCES
                // fc persistent block = the appended region after the grid FD_DU arena
                // (TempMemSize grew by FC_PERSIST_COUNT). Inner-call scratch reuses s_temp
                // sequentially — the ID-gradient's need is the region's high-water mark.
                T* s_XmatsHom   = &s_XITemp[grid::FD_DU_MAX_SHARED_MEM_COUNT];
                T* s_fext       = s_XmatsHom + grid::XHOM_T_COUNT;
                T* s_dtau_dfext = s_fext + FEXT_COUNT;            // NQ x 6NB, [v + NQ*(6i+r)]
                T* s_dfext_dfc  = s_dtau_dfext + NQ * FEXT_COUNT; // 6NB x FC, [row + 6NB*col]
                T* s_dtau_dfc   = s_dfext_dfc + FEXT_COUNT * FC;  // NQ x FC,  [k + NQ*c]
                T* s_dfext_dq   = s_dtau_dfc + NQ * FC;           // 6NB x NQ, [row + 6NB*v]
                T* s_dtau_dq    = s_dfext_dq + FEXT_COUNT * NQ;   // NQ x NQ,  [v*NQ + k]
                buildContactFext<T>(s_fext, &s_u[NU], s_q, s_XmatsHom, s_temp, d_robotModel, d_f_ext);
                d_f_ext = s_fext;
#endif
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[6], d_f_ext, GRAVITY<T>());
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
                // finish WRITES s_qdd (parallel over rows); inner_vaf READS it under a
                // different thread mapping — GRiD's own generated finish→vaf composition
                // emits this sync, our hand-rolled copy dropped it (racecheck WAW family,
                // GRiD reply 2026-08-02; bit-identical, ordering-hazard-only fix).
                __syncthreads();
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, GRAVITY<T>());
                // Inverse-dynamics gradient is not f_ext-threaded in GRiD (matches prior GATO behavior,
                // which also omitted d_f_ext on the gradient). d_temp_spill == nullptr (USE_DA_DF_SPILL=false).
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, /*d_temp_spill*/nullptr, GRAVITY<T>());

                // Option A: GATO keeps its own hand-composition df_du = -Minv * dc_du and packs
                // dqdd/du = Minv into s_df_du[2*NQ*NQ..] itself (the integrator gradient reads it there).
                for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < 2 * NQ * NQ; ind += blockDim.x * blockDim.y) {
                        int row = ind % NQ;
                        int dc_col_offset = ind - row;
                        // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                        T val = static_cast<T>(0);
#pragma unroll
                        for (int col = 0; col < NQ; col++) {
                                int index = (row <= col) * (col * NQ + row) + (row > col) * (row * NQ + col);
                                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                        }
                        s_df_du[ind] = -val;
                        if (INCLUDE_DU && ind < NQ * NQ) {
                                int col = ind / NQ;
                                int index = (row <= col) * (col * NQ + row) + (row > col) * (row * NQ + col);
                                s_df_du[ind + 2 * NQ * NQ] = s_Minv[index];
                        }
                }
                __syncthreads();
#if GATO_CONTACT_FORCES
                // CL-3a: append dqdd/dfc = dqdd_dfext @ dfext_dfc = -Minv @ (dtau_dfext @ dfext_dfc)
                // as CONTROL columns NU..NU+FC-1 of s_df_du (offset 3NQ^2, [c*NQ + r]) — the
                // integrator gradient already iterates CONTROL_SIZE columns and consumes them
                // as B's fc columns with no further changes. Reference composition:
                // grid::f_ext_gradient_device (dqdd_dfext = -Minv @ dtau_dfext).
                {
                        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                        const int nth = blockDim.x * blockDim.y;
                        // ID gradient is done -> s_temp is free scratch again; XImats/XmatsHom still live.
                        // dtau_dfext (== -J^T, columns are the local body Jacobians) feeds BOTH the
                        // fc control columns and the W2 dq correction, so it is computed either way.
                        grid::f_ext_gradient_jacobianT_inner<T>(s_dtau_dfext, s_q, s_XImats, s_topology_helpers, s_temp);
                        __syncthreads();
                        if (INCLUDE_DU) {
                                grid::f_ext_body_jacobian_dfc_inner<T>(s_dfext_dfc, s_q, s_XmatsHom, s_topology_helpers, s_temp,
                                                                       /*d_workspace*/nullptr, /*s_linalg_smem*/nullptr);
                                __syncthreads();
                                for (int ind = tid; ind < NQ * FC; ind += nth) {  // dtau_dfc = dtau_dfext @ dfext_dfc
                                        int k = ind % NQ; int c = ind / NQ;
                                        T acc = static_cast<T>(0);
                                        for (int i = 0; i < FEXT_COUNT; i++) {
                                                acc += s_dtau_dfext[k + NQ * i] * s_dfext_dfc[i + FEXT_COUNT * c];
                                        }
                                        s_dtau_dfc[ind] = acc;
                                }
                                __syncthreads();
                                for (int ind = tid; ind < NQ * FC; ind += nth) {  // -Minv (sym-upper) @ dtau_dfc
                                        int r = ind % NQ; int c = ind / NQ;
                                        T val = static_cast<T>(0);
#pragma unroll
                                        for (int col = 0; col < NQ; col++) {
                                                int index = (r <= col) * (col * NQ + r) + (r > col) * (r * NQ + col);
                                                val += s_Minv[index] * s_dtau_dfc[c * NQ + col];
                                        }
                                        s_df_du[3 * NQ * NQ + ind] = -val;
                                }
                                __syncthreads();
                        }
                        // W2: fold the dfext/dq chain term into the A-block (dqdd/dq) — applies
                        // to the dq columns, so it is NOT gated on INCLUDE_DU.
                        addContactChainCorrection<T>(s_df_du, s_dfext_dq, s_dtau_dq, &s_u[NU], s_dtau_dfext,
                                                     s_Minv, s_q, s_XmatsHom, s_topology_helpers, s_temp);
                }
#endif
        }

        __host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT + FC_PERSIST_COUNT;
        }

#if USE_EXACT_HESSIAN
        // ===================================================================
        // Exact-Hessian (SO-SQP) lambda^T d2a/dz2 contraction  [USE_EXACT_HESSIAN]
        // ===================================================================

        // Shared elements for exactHessianContraction: w (NQ) + the all-smem
        // grid::fdsva_so_device arena (qdd + Minv + df_du + df2 + idsva_so +
        // XImats + the SO temp pool, whose element count == the per-timestep SO
        // workspace band the generated TIER_LITE path would use instead).
        template<typename T>
        __host__ __device__ inline unsigned exactHessianSO_TempMemCt()
        {
                return NQ + NQ + NQ * NQ + 2 * NQ * NQ + 2 * grid::SECOND_ORDER_TENSOR_SIZE + XIMATS_COUNT
                       + (unsigned)(grid::GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>() / sizeof(T));
        }

        // Adds E = sum_i w_i * d2a_i/dz2 (z = [q; qd; u], column-major (NX+NU)^2)
        // into the stage block s_P — the lagged-lambda exact-Hessian term.
        //
        // lam = GATO's stored Schur multiplier for constraint row k+1. Sign:
        // computeDz recovers dz_x = -Q^-1(q - lam_k + A^T lam_kp1), so the QP
        // multiplier is mu = -lam; with c = x_next - F(x, u) the Lagrangian stage
        // Hessian addition is +mu^T d2c = -mu^T d2F = +lam^T d2F. F's second
        // derivative flows only through the acceleration, so the per-row-family
        // integrator weights (trapezoidal: q-rows 0.5*dt^2, v-rows dt; euler:
        // (0, dt); semi-implicit: (dt^2, dt)) collapse the whole contraction to
        // ONE accel-index weight vector w_i = c_q*lam_q[i] + c_v*lam_v[i].
        // lam == 0 (first cold iteration / after reset_dual) reduces exactly to GN.
        //
        // d2a block layout (grid fdsva_so, [dqdq | dvdq | dvdv | dtdq] stacked
        // NQ^3): block[i*NQ^2 + j*NQ + k] = d2(qdd_i)/d(first_j)d(second_k); the
        // qv/qu twins are the jk-transposed reads of dvdq/dtdq, uu/uv/vu are
        // identically zero (qdd linear in tau). Symmetric blocks are read at the
        // canonical (lo, hi) wedge so E is bitwise symmetric even where the
        // generated tensor is only ULP-symmetric.
        //
        // KNOWN CAVEAT: grid's fdsva_so does not thread f_ext, so with a nonzero
        // wrench the SO term is evaluated at f_ext = 0 while A/B (first-order
        // path) carry it — the projection still guarantees PSD; the curvature
        // correction is merely approximate under external wrenches.
        //
        // __noinline__: cicc-cliff guard (this lands in the setup_kkt TU).
        template<typename T, unsigned INTEGRATOR_TYPE = 2>
        __device__ __noinline__ void exactHessianContraction(T* s_P, const T* s_xux, const T* d_lambda_kp1, T dt, T* s_so, void* d_dynMem_const)
        {
                constexpr int PDIM = NX + NU;
                constexpr int NQ2 = NQ * NQ;
                constexpr int NQ3 = NQ * NQ * NQ;
                T*            s_w = s_so;
                T*            s_qdd = s_w + NQ;
                T*            s_Minv = s_qdd + NQ;
                T*            s_df_du = s_Minv + NQ2;
                T*            s_df2 = s_df_du + 2 * NQ2;
                T*            s_idsva_so = s_df2 + grid::SECOND_ORDER_TENSOR_SIZE;
                T*            s_XImats = s_idsva_so + grid::SECOND_ORDER_TENSOR_SIZE;
                T*            s_sotemp = s_XImats + XIMATS_COUNT;

                for (int i = threadIdx.x; i < NQ; i += blockDim.x) {
                        const T cq = (INTEGRATOR_TYPE == 0) ? static_cast<T>(0) : (INTEGRATOR_TYPE == 1 ? dt * dt : static_cast<T>(0.5) * dt * dt);
                        s_w[i] = cq * d_lambda_kp1[i] + dt * d_lambda_kp1[NQ + i];
                }
                // no barrier needed: s_w is disjoint from the fdsva arena, and the
                // contraction reads below are ordered by fdsva's internal barriers
                grid::fdsva_so_device<T, true, false, true>(s_df2, s_idsva_so, s_Minv, s_df_du, s_qdd, s_xux, s_xux + NQ, s_xux + NX, s_XImats, /*s_topology_helpers*/nullptr, s_sotemp,
                                                            /*d_workspace*/nullptr, /*d_fd_grad_spill*/nullptr, /*s_fdsva_temp*/nullptr, (const grid::robotModel<T>*)d_dynMem_const, GRAVITY<T>());
                __syncthreads();  // fdsva tensor writes -> contraction reads

                const T* d2a_dqdq = s_df2;
                const T* d2a_dvdq = s_df2 + NQ3;
                const T* d2a_dvdv = s_df2 + 2 * NQ3;
                const T* d2a_dtdq = s_df2 + 3 * NQ3;
                for (int idx = threadIdx.x; idx < PDIM * PDIM; idx += blockDim.x) {
                        const int a = idx % PDIM, b = idx / PDIM;
                        const int ablk = a / NQ, aj = a % NQ;
                        const int bblk = b / NQ, bk = b % NQ;
                        if (ablk >= 1 && bblk >= 1 && ablk + bblk >= 3) continue;  // uu/uv/vu = 0
                        const int lo = (aj < bk) ? aj : bk;
                        const int hi = (aj < bk) ? bk : aj;
                        T         acc = static_cast<T>(0);
                        for (int i = 0; i < NQ; i++) {
                                T d2;
                                if (ablk == 0 && bblk == 0)      d2 = d2a_dqdq[i * NQ2 + lo * NQ + hi];
                                else if (ablk == 1 && bblk == 1) d2 = d2a_dvdv[i * NQ2 + lo * NQ + hi];
                                else if (ablk == 1 && bblk == 0) d2 = d2a_dvdq[i * NQ2 + aj * NQ + bk];
                                else if (ablk == 0 && bblk == 1) d2 = d2a_dvdq[i * NQ2 + bk * NQ + aj];
                                else if (ablk == 2 && bblk == 0) d2 = d2a_dtdq[i * NQ2 + aj * NQ + bk];
                                else                             d2 = d2a_dtdq[i * NQ2 + bk * NQ + aj];  // (0,2)
                                acc += s_w[i] * d2;
                        }
                        s_P[idx] += acc;
                }
                __syncthreads();
        }
#endif  // USE_EXACT_HESSIAN

        // ===================================================================
        // grid_plant::tracking_cost ADAPTER
        //
        // Bridges GATO's scalar cost weights + device-constexpr joint limits to the
        // per-ELEMENT buffer contract of grid_plant::tracking_cost[_gradient/_hessian]
        // (the composable preset that owns the EE-pose + quadratic + barrier cost math).
        // All buffers are carved from a caller scratch slab and built cooperatively.
        // ===================================================================

#ifdef GRID_PLANT_HAS_TRACKING_COST
        // ─── FIXED BASE: the generated grid_plant::tracking_cost* preset ─────

        // Worst-case scratch (T elements) for the value / grad+hess adapters.
        template<typename T>
        __host__ __device__ constexpr unsigned trackingCostValue_TempMemCt()
        {
                // weights/targets (s_Q+s_R+s_W+s_x_des+s_u_des+s_ee_des) + bounds
                // (q/qd/u lower+upper) + s_eePos(6*NEE) + value EE-pose arena.
                return (2 * NX + 2 * NU + 6) + (4 * NQ + 2 * NU) + 6 * NEE
                       + grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned trackingCostGradHess_TempMemCt()
        {
                // + s_end_effector_pose_gradient (6*NUM_VEL*NEE) + the larger gradient arena
                // (+ the NU-wide generated-R/r redirect scratch under GATO_CONTACT_FORCES).
                return (2 * NX + 2 * NU + 6) + (4 * NQ + 2 * NU) + 6 * NEE + 6 * NQ * NEE
                       + (FC > 0 ? (unsigned)(NU * NU + NU) : 0u)
                       + grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT;
        }

        // Build the per-element weight/target/bound buffers. ee_weight = q_cost (running)
        // or N_cost (terminal). u_cost=0 (+ caller's mu_u=0) disables the control reg/barrier
        // on the terminal value knot. s_eePos_traj is the 6-wide reference (only xyz used).
        template<typename T>
        __device__ void buildTrackingCostBuffers(
            T* s_Q, T* s_R, T* s_W, T* s_x_des, T* s_u_des, T* s_ee_des,
            T* s_q_lo, T* s_q_hi, T* s_qd_lo, T* s_qd_hi, T* s_u_lo, T* s_u_hi,
            const T* s_eePos_traj, T qd_cost, T u_cost, T ee_weight,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr)
        {
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                // s_Q q-block: q_pos_cost toward d_q_nom (default 0/nullptr = the historic
                // EE-only cost — the q-block was hardcoded 0 and the limit barrier was the
                // only q-space term; q_pos_cost is the nullspace posture anchor, PDDP ask
                // 2026-08-02). qd_cost on the qd-block. Per-joint overrides (nullable
                // device vectors, PDDP round-5: tune single-joint effort/anchor gains —
                // the q7 Nyquist story): d_q_pos_w_vec (NQ) over q_pos_cost,
                // d_u_cost_vec (NU actuated) over u_cost. Terminal knots (no control
                // reg) pass d_u_cost_vec = nullptr alongside u_cost = 0 — caller's job.
                for (int i = tid; i < NX; i += nth) {
                        s_Q[i] = (i < NQ) ? (d_q_pos_w_vec != nullptr ? d_q_pos_w_vec[i] : q_pos_cost) : qd_cost;
                        s_x_des[i] = (i < NQ && d_q_nom != nullptr) ? d_q_nom[i] : static_cast<T>(0);
                }
                for (int i = tid; i < NU; i += nth) {
                        s_R[i] = (d_u_cost_vec != nullptr) ? d_u_cost_vec[i] : u_cost;
                        s_u_des[i] = static_cast<T>(0);
                }
                for (int i = tid; i < 3;  i += nth) { s_W[i] = ee_weight; s_ee_des[i] = s_eePos_traj[i]; }
                // de-interleave the constexpr [NQ][2] limits into contiguous lower/upper buffers.
                for (int i = tid; i < NQ; i += nth) {
                        s_q_lo[i]  = JOINT_LIMITS<T>()[i][0]; s_q_hi[i]  = JOINT_LIMITS<T>()[i][1];
                        s_qd_lo[i] = VEL_LIMITS<T>()[i][0];   s_qd_hi[i] = VEL_LIMITS<T>()[i][1];
                }
                for (int i = tid; i < NU; i += nth) { s_u_lo[i] = CTRL_LIMITS<T>()[i][0]; s_u_hi[i] = CTRL_LIMITS<T>()[i][1]; }
        }

        // VALUE adapter (replaces trackingcost). Returns the total scalar cost for one knot.
        // is_terminal picks N_cost EE weight + drops the control reg/barrier (matches GATO's
        // per-knot terminal handling). s_temp >= trackingCostValue_TempMemCt().
        template<typename T>
        __device__ T trackingCostValue(
            const T* s_x, const T* s_u, const T* s_eePos_traj, T* s_temp,
            const grid::robotModel<T>* d_robotModel,
            T q_cost, T qd_cost, T u_cost, T N_cost,
            T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, bool is_terminal,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            T fc_cost = static_cast<T>(0),
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr,
            const T* d_fc_ref = nullptr)
        {
                T* s_Q = s_temp;            T* s_R = s_Q + NX;          T* s_W = s_R + NU;
                T* s_x_des = s_W + 3;       T* s_u_des = s_x_des + NX;  T* s_ee_des = s_u_des + NU;
                T* s_q_lo = s_ee_des + 3;   T* s_q_hi = s_q_lo + NQ;
                T* s_qd_lo = s_q_hi + NQ;   T* s_qd_hi = s_qd_lo + NQ;
                T* s_u_lo = s_qd_hi + NQ;   T* s_u_hi = s_u_lo + NU;
                T* s_eePos = s_u_hi + NU;   T* s_scratch = s_eePos + 6 * NEE;

                const T ee_w = is_terminal ? N_cost : q_cost;
                const T u_w  = is_terminal ? static_cast<T>(0) : u_cost;        // terminal knot: no control reg
                const T mu_u = is_terminal ? static_cast<T>(0) : ctrl_lim_cost; // terminal knot: no ctrl barrier
                buildTrackingCostBuffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
                                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
                                            s_eePos_traj, qd_cost, u_w, ee_w, q_pos_cost, d_q_nom,
                                            is_terminal ? nullptr : d_u_cost_vec, d_q_pos_w_vec);
                __syncthreads();
                __shared__ T s_out[1];
                grid_plant::tracking_cost<T, 0>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, mu_u,
                                                s_eePos, s_scratch, d_robotModel);
                __syncthreads();
                T total = s_out[0];
#if GATO_CONTACT_FORCES
                // fc regularization (0.5*fc_cost*|fc - fc_ref|^2; d_fc_ref nullptr = zero
                // reference, bitwise the historic pure regularization; terminal knot has no
                // control slot, matching the u-reg drop). Uniform per-thread compute — no
                // shared write.
                if (!is_terminal) {
                        T fcreg = static_cast<T>(0);
                        for (int j = 0; j < FC; j++) {
                                T e = s_u[NU + j] - (d_fc_ref ? d_fc_ref[j] : static_cast<T>(0));
                                fcreg += e * e;
                        }
                        total += static_cast<T>(0.5) * fc_cost * fcreg;
                }
#else
                (void)fc_cost;
                (void)d_fc_ref;
#endif
                return total;
        }

        // GRAD+HESS adapter (replaces trackingCostGradientAndHessian). Writes s_qk/s_Qk
        // (state block) + s_rk/s_Rk (input block). ee_weight = q_cost (running, at state
        // s_x) or N_cost (terminal, at state x_{k+1} — see _lastblock rewire / PR #17).
        // For a terminal R-less call, pass throwaway s_rk/s_Rk. s_temp >= trackingCostGradHess_TempMemCt().
        template<typename T>
        __device__ void trackingCostGradHess(
            const T* s_x, const T* s_u, const T* s_eePos_traj,
            T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_temp,
            const grid::robotModel<T>* d_robotModel,
            T qd_cost, T u_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T ee_weight,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            T fc_cost = static_cast<T>(0),
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr,
            const T* d_fc_ref = nullptr)
        {
                T* s_Q = s_temp;            T* s_R = s_Q + NX;          T* s_W = s_R + NU;
                T* s_x_des = s_W + 3;       T* s_u_des = s_x_des + NX;  T* s_ee_des = s_u_des + NU;
                T* s_q_lo = s_ee_des + 3;   T* s_q_hi = s_q_lo + NQ;
                T* s_qd_lo = s_q_hi + NQ;   T* s_qd_hi = s_qd_lo + NQ;
                T* s_u_lo = s_qd_hi + NQ;   T* s_u_hi = s_u_lo + NU;
                T* s_eePos = s_u_hi + NU;   T* s_eePosGrad = s_eePos + 6 * NEE;
#if GATO_CONTACT_FORCES
                // The generated grid_plant cost is NU(=actuated)-wide with R layout [r + NU*c];
                // the caller's s_Rk/s_rk are CONTROL_SIZE-wide. Redirect the generated writes
                // to NU-wide scratch, then scatter + fill the fc block below.
                T* s_R7 = s_eePosGrad + 6 * NQ * NEE;
                T* s_r7 = s_R7 + NU * NU;
                T* s_scratch = s_r7 + NU;
                T* s_Rk_gen = s_R7;
                T* s_rk_gen = s_r7;
#else
                T* s_scratch = s_eePosGrad + 6 * NQ * NEE;
                T* s_Rk_gen = s_Rk;
                T* s_rk_gen = s_rk;
#endif

                buildTrackingCostBuffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
                                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
                                            s_eePos_traj, qd_cost, u_cost, ee_weight, q_pos_cost, d_q_nom,
                                            d_u_cost_vec, d_q_pos_w_vec);
                __syncthreads();
                // Block layout note: grid_plant writes s_Qk/s_Rk COLUMN-major; GATO's old code
                // wrote row-major (i*NX+j). The tracking Hessian (diag(qd,barrier) + JᵀWJ) is
                // SYMMETRIC, so the two layouts are bit-identical — do NOT "transpose-fix" this.
                grid_plant::tracking_cost_gradient<T, 0>(s_qk, s_rk_gen, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                         s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                         s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
                                                         s_eePos, s_eePosGrad, s_scratch, d_robotModel);
                __syncthreads();
                grid_plant::tracking_cost_hessian<T, 0>(s_Qk, s_Rk_gen, s_x, s_u,
                                                        s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                        s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
                                                        s_eePosGrad, s_scratch, d_robotModel);
                __syncthreads();
#if GATO_CONTACT_FORCES
                {
                        // Scatter the NU-wide generated R/r into the CONTROL_SIZE-wide caller
                        // buffers: actuated block verbatim, fc diag = fc_cost, cross terms 0,
                        // fc gradient rows = fc_cost * (fc - fc_ref) (d_fc_ref nullptr = zero
                        // reference — the historic pure regularization, bitwise).
                        constexpr int CS = NU + FC;
                        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                        const int nth = blockDim.x * blockDim.y;
                        for (int ind = tid; ind < CS * CS; ind += nth) {
                                int r = ind % CS; int c = ind / CS;
                                s_Rk[ind] = (r < NU && c < NU) ? s_R7[r + NU * c]
                                          : (r == c ? fc_cost : static_cast<T>(0));
                        }
                        for (int j = tid; j < CS; j += nth) {
                                s_rk[j] = (j < NU) ? s_r7[j]
                                        : fc_cost * (s_u[j] - (d_fc_ref ? d_fc_ref[j - NU] : static_cast<T>(0)));
                        }
                        __syncthreads();
                }
#endif
        }

#else  // !GRID_PLANT_HAS_TRACKING_COST — FLOATING BASE (CL-3 W3.4)
        // ─── FLOATING BASE: composed tracking cost ───────────────────────────
        //
        // The generated tracking preset is fixed-base gated upstream (its
        // per-term inners disagree on frame: quadratic_state_cost differences
        // the quaternion componentwise while ee_pos_cost is tangent — GRiD
        // ASK 3 requests the upstream floating preset; this composition is the
        // GATO-side interim and the convention reference). Terms, all TANGENT
        // (gradient 2·NV, Hessian 2·NV × 2·NV col-major, R/r actuated NU):
        //   - EE position tracking: grid_plant::ee_pos_cost[_gradient] (already
        //     tangent; on go2 the "EE" is the torso imu frame, so this doubles
        //     as the base-position cost) + a Gauss-Newton JᵀWJ q-block composed
        //     from the Jacobian the gradient call produces (PSD by construction,
        //     matching the GN-SQP pairing; exact-Hessian is fixed-base only).
        //   - velocity reg qd_cost on ALL NV tangent velocities (base damping
        //     included — the floating analog of the historic full-qd reg).
        //   - control reg + posture anchor + barriers on the ACTUATED slots
        //     only (tangent q slots 6..NV, stored q slots 7..NQ; the NU-row
        //     limit tables align with the actuated joints by construction).
        //     d_q_nom / d_q_pos_w_vec are STORED-q indexed (base slots unread).
        //   - fc terms: none (fc builds are fixed-base only, asserted above).
        static_assert(NQ == NV + 1, "floating tracking composition expects the free-flyer layout");

        template<typename T>
        __host__ __device__ constexpr unsigned trackingCostValue_TempMemCt()
        {
                // W(3) + eePos(6*NEE) + max(EE value arena incl. topology ints,
                // the per-slot partial buffer NV + 4*NU reused from the arena)
                constexpr unsigned arena =
                    (unsigned)grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT
                    + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T))
                    + (unsigned)(grid::GRID_EE_LINALG_SHARED_BYTES<T>() > 0 ? (grid::GRID_EE_LINALG_SHARED_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T) : 0);
                constexpr unsigned partials = (unsigned)(NV + 4 * NU);
                return 3 + 6 * NEE + (arena > partials ? arena : partials);
        }
        template<typename T>
        __host__ __device__ constexpr unsigned trackingCostGradHess_TempMemCt()
        {
                // W(3) + eePos(6*NEE) + J(6*NV*NEE) + EE gradient arena
                constexpr unsigned arena =
                    (unsigned)grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT
                    + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T))
                    + (unsigned)(grid::GRID_EE_LINALG_SHARED_BYTES<T>() > 0 ? (grid::GRID_EE_LINALG_SHARED_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T) : 0);
                return 3 + 6 * NEE + 6 * NV * NEE + arena;
        }

        // VALUE (same signature as the fixed adapter). s_x is the STORED state
        // [q(NQ); qd(NV)]; s_u the actuated control.
        template<typename T>
        __device__ T trackingCostValue(
            const T* s_x, const T* s_u, const T* s_eePos_traj, T* s_temp,
            const grid::robotModel<T>* d_robotModel,
            T q_cost, T qd_cost, T u_cost, T N_cost,
            T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, bool is_terminal,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            T fc_cost = static_cast<T>(0),
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr,
            const T* d_fc_ref = nullptr)
        {
                (void)fc_cost; (void)d_fc_ref;
                T* s_W = s_temp;
                T* s_eePos = s_W + 3;
                T* s_arena = s_eePos + 6 * NEE;  // EE value arena; reused as the partial buffer after
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                const T ee_w = is_terminal ? N_cost : q_cost;
                const T u_w  = is_terminal ? static_cast<T>(0) : u_cost;
                const T mu_u = is_terminal ? static_cast<T>(0) : ctrl_lim_cost;
                for (int r = tid; r < 3; r += nth) { s_W[r] = ee_w; }
                __syncthreads();
                __shared__ T s_out[1];
                grid_plant::ee_pos_cost<T, 0, /*ACCUMULATE=*/false>(s_out, s_x, s_eePos_traj, s_W, s_eePos, s_arena, d_robotModel);
                __syncthreads();
                // per-slot partials in a FIXED layout (deterministic serial sum below):
                // [0,NV) qd reg | [NV,NV+NU) u reg+barrier | +NU anchor | +NU q barrier | +NU qd barrier
                T* s_part = s_arena;
                constexpr int NP = NV + 4 * NU;
                for (int i = tid; i < NP; i += nth) {
                        T v;
                        if (i < NV) {
                                const T qd = s_x[NQ + i];
                                v = static_cast<T>(0.5) * qd_cost * qd * qd;
                        } else if (i < NV + NU) {
                                const int j = i - NV;
                                const T uw = (d_u_cost_vec != nullptr && !is_terminal) ? d_u_cost_vec[j] : u_w;
                                const T uj = s_u[j];
                                v = static_cast<T>(0.5) * uw * uj * uj
                                    + grid_plant::grid_plant_log_barrier<T>(uj, CTRL_LIMITS<T>()[j][0], CTRL_LIMITS<T>()[j][1], mu_u);
                        } else if (i < NV + 2 * NU) {
                                const int j = i - NV - NU;
                                const T w = (d_q_pos_w_vec != nullptr) ? d_q_pos_w_vec[7 + j] : q_pos_cost;
                                const T e = s_x[7 + j] - (d_q_nom != nullptr ? d_q_nom[7 + j] : static_cast<T>(0));
                                v = static_cast<T>(0.5) * w * e * e;
                        } else if (i < NV + 3 * NU) {
                                const int j = i - NV - 2 * NU;
                                v = grid_plant::grid_plant_log_barrier<T>(s_x[7 + j], JOINT_LIMITS<T>()[j][0], JOINT_LIMITS<T>()[j][1], q_lim_cost);
                        } else {
                                const int j = i - NV - 3 * NU;
                                v = grid_plant::grid_plant_log_barrier<T>(s_x[NQ + 6 + j], VEL_LIMITS<T>()[j][0], VEL_LIMITS<T>()[j][1], vel_lim_cost);
                        }
                        s_part[i] = v;
                }
                __syncthreads();
                if (tid == 0) {
                        T acc = s_out[0];
                        for (int i = 0; i < NP; i++) { acc += s_part[i]; }
                        s_out[0] = acc;
                }
                __syncthreads();
                return s_out[0];
        }

        // GRAD+HESS (same signature as the fixed adapter). Outputs TANGENT
        // blocks: s_Qk 2NV×2NV col-major, s_qk 2NV, s_Rk NU×NU, s_rk NU.
        template<typename T>
        __device__ void trackingCostGradHess(
            const T* s_x, const T* s_u, const T* s_eePos_traj,
            T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_temp,
            const grid::robotModel<T>* d_robotModel,
            T qd_cost, T u_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T ee_weight,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            T fc_cost = static_cast<T>(0),
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr,
            const T* d_fc_ref = nullptr)
        {
                (void)fc_cost; (void)d_fc_ref;
                constexpr int TS = NV * 2;  // tangent state size
                T* s_W = s_temp;
                T* s_eePos = s_W + 3;
                T* s_J = s_eePos + 6 * NEE;     // 6*NV*NEE, layout [6*vi + row]
                T* s_arena = s_J + 6 * NV * NEE;
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                for (int i = tid; i < TS * TS; i += nth) { s_Qk[i] = static_cast<T>(0); }
                for (int i = tid; i < TS; i += nth) { s_qk[i] = static_cast<T>(0); }
                for (int i = tid; i < NU * NU; i += nth) { s_Rk[i] = static_cast<T>(0); }
                for (int i = tid; i < NU; i += nth) { s_rk[i] = static_cast<T>(0); }
                for (int r = tid; r < 3; r += nth) { s_W[r] = ee_weight; }
                __syncthreads();
                // tangent EE gradient adds into the zeroed dq block (ACCUMULATE:
                // the non-accumulate variant zeroes an NX-long tail past our
                // TS-long buffer). Ends on a barrier; s_J holds J afterwards.
                grid_plant::ee_pos_cost_gradient<T, 0, /*ACCUMULATE=*/true>(s_qk, s_x, s_eePos_traj, s_W, s_eePos, s_J, s_arena, d_robotModel);
                // Gauss-Newton EE q-block from the SAME Jacobian: Q[vi,vj] +=
                // sum_r W[r]·J[6vi+r]·J[6vj+r] (position rows 0..2; PSD).
                for (int i = tid; i < NV * NV; i += nth) {
                        const int vi = i % NV, vj = i / NV;
                        T acc = static_cast<T>(0);
#pragma unroll
                        for (int r = 0; r < 3; r++) { acc += s_W[r] * s_J[6 * vi + r] * s_J[6 * vj + r]; }
                        s_Qk[vj * TS + vi] = acc;
                }
                __syncthreads();
                // diagonal terms + gradients (anchor/barriers actuated-only)
                for (int i = tid; i < TS; i += nth) {
                        T g = static_cast<T>(0), h = static_cast<T>(0);
                        if (i < NV) {
                                if (i >= 6) {
                                        const int j = i - 6;
                                        const T w = (d_q_pos_w_vec != nullptr) ? d_q_pos_w_vec[7 + j] : q_pos_cost;
                                        const T qj = s_x[7 + j];
                                        g += w * (qj - (d_q_nom != nullptr ? d_q_nom[7 + j] : static_cast<T>(0)));
                                        h += w;
                                        g += grid_plant::grid_plant_log_barrier_grad<T>(qj, JOINT_LIMITS<T>()[j][0], JOINT_LIMITS<T>()[j][1], q_lim_cost);
                                        h += grid_plant::grid_plant_log_barrier_hess<T>(qj, JOINT_LIMITS<T>()[j][0], JOINT_LIMITS<T>()[j][1], q_lim_cost);
                                }
                        } else {
                                const T qd = s_x[NQ + (i - NV)];
                                g += qd_cost * qd;
                                h += qd_cost;
                                if (i >= NV + 6) {
                                        const int j = i - NV - 6;
                                        g += grid_plant::grid_plant_log_barrier_grad<T>(qd, VEL_LIMITS<T>()[j][0], VEL_LIMITS<T>()[j][1], vel_lim_cost);
                                        h += grid_plant::grid_plant_log_barrier_hess<T>(qd, VEL_LIMITS<T>()[j][0], VEL_LIMITS<T>()[j][1], vel_lim_cost);
                                }
                        }
                        s_qk[i] += g;
                        s_Qk[i * TS + i] += h;
                }
                for (int j = tid; j < NU; j += nth) {
                        const T uw = (d_u_cost_vec != nullptr) ? d_u_cost_vec[j] : u_cost;
                        const T uj = s_u[j];
                        s_rk[j] = uw * uj
                                  + grid_plant::grid_plant_log_barrier_grad<T>(uj, CTRL_LIMITS<T>()[j][0], CTRL_LIMITS<T>()[j][1], ctrl_lim_cost);
                        s_Rk[j * NU + j] = uw
                                  + grid_plant::grid_plant_log_barrier_hess<T>(uj, CTRL_LIMITS<T>()[j][0], CTRL_LIMITS<T>()[j][1], ctrl_lim_cost);
                }
                __syncthreads();
        }
#endif  // GRID_PLANT_HAS_TRACKING_COST

        // ===================================================================
        // RAW EE-pose evaluators (constraint row-group layer)
        //
        // Caller-scratch mirrors of grid_plant::ee_pos_cost[_gradient] minus the
        // cost math: carve the GRiD EE arena from s_scratch and call the
        // composable _inner entry points (NOT the auto-allocating _device
        // variants, which claim the kernel's dynamic-smem arena at offset 0).
        // s_scratch must be 16B-aligned and hold >= eePos[Grad]_TempMemCt()
        // elements. Outputs: s_pose = 6*NEE (position = rows 0..2 per EE);
        // s_grad = 6*NQ*NEE, layout [6*NQ*ee + 6*vi + row] (J_p = rows 0..2).
        // Upstream ask logged: first-class raw evaluators in grid_plant.
        // ===================================================================

        // Counts include the topology-helper int region (0 on fixed serial
        // chains — identical values there) and the constexpr linalg bytes pad.
        template<typename T>
        __host__ __device__ constexpr unsigned eePos_TempMemCt()
        {
                return grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T))
                       + (unsigned)(grid::GRID_EE_LINALG_SHARED_BYTES<T>() > 0 ? (grid::GRID_EE_LINALG_SHARED_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T) : 0);
        }
        template<typename T>
        __host__ __device__ constexpr unsigned eePosGrad_TempMemCt()
        {
                return grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T))
                       + (unsigned)(grid::GRID_EE_LINALG_SHARED_BYTES<T>() > 0 ? (grid::GRID_EE_LINALG_SHARED_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T) : 0);
        }

        // Carve helper for the two raw evaluators: XmatsHom at 0 (emitted
        // count — 144 on the vendored arms, per-body on branched/floating
        // robots), temp after it, topology ints past the T region (nullptr on
        // fixed serial chains, count 0 — pointer values identical there).
        template<typename T, unsigned TOTAL_T_CT>
        __device__ __forceinline__ void eeCarve(T* s_scratch, T** s_XmatsHom, T** s_temp, int** s_topology_helpers, unsigned char** s_linalg_smem)
        {
                using namespace grid;
                *s_XmatsHom = s_scratch;
                *s_temp = s_scratch + XHOM_T_COUNT;
                *s_topology_helpers = (TOPOLOGY_HELPERS_COUNT > 0)
                                          ? reinterpret_cast<int*>(s_scratch + TOTAL_T_CT) : nullptr;
                *s_linalg_smem = nullptr;
                if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                        size_t off = grid_align_up((TOTAL_T_CT * sizeof(T)) + TOPOLOGY_HELPERS_COUNT * sizeof(int),
                                                   static_cast<size_t>(16));
                        *s_linalg_smem = reinterpret_cast<unsigned char*>(s_scratch) + off;
                }
        }

        template<typename T>
        __device__ void eePos(T* s_pose, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                using namespace grid;
                T *s_XmatsHom, *s_temp;
                int* s_topology_helpers;
                unsigned char* s_linalg_smem;
                eeCarve<T, (unsigned)END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT>(s_scratch, &s_XmatsHom, &s_temp, &s_topology_helpers, &s_linalg_smem);
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                GATO_GRID_EE_POSE_INNER<T, true>(s_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
        }

        template<typename T>
        __device__ void eePosGrad(T* s_pose, T* s_grad, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                using namespace grid;
                T *s_XmatsHom, *s_temp;
                int* s_topology_helpers;
                unsigned char* s_linalg_smem;
                eeCarve<T, (unsigned)END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT>(s_scratch, &s_XmatsHom, &s_temp, &s_topology_helpers, &s_linalg_smem);
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                GATO_GRID_EE_POSE_INNER<T, true>(s_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                GATO_GRID_EE_POSE_GRAD_INNER<T, true>(s_grad, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
        }

        // ===================================================================
        // Environment-collision clearance evaluators (CL-2). Caller-scratch
        // adapters over the generated grid_collision:: family — the generated
        // *_device wrappers carve `extern __shared__` FROM OFFSET 0, which
        // would alias any consumer kernel's own dynamic-smem layout (same
        // reason eePos/eePosGrad exist above). Layout mirrors
        // grid::multi_target_position[_gradient]_device exactly.
        //
        // d_i(q) = signed clearance of collision sphere i to the NEAREST
        // environment obstacle (min over obstacles; +1e30 when env is empty).
        // The argmin re-picks at every call site's linearization point q —
        // within one QP/ADMM inner loop q is constant, so the active obstacle
        // is frozen per quadratic model by construction (the upstream
        // non-smoothness caveat); across SQP iterations it relinearizes like
        // every other nonlinearity. If the argmin switch ever bites, the
        // generated collision_distance_pairs[_gradient] (per-(sphere,obstacle)
        // rows, each smooth in q) is the principled fallback.
        // ===================================================================

        inline constexpr int NCC = grid_collision::NUM_COLLISION_SPHERES;

        // Multi-target FK arena T counts (XmatsHom + FK temp), robot-generic:
        // the TIER_LITE inline-workspace byte count IS the TIER_SHARED smem
        // temp count (the wrappers' own arena asserts pin this identity).
        // Vendored arms: 144+144 (value) / 144+570 (gradient) — the historic
        // hardcoded carve, byte-identical there.
        template<typename T>
        __host__ __device__ constexpr unsigned mtPos_T_Ct()
        {
                return (unsigned)grid::XHOM_T_COUNT
                       + (unsigned)(grid::MULTI_TARGET_POSITION_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned mtPosGrad_T_Ct()
        {
                return (unsigned)grid::XHOM_T_COUNT
                       + (unsigned)(grid::MULTI_TARGET_POSITION_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned mtPosArenaCt()
        {
                return mtPos_T_Ct<T>()
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T))
                       + (unsigned)(grid::GRID_EE_LINALG_SHARED_BYTES<T>() > 0 ? (grid::GRID_EE_LINALG_SHARED_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T) : 0);
        }
        template<typename T>
        __host__ __device__ constexpr unsigned mtPosGradArenaCt()
        {
                return mtPosGrad_T_Ct<T>()
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T))
                       + (unsigned)(grid::GRID_EE_LINALG_SHARED_BYTES<T>() > 0 ? (grid::GRID_EE_LINALG_SHARED_BYTES<T>() + 16 + sizeof(T) - 1) / sizeof(T) : 0);
        }

        template<typename T>
        __host__ __device__ constexpr unsigned collisionDist_TempMemCt()
        {
                // FK arena + sphere pos/radii + normals + align slop
                return mtPosArenaCt<T>() + 3 * NCC + NCC + 3 * NCC + 16 / sizeof(T) + 1;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned collisionDistGrad_TempMemCt()
        {
                // FK arena + pos/radii/normals + dp/dq batch (NV tangent columns)
                return mtPosGradArenaCt<T>() + 3 * NCC + NCC + 3 * NCC + 3 * NV * NCC + 16 / sizeof(T) + 1;
        }

        // s_dist: NCC per-sphere clearances. s_q is the STORED configuration
        // (quaternion layout on floating base). ALL threads must call (barriers).
        template<typename T>
        __device__ void collisionDist(T* s_dist, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel,
                                      const grid_collision::Environment<T>& env)
        {
                using namespace grid;
                T *s_XmatsHom, *s_temp;
                int* s_topology_helpers;
                unsigned char* s_linalg_smem;
                eeCarve<T, mtPos_T_Ct<T>()>(s_scratch, &s_XmatsHom, &s_temp, &s_topology_helpers, &s_linalg_smem);
                T* s_pos = s_scratch + mtPosArenaCt<T>();
                T* s_r = s_pos + 3 * NCC;
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                multi_target_position_inner<T, true>(s_pos, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                grid_collision::load_collision_radii<T>(s_r);
                __syncthreads();
                for (int i = threadIdx.x; i < NCC; i += blockDim.x) {
                        T nx, ny, nz;
                        s_dist[i] = grid_collision::grid_cc_nearest_obstacle<T>(env, s_pos[3 * i], s_pos[3 * i + 1], s_pos[3 * i + 2], s_r[i], &nx, &ny, &nz);
                }
                __syncthreads();
        }

        // s_dist: NCC clearances; s_ddist: NCC x NV sphere-major TANGENT
        // Jacobian (s_ddist[i*NV + vi] = n_i^T dp_i/dq_vi; the generated
        // multi-target gradient is NV-columned — fixed base NV == NQ).
        // ALL threads must call.
        template<typename T>
        __device__ void collisionDistGrad(T* s_dist, T* s_ddist, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel,
                                          const grid_collision::Environment<T>& env)
        {
                using namespace grid;
                T *s_XmatsHom, *s_temp;
                int* s_topology_helpers;
                unsigned char* s_linalg_smem;
                eeCarve<T, mtPosGrad_T_Ct<T>()>(s_scratch, &s_XmatsHom, &s_temp, &s_topology_helpers, &s_linalg_smem);
                T* s_pos = s_scratch + mtPosGradArenaCt<T>();
                T* s_r = s_pos + 3 * NCC;
                T* s_normal = s_r + NCC;
                T* s_pos_grad = s_normal + 3 * NCC;
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                multi_target_position_inner<T, true>(s_pos, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                grid_collision::load_collision_radii<T>(s_r);
                __syncthreads();
                for (int i = threadIdx.x; i < NCC; i += blockDim.x) {
                        T nx, ny, nz;
                        s_dist[i] = grid_collision::grid_cc_nearest_obstacle<T>(env, s_pos[3 * i], s_pos[3 * i + 1], s_pos[3 * i + 2], s_r[i], &nx, &ny, &nz);
                        s_normal[3 * i + 0] = nx;
                        s_normal[3 * i + 1] = ny;
                        s_normal[3 * i + 2] = nz;
                }
                __syncthreads();
                multi_target_position_gradient_inner<T, true>(s_pos_grad, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                for (int ind = threadIdx.x; ind < NCC * NV; ind += blockDim.x) {
                        const int vi = ind % NV;
                        const int i = ind / NV;
                        const int jb = 3 * (NV * i + vi);
                        s_ddist[i * NV + vi] = s_normal[3 * i + 0] * s_pos_grad[jb + 0] + s_normal[3 * i + 1] * s_pos_grad[jb + 1] + s_normal[3 * i + 2] * s_pos_grad[jb + 2];
                }
                __syncthreads();
        }

}  // namespace plant
}  // namespace gato
