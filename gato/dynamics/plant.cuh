#pragma once

#include <stdio.h>
#include "grid.cuh"
#include "glass.cuh"
#include "settings.h"
#include "utils/linalg.cuh"
#include "cost_menu.cuh"   // gato::menu::Buffers, the cost-menu view the adapters below read
#include "com_term.cuh"    // gato::com::Buffers, the centre-of-mass term view

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
        inline constexpr int NCF         = grid::NUM_CONTACT_FRAMES; // baked contact frames (feet / the arm EE)
        inline constexpr int EE_POS_SIZE = 6;                      // pose size (xyz + orientation)

        // Shared-memory extents of the grid:: inner buffers this adapter carves by hand.
        // XImats is per-BODY (72*NUM_BODIES == 72*NUM_JOINTS on fixed base, where the
        // hand-carved paths live; floating NUM_JOINTS is nq and must not size it) —
        // use the emitted count.
        inline constexpr int XIMATS_COUNT = grid::DYNAMICS_XI_T_COUNT;  // X + I (6x6 each) per body
        inline constexpr int VAF_COUNT    = 18 * grid::NUM_BODIES;      // v, a, f (6 each) per body

        // A serial chain can still need integer helpers when motion subspaces
        // differ (negative or oblique axes; the count rule is in GRiD's header).
        // Appended after the T arena; the footprint in T elements is the
        // generated TOPOLOGY_HELPERS_T_COUNT (grid-rbd 0.5.4).
        template<typename T>
        __host__ __device__ constexpr unsigned topology_helpers_smem_ct()
        {
                return static_cast<unsigned>(grid::TOPOLOGY_HELPERS_T_COUNT<T>());
        }

        template<typename T>
        __device__ __forceinline__ int* topology_helpers_at(T* tail)
        {
                return grid::TOPOLOGY_HELPERS_COUNT > 0 ? reinterpret_cast<int*>(tail) : nullptr;
        }

#if GATO_CONTACT_FORCES
    #if USE_EXACT_HESSIAN
        #error "GATO_CONTACT_FORCES x USE_EXACT_HESSIAN unsupported (fdsva_so is not fc-aware; W2+ item)"
    #endif
        // CL-3a: NU stays the ACTUATED width (the generated grid_plant cost code is
        // NU-wide); gato::constants::CONTROL_SIZE = NU + FC is the solver-facing width.
        // The fixed-base adapters below carry the fc chain in generalized dims (NQ ==
        // NV there); the floating base consumes fc through the grid step twins in
        // grid_plant_step.cuh  and the composed cost below.
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
        inline constexpr int CS = NU + FC;   // solver-facing control width (== gato::constants::CONTROL_SIZE)
        static_assert(CS == (int)gato::constants::CONTROL_SIZE, "plant CS must be the solver's CONTROL_SIZE");

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
        void* initialize_dynamics_const_mem()
        {
                grid::robotModel<T>* d_robotModel = grid::init_robotModel<T>();
                return (void*)d_robotModel;
        }

        template<typename T>
        void free_dynamics_const_mem(void* d_dynMem_const)
        {
                // grid::free_robotModel was removed (folded into close_grid); free the model directly.
                cudaFree((grid::robotModel<T>*)d_dynMem_const);
        }

        // (Minv @ X)[r, c] with Minv stored SYMMETRIC-UPPER (GRiD layout): the
        // one product all three Jacobian compositions below use (dq/du/dfc).
        template<typename T>
        __device__ __forceinline__ T minv_sym_upper_dot(const T* s_Minv, const T* x_col, int r)
        {
                T val = static_cast<T>(0);
#pragma unroll
                for (int col = 0; col < NQ; col++) {
                        const int index = (r <= col) * (col * NQ + r) + (r > col) * (r * NQ + col);
                        val += s_Minv[index] * x_col[col];
                }
                return val;
        }

#if GATO_CONTACT_FORCES
        // CL-3a: map the per-knot contact wrench decision variables (the fc tail of the
        // control slice, world-aligned [n; f] per contact frame) to the body-major
        // per-body wrench array the dynamics inners consume, ADDING the known external
        // band (P4.6 d_f_ext) if present. s_fc_scratch layout: XmatsHom | fext | temp
        // (temp = the caller's ordinary scratch region, reused sequentially — safe:
        // strictly-ordered inner calls, same pattern as contact_debug.cuh).
        template<typename T>
        __device__ void build_contact_fext(T* s_fext, const T* s_fc, const T* s_q,
                                         T* s_XmatsHom, T* s_temp, int* s_topology_helpers,
                                         grid::robotModel<T>* d_robotModel, const T* d_f_ext_band)
        {
                grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                __syncthreads();
                grid::f_ext_body_inner<T>(s_fext, s_fc, s_q, s_XmatsHom, s_topology_helpers, s_temp,
                                          /*d_workspace*/nullptr);
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
        __device__ __noinline__ void add_contact_chain_correction(T* s_df_du, T* s_dfext_dq, T* s_dtau_dq,
                                                               const T* s_fc, const T* s_dtau_dfext,
                                                               const T* s_Minv, const T* s_q, const T* s_XmatsHom,
                                                               int* s_topology_helpers, T* s_temp)
        {
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                // dfext/dq at fixed f_c (needs the LOCAL body Jacobians == dtau_dfext's columns)
                grid::f_ext_body_jacobian_dq_inner<T>(s_dfext_dq, s_fc, s_dtau_dfext, s_q, s_XmatsHom,
                                                      s_topology_helpers, s_temp,
                                                      /*d_workspace*/nullptr);
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
                        s_df_du[ind] -= minv_sym_upper_dot<T>(s_Minv, s_dtau_dq + c * NQ, r);
                }
                __syncthreads();
        }
#endif

        template<typename T>
        __device__ void forward_dynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext = nullptr)
        {
                int* s_topology_helpers = topology_helpers_at(
                    s_XITemp + grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT + FC_PERSIST_COUNT);
                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[XIMATS_COUNT];
#if GATO_CONTACT_FORCES
                // fc persistent block = the appended region AFTER the grid FD arena, so
                // s_temp keeps its usual extent in front of it (TempMemSize grew by
                // FC_PERSIST_COUNT below).
                T* s_XmatsHom = &s_XITemp[grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT];
                T* s_fext = s_XmatsHom + grid::XHOM_T_COUNT;
                build_contact_fext<T>(s_fext, &s_u[NU], s_q, s_XmatsHom, s_temp, s_topology_helpers,
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

        template<typename T>
        __host__ __device__ constexpr unsigned forward_dynamics_smem_ct()
        {
                return grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT + FC_PERSIST_COUNT + topology_helpers_smem_ct<T>();
        }

        template<typename T, bool INCLUDE_DU = true>
        __device__ void forward_dynamics_and_gradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const, T* d_f_ext = nullptr)
        {
                T*                   s_XITemp = s_temp_in;
                grid::robotModel<T>* d_robotModel = (grid::robotModel<T>*)d_dynMem_const;

                int* s_topology_helpers = topology_helpers_at(
                    s_XITemp + grid::FD_DU_MAX_SHARED_MEM_COUNT + FC_PERSIST_COUNT);
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
                build_contact_fext<T>(s_fext, &s_u[NU], s_q, s_XmatsHom, s_temp, s_topology_helpers, d_robotModel, d_f_ext);
                d_f_ext = s_fext;
#endif
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                // s_v does not change between these two inners: a fused minv+ID call could skip its recompute (perf, not done)
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
                        s_df_du[ind] = -minv_sym_upper_dot<T>(s_Minv, s_dc_du + dc_col_offset, row);
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
                                                                       /*d_workspace*/nullptr);
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
                                        s_df_du[3 * NQ * NQ + ind] = -minv_sym_upper_dot<T>(s_Minv, s_dtau_dfc + c * NQ, r);
                                }
                                __syncthreads();
                        }
                        // W2: fold the dfext/dq chain term into the A-block (dqdd/dq) — applies
                        // to the dq columns, so it is NOT gated on INCLUDE_DU.
                        add_contact_chain_correction<T>(s_df_du, s_dfext_dq, s_dtau_dq, &s_u[NU], s_dtau_dfext,
                                                     s_Minv, s_q, s_XmatsHom, s_topology_helpers, s_temp);
                }
#endif
        }

        template<typename T>
        __host__ __device__ constexpr unsigned forward_dynamics_gradient_smem_ct()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT + FC_PERSIST_COUNT + topology_helpers_smem_ct<T>();
        }

#if USE_EXACT_HESSIAN
        // ===================================================================
        // Exact-Hessian (SO-SQP) lambda^T d2a/dz2 contraction  [USE_EXACT_HESSIAN]
        // ===================================================================

        // Shared elements for exact_hessian_contraction: w (NQ) + the all-smem
        // grid::fdsva_so_device arena (qdd + Minv + df_du + df2 + idsva_so +
        // XImats + the SO temp pool, whose element count == the per-timestep SO
        // workspace band the generated TIER_LITE path would use instead).
        template<typename T>
        __host__ __device__ constexpr unsigned exact_hessian_so_numeric_ct()
        {
                return NQ + NQ + NQ * NQ + 2 * NQ * NQ + 2 * grid::SECOND_ORDER_TENSOR_SIZE + XIMATS_COUNT
                       + (unsigned)(grid::GRID_SO_WORKSPACE_BYTES_PER_TIMESTEP<T>() / sizeof(T));
        }
        // The topology helper ints follow the numeric arena (topology_helpers_smem_ct);
        // exact_hessian_contraction carves them at exact_hessian_so_numeric_ct and
        // checks that the sum fits this sizer.
        template<typename T>
        __host__ __device__ constexpr unsigned exact_hessian_so_smem_ct()
        {
                return exact_hessian_so_numeric_ct<T>() + topology_helpers_smem_ct<T>();
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
        __device__ __noinline__ void exact_hessian_contraction(T* s_P, const T* s_xux, const T* d_lambda_kp1, T dt, T* s_so, void* d_dynMem_const)
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
                static_assert(exact_hessian_so_numeric_ct<T>() + topology_helpers_smem_ct<T>() <= exact_hessian_so_smem_ct<T>(),
                              "exact-Hessian SO arena must have room for the topology helpers after the numeric scratch");
                int* s_topology_helpers = topology_helpers_at(s_so + exact_hessian_so_numeric_ct<T>());
                static_assert(INTEGRATOR_TYPE <= 2, "the lagged-lambda contraction weights are those of the single-stage integrators (0, 1, 2)");

                for (int i = threadIdx.x; i < NQ; i += blockDim.x) {
                        const T cq = (INTEGRATOR_TYPE == 0) ? static_cast<T>(0) : (INTEGRATOR_TYPE == 1 ? dt * dt : static_cast<T>(0.5) * dt * dt);
                        s_w[i] = cq * d_lambda_kp1[i] + dt * d_lambda_kp1[NQ + i];
                }
                // no barrier needed: s_w is disjoint from the fdsva arena, and the
                // contraction reads below are ordered by fdsva's internal barriers
                grid::fdsva_so_device<T, true, false, true>(s_df2, s_idsva_so, s_Minv, s_df_du, s_qdd, s_xux, s_xux + NQ, s_xux + NX, s_XImats, s_topology_helpers, s_sotemp,
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

// The cost menu (grid_plant::tracking_cost_menu*, grid-rbd 0.5.5) is emitted for
// fixed-base plants and for free-flyer plants with single-axis joints. On the
// contact-force variant the adapter below composes the fc tail (the wrench
// regularization of the default preset) around the generated menu.
#if defined(GRID_PLANT_HAS_COST_MENU)
#define GATO_MENU_AVAILABLE 1
#else
#define GATO_MENU_AVAILABLE 0
#endif

#ifdef GRID_PLANT_HAS_TRACKING_COST
        // ─── FIXED BASE: the generated grid_plant::tracking_cost* preset ─────

        // Worst-case scratch (T elements) for the value / grad+hess adapters.
        template<typename T>
        __host__ __device__ constexpr unsigned tracking_cost_value_smem_ct()
        {
                // weights/targets (s_Q+s_R+s_W+s_x_des+s_u_des+s_ee_des) + bounds
                // (q/qd/u lower+upper) + s_eePos(6*NEE) + value EE-pose arena.
                return (2 * NX + 2 * NU + 6) + (4 * NQ + 2 * NU) + 6 * NEE
                       + grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned tracking_cost_grad_hess_smem_ct()
        {
                // + s_end_effector_pose_gradient (6*NUM_VEL*NEE) + the larger gradient arena
                // (+ the NU*NU tracking_cost_hessian_fc base-composition scratch under
                // GATO_CONTACT_FORCES — GRiD ASK 1).
                return (2 * NX + 2 * NU + 6) + (4 * NQ + 2 * NU) + 6 * NEE + 6 * NQ * NEE
                       + (FC > 0 ? (unsigned)(NU * NU) : 0u)
                       + grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT;
        }

        // Build the per-element weight/target/bound buffers. ee_weight = q_cost (running)
        // or N_cost (terminal). u_cost=0 (+ caller's mu_u=0) disables the control reg/barrier
        // on the terminal value knot. s_eePos_traj is the 6-wide reference (only xyz used).
        template<typename T>
        __device__ void build_tracking_cost_buffers(
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
        // per-knot terminal handling). s_temp >= tracking_cost_value_smem_ct().
        template<typename T>
        __device__ T tracking_cost_value(
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
                build_tracking_cost_buffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
                                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
                                            s_eePos_traj, qd_cost, u_w, ee_w, q_pos_cost, d_q_nom,
                                            is_terminal ? nullptr : d_u_cost_vec, d_q_pos_w_vec);
                __syncthreads();
                __shared__ T s_out[1];
#if GATO_CONTACT_FORCES
                // Generated fc-aware preset (GRiD ASK 1, @27267bc): base tracking cost
                // + 0.5*fc_cost*|fc - fc_ref|^2 over the fc tail of the CONTROL_SIZE-wide
                // s_u, single-sourced with the gradient/hessian overloads below. Terminal
                // knot passes fc_cost = 0 (the emitted weight-selection contract, matching
                // the u-reg drop); d_fc_ref nullptr = zero reference (pure regularization,
                // bitwise the historic hand-rolled scatter).
                grid_plant::tracking_cost_fc<T, 0>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                   s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                   s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, mu_u,
                                                   is_terminal ? static_cast<T>(0) : fc_cost, d_fc_ref,
                                                   s_eePos, s_scratch, d_robotModel);
#else
                (void)fc_cost;
                (void)d_fc_ref;
                grid_plant::tracking_cost<T, 0>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, mu_u,
                                                s_eePos, s_scratch, d_robotModel);
#endif
                __syncthreads();
                return s_out[0];
        }

        // GRAD+HESS adapter (replaces trackingCostGradientAndHessian). Writes s_qk/s_Qk
        // (state block) + s_rk/s_Rk (input block). ee_weight = q_cost (running, at state
        // s_x) or N_cost (terminal, at state x_{k+1} — see _lastblock rewire / PR #17).
        // For a terminal R-less call, pass throwaway s_rk/s_Rk. s_temp >= tracking_cost_grad_hess_smem_ct().
        template<typename T>
        __device__ void tracking_cost_grad_hess(
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
                // NU*NU scratch for tracking_cost_hessian_fc's base input-hessian
                // composition (the generated overload scatters it to CONTROL_SIZE
                // strides itself — GRiD ASK 1).
                T* s_R_nu = s_eePosGrad + 6 * NQ * NEE;
                T* s_scratch = s_R_nu + NU * NU;
#else
                T* s_scratch = s_eePosGrad + 6 * NQ * NEE;
#endif

                build_tracking_cost_buffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
                                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
                                            s_eePos_traj, qd_cost, u_cost, ee_weight, q_pos_cost, d_q_nom,
                                            d_u_cost_vec, d_q_pos_w_vec);
                __syncthreads();
                // Block layout note: grid_plant writes s_Qk/s_Rk COLUMN-major; GATO's old code
                // wrote row-major (i*NX+j). The tracking Hessian (diag(qd,barrier) + JᵀWJ) is
                // SYMMETRIC, so the two layouts are bit-identical — do NOT "transpose-fix" this.
#if GATO_CONTACT_FORCES
                // Generated fc-aware overloads (GRiD ASK 1): CONTROL_SIZE-wide s_rk/s_Rk
                // written directly — actuated block exactly the base preset, fc gradient
                // rows = fc_cost*(fc - fc_ref), fc Hessian diag = fc_cost, cross terms 0
                // (bitwise the historic hand-rolled scatter).
                grid_plant::tracking_cost_gradient_fc<T, 0>(s_qk, s_rk, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                            s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                            s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
                                                            fc_cost, d_fc_ref,
                                                            s_eePos, s_eePosGrad, s_scratch, d_robotModel);
                __syncthreads();
                grid_plant::tracking_cost_hessian_fc<T, 0>(s_Qk, s_Rk, s_x, s_u,
                                                           s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                           s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
                                                           fc_cost, s_R_nu,
                                                           s_eePosGrad, s_scratch, d_robotModel);
                __syncthreads();
#else
                grid_plant::tracking_cost_gradient<T, 0>(s_qk, s_rk, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                         s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                         s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
                                                         s_eePos, s_eePosGrad, s_scratch, d_robotModel);
                __syncthreads();
                grid_plant::tracking_cost_hessian<T, 0>(s_Qk, s_Rk, s_x, s_u,
                                                        s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                        s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, ctrl_lim_cost,
                                                        s_eePosGrad, s_scratch, d_robotModel);
                __syncthreads();
#endif
        }


#if USE_EXACT_HESSIAN
        // ===================================================================
        // Exact curvature of the end-effector position term (0.1.5).
        // The tracking cost's Hessian is Gauss-Newton: w J^T J for the term
        // 0.5 w |p(q) - p_des|^2. Its exact Hessian adds the residual-weighted
        // second derivative of the frame position, sum_r w (p_r - p_des_r) H_r,
        // with H the generated end_effector_pose_hessian (d^2 pose / dv^2, the
        // pinocchio LOCAL_WORLD_ALIGNED kinematic Hessian; rows 0..2 are the
        // position). The term is indefinite in general, so it lives on the
        // exact-Hessian path, before the stage-block PSD projection, behind
        // its own runtime switch (set_cost_curvature). Fixed base only: the
        // tangent is q and the Hessian's position rows are d^2 p / dq^2.
        // Layout of the emitted Hessian: H[row * NV * NV + vi * NV + vj], both
        // orders of the pair filled (the inner writes out_base + row * NV * NV).
        template<typename T>
        __host__ __device__ constexpr unsigned ee_curvature_smem_ct()
        {
                // XmatsHom | pose (6 NEE) | J (6 NV NEE) | H (6 NV NV NEE) | the inner's scratch
                // (the generated auto-allocating count bounds it) | topology helpers
                return (unsigned)grid::XHOM_T_COUNT + 6u * NEE + 6u * NV * NEE + 6u * NV * NV * NEE
                     + (unsigned)grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT
                     + topology_helpers_smem_ct<T>();
        }

        // s_P: a column-major block with leading dimension ld whose first NV
        // rows and columns are the q block; adds the curvature term into it.
        // s_x: the stored state (q first); s_ee_des: the 3 target coordinates;
        // w: the end-effector weight of this knot. Ends on a barrier.
        template<typename T>
        __device__ __noinline__ void add_ee_position_curvature(T* s_P, int ld, const T* s_x, const T* s_ee_des, T w,
                                                               T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                T* s_XmatsHom = s_scratch;
                T* s_pose     = s_XmatsHom + grid::XHOM_T_COUNT;
                T* s_J        = s_pose + 6 * NEE;
                T* s_H        = s_J + 6 * NV * NEE;
                T* s_temp     = s_H + 6 * NV * NV * NEE;
                int* s_topology_helpers = topology_helpers_at(s_temp + grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT);
                grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_x, d_robotModel, s_temp);
                __syncthreads();
                grid::end_effector_pose_inner_EE<T, true>(s_pose, s_x, s_XmatsHom, s_topology_helpers, s_temp, nullptr);
                __syncthreads();
                grid::end_effector_pose_hessian_inner_EE<T, true>(s_H, s_J, s_x, s_XmatsHom, s_topology_helpers, s_temp, nullptr, d_robotModel);
                __syncthreads();
                for (int i = threadIdx.x; i < NV * NV; i += blockDim.x) {
                        const int vi = i % NV, vj = i / NV;
                        T acc = static_cast<T>(0);
#pragma unroll
                        for (int r = 0; r < 3; r++) {
                                acc += w * (s_pose[r] - s_ee_des[r]) * s_H[r * NV * NV + vi * NV + vj];
                        }
                        s_P[vj * ld + vi] += acc;
                }
                __syncthreads();
        }
#endif  // USE_EXACT_HESSIAN

#else  // !GRID_PLANT_HAS_TRACKING_COST — FLOATING BASE (CL-3 W3.4)
        // ─── FLOATING BASE: composed tracking cost ───────────────────────────
        //
        // The full tracking preset is fixed-base gated upstream; the floating
        // composition lives here, built from generated per-term surfaces.
        // Since 2026-08-13 the quadratic state part goes through the GRiD
        // ASK 3 log-map preset (grid_plant::quadratic_state_cost_tangent*,
        // oracle-gated vs RBDReference.quadratic_state_cost_tangent). Terms,
        // all TANGENT (gradient 2·NV, Hessian 2·NV × 2·NV col-major, R/r
        // actuated NU):
        //   - EE position tracking: grid_plant::ee_pos_cost[_gradient] (already
        //     tangent; on go2 the "EE" is the torso imu frame, so this doubles
        //     as the base-position cost) + a Gauss-Newton JᵀWJ q-block composed
        //     from the Jacobian the gradient call produces (PSD by construction,
        //     matching the GN-SQP pairing; exact-Hessian is fixed-base only).
        //   - quadratic state cost via the tangent preset: velocity reg qd_cost
        //     on ALL NV tangent velocities + posture anchor on the actuated q
        //     slots; base pose rows carry weight 0 today (base-pose tracking =
        //     weight the 6 base slots + a real x_des — no new code).
        //   - control reg + barriers on the ACTUATED slots only (tangent q
        //     slots 6..NV, stored q slots 7..NQ; the NU-row limit tables align
        //     with the actuated joints by construction). d_q_nom /
        //     d_q_pos_w_vec are STORED-q indexed (base slots unread).
        //   - fc terms (fc builds): 0.5*fc_cost*|fc - fc_ref|^2 over the
        //     fc tail of the CONTROL_SIZE-wide control on running knots (the
        //     terminal knot has no control), gradient rows NU.. and a diagonal
        //     Hessian block — the same term the arms' generated fc preset adds.
        //     R/r are CS = NU + FC wide (== NU on default builds: bitwise the
        //     pre-Wave-F composition there).
        static_assert(NQ == NV + 1, "floating tracking composition expects the free-flyer layout");

        template<typename T>
        __host__ __device__ constexpr unsigned tracking_cost_value_smem_ct()
        {
                // W(3) + ee_pos(6*NEE) + Q_diag(2*NV) + x_des(NX) + max(EE value
                // arena incl. topology ints, the tangent state-cost value
                // scratch (54), the per-slot partial buffer 3*NU + FC)
                constexpr unsigned arena =
                    (unsigned)grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT
                    + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T));
                constexpr unsigned partials = (unsigned)(3 * NU + FC);
                constexpr unsigned tsc = 54u;
                constexpr unsigned mx0 = arena > partials ? arena : partials;
                return 3 + 6 * NEE + 2 * NV + NX + (mx0 > tsc ? mx0 : tsc);
        }
        template<typename T>
        __host__ __device__ constexpr unsigned tracking_cost_grad_hess_smem_ct()
        {
                // W(3) + ee_pos(6*NEE) + J(6*NV*NEE) + Q_diag(2*NV) + x_des(NX)
                // + max(EE gradient arena, tangent state-cost GN scratch (90))
                constexpr unsigned arena =
                    (unsigned)grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT
                    + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T));
                constexpr unsigned tsc = 90u;
                return 3 + 6 * NEE + 6 * NV * NEE + 2 * NV + NX + (arena > tsc ? arena : tsc);
        }

        // Shared builder for the tangent state-cost inputs (GRiD ASK 3,
        // landed 2026-08-13): diagonal tangent weights s_Q (2*NV) and the
        // stored anchor state s_x_des (NX). The base pose is a weight change:
        // the first six entries of d_q_pos_w_vec (stored-q indexed; the base
        // slots were unread before 2026-10-23) are the base tangent weights
        // (position, then orientation) and the first seven entries of d_q_nom
        // the stored base pose (position, quaternion xyzw) they pull toward.
        // Without them the base rows weigh 0 and x_des holds the identity pose
        // (slot 6 = 1), so the log map stays finite at zero weight (0 * NaN =
        // NaN) and the path is bit-identical to the former one. Strided, no
        // trailing sync.
        template<typename T>
        __device__ __forceinline__ void build_tangent_state_cost_inputs(
            T* s_Q, T* s_x_des, T qd_cost, T q_pos_cost,
            const T* d_q_nom, const T* d_q_pos_w_vec, int tid, int nth)
        {
                for (int i = tid; i < 2 * NV + NX; i += nth) {
                        if (i < 2 * NV) {
                                s_Q[i] = (i < 6) ? ((d_q_pos_w_vec != nullptr) ? d_q_pos_w_vec[i] : static_cast<T>(0))
                                         : (i < NV ? ((d_q_pos_w_vec != nullptr) ? d_q_pos_w_vec[7 + i - 6] : q_pos_cost)
                                                   : qd_cost);
                        } else {
                                const int j = i - 2 * NV;
                                T v = static_cast<T>(0);
                                if (j < 7) v = (d_q_nom != nullptr) ? d_q_nom[j] : ((j == 6) ? static_cast<T>(1) : static_cast<T>(0));
                                else if (j < NQ) v = (d_q_nom != nullptr) ? d_q_nom[j] : static_cast<T>(0);
                                s_x_des[j] = v;
                        }
                }
        }

        // VALUE (same signature as the fixed adapter). s_x is the STORED state
        // [q(NQ); qd(NV)]; s_u the actuated control.
        template<typename T>
        __device__ T tracking_cost_value(
            const T* s_x, const T* s_u, const T* s_eePos_traj, T* s_temp,
            const grid::robotModel<T>* d_robotModel,
            T q_cost, T qd_cost, T u_cost, T N_cost,
            T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, bool is_terminal,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            T fc_cost = static_cast<T>(0),
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr,
            const T* d_fc_ref = nullptr)
        {
                T* s_W = s_temp;
                T* s_eePos = s_W + 3;
                T* s_Q = s_eePos + 6 * NEE;      // tangent diag weights (2*NV)
                T* s_x_des = s_Q + 2 * NV;       // stored anchor state (NX)
                T* s_arena = s_x_des + NX;       // EE value arena; preset scratch + partial buffer after
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                const T ee_w = is_terminal ? N_cost : q_cost;
                const T u_w  = is_terminal ? static_cast<T>(0) : u_cost;
                const T mu_u = is_terminal ? static_cast<T>(0) : ctrl_lim_cost;
                for (int r = tid; r < 3; r += nth) { s_W[r] = ee_w; }
                build_tangent_state_cost_inputs<T>(s_Q, s_x_des, qd_cost, q_pos_cost,
                                               d_q_nom, d_q_pos_w_vec, tid, nth);
                __syncthreads();
                __shared__ T s_out[1];
                grid_plant::ee_pos_cost<T, 0, /*ACCUMULATE=*/false>(s_out, s_x, s_eePos_traj, s_W, s_eePos, s_arena, d_robotModel);
                __syncthreads();
                // posture anchor (actuated q) + qd reg via the generated log-map
                // tangent preset (GRiD ASK 3; serial fixed-order sum inside —
                // the merit stays deterministic). Base rows weight 0 today.
                grid_plant::quadratic_state_cost_tangent<T, /*ACCUMULATE=*/true>(s_out, s_x, s_x_des, s_Q, s_arena);
                __syncthreads();
                // remaining per-slot partials in a FIXED layout (deterministic
                // serial sum below): [0,NU) u reg+barrier | +NU q barrier | +NU qd barrier
                // | +3NU fc reg (fc builds; 0 on the terminal knot — it has no control)
                T* s_part = s_arena;
                constexpr int NP = 3 * NU + FC;
                for (int i = tid; i < NP; i += nth) {
                        T v;
                        if (i < NU) {
                                const int j = i;
                                const T uw = (d_u_cost_vec != nullptr && !is_terminal) ? d_u_cost_vec[j] : u_w;
                                const T uj = s_u[j];
                                v = static_cast<T>(0.5) * uw * uj * uj
                                    + grid_plant::grid_plant_log_barrier<T>(uj, CTRL_LIMITS<T>()[j][0], CTRL_LIMITS<T>()[j][1], mu_u);
                        } else if (i < 2 * NU) {
                                const int j = i - NU;
                                v = grid_plant::grid_plant_log_barrier<T>(s_x[7 + j], JOINT_LIMITS<T>()[j][0], JOINT_LIMITS<T>()[j][1], q_lim_cost);
                        } else if (i < 3 * NU) {
                                const int j = i - 2 * NU;
                                v = grid_plant::grid_plant_log_barrier<T>(s_x[NQ + 6 + j], VEL_LIMITS<T>()[j][0], VEL_LIMITS<T>()[j][1], vel_lim_cost);
                        } else {
                                const int j = i - 3 * NU;
                                if (is_terminal) {
                                        v = static_cast<T>(0);
                                } else {
                                        const T e = s_u[NU + j] - ((d_fc_ref != nullptr) ? d_fc_ref[j] : static_cast<T>(0));
                                        v = static_cast<T>(0.5) * fc_cost * e * e;
                                }
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
        // blocks: s_Qk 2NV×2NV col-major, s_qk 2NV, s_Rk CS×CS col-major, s_rk CS
        // (CS = CONTROL_SIZE: the actuated block first, then the fc rows on fc builds).
        template<typename T>
        __device__ void tracking_cost_grad_hess(
            const T* s_x, const T* s_u, const T* s_eePos_traj,
            T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_temp,
            const grid::robotModel<T>* d_robotModel,
            T qd_cost, T u_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T ee_weight,
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr,
            T fc_cost = static_cast<T>(0),
            const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr,
            const T* d_fc_ref = nullptr)
        {
                constexpr int TS = NV * 2;  // tangent state size
                T* s_W = s_temp;
                T* s_eePos = s_W + 3;
                T* s_J = s_eePos + 6 * NEE;     // 6*NV*NEE, layout [6*vi + row]
                T* s_Q = s_J + 6 * NV * NEE;    // tangent diag weights (2*NV)
                T* s_x_des = s_Q + 2 * NV;      // stored anchor state (NX)
                T* s_arena = s_x_des + NX;
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                for (int i = tid; i < TS * TS; i += nth) { s_Qk[i] = static_cast<T>(0); }
                for (int i = tid; i < TS; i += nth) { s_qk[i] = static_cast<T>(0); }
                for (int i = tid; i < CS * CS; i += nth) { s_Rk[i] = static_cast<T>(0); }
                for (int i = tid; i < CS; i += nth) { s_rk[i] = static_cast<T>(0); }
                for (int r = tid; r < 3; r += nth) { s_W[r] = ee_weight; }
                build_tangent_state_cost_inputs<T>(s_Q, s_x_des, qd_cost, q_pos_cost,
                                               d_q_nom, d_q_pos_w_vec, tid, nth);
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
                // posture anchor (actuated q) + qd reg via the generated log-map
                // tangent preset (GRiD ASK 3): exact-J_diff gradient + GN hessian
                // ACCUMULATEd onto the zeroed/EE-GN blocks. Base rows weight 0
                // today — base-pose tracking is a weight change (see
                // build_tangent_state_cost_inputs). Both calls end on a barrier.
                grid_plant::quadratic_state_cost_tangent_gradient<T, /*ACCUMULATE=*/true>(s_qk, s_x, s_x_des, s_Q, s_arena);
                grid_plant::quadratic_state_cost_tangent_hessian<T, /*ACCUMULATE=*/true, /*GAUSS_NEWTON=*/true>(s_Qk, s_x, s_x_des, s_Q, s_arena);
                // barrier terms (actuated-only) + u reg
                for (int i = tid; i < TS; i += nth) {
                        T g = static_cast<T>(0), h = static_cast<T>(0);
                        if (i >= 6 && i < NV) {
                                const int j = i - 6;
                                const T qj = s_x[7 + j];
                                g = grid_plant::grid_plant_log_barrier_grad<T>(qj, JOINT_LIMITS<T>()[j][0], JOINT_LIMITS<T>()[j][1], q_lim_cost);
                                h = grid_plant::grid_plant_log_barrier_hess<T>(qj, JOINT_LIMITS<T>()[j][0], JOINT_LIMITS<T>()[j][1], q_lim_cost);
                        } else if (i >= NV + 6) {
                                const int j = i - NV - 6;
                                const T qd = s_x[NQ + 6 + j];
                                g = grid_plant::grid_plant_log_barrier_grad<T>(qd, VEL_LIMITS<T>()[j][0], VEL_LIMITS<T>()[j][1], vel_lim_cost);
                                h = grid_plant::grid_plant_log_barrier_hess<T>(qd, VEL_LIMITS<T>()[j][0], VEL_LIMITS<T>()[j][1], vel_lim_cost);
                        }
                        s_qk[i] += g;
                        s_Qk[i * TS + i] += h;
                }
                for (int j = tid; j < CS; j += nth) {
                        if (j < NU) {
                                const T uw = (d_u_cost_vec != nullptr) ? d_u_cost_vec[j] : u_cost;
                                const T uj = s_u[j];
                                s_rk[j] = uw * uj
                                          + grid_plant::grid_plant_log_barrier_grad<T>(uj, CTRL_LIMITS<T>()[j][0], CTRL_LIMITS<T>()[j][1], ctrl_lim_cost);
                                s_Rk[j * CS + j] = uw
                                          + grid_plant::grid_plant_log_barrier_hess<T>(uj, CTRL_LIMITS<T>()[j][0], CTRL_LIMITS<T>()[j][1], ctrl_lim_cost);
                        } else {
                                // fc regularization rows (fc builds): fc_cost*(fc - fc_ref), diag fc_cost
                                const int c = j - NU;
                                const T e = s_u[j] - ((d_fc_ref != nullptr) ? d_fc_ref[c] : static_cast<T>(0));
                                s_rk[j] = fc_cost * e;
                                s_Rk[j * CS + j] = fc_cost;
                        }
                }
                __syncthreads();
        }
#endif  // GRID_PLANT_HAS_TRACKING_COST

#if GATO_MENU_AVAILABLE
        // ===================================================================
        // COST MENU adapter (grid_plant::tracking_cost_menu*, grid-rbd 0.5.5)
        //
        // When the host turns the menu on it replaces the tracking objective:
        // every target, weight, bound and activation is a per-knot buffer
        // (gato::menu::Buffers; one copy shared across the batch or one per
        // solve). The joint-limit barriers keep GATO's constexpr limit tables
        // and scalar mu, as in the default preset. The linear features are
        // full-rank (MX = MENU_NLX, MU = MENU_NU; a row is off through
        // activation NONE or a zero weight). Position, orientation and twist act
        // on cost frame 0 (the end-effector target; the imu on the Go2), the
        // second position term on cost frame 1 when the plant bakes one. The
        // terminal knot has no control: its input and linear-u terms and its
        // torque barrier are off, as in the default path.
        //
        // One adapter for both bases. On a floating base the preset is in the
        // tangent layout (GRID_PLANT_MENU_TANGENT): the state term is the
        // log-map residual, the weights and bounds of the state term are
        // tangent-indexed (2 NV; the base pose carries no bound), the input is
        // the NV - 6 actuated torques, the position barrier bounds the NQ - 7
        // joint positions, the velocity barrier the NV tangent velocities (the
        // base rows unbounded), and the linear x feature reads [q[7:]; qd].
        // The outputs are the KKT blocks the kernels consume: STATE_SIZE wide.
        // ===================================================================
        inline constexpr bool MENU_TANGENT = (grid_plant::GRID_PLANT_MENU_TANGENT != 0);
        inline constexpr int  MENU_NU  = grid_plant::GRID_PLANT_MENU_NU;    // the preset's input width
        inline constexpr int  MENU_NLX = grid_plant::GRID_PLANT_MENU_NLX;   // the linear x feature's variable width
        inline constexpr int  MENU_TS  = MENU_TANGENT ? 2 * NV : NX;       // the state block width
        inline constexpr int  MENU_NQB = MENU_TANGENT ? NQ - 7 : NQ;       // joint positions under the position barrier
        inline constexpr int  MENU_NVB = NV;                                // velocities under the velocity barrier
        inline constexpr int  MENU_CF2 = (grid_plant::GRID_PLANT_NUM_COST_FRAMES > 1) ? 1 : -1;
        static_assert(MENU_NU == NU, "the menu's input width must be the plant's actuated width");
        static_assert(MENU_TS == (int)gato::constants::STATE_SIZE, "the menu's state block must be the KKT state size");
        static_assert(MENU_TANGENT == (bool)gato::constants::FLOATING_BASE, "the menu layout must follow the plant's base");

        __host__ __device__ constexpr gato::menu::Dims menu_dims()
        {
                return gato::menu::Dims{static_cast<uint32_t>(NX), static_cast<uint32_t>(MENU_TS), static_cast<uint32_t>(MENU_NU), static_cast<uint32_t>(MENU_NLX)};
        }
        template<typename T>
        __host__ __device__ constexpr unsigned menu_act_ct()
        {       // the knot's activation ints, in T slots
                return (unsigned)((gato::menu::NUM_TERMS * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned menu_scratch_ct()
        {       // the preset's s_scratch: the frame_eval arena (16-byte aligned, hence the slack) and the
                // larger of the tangent state term's scratch (NV + 72 + 2 NV) and the linear feature widths
                constexpr unsigned arena = (unsigned)((grid_plant::FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES<T>() + sizeof(T) - 1) / sizeof(T)) + (unsigned)(16 / sizeof(T));
                constexpr unsigned terms = (unsigned)(NV + 72 + MENU_TS) > (unsigned)MENU_NLX ? (unsigned)(NV + 72 + MENU_TS) : (unsigned)MENU_NLX;
                return arena > terms ? arena : terms;
        }
        // s_temp element count of menu_cost_value / menu_cost_grad_hess (fc builds add the
        // NU-wide input blocks the generated preset writes before the scatter into CONTROL_SIZE)
        template<typename T>
        __host__ __device__ constexpr unsigned menu_cost_smem_ct()
        {
                return (2 * MENU_NQB + 2 * MENU_NVB + 2 * MENU_NU) + menu_act_ct<T>() + (unsigned)grid_plant::GRID_PLANT_FRAME_SCRATCH_COUNT
                       + (FC > 0 ? (unsigned)(MENU_NU + MENU_NU * MENU_NU) : 0u) + menu_scratch_ct<T>();
        }

        template<typename T>
        struct MenuCarve { T* s_q_lo; T* s_q_hi; T* s_qd_lo; T* s_qd_hi; T* s_u_lo; T* s_u_hi; int* s_act; T* s_frame; T* s_r_nu; T* s_R_nu; T* s_scratch; };

        template<typename T>
        __device__ __forceinline__ MenuCarve<T> menu_carve(T* s_temp)
        {
                MenuCarve<T> c;
                c.s_q_lo = s_temp;                   c.s_q_hi = c.s_q_lo + MENU_NQB;
                c.s_qd_lo = c.s_q_hi + MENU_NQB;     c.s_qd_hi = c.s_qd_lo + MENU_NVB;
                c.s_u_lo = c.s_qd_hi + MENU_NVB;     c.s_u_hi = c.s_u_lo + MENU_NU;
                c.s_act = reinterpret_cast<int*>(c.s_u_hi + MENU_NU);
                c.s_frame = c.s_u_hi + MENU_NU + menu_act_ct<T>();
                c.s_r_nu = c.s_frame + grid_plant::GRID_PLANT_FRAME_SCRATCH_COUNT;   // fc builds only (NU and NU*NU)
                c.s_R_nu = c.s_r_nu + MENU_NU;
                const uintptr_t a = reinterpret_cast<uintptr_t>(FC > 0 ? c.s_R_nu + MENU_NU * MENU_NU : c.s_r_nu);
                c.s_scratch = reinterpret_cast<T*>((a + 15u) & ~static_cast<uintptr_t>(15u));
                return c;
        }

        // the limit buffers and the knot's activations (terminal: input terms off); act_override
        // (NUM_TERMS ints) replaces the knot's table, for the per-term readout. Ends without a barrier.
        template<typename T>
        __device__ __forceinline__ void menu_prepare(const MenuCarve<T>& c, const gato::menu::Buffers<T>& m, uint32_t knot_idx, bool is_terminal, const int* act_override = nullptr)
        {
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                for (int i = tid; i < MENU_NQB; i += nth) { c.s_q_lo[i] = JOINT_LIMITS<T>()[i][0]; c.s_q_hi[i] = JOINT_LIMITS<T>()[i][1]; }
                for (int i = tid; i < MENU_NVB; i += nth) {
                        if (MENU_TANGENT && i < 6) {   // the base twist carries no limit
                                c.s_qd_lo[i] = -static_cast<T>(INFINITY); c.s_qd_hi[i] = static_cast<T>(INFINITY);
                        } else {
                                const int j = MENU_TANGENT ? i - 6 : i;
                                c.s_qd_lo[i] = VEL_LIMITS<T>()[j][0]; c.s_qd_hi[i] = VEL_LIMITS<T>()[j][1];
                        }
                }
                for (int i = tid; i < MENU_NU; i += nth) { c.s_u_lo[i] = CTRL_LIMITS<T>()[i][0]; c.s_u_hi[i] = CTRL_LIMITS<T>()[i][1]; }
                for (int t = tid; t < gato::menu::NUM_TERMS; t += nth) {
                        int a = act_override ? act_override[t] : m.activation[knot_idx * gato::menu::NUM_TERMS + t];
                        if (is_terminal && (t == grid_plant::GRID_PLANT_MENU_INPUT || t == grid_plant::GRID_PLANT_MENU_LINEAR_U)) { a = grid_plant::GRID_PLANT_ACTIVATION_NONE; }
                        c.s_act[t] = a;
                }
        }

        // the preset's parameter block for one (solve, knot)
        template<typename T>
        __device__ __forceinline__ grid_plant::cost_menu_params<T> menu_params(const MenuCarve<T>& c, const gato::menu::Buffers<T>& m, uint32_t solve_idx, uint32_t knot_idx,
                                                                                 T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, bool is_terminal)
        {
                namespace mn = gato::menu;
                grid_plant::cost_menu_params<T> p;
                const auto at = [&](int f) { return m.at(f, solve_idx, knot_idx, mn::field_size(f, menu_dims())); };
                p.x_des = at(mn::X_DES);   p.u_des = at(mn::U_DES);   p.ee_des = at(mn::EE_DES);
                p.Q = at(mn::Q);           p.R = at(mn::R);           p.W = at(mn::W);
                p.q_lower = c.s_q_lo;      p.q_upper = c.s_q_hi;      p.mu_q = q_lim_cost;
                p.qd_lower = c.s_qd_lo;    p.qd_upper = c.s_qd_hi;    p.mu_qd = vel_lim_cost;
                p.u_lower = c.s_u_lo;      p.u_upper = c.s_u_hi;      p.mu_u = is_terminal ? static_cast<T>(0) : ctrl_lim_cost;
                p.activation = c.s_act;    p.eps = at(mn::EPS);
                p.x_lower = at(mn::X_LOWER);   p.x_upper = at(mn::X_UPPER);   p.ub_lower = at(mn::UB_LOWER); p.ub_upper = at(mn::UB_UPPER);
                p.Ax = at(mn::AX); p.bx = at(mn::BX); p.tx = at(mn::TX); p.Wx = at(mn::WX); p.lx_lower = at(mn::LX_LOWER); p.lx_upper = at(mn::LX_UPPER);
                p.Au = at(mn::AU); p.bu = at(mn::BU); p.tu = at(mn::TU); p.Wu = at(mn::WU); p.lu_lower = at(mn::LU_LOWER); p.lu_upper = at(mn::LU_UPPER);
                p.R_des = at(mn::R_DES);   p.Wo = at(mn::WO);
                p.v_des = at(mn::V_DES);   p.Wv = at(mn::WV);         p.v_lower = at(mn::V_LOWER);   p.v_upper = at(mn::V_UPPER);
                p.cf_des = at(mn::CF_DES); p.Wcf = at(mn::WCF);       p.cf_lower = at(mn::CF_LOWER); p.cf_upper = at(mn::CF_UPPER);
                return p;
        }

        // VALUE: the menu's total cost at one knot. s_x is the STORED state, s_u the control
        // (the actuated torques, then the fc wrench slots on fc builds, whose regularization
        // 0.5 fc_cost |fc - fc_ref|^2 composes here as in the default preset; 0 on the terminal
        // knot). s_temp >= menu_cost_smem_ct().
        template<typename T>
        __device__ T menu_cost_value(const T* s_x, const T* s_u, const gato::menu::Buffers<T>& m, uint32_t solve_idx, uint32_t knot_idx, bool is_terminal,
                                     T* s_temp, const grid::robotModel<T>* d_robotModel, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost,
                                     const int* act_override = nullptr, T fc_cost = static_cast<T>(0), const T* d_fc_ref = nullptr)
        {
                const MenuCarve<T> c = menu_carve<T>(s_temp);
                menu_prepare<T>(c, m, knot_idx, is_terminal, act_override);
                __syncthreads();
                const grid_plant::cost_menu_params<T> p = menu_params<T>(c, m, solve_idx, knot_idx, q_lim_cost, vel_lim_cost, ctrl_lim_cost, is_terminal);
                __shared__ T s_out[1];
                grid_plant::tracking_cost_menu<T, MENU_NLX, MENU_NU, 0, MENU_CF2>(s_out, s_x, s_u, p, c.s_frame, c.s_scratch, d_robotModel);
                __syncthreads();
#if GATO_CONTACT_FORCES
                if (!is_terminal && fc_cost != static_cast<T>(0)) {
                        // fixed-order serial sum, as the default preset's partials (the merit stays deterministic)
                        if (threadIdx.x == 0 && threadIdx.y == 0) {
                                T acc = static_cast<T>(0);
                                for (int j = 0; j < FC; ++j) {
                                        const T e = s_u[NU + j] - ((d_fc_ref != nullptr) ? d_fc_ref[j] : static_cast<T>(0));
                                        acc += static_cast<T>(0.5) * fc_cost * e * e;
                                }
                                s_out[0] += acc;
                        }
                        __syncthreads();
                }
#else
                (void)fc_cost; (void)d_fc_ref;
#endif
                return s_out[0];
        }

        // GRAD+HESS: s_Qk/s_qk (state block, STATE_SIZE) and s_Rk/s_rk (input block, CONTROL_SIZE:
        // the actuated block from the generated preset, then the fc rows fc_cost (fc - fc_ref) and
        // diagonal fc_cost on fc builds), Gauss-Newton. s_temp >= menu_cost_smem_ct().
        template<typename T>
        __device__ void menu_cost_grad_hess(const T* s_x, const T* s_u, T* s_Qk, T* s_qk, T* s_Rk, T* s_rk,
                                            const gato::menu::Buffers<T>& m, uint32_t solve_idx, uint32_t knot_idx, bool is_terminal,
                                            T* s_temp, const grid::robotModel<T>* d_robotModel, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost,
                                            T fc_cost = static_cast<T>(0), const T* d_fc_ref = nullptr)
        {
                const MenuCarve<T> c = menu_carve<T>(s_temp);
                menu_prepare<T>(c, m, knot_idx, is_terminal);
                __syncthreads();
                const grid_plant::cost_menu_params<T> p = menu_params<T>(c, m, solve_idx, knot_idx, q_lim_cost, vel_lim_cost, ctrl_lim_cost, is_terminal);
#if GATO_CONTACT_FORCES
                // the generated preset writes NU-wide input blocks; scatter them into the CONTROL_SIZE layout
                grid_plant::tracking_cost_menu_gradient<T, MENU_NLX, MENU_NU, 0, MENU_CF2>(s_qk, c.s_r_nu, s_x, s_u, p, c.s_frame, c.s_scratch, d_robotModel);
                __syncthreads();
                grid_plant::tracking_cost_menu_hessian<T, MENU_NLX, MENU_NU, 0, MENU_CF2>(s_Qk, c.s_R_nu, s_x, s_u, p, c.s_frame, c.s_scratch, d_robotModel);
                __syncthreads();
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                const T fcw = is_terminal ? static_cast<T>(0) : fc_cost;
                for (int j = tid; j < CS; j += nth) {
                        if (j < NU) {
                                s_rk[j] = c.s_r_nu[j];
                        } else {
                                const int cidx = j - NU;
                                const T e = s_u[j] - ((d_fc_ref != nullptr) ? d_fc_ref[cidx] : static_cast<T>(0));
                                s_rk[j] = fcw * e;
                        }
                }
                for (int i = tid; i < CS * CS; i += nth) {
                        const int r = i % CS, col = i / CS;
                        T v = static_cast<T>(0);
                        if (r < NU && col < NU) { v = c.s_R_nu[col * NU + r]; }
                        else if (r == col) { v = fcw; }
                        s_Rk[i] = v;
                }
                __syncthreads();
#else
                (void)fc_cost; (void)d_fc_ref;
                grid_plant::tracking_cost_menu_gradient<T, MENU_NLX, MENU_NU, 0, MENU_CF2>(s_qk, s_rk, s_x, s_u, p, c.s_frame, c.s_scratch, d_robotModel);
                __syncthreads();
                grid_plant::tracking_cost_menu_hessian<T, MENU_NLX, MENU_NU, 0, MENU_CF2>(s_Qk, s_Rk, s_x, s_u, p, c.s_frame, c.s_scratch, d_robotModel);
                __syncthreads();
#endif
        }
#endif  // GATO_MENU_AVAILABLE

        // ===================================================================
        // Centre-of-mass term ADAPTER (dynamics/com_term.cuh, 0.1.5): the generated
        // plant-layer com_cost (value, gradient over x, Gauss-Newton Hessian) on
        // modules whose header bakes the centroidal families. Added on top of the
        // objective in force (tracking cost or menu); the gradient's q block maps
        // onto the first NV tangent slots on both bases (the generated term
        // follows pinocchio's tangent convention).
#ifdef GRID_PLANT_HAS_COM_COST
#define GATO_COM_AVAILABLE 1
#else
#define GATO_COM_AVAILABLE 0
#endif
#if GATO_COM_AVAILABLE
        template<typename T>
        __host__ __device__ constexpr unsigned com_term_smem_ct()
        {
                // W (3) | target (3) | out (4, keeps the scratch 16-byte aligned) | grad (NX) | hess (NX NX) | the generated arena
                constexpr unsigned arena = (unsigned)((grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>() + sizeof(T) - 1) / sizeof(T)) + 4u;
                return 3u + 3u + 4u + (unsigned)NX + (unsigned)(NX * NX) + arena;
        }
        template<typename T>
        __device__ __forceinline__ T* com_term_arena(T* s_temp) { return s_temp + 3 + 3 + 4 + NX + NX * NX; }

        // the term's value at x; s_temp >= com_term_smem_ct(). Ends on a barrier.
        template<typename T>
        __device__ T com_term_value(const T* s_x, const gato::com::Buffers<T>& c, uint32_t solve_idx, uint32_t knot_idx,
                                    T* s_temp, const grid::robotModel<T>* d_robotModel)
        {
                T* s_W = s_temp; T* s_des = s_W + 3; T* s_out = s_des + 3;
                const T* d_des = c.at(solve_idx, knot_idx);
                for (int i = threadIdx.x; i < 3; i += blockDim.x) { s_W[i] = c.w[i]; s_des[i] = d_des[i]; }
                __syncthreads();
                T* s_arena = reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(com_term_arena(s_temp)) + 15u) & ~static_cast<uintptr_t>(15u));
                grid_plant::com_cost<T, /*ACCUMULATE=*/false>(s_out, s_x, s_des, s_W, s_arena, d_robotModel);
                __syncthreads();
                return s_out[0];
        }

        // gradient (into the first NV tangent slots of s_qk) and Gauss-Newton Hessian
        // (into the NV x NV q block of s_Qk, leading dimension ts), accumulated.
        // s_temp >= com_term_smem_ct(). Ends on a barrier.
        template<typename T>
        __device__ void com_term_grad_hess(const T* s_x, const gato::com::Buffers<T>& c, uint32_t solve_idx, uint32_t knot_idx,
                                           T* s_Qk, T* s_qk, int ts, T* s_temp, const grid::robotModel<T>* d_robotModel)
        {
                T* s_W = s_temp; T* s_des = s_W + 3; T* s_grad = s_des + 3 + 4; T* s_hess = s_grad + NX;
                const T* d_des = c.at(solve_idx, knot_idx);
                for (int i = threadIdx.x; i < 3; i += blockDim.x) { s_W[i] = c.w[i]; s_des[i] = d_des[i]; }
                __syncthreads();
                T* s_arena = reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(com_term_arena(s_temp)) + 15u) & ~static_cast<uintptr_t>(15u));
                grid_plant::com_cost_gradient<T, /*ACCUMULATE=*/false>(s_grad, s_x, s_des, s_W, s_arena, d_robotModel);
                __syncthreads();
                grid_plant::com_cost_hessian<T, /*ACCUMULATE=*/false>(s_hess, s_x, s_W, s_arena, d_robotModel);
                __syncthreads();
                for (int i = threadIdx.x; i < NV; i += blockDim.x) { s_qk[i] += s_grad[i]; }
                for (int i = threadIdx.x; i < NV * NV; i += blockDim.x) {
                        const int vi = i % NV, vj = i / NV;
                        s_Qk[vj * ts + vi] += s_hess[vj * NX + vi];
                }
                __syncthreads();
        }
#endif  // GATO_COM_AVAILABLE

        // ===================================================================
        // RAW EE-pose evaluators (constraint row-group layer)
        //
        // Thin forwards to the generated grid_plant::ee_pos[_gradient]
        // (GRiD ASK 2, landed 2026-08-13): caller-scratch, no cost coupling,
        // arena carved by the emitted code itself. s_scratch must be
        // 16B-aligned and hold >= ee_pos[_grad]_smem_ct() elements.
        // Outputs: s_pose = 6*NEE (position = rows 0..2 per EE);
        // s_grad = 6*NV*NEE, layout [6*NV*ee + 6*vi + row] (J_p = rows 0..2;
        // tangent d/dv convention — NV == NQ on fixed base).
        // ===================================================================

        // Counts include the topology-helper int region (0 on fixed serial
        // chains — identical values there).
        template<typename T>
        __host__ __device__ constexpr unsigned ee_pos_smem_ct()
        {
                return grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned ee_pos_grad_smem_ct()
        {
                return grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        }

        // Contact-frame positions (CL-4 §1.3, GRiD contact_frame_positions surface
        // 2026-09-26): world positions of the NCF baked contact ORIGINS (the points
        // f_ext_body takes the wrench about) and their 3 x NV tangent Jacobians,
        // layout s_dpos[3*NV*f + 3*vi + axis] in the [v_lin; omega; joints] LOCAL
        // chart on a floating base — the KKT tangent chart. Caller-scratch like
        // ee_pos: 16B-aligned, >= contact_pos[_grad]_smem_ct() elements. The
        // emitted *_COUNT is the documented contract of the plant wrappers (the
        // shared-tier arena incl. the topology ints + alignment slack — the
        // wrappers hard-carve that layout whatever GRID_DEFAULT_RESOURCE_TIER
        // says, so the tier-dependent BYTES sizer is NOT the right bound); the
        // constexpr linalg bytes pad rides along as for the EE pair.
        template<typename T>
        __host__ __device__ constexpr unsigned contact_pos_smem_ct()
        {
                return (unsigned)grid::CONTACT_FRAME_POSITIONS_DYNAMIC_SHARED_MEM_COUNT;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned contact_pos_grad_smem_ct()
        {
                return (unsigned)grid::CONTACT_FRAME_POSITIONS_GRADIENT_DYNAMIC_SHARED_MEM_COUNT;
        }

        // Carve helper for the remaining hand-carved inner compositions (the
        // collision multi-target FK below — no upstream raw multi-target
        // evaluator exists; the EE pair moved to grid_plant::ee_pos*):
        // XmatsHom at 0, temp after it, topology ints past the T region
        // (nullptr on fixed serial chains, count 0 — pointer values identical
        // there).
        template<typename T, unsigned TOTAL_T_CT>
        __device__ __forceinline__ void ee_carve(T* s_scratch, T** s_XmatsHom, T** s_temp, int** s_topology_helpers)
        {
                using namespace grid;
                *s_XmatsHom = s_scratch;
                *s_temp = s_scratch + XHOM_T_COUNT;
                *s_topology_helpers = (TOPOLOGY_HELPERS_COUNT > 0)
                                          ? reinterpret_cast<int*>(s_scratch + TOTAL_T_CT) : nullptr;
        }

        template<typename T>
        __device__ void ee_pos(T* s_pose, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                grid_plant::ee_pos<T>(s_pose, s_q, s_scratch, d_robotModel);
        }

        template<typename T>
        __device__ void ee_pos_grad(T* s_pose, T* s_grad, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                grid_plant::ee_pos_gradient<T>(s_pose, s_grad, s_q, s_scratch, d_robotModel);
        }

        // s_pos: 3*NCF world positions; s_dpos: 3*NV*NCF tangent Jacobian (see the
        // sizers above). s_q is the STORED configuration. ALL threads call (barriers).
        template<typename T>
        __device__ void contact_pos(T* s_pos, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                grid_plant::contact_frame_positions<T>(s_pos, s_q, s_scratch, d_robotModel);
        }

        template<typename T>
        __device__ void contact_pos_grad(T* s_pos, T* s_dpos, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                grid_plant::contact_frame_positions_gradient<T>(s_pos, s_dpos, s_q, s_scratch, d_robotModel);
        }

        // ===================================================================
        // Environment-collision clearance evaluators (CL-2). Caller-scratch
        // adapters over the generated grid_collision:: family — the generated
        // *_device wrappers carve `extern __shared__` FROM OFFSET 0, which
        // would alias any consumer kernel's own dynamic-smem layout (same
        // reason ee_pos/ee_pos_grad exist above). Layout mirrors
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
        __host__ __device__ constexpr unsigned mt_pos_t_ct()
        {
                return (unsigned)grid::XHOM_T_COUNT
                       + (unsigned)(grid::MULTI_TARGET_POSITION_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned mt_pos_grad_t_ct()
        {
                return (unsigned)grid::XHOM_T_COUNT
                       + (unsigned)(grid::MULTI_TARGET_POSITION_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned mt_pos_arena_ct()
        {
                return mt_pos_t_ct<T>()
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        }
        template<typename T>
        __host__ __device__ constexpr unsigned mt_pos_grad_arena_ct()
        {
                return mt_pos_grad_t_ct<T>()
                       + (unsigned)((grid::TOPOLOGY_HELPERS_COUNT * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        }

        template<typename T>
        __host__ __device__ constexpr unsigned collision_dist_smem_ct()
        {
                // FK arena + sphere pos/radii + normals + align slop
                return mt_pos_arena_ct<T>() + 3 * NCC + NCC + 3 * NCC + 16 / sizeof(T) + 1;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned collision_dist_grad_smem_ct()
        {
                // FK arena + pos/radii/normals + dp/dq batch (NV tangent columns)
                return mt_pos_grad_arena_ct<T>() + 3 * NCC + NCC + 3 * NCC + 3 * NV * NCC + 16 / sizeof(T) + 1;
        }

        // s_dist: NCC per-sphere clearances. s_q is the STORED configuration
        // (quaternion layout on floating base). ALL threads must call (barriers).
        template<typename T>
        __device__ void collision_dist(T* s_dist, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel,
                                      const grid_collision::Environment<T>& env)
        {
                using namespace grid;
                T *s_XmatsHom, *s_temp;
                int* s_topology_helpers;
                ee_carve<T, mt_pos_t_ct<T>()>(s_scratch, &s_XmatsHom, &s_temp, &s_topology_helpers);
                T* s_pos = s_scratch + mt_pos_arena_ct<T>();
                T* s_r = s_pos + 3 * NCC;
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                multi_target_position_inner<T, true>(s_pos, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr);
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
        __device__ void collision_dist_grad(T* s_dist, T* s_ddist, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel,
                                          const grid_collision::Environment<T>& env)
        {
                using namespace grid;
                T *s_XmatsHom, *s_temp;
                int* s_topology_helpers;
                ee_carve<T, mt_pos_grad_t_ct<T>()>(s_scratch, &s_XmatsHom, &s_temp, &s_topology_helpers);
                T* s_pos = s_scratch + mt_pos_grad_arena_ct<T>();
                T* s_r = s_pos + 3 * NCC;
                T* s_normal = s_r + NCC;
                T* s_pos_grad = s_normal + 3 * NCC;
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                multi_target_position_inner<T, true>(s_pos, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr);
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
                multi_target_position_gradient_inner<T, true>(s_pos_grad, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr);
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
