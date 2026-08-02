#pragma once

#include <stdio.h>
#include "grid.cuh"
#include "glass.cuh"
#include "settings.h"
#include "utils/linalg.cuh"

using namespace sqp;

// Generic plant adapter: everything robot-specific comes from the generated
// grid.cuh (dimensions, dynamics, grid_plant costs) and the generated
// limits.cuh (URDF <limit> tables). Both are resolved through the per-robot
// include dir that CMake adds for each (plant, N) module — this file is
// robot-agnostic and shared by all plants.

namespace gato {
namespace plant {

        // Dimension aliases (the generated grid.cuh now exposes NUM_JOINTS / NUM_POS /
        // NUM_VEL / NUM_EES instead of the old NQ / NX / NU / NEE / EE_POS_SIZE).
        inline constexpr int NQ          = grid::NUM_JOINTS;       // joints (fixed base: NUM_POS)
        inline constexpr int NX          = 2 * grid::NUM_JOINTS;   // state = [q; qd]
        inline constexpr int NU          = grid::NUM_JOINTS;       // controls
        inline constexpr int NEE         = grid::NUM_EES;          // end-effector count
        inline constexpr int EE_POS_SIZE = 6;                      // pose size (xyz + orientation)

        // Shared-memory extents of the grid:: inner buffers this adapter carves by hand.
        inline constexpr int XIMATS_COUNT = 72 * grid::NUM_JOINTS; // X + I (6x6 each) per joint
        inline constexpr int VAF_COUNT    = 18 * grid::NUM_BODIES; // v, a, f (6 each) per body

#if GATO_CONTACT_FORCES
    #if USE_EXACT_HESSIAN
        #error "GATO_CONTACT_FORCES x USE_EXACT_HESSIAN unsupported (fdsva_so is not fc-aware; W2+ item)"
    #endif
        // CL-3a: NU stays the ACTUATED width (the generated grid_plant cost code is
        // NU-wide); gato::constants::CONTROL_SIZE = NU + FC is the solver-facing width.
        inline constexpr int FC         = 6 * grid::NUM_CONTACT_FRAMES; // wrench slots appended to u
        inline constexpr int FEXT_COUNT = 6 * grid::NUM_BODIES;         // per-body wrench array
        // Persistent fc scratch APPENDED after the FD/FD_DU arenas (inner-call scratch
        // reuses the existing temp region — ID-gradient's need is the high-water mark):
        // XmatsHom + fext + dtau_dfext + dfext_dfc + dtau_dfc.
        inline constexpr int FC_PERSIST_COUNT =
            grid::XHOM_T_COUNT + FEXT_COUNT + NQ * FEXT_COUNT + FEXT_COUNT * FC + NQ * FC;
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

        static_assert(sizeof(JOINT_LIMITS_DATA<float>) == sizeof(float) * 2 * NQ,
                      "limits.cuh joint table does not match grid.cuh NUM_JOINTS");
        static_assert(sizeof(VEL_LIMITS_DATA<float>) == sizeof(float) * 2 * NQ,
                      "limits.cuh velocity table does not match grid.cuh NUM_JOINTS");
        static_assert(sizeof(CTRL_LIMITS_DATA<float>) == sizeof(float) * 2 * NQ,
                      "limits.cuh effort table does not match grid.cuh NUM_JOINTS");

        template<class T>
        __host__ __device__ constexpr const T (&JOINT_LIMITS())[NQ][2]
        {
                return JOINT_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&VEL_LIMITS())[NQ][2]
        {
                return VEL_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&CTRL_LIMITS())[NQ][2]
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
                if (INCLUDE_DU) {
                        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                        const int nth = blockDim.x * blockDim.y;
                        // ID gradient is done -> s_temp is free scratch again; XImats/XmatsHom still live.
                        grid::f_ext_gradient_jacobianT_inner<T>(s_dtau_dfext, s_q, s_XImats, s_topology_helpers, s_temp);
                        __syncthreads();
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
            T q_pos_cost = static_cast<T>(0), const T* d_q_nom = nullptr)
        {
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                // s_Q q-block: q_pos_cost toward d_q_nom (default 0/nullptr = the historic
                // EE-only cost — the q-block was hardcoded 0 and the limit barrier was the
                // only q-space term; q_pos_cost is the nullspace posture anchor, PDDP ask
                // 2026-08-02). qd_cost on the qd-block.
                for (int i = tid; i < NX; i += nth) {
                        s_Q[i] = (i < NQ) ? q_pos_cost : qd_cost;
                        s_x_des[i] = (i < NQ && d_q_nom != nullptr) ? d_q_nom[i] : static_cast<T>(0);
                }
                for (int i = tid; i < NU; i += nth) { s_R[i] = u_cost; s_u_des[i] = static_cast<T>(0); }
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
            T fc_cost = static_cast<T>(0))
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
                                            s_eePos_traj, qd_cost, u_w, ee_w, q_pos_cost, d_q_nom);
                __syncthreads();
                __shared__ T s_out[1];
                grid_plant::tracking_cost<T, 0>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, mu_u,
                                                s_eePos, s_scratch, d_robotModel);
                __syncthreads();
                T total = s_out[0];
#if GATO_CONTACT_FORCES
                // fc regularization (0.5*fc_cost*|fc|^2; terminal knot has no control slot,
                // matching the u-reg drop). Uniform per-thread compute — no shared write.
                if (!is_terminal) {
                        T fcreg = static_cast<T>(0);
                        for (int j = 0; j < FC; j++) { fcreg += s_u[NU + j] * s_u[NU + j]; }
                        total += static_cast<T>(0.5) * fc_cost * fcreg;
                }
#else
                (void)fc_cost;
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
            T fc_cost = static_cast<T>(0))
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
                                            s_eePos_traj, qd_cost, u_cost, ee_weight, q_pos_cost, d_q_nom);
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
                        // fc gradient rows = fc_cost * fc (fc reference is 0).
                        constexpr int CS = NU + FC;
                        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                        const int nth = blockDim.x * blockDim.y;
                        for (int ind = tid; ind < CS * CS; ind += nth) {
                                int r = ind % CS; int c = ind / CS;
                                s_Rk[ind] = (r < NU && c < NU) ? s_R7[r + NU * c]
                                          : (r == c ? fc_cost : static_cast<T>(0));
                        }
                        for (int j = tid; j < CS; j += nth) {
                                s_rk[j] = (j < NU) ? s_r7[j] : fc_cost * s_u[j];
                        }
                        __syncthreads();
                }
#endif
        }

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

        template<typename T>
        __host__ __device__ constexpr unsigned eePos_TempMemCt()
        {
                return grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned eePosGrad_TempMemCt()
        {
                return grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT;
        }

        template<typename T>
        __device__ void eePos(T* s_pose, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                using namespace grid;
                unsigned char* s_arena = reinterpret_cast<unsigned char*>(s_scratch);
                size_t         off = grid_align_up(0, alignof(T));
                T*             s_XmatsHom = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(144);   // XmatsHom incl. the fixed EE + world slots
                off = grid_align_up(off, alignof(T));
                T* s_temp = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(32);
                int*           s_topology_helpers = nullptr;
                unsigned char* s_linalg_smem = nullptr;
                if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                        off = grid_align_up(off, static_cast<size_t>(16));
                        s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, off);
                }
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_inner_EE<T, true>(s_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
        }

        template<typename T>
        __device__ void eePosGrad(T* s_pose, T* s_grad, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel)
        {
                using namespace grid;
                unsigned char* s_arena = reinterpret_cast<unsigned char*>(s_scratch);
                size_t         off = grid_align_up(0, alignof(T));
                T*             s_XmatsHom = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(144);   // XmatsHom incl. the fixed EE + world slots
                off = grid_align_up(off, alignof(T));
                T* s_temp = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(168);
                int*           s_topology_helpers = nullptr;
                unsigned char* s_linalg_smem = nullptr;
                if (static_cast<size_t>(GRID_EE_LINALG_SHARED_BYTES<T>()) > 0) {
                        off = grid_align_up(off, static_cast<size_t>(16));
                        s_linalg_smem = grid_arena_ptr<unsigned char>(s_arena, off);
                }
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                end_effector_pose_inner_EE<T, true>(s_pose, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
                __syncthreads();
                end_effector_pose_gradient_inner_EE<T, true>(s_grad, s_q, s_XmatsHom, nullptr, s_topology_helpers, s_temp, nullptr, s_linalg_smem);
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

        template<typename T>
        __host__ __device__ constexpr unsigned collisionDist_TempMemCt()
        {
                // XmatsHom + FK temp (value arena) + sphere pos/radii + normals + align slop
                return 144 + 144 + 3 * NCC + NCC + 3 * NCC + 16 / sizeof(T) + 1;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned collisionDistGrad_TempMemCt()
        {
                // XmatsHom + FK temp (gradient arena) + pos/radii/normals + dp/dq batch
                return 144 + 570 + 3 * NCC + NCC + 3 * NCC + 3 * NQ * NCC + 16 / sizeof(T) + 1;
        }

        // s_dist: NCC per-sphere clearances. ALL threads must call (barriers).
        template<typename T>
        __device__ void collisionDist(T* s_dist, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel,
                                      const grid_collision::Environment<T>& env)
        {
                using namespace grid;
                unsigned char* s_arena = reinterpret_cast<unsigned char*>(s_scratch);
                size_t         off = grid_align_up(0, alignof(T));
                T*             s_XmatsHom = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(144);
                off = grid_align_up(off, alignof(T));
                T* s_temp = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(144);
                T* s_pos = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(3 * NCC);
                T* s_r = grid_arena_ptr<T>(s_arena, off);
                int* s_topology_helpers = nullptr;
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                multi_target_position_inner<T, true>(s_pos, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, nullptr);
                grid_collision::load_collision_radii<T>(s_r);
                __syncthreads();
                for (int i = threadIdx.x; i < NCC; i += blockDim.x) {
                        T nx, ny, nz;
                        s_dist[i] = grid_collision::grid_cc_nearest_obstacle<T>(env, s_pos[3 * i], s_pos[3 * i + 1], s_pos[3 * i + 2], s_r[i], &nx, &ny, &nz);
                }
                __syncthreads();
        }

        // s_dist: NCC clearances; s_ddist: NCC x NQ sphere-major Jacobian
        // (s_ddist[i*NQ + vi] = n_i^T dp_i/dq_vi). ALL threads must call.
        template<typename T>
        __device__ void collisionDistGrad(T* s_dist, T* s_ddist, const T* s_q, T* s_scratch, const grid::robotModel<T>* d_robotModel,
                                          const grid_collision::Environment<T>& env)
        {
                using namespace grid;
                unsigned char* s_arena = reinterpret_cast<unsigned char*>(s_scratch);
                size_t         off = grid_align_up(0, alignof(T));
                T*             s_XmatsHom = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(144);
                off = grid_align_up(off, alignof(T));
                T* s_temp = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(570);
                T* s_pos = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(3 * NCC);
                T* s_r = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(NCC);
                T* s_normal = grid_arena_ptr<T>(s_arena, off);
                off += sizeof(T) * static_cast<size_t>(3 * NCC);
                T* s_pos_grad = grid_arena_ptr<T>(s_arena, off);
                int* s_topology_helpers = nullptr;
                load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, d_robotModel, s_temp);
                multi_target_position_inner<T, true>(s_pos, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, nullptr);
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
                multi_target_position_gradient_inner<T, true>(s_pos_grad, s_q, s_XmatsHom, s_topology_helpers, s_temp, nullptr, nullptr);
                __syncthreads();
                for (int ind = threadIdx.x; ind < NCC * NQ; ind += blockDim.x) {
                        const int vi = ind % NQ;
                        const int i = ind / NQ;
                        const int jb = 3 * (NQ * i + vi);
                        s_ddist[i * NQ + vi] = s_normal[3 * i + 0] * s_pos_grad[jb + 0] + s_normal[3 * i + 1] * s_pos_grad[jb + 1] + s_normal[3 * i + 2] * s_pos_grad[jb + 2];
                }
                __syncthreads();
        }

}  // namespace plant
}  // namespace gato
