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

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext = nullptr)
        {
                // TOPOLOGY_HELPERS_COUNT == 0 for fixed serial chains; the inners never
                // dereference a zero-count helper buffer, so a null pointer is safe.
                int* s_topology_helpers = nullptr;
                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[XIMATS_COUNT];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, (grid::robotModel<T>*)d_dynMem_const, s_temp);
                __syncthreads();

                // d_workspace == nullptr: at the in-smem placement default the inner keeps all
                // scratch in shared memory (INNER_WORKSPACE_BYTES == 0). d_f_ext is body-major
                // 6*NUM_BODIES (or nullptr); gravity is now the last argument.
                grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr, d_f_ext, gato::plant::GRAVITY<T>());
        }

        __host__ __device__ constexpr unsigned forwardDynamics_TempMemSize_Shared()
        {
                return grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_COUNT;
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
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[6], d_f_ext, GRAVITY<T>());
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
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
        }

        __host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT;
        }

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
                // + s_end_effector_pose_gradient (6*NUM_VEL*NEE) + the larger gradient arena.
                return (2 * NX + 2 * NU + 6) + (4 * NQ + 2 * NU) + 6 * NEE + 6 * NQ * NEE
                       + grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT;
        }

        // Build the per-element weight/target/bound buffers. ee_weight = q_cost (running)
        // or N_cost (terminal). u_cost=0 (+ caller's mu_u=0) disables the control reg/barrier
        // on the terminal value knot. s_eePos_traj is the 6-wide reference (only xyz used).
        template<typename T>
        __device__ void buildTrackingCostBuffers(
            T* s_Q, T* s_R, T* s_W, T* s_x_des, T* s_u_des, T* s_ee_des,
            T* s_q_lo, T* s_q_hi, T* s_qd_lo, T* s_qd_hi, T* s_u_lo, T* s_u_hi,
            const T* s_eePos_traj, T qd_cost, T u_cost, T ee_weight)
        {
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int nth = blockDim.x * blockDim.y;
                // s_Q: 0 on the q-block (EE term carries position cost), qd_cost on the qd-block.
                for (int i = tid; i < NX; i += nth) { s_Q[i] = (i < NQ) ? static_cast<T>(0) : qd_cost; s_x_des[i] = static_cast<T>(0); }
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
            T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, bool is_terminal)
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
                                            s_eePos_traj, qd_cost, u_w, ee_w);
                __syncthreads();
                __shared__ T s_out[1];
                grid_plant::tracking_cost<T, 0>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des,
                                                s_Q, s_R, s_W, s_q_lo, s_q_hi, q_lim_cost,
                                                s_qd_lo, s_qd_hi, vel_lim_cost, s_u_lo, s_u_hi, mu_u,
                                                s_eePos, s_scratch, d_robotModel);
                __syncthreads();
                return s_out[0];
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
            T qd_cost, T u_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T ee_weight)
        {
                T* s_Q = s_temp;            T* s_R = s_Q + NX;          T* s_W = s_R + NU;
                T* s_x_des = s_W + 3;       T* s_u_des = s_x_des + NX;  T* s_ee_des = s_u_des + NU;
                T* s_q_lo = s_ee_des + 3;   T* s_q_hi = s_q_lo + NQ;
                T* s_qd_lo = s_q_hi + NQ;   T* s_qd_hi = s_qd_lo + NQ;
                T* s_u_lo = s_qd_hi + NQ;   T* s_u_hi = s_u_lo + NU;
                T* s_eePos = s_u_hi + NU;   T* s_eePosGrad = s_eePos + 6 * NEE;
                T* s_scratch = s_eePosGrad + 6 * NQ * NEE;

                buildTrackingCostBuffers<T>(s_Q, s_R, s_W, s_x_des, s_u_des, s_ee_des,
                                            s_q_lo, s_q_hi, s_qd_lo, s_qd_hi, s_u_lo, s_u_hi,
                                            s_eePos_traj, qd_cost, u_cost, ee_weight);
                __syncthreads();
                // Block layout note: grid_plant writes s_Qk/s_Rk COLUMN-major; GATO's old code
                // wrote row-major (i*NX+j). The tracking Hessian (diag(qd,barrier) + JᵀWJ) is
                // SYMMETRIC, so the two layouts are bit-identical — do NOT "transpose-fix" this.
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

}  // namespace plant
}  // namespace gato
