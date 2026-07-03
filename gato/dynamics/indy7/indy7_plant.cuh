#pragma once

#include <stdio.h>
#include "grid.cuh"
#include "glass.cuh"
#include "settings.h"
#include "utils/linalg.cuh"
// #include <random>
// #define RANDOM_MEAN 0
// #define RANDOM_STDEV 0.001
// std::default_random_engine randEng(time(0)); //seed
// std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

using namespace sqp;

namespace gato {
namespace plant {

        // Dimension aliases (the generated grid.cuh now exposes NUM_JOINTS / NUM_POS /
        // NUM_VEL / NUM_EES instead of the old NQ / NX / NU / NEE / EE_POS_SIZE).
        inline constexpr int NQ          = grid::NUM_JOINTS;       // joints (fixed base: NUM_POS)
        inline constexpr int NX          = 2 * grid::NUM_JOINTS;   // state = [q; qd]
        inline constexpr int NU          = grid::NUM_JOINTS;       // controls
        inline constexpr int NEE         = grid::NUM_EES;          // end-effector count
        inline constexpr int EE_POS_SIZE = 6;                      // pose size (xyz + orientation)

        template<class T>
        __host__ __device__ constexpr T PI()
        {
                return static_cast<T>(3.14159);
        }
        template<class T>
        __host__ __device__ constexpr T GRAVITY()
        {
                // -9.81: match the physical downward gravity of the pinocchio MPC sim.
                // grid.cuh treats +g as upward base accel (= -pinocchio); a positive
                // scalar inverts the solver's gravity compensation (see iiwa14_plant.cuh).
                return static_cast<T>(-9.81);
        }

        // template<class T>
        // __host__ __device__ constexpr T COST_Q()
        // {
        //         return static_cast<T>(q_COST);
        // }

        // template<class T>
        // __host__ __device__ constexpr T COST_QD()
        // {
        //         return static_cast<T>(dq_COST);
        // }

        // template<class T>
        // __host__ __device__ constexpr T COST_R()
        // {
        //         return static_cast<T>(u_COST);
        // }

        // template<class T>
        // __host__ __device__ constexpr T COST_TERMINAL()
        // {
        //         return static_cast<T>(N_COST);
        // }

        // template<class T>
        // __host__ __device__ constexpr T COST_BARRIER()
        // {
        //         return static_cast<T>(q_lim_COST);
        // }

        template<class T>
        __host__ __device__ constexpr T JOINT_LIMIT_MARGIN()
        {
                return static_cast<T>(-0.1);
        }

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[6][2] = {
            // from indy7.urdf
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-3.7520 - JOINT_LIMIT_MARGIN<T>(), 3.7520 + JOINT_LIMIT_MARGIN<T>()}   // joint 5
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[6][2] = {
            // from indy7.urdf
            {-2.61 - JOINT_LIMIT_MARGIN<T>(), 2.61 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-2.61 - JOINT_LIMIT_MARGIN<T>(), 2.61 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-2.61 - JOINT_LIMIT_MARGIN<T>(), 2.61 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-3.14 - JOINT_LIMIT_MARGIN<T>(), 3.14 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-3.14 - JOINT_LIMIT_MARGIN<T>(), 3.14 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-3.14 - JOINT_LIMIT_MARGIN<T>(), 3.14 + JOINT_LIMIT_MARGIN<T>()}   // joint 5
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[6][2] = {
            // from indy7.urdf
            {-431.97 - JOINT_LIMIT_MARGIN<T>(), 431.97 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-431.97 - JOINT_LIMIT_MARGIN<T>(), 431.97 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-197.23 - JOINT_LIMIT_MARGIN<T>(), 197.23 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()}   // joint 5
        };

        template<class T>
        __host__ __device__ constexpr const T (&JOINT_LIMITS())[6][2]
        {
                return JOINT_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&VEL_LIMITS())[6][2]
        {
                return VEL_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&CTRL_LIMITS())[6][2]
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

        template<class T>
        __device__ T jointBarrier(T q, T q_min, T q_max)
        {
                T       dist_min = q - q_min;
                T       dist_max = q_max - q;
                dist_min = (dist_min <= 1e-10) ? 1e-10 : dist_min;
                dist_max = (dist_max <= 1e-10) ? 1e-10 : dist_max;
                return -log(dist_min) - log(dist_max);
        }

        template<class T>
        __device__ T jointBarrierGradient(T q, T q_min, T q_max)
        {
                T       dist_min = q - q_min;
                T       dist_max = q_max - q;
                dist_min = (dist_min <= 1e-6) ? 1e-6 : dist_min;
                dist_max = (dist_max <= 1e-6) ? 1e-6 : dist_max;
                return (-1 / dist_min) + (1 / dist_max);
        }

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext = nullptr)
        {
                // TOPOLOGY_HELPERS_COUNT == 0 for indy7 (fixed serial chain); inners never deref it.
                int* s_topology_helpers = nullptr;
                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[432];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, (grid::robotModel<T>*)d_dynMem_const, s_temp);
                __syncthreads();

                // d_workspace == nullptr (in-smem placement default); d_f_ext is body-major 6*NUM_BODIES
                // (or nullptr); gravity is now the last argument.
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

                int* s_topology_helpers = nullptr;   // TOPOLOGY_HELPERS_COUNT == 0 (indy7 fixed chain)
                T* s_XImats = s_XITemp;
                T* s_vaf = &s_XITemp[432];
                T* s_dc_du = &s_vaf[108];
                T* s_Minv = &s_dc_du[72];
                T* s_temp = &s_Minv[36];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp, /*d_workspace*/nullptr);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[6], d_f_ext, GRAVITY<T>());
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, d_f_ext, GRAVITY<T>());
                // Inverse-dynamics gradient is not f_ext-threaded in GRiD (matches prior GATO behavior).
                // d_temp_spill == nullptr (USE_DA_DF_SPILL=false).
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, /*d_temp_spill*/nullptr, GRAVITY<T>());

                // Option A: GATO keeps its own hand-composition df_du = -Minv * dc_du and packs
                // dqdd/du = Minv into s_df_du[72..] itself (the integrator gradient reads it at offset 72).
                for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < 72; ind += blockDim.x * blockDim.y) {
                        int row = ind % 6;
                        int dc_col_offset = ind - row;
                        // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                        T val = static_cast<T>(0);
#pragma unroll
                        for (int col = 0; col < 6; col++) {
                                int index = (row <= col) * (col * 6 + row) + (row > col) * (row * 6 + col);
                                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                        }
                        s_df_du[ind] = -val;
                        if (INCLUDE_DU && ind < 36) {
                                int col = ind / 6;
                                int index = (row <= col) * (col * 6 + row) + (row > col) * (row * 6 + col);
                                s_df_du[ind + 72] = s_Minv[index];
                        }
                }
                __syncthreads();
        }

        __host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT;
        }

        // ===================================================================
        // grid_plant::tracking_cost ADAPTER (replaces the hand-rolled cost below)
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
                return (2 * NX + 2 * NU + 6) + (4 * NQ + 2 * NU) + 6 * NEE
                       + grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT;
        }
        template<typename T>
        __host__ __device__ constexpr unsigned trackingCostGradHess_TempMemCt()
        {
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
                for (int i = tid; i < NX; i += nth) { s_Q[i] = (i < NQ) ? static_cast<T>(0) : qd_cost; s_x_des[i] = static_cast<T>(0); }
                for (int i = tid; i < NU; i += nth) { s_R[i] = u_cost; s_u_des[i] = static_cast<T>(0); }
                for (int i = tid; i < 3;  i += nth) { s_W[i] = ee_weight; s_ee_des[i] = s_eePos_traj[i]; }
                for (int i = tid; i < NQ; i += nth) {
                        s_q_lo[i]  = JOINT_LIMITS<T>()[i][0]; s_q_hi[i]  = JOINT_LIMITS<T>()[i][1];
                        s_qd_lo[i] = VEL_LIMITS<T>()[i][0];   s_qd_hi[i] = VEL_LIMITS<T>()[i][1];
                }
                for (int i = tid; i < NU; i += nth) { s_u_lo[i] = CTRL_LIMITS<T>()[i][0]; s_u_hi[i] = CTRL_LIMITS<T>()[i][1]; }
        }

        // VALUE adapter (replaces trackingcost). Returns the total scalar cost for one knot.
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

        // GRAD+HESS adapter (replaces trackingCostGradientAndHessian). ee_weight = q_cost
        // (running, at s_x) or N_cost (terminal, at x_{k+1} — see _lastblock rewire / PR #17).
        // For a terminal R-less call, pass throwaway s_rk/s_Rk.
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
                // grid_plant writes s_Qk/s_Rk COLUMN-major; GATO's old code was row-major
                // (i*NX+j). The tracking Hessian is SYMMETRIC, so identical — do NOT "fix".
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

}  // namespace plant
}  // namespace gato
