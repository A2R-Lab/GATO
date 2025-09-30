#pragma once

#include <stdio.h>
#include "iiwa14_grid.cuh"
#include "iiwa14_fext.cuh"
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

        template<class T>
        __host__ __device__ constexpr T PI()
        {
                return static_cast<T>(3.14159);
        }
        template<class T>
        __host__ __device__ constexpr T GRAVITY()
        {
                return static_cast<T>(9.81);
        }

        template<class T>
        __host__ __device__ constexpr T JOINT_LIMIT_MARGIN()
        {
                return static_cast<T>(-0.1);
        }

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[7][2] = {
            // from iiwa14.urdf
            {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-2.09440 - JOINT_LIMIT_MARGIN<T>(), 2.09440 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-2.09440 - JOINT_LIMIT_MARGIN<T>(), 2.09440 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-2.09440 - JOINT_LIMIT_MARGIN<T>(), 2.09440 + JOINT_LIMIT_MARGIN<T>()},  // joint 5
            {-3.05433 - JOINT_LIMIT_MARGIN<T>(), 3.05433 + JOINT_LIMIT_MARGIN<T>()}   // joint 6
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[7][2] = {
            // from iiwa14.urdf
            {-1.48353 - JOINT_LIMIT_MARGIN<T>(), 1.48353 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-1.48353 - JOINT_LIMIT_MARGIN<T>(), 1.48353 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-1.74533 - JOINT_LIMIT_MARGIN<T>(), 1.74533 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-1.30900 - JOINT_LIMIT_MARGIN<T>(), 1.30900 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-2.26893 - JOINT_LIMIT_MARGIN<T>(), 2.26893 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-2.35619 - JOINT_LIMIT_MARGIN<T>(), 2.35619 + JOINT_LIMIT_MARGIN<T>()},   // joint 5
            {-2.35619 - JOINT_LIMIT_MARGIN<T>(), 2.35619 + JOINT_LIMIT_MARGIN<T>()}   // joint 6
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[7][2] = {
            // from iiwa14.urdf
            {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-110.0 - JOINT_LIMIT_MARGIN<T>(), 110.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-40.0 - JOINT_LIMIT_MARGIN<T>(), 40.0 + JOINT_LIMIT_MARGIN<T>()},   // joint 5
            {-40.0 - JOINT_LIMIT_MARGIN<T>(), 40.0 + JOINT_LIMIT_MARGIN<T>()}   // joint 6
        };

        template<class T>
        __host__ __device__ constexpr const T (&JOINT_LIMITS())[7][2]
        {
                return JOINT_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&VEL_LIMITS())[7][2]
        {
                return VEL_LIMITS_DATA<T>;
        }

        template<class T>
        __host__ __device__ constexpr const T (&CTRL_LIMITS())[7][2]
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
                grid::free_robotModel((grid::robotModel<T>*)d_dynMem_const);
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
                // Sign-preserving clamp so gradient points back inside even when violated
                T dist_min = q - q_min;
                T dist_max = q_max - q;
                const T eps = static_cast<T>(1e-6);

                // If inside and very close to boundary, grow magnitude to eps (same sign)
                // If outside, keep sign and ensure magnitude at least eps
                if (dist_min >= static_cast<T>(0)) {
                        if (dist_min < eps) dist_min = eps;
                } else {
                        if (dist_min > -eps) dist_min = -eps;
                }

                if (dist_max >= static_cast<T>(0)) {
                        if (dist_max < eps) dist_max = eps;
                } else {
                        if (dist_max > -eps) dist_max = -eps;
                }

                return (-static_cast<T>(1) / dist_min) + (static_cast<T>(1) / dist_max);
        }

        // Second derivative (Hessian) of the joint barrier:
        // d^2/dq^2 [-log(q - q_min) - log(q_max - q)]
        // = 1/(q - q_min)^2 + 1/(q_max - q)^2
        template<class T>
        __device__ T jointBarrierHessian(T q, T q_min, T q_max)
        {
                // Use magnitude clamp for curvature; sign does not affect squared term
                T dist_min = q - q_min;
                T dist_max = q_max - q;
                const T eps = static_cast<T>(1e-6);

                T abs_min = dist_min >= static_cast<T>(0) ? dist_min : -dist_min;
                T abs_max = dist_max >= static_cast<T>(0) ? dist_max : -dist_max;
                if (abs_min < eps) abs_min = eps;
                if (abs_max < eps) abs_max = eps;

                return static_cast<T>(1.0) / (abs_min * abs_min) + static_cast<T>(1.0) / (abs_max * abs_max);
        }

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const)
        {

                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[1008];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, (grid::robotModel<T>*)d_dynMem_const, s_temp);
                __syncthreads();

                grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gato::plant::GRAVITY<T>());
        }

        // Add external wrench
        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext)
        {

                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[1008];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, (grid::robotModel<T>*)d_dynMem_const, s_temp);
                __syncthreads();

                grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gato::plant::GRAVITY<T>(), d_f_ext);
        }

        __host__ __device__ constexpr unsigned forwardDynamics_TempMemSize_Shared()
        {
                return grid::FD_DYNAMIC_SHARED_MEM_COUNT;
        }

        template<typename T, bool INCLUDE_DU = true>
        __device__ void forwardDynamicsAndGradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const)
        {
                T*                   s_XITemp = s_temp_in;
                grid::robotModel<T>* d_robotModel = (grid::robotModel<T>*)d_dynMem_const;

                T* s_XImats = s_XITemp;
                T* s_vaf = &s_XITemp[504];
                T* s_dc_du = &s_vaf[126];
                T* s_Minv = &s_dc_du[98];
                T* s_temp = &s_Minv[49];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[6], GRAVITY<T>());
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>());
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>());

                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                        int row = ind % 7; 
                        int dc_col_offset = ind - row;
                        // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                        T val = static_cast<T>(0);
                        #pragma unroll
                        for(int col = 0; col < 7; col++) {
                                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                        }
                        s_df_du[ind] = -val;

                        if (INCLUDE_DU && ind < 49){
                                int col = ind / 7;
                                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                                s_df_du[ind + 98] = s_Minv[index];
                        }
                }
                __syncthreads();
        }

        // Add external wrench
        template<typename T, bool INCLUDE_DU = true>
        __device__ void forwardDynamicsAndGradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const, T* d_f_ext)
        {
                T*                   s_XITemp = s_temp_in;
                grid::robotModel<T>* d_robotModel = (grid::robotModel<T>*)d_dynMem_const;

                T* s_XImats = s_XITemp;
                T* s_vaf = &s_XITemp[504];
                T* s_dc_du = &s_vaf[126];
                T* s_Minv = &s_dc_du[98];
                T* s_temp = &s_Minv[49];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[6], GRAVITY<T>(), d_f_ext);
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>(), d_f_ext);
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>());

                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                        int row = ind % 7; 
                        int dc_col_offset = ind - row;
                        // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                        T val = static_cast<T>(0);
                        #pragma unroll
                        for(int col = 0; col < 7; col++) {
                                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                        }
                        s_df_du[ind] = -val;

                        if (INCLUDE_DU && ind < 49){
                                int col = ind / 7;
                                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                                s_df_du[ind + 98] = s_Minv[index];
                        }
                }
                __syncthreads();
        }

        __host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT;
        }

        template<typename T>
        __device__ T trackingcost(uint32_t                   state_size,
                                  uint32_t                   control_size,
                                  uint32_t                   knot_points,
                                  T*                         s_xu,
                                  T*                         s_eePos_traj,
                                  T*                         s_temp,
                                  const grid::robotModel<T>* d_robotModel,
                                  T                          q_cost,
                                  T                          qd_cost,
                                  T                          u_cost,
                                  T                          N_cost,
                                  T                          q_lim_cost,
                                  T                          vel_lim_cost,
                                  T                          ctrl_lim_cost)
        {
                T              err;
                const uint32_t threadsNeeded = state_size / 2 + control_size * (blockIdx.x < knot_points - 1);

                T* s_cost_vec = s_temp;
                T* s_eePos_cost = s_cost_vec + threadsNeeded + 3;
                T* s_scratch = s_eePos_cost + 6;

                grid::end_effector_pose_device<T>(s_eePos_cost, s_xu, s_scratch, d_robotModel);

                for (int i = threadIdx.x; i < threadsNeeded; i += blockDim.x) {
                        if (i < state_size / 2) {
                                err = s_xu[i + state_size / 2];
                                s_cost_vec[i] = static_cast<T>(0.5) * qd_cost * err * err;
                                s_cost_vec[i] += q_lim_cost * jointBarrier(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                s_cost_vec[i] += vel_lim_cost * jointBarrier(s_xu[i + state_size / 2], VEL_LIMITS<T>()[i][0], VEL_LIMITS<T>()[i][1]);
                        } else {
                                err = s_xu[i + state_size / 2];
                                s_cost_vec[i] = static_cast<T>(0.5) * u_cost * err * err;
                                s_cost_vec[i] += ctrl_lim_cost * jointBarrier(s_xu[i + state_size / 2], CTRL_LIMITS<T>()[i - state_size / 2][0], CTRL_LIMITS<T>()[i - state_size / 2][1]);
                        }
                }
#pragma unroll
                for (int i = threadIdx.x; i < 3; i += blockDim.x) {
                        err = s_eePos_cost[i] - s_eePos_traj[i];
                        if (blockIdx.x == KNOT_POINTS - 1) {
                                s_cost_vec[threadsNeeded + i] = 0.5 * N_cost * err * err;
                        } else {
                                s_cost_vec[threadsNeeded + i] = 0.5 * q_cost * err * err;
                        }
                }
                __syncthreads();

                block::reduce<T>(threadsNeeded + 3, s_cost_vec);
                __syncthreads();

                return s_cost_vec[0];
        }

        __host__ unsigned trackingcost_TempMemCt_Shared(uint32_t state_size, uint32_t control_size, uint32_t knot_points)
        {
                // Worst-case per-knot temp shared memory for trackingcost():
                // threadsNeeded (NQ + NU on non-terminal knots) + 3 (position error terms)
                // + EE pose size (6) + EE pose dynamic shared mem used by the GRiD device call.
                // Using grid constants keeps it consistent with compile-time configuration.
                return grid::NQ / 2 + grid::NU + 2 * grid::NEE + grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT;
        }

        template<typename T, bool computeR = true>
        __device__ void trackingCostGradientAndHessian(uint32_t state_size,
                                                       uint32_t control_size,
                                                       T*       s_xu,
                                                       T*       s_eePos_traj,
                                                       T*       s_Qk,
                                                       T*       s_qk,
                                                       T*       s_Rk,
                                                       T*       s_rk,
                                                       T*       s_temp,
                                                       void*    d_robotModel,
                                                       T        q_cost,
                                                       T        qd_cost,
                                                       T        u_cost,
                                                       T        N_cost,
                                                       T        q_lim_cost,
                                                       T        vel_lim_cost,
                                                       T        ctrl_lim_cost)
        {
                T* s_eePos = s_temp;
                T* s_eePos_grad = s_eePos + 6;
                T* s_scratch = s_eePos_grad + (6 * grid::NQ);

                const uint32_t threads_needed = grid::NX + grid::NU * computeR;

                grid::end_effector_pose_device<T>(s_eePos, s_xu, s_scratch, (grid::robotModel<T>*)d_robotModel);
                grid::end_effector_pose_gradient_device<T>(s_eePos_grad, s_xu, s_scratch, (grid::robotModel<T>*)d_robotModel);

                // Gradient (qk, rk)
                for (int i = threadIdx.x; i < threads_needed; i += blockDim.x) {
                        if (i < grid::NX) {
                                if (i < grid::NQ) {
                                        // tracking err
                                        s_qk[i] = (s_eePos_grad[6 * i + 0] * (s_eePos[0] - s_eePos_traj[0]) + s_eePos_grad[6 * i + 1] * (s_eePos[1] - s_eePos_traj[1])
                                                   + s_eePos_grad[6 * i + 2] * (s_eePos[2] - s_eePos_traj[2]))
                                                  * (blockIdx.x == KNOT_POINTS - 1 ? N_cost : q_cost);
                                        // joint barrier
                                        s_qk[i] += q_lim_cost * jointBarrierGradient(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                } else {
                                        s_qk[i] = qd_cost * s_xu[i];
                                        s_qk[i] += vel_lim_cost * jointBarrierGradient(s_xu[i], VEL_LIMITS<T>()[i - grid::NQ][0], VEL_LIMITS<T>()[i - grid::NQ][1]);
                                }
                        } else {
                                s_rk[i - grid::NX] = u_cost * s_xu[i];
                                s_rk[i - grid::NX] += ctrl_lim_cost * jointBarrierGradient(s_xu[i], CTRL_LIMITS<T>()[i - grid::NX][0], CTRL_LIMITS<T>()[i - grid::NX][1]);
                        }
                }
                __syncthreads();

                // Hessian (Qk, Rk)
                for (int i = threadIdx.x; i < threads_needed; i += blockDim.x) {
                        if (i < grid::NX) {
                                for (int j = 0; j < grid::NX; j++) {
                                        if (j < grid::NQ && i < grid::NQ) {
                                                // tracking err
                                                s_Qk[i * grid::NX + j] = ((s_eePos_grad[6 * i + 0] * (s_eePos[0] - s_eePos_traj[0]) + s_eePos_grad[6 * i + 1] * (s_eePos[1] - s_eePos_traj[1])
                                                                           + s_eePos_grad[6 * i + 2] * (s_eePos[2] - s_eePos_traj[2]))
                                                                          * (s_eePos_grad[6 * j + 0] * (s_eePos[0] - s_eePos_traj[0]) + s_eePos_grad[6 * j + 1] * (s_eePos[1] - s_eePos_traj[1])
                                                                             + s_eePos_grad[6 * j + 2] * (s_eePos[2] - s_eePos_traj[2]))) * (blockIdx.x == KNOT_POINTS - 1 ? N_cost : q_cost);

                                                // Add exact diagonal barrier Hessian for joint limits
                                                if (i == j) {
                                                        s_Qk[i * grid::NX + j] += q_lim_cost * jointBarrierHessian<T>(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                                }

                                        } else {
                                                // joint velocity reg
                                                s_Qk[i * grid::NX + j] = (i == j) ? qd_cost : static_cast<T>(0);
                                                if (i == j) {
                                                        // Add exact diagonal barrier Hessian for velocity limits
                                                        s_Qk[i * grid::NX + j] += vel_lim_cost * jointBarrierHessian<T>(s_xu[i], VEL_LIMITS<T>()[i - grid::NQ][0], VEL_LIMITS<T>()[i - grid::NQ][1]);
                                                }
                                        }
                                }
                        } else {
                                uint32_t offset = i - grid::NX;
                                for (int j = 0; j < grid::NU; j++) { 
                                        s_Rk[offset * grid::NU + j] = (offset == j) ? u_cost : static_cast<T>(0);
                                        if (offset == j) {
                                                // Add exact diagonal barrier Hessian for control limits
                                                s_Rk[offset * grid::NU + j] += ctrl_lim_cost * jointBarrierHessian<T>(s_xu[i], CTRL_LIMITS<T>()[offset][0], CTRL_LIMITS<T>()[offset][1]);
                                        }
                                }
                        }
                }
                __syncthreads();
        }

        template<typename T>
        __device__ void trackingCostGradientAndHessian_lastblock(uint32_t state_size,
                                                                 uint32_t control_size,
                                                                 T*       s_xux,
                                                                 T*       s_eePos_traj,
                                                                 T*       s_Qk,
                                                                 T*       s_qk,
                                                                 T*       s_Rk,
                                                                 T*       s_rk,
                                                                 T*       s_Qkp1,
                                                                 T*       s_qkp1,
                                                                 T*       s_temp,
                                                                 void*    d_dynMem_const,
                                                                 T        q_cost,
                                                                 T        qd_cost,
                                                                 T        u_cost,
                                                                 T        N_cost,
                                                                 T        q_lim_cost,
                                                                 T        vel_lim_cost,
                                                                 T        ctrl_lim_cost)
        {
                trackingCostGradientAndHessian<T>(state_size, control_size, s_xux, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp, d_dynMem_const, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
                trackingCostGradientAndHessian<T, false>(
                    state_size, control_size, s_xux, &s_eePos_traj[6], s_Qkp1, s_qkp1, nullptr, nullptr, s_temp, d_dynMem_const, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
        }

        __host__ __device__ constexpr unsigned trackingCostGradientAndHessian_TempMemSize_Shared()
        {
                return grid::DEE_POS_DYNAMIC_SHARED_MEM_COUNT + 6 + 6 * grid::NQ;
        }
}  // namespace plant
}  // namespace gato
