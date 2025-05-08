#pragma once

#include <stdio.h>
#include "indy7_grid.cuh"
#include "indy7_fext.cuh"
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
        __host__ __device__ constexpr T COST_Q()
        {
                return static_cast<T>(q_COST);
        }

        template<class T>
        __host__ __device__ constexpr T COST_QD()
        {
                return static_cast<T>(dq_COST);
        }

        template<class T>
        __host__ __device__ constexpr T COST_R()
        {
                return static_cast<T>(u_COST);
        }

        template<class T>
        __host__ __device__ constexpr T COST_TERMINAL()
        {
                return static_cast<T>(N_COST);
        }

        template<class T>
        __host__ __device__ constexpr T COST_BARRIER()
        {
                return static_cast<T>(q_lim_COST);
        }

        template<class T>
        __host__ __device__ constexpr T JOINT_LIMIT_MARGIN()
        {
                return static_cast<T>(-0.5);
        }

        __device__ constexpr float JOINT_LIMITS_DATA[6][2] = {
            // from indy7.urdf
            {-3.0543f, 3.0543f},  // joint 0
            {-3.0543f, 3.0543f},  // joint 1
            {-3.0543f, 3.0543f},  // joint 2
            {-3.0543f, 3.0543f},  // joint 3
            {-3.0543f, 3.0543f},  // joint 4
            {-3.7520f, 3.7520f},  // joint 5
        };

        template<class T>
        __host__ __device__ constexpr const float (&JOINT_LIMITS())[6][2]
        {
                return JOINT_LIMITS_DATA;
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
                const T margin = JOINT_LIMIT_MARGIN<T>();
                T       dist_min = q - (q_min + margin);
                T       dist_max = (q_max - margin) - q;
                return -log(dist_min) - log(dist_max);
        }

        template<class T>
        __device__ T jointBarrierGradient(T q, T q_min, T q_max)
        {
                const T margin = JOINT_LIMIT_MARGIN<T>();
                T       dist_min = q - (q_min + margin);
                T       dist_max = (q_max - margin) - q;
                return -1 / dist_min + 1 / dist_max;
        }

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const)
        {

                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[864];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, (grid::robotModel<float>*)d_dynMem_const, s_temp);
                __syncthreads();

                grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gato::plant::GRAVITY<T>());
        }

        // Add external wrench
        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext)
        {

                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[864];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, (grid::robotModel<float>*)d_dynMem_const, s_temp);
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
                T* s_vaf = &s_XITemp[432];
                T* s_dc_du = &s_vaf[108];
                T* s_Minv = &s_dc_du[72];
                T* s_temp = &s_Minv[36];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[6], GRAVITY<T>());
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>());
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>());

                for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < 72; ind += blockDim.x * blockDim.y) {
                        int row = ind % 6;
                        int dc_col_offset = ind - row;
                        // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                        T val = static_cast<T>(0);
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
        }

        // Add external wrench
        template<typename T, bool INCLUDE_DU = true>
        __device__ void forwardDynamicsAndGradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const, T* d_f_ext)
        {
                T*                   s_XITemp = s_temp_in;
                grid::robotModel<T>* d_robotModel = (grid::robotModel<T>*)d_dynMem_const;

                T* s_XImats = s_XITemp;
                T* s_vaf = &s_XITemp[432];
                T* s_dc_du = &s_vaf[108];
                T* s_Minv = &s_dc_du[72];
                T* s_temp = &s_Minv[36];
                grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                // TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
                grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[6], GRAVITY<T>(), d_f_ext);
                grid::forward_dynamics_finish<T>(s_qdd, s_u, s_c, s_Minv);
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>(), d_f_ext);
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>());

                // 6x12 elements
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
        }

        __host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT;
        }

        template<typename T>
        __device__ T trackingcost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T* s_xu, T* s_eePos_traj, T* s_temp, const grid::robotModel<T>* d_robotModel)
        {
                T              err;
                const uint32_t threadsNeeded = state_size / 2 + control_size * (blockIdx.x < knot_points - 1);

                T* s_cost_vec = s_temp;
                T* s_eePos_cost = s_cost_vec + threadsNeeded + 3;
                T* s_scratch = s_eePos_cost + 6;

                grid::end_effector_positions_device<T>(s_eePos_cost, s_xu, s_scratch, d_robotModel);

                for (int i = threadIdx.x; i < threadsNeeded; i += blockDim.x) {
                        if (i < state_size / 2) {
                                err = s_xu[i + state_size / 2];
                                s_cost_vec[i] = static_cast<T>(0.5) * COST_QD<T>() * err * err;
                                s_cost_vec[i] += COST_BARRIER<T>() * jointBarrier(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                        } else {
                                err = s_xu[i + state_size / 2];
                                s_cost_vec[i] = static_cast<T>(0.5) * COST_R<T>() * err * err;
                        }
                }
#pragma unroll
                for (int i = threadIdx.x; i < 3; i += blockDim.x) {
                        err = s_eePos_cost[i] - s_eePos_traj[i];
                        if (blockIdx.x == KNOT_POINTS - 1) {
                                s_cost_vec[threadsNeeded + i] = 0.5 * COST_TERMINAL<T>() * err * err;
                        } else {
                                s_cost_vec[threadsNeeded + i] = 0.5 * COST_Q<T>() * err * err;
                        }
                }
                __syncthreads();

                block::reduce<T>(threadsNeeded + 3, s_cost_vec);
                __syncthreads();

                return s_cost_vec[0];
        }

        __host__ unsigned trackingcost_TempMemCt_Shared(uint32_t state_size, uint32_t control_size, uint32_t knot_points)
        {
                return grid::NQ / 2 + grid::NU + 2 * grid::NEE + grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT;
        }

        template<typename T, bool computeR = true>
        __device__ void trackingCostGradientAndHessian(uint32_t state_size, uint32_t control_size, T* s_xu, T* s_eePos_traj, T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_temp, void* d_robotModel)
        {
                T* s_eePos = s_temp;
                T* s_eePos_grad = s_eePos + 6;
                T* s_scratch = s_eePos_grad + (6 * grid::NQ);

                const uint32_t threads_needed = grid::NX + grid::NU * computeR;

                grid::end_effector_positions_device<T>(s_eePos, s_xu, s_scratch, (grid::robotModel<T>*)d_robotModel);
                grid::end_effector_positions_gradient_device<T>(s_eePos_grad, s_xu, s_scratch, (grid::robotModel<T>*)d_robotModel);

                // Gradient (qk, rk)
                for (int i = threadIdx.x; i < threads_needed; i += blockDim.x) {
                        if (i < grid::NX) {
                                if (i < grid::NQ) {
                                        // tracking err
                                        s_qk[i] = (s_eePos_grad[6 * i + 0] * (s_eePos[0] - s_eePos_traj[0]) + s_eePos_grad[6 * i + 1] * (s_eePos[1] - s_eePos_traj[1])
                                                   + s_eePos_grad[6 * i + 2] * (s_eePos[2] - s_eePos_traj[2]))
                                                  * (blockIdx.x == KNOT_POINTS - 1 ? COST_TERMINAL<T>() : COST_Q<T>());
                                        // joint barrier
                                        s_qk[i] += COST_BARRIER<T>() * jointBarrierGradient(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                } else {
                                        s_qk[i] = COST_QD<T>() * s_xu[i];
                                }
                        } else {
                                s_rk[i - grid::NX] = COST_R<T>() * s_xu[i];
                        }
                }
                // __syncthreads();

                // Hessian (Qk, Rk)
                for (int i = threadIdx.x; i < threads_needed; i += blockDim.x) {
                        if (i < grid::NX) {
                                for (int j = 0; j < grid::NX; j++) {
                                        if (j < grid::NQ && i < grid::NQ) {
                                                // tracking err
                                                s_Qk[i * grid::NX + j] = (s_eePos_grad[6 * i + 0] * s_eePos_grad[6 * j + 0] + s_eePos_grad[6 * i + 1] * s_eePos_grad[6 * j + 1]
                                                                          + s_eePos_grad[6 * i + 2] * s_eePos_grad[6 * j + 2])
                                                                         * (blockIdx.x == KNOT_POINTS - 1 ? COST_TERMINAL<T>() : COST_Q<T>());
                                                // joint barrier
                                                if (i == j) {
                                                        const T margin = JOINT_LIMIT_MARGIN<T>();
                                                        T       dist_min = s_xu[i] - (JOINT_LIMITS<T>()[i][0] + margin);
                                                        T       dist_max = (JOINT_LIMITS<T>()[i][1] - margin) - s_xu[i];
                                                        s_Qk[i * state_size + j] += COST_BARRIER<T>() * (1 / (dist_min * dist_min) + 1 / (dist_max * dist_max));
                                                }
                                        } else {
                                                // joint velocity reg
                                                s_Qk[i * grid::NX + j] = (i == j) ? COST_QD<T>() : static_cast<T>(0);
                                        }
                                }
                        } else {
                                uint32_t offset = i - grid::NX;
                                for (int j = 0; j < grid::NU; j++) { s_Rk[offset * grid::NU + j] = (offset == j) ? COST_R<T>() : static_cast<T>(0); }
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
                                                                 void*    d_dynMem_const)
        {
                trackingCostGradientAndHessian<T>(state_size, control_size, s_xux, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp, d_dynMem_const);
                trackingCostGradientAndHessian<T, false>(state_size, control_size, s_xux, &s_eePos_traj[6], s_Qkp1, s_qkp1, nullptr, nullptr, s_temp, d_dynMem_const);
        }

        __host__ __device__ constexpr unsigned trackingCostGradientAndHessian_TempMemSize_Shared()
        {
                return grid::DEE_POS_DYNAMIC_SHARED_MEM_COUNT + 6 + 6 * grid::NQ;
        }
}  // namespace plant
}  // namespace gato
