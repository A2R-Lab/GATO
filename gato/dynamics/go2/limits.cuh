#pragma once
// Joint/velocity/effort limit tables for go2, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[12][2] = {
            {-1.0472 - JOINT_LIMIT_MARGIN<T>(), 1.0472 + JOINT_LIMIT_MARGIN<T>()},  // FL_hip_joint
            {-1.5708 - JOINT_LIMIT_MARGIN<T>(), 3.4907 + JOINT_LIMIT_MARGIN<T>()},  // FL_thigh_joint
            {-2.7227 - JOINT_LIMIT_MARGIN<T>(), -0.83776 + JOINT_LIMIT_MARGIN<T>()},  // FL_calf_joint
            {-1.0472 - JOINT_LIMIT_MARGIN<T>(), 1.0472 + JOINT_LIMIT_MARGIN<T>()},  // FR_hip_joint
            {-1.5708 - JOINT_LIMIT_MARGIN<T>(), 3.4907 + JOINT_LIMIT_MARGIN<T>()},  // FR_thigh_joint
            {-2.7227 - JOINT_LIMIT_MARGIN<T>(), -0.83776 + JOINT_LIMIT_MARGIN<T>()},  // FR_calf_joint
            {-1.0472 - JOINT_LIMIT_MARGIN<T>(), 1.0472 + JOINT_LIMIT_MARGIN<T>()},  // RL_hip_joint
            {-0.5236 - JOINT_LIMIT_MARGIN<T>(), 4.5379 + JOINT_LIMIT_MARGIN<T>()},  // RL_thigh_joint
            {-2.7227 - JOINT_LIMIT_MARGIN<T>(), -0.83776 + JOINT_LIMIT_MARGIN<T>()},  // RL_calf_joint
            {-1.0472 - JOINT_LIMIT_MARGIN<T>(), 1.0472 + JOINT_LIMIT_MARGIN<T>()},  // RR_hip_joint
            {-0.5236 - JOINT_LIMIT_MARGIN<T>(), 4.5379 + JOINT_LIMIT_MARGIN<T>()},  // RR_thigh_joint
            {-2.7227 - JOINT_LIMIT_MARGIN<T>(), -0.83776 + JOINT_LIMIT_MARGIN<T>()}  // RR_calf_joint
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[12][2] = {
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // FL_hip_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // FL_thigh_joint
            {-15.7 - JOINT_LIMIT_MARGIN<T>(), 15.7 + JOINT_LIMIT_MARGIN<T>()},  // FL_calf_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // FR_hip_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // FR_thigh_joint
            {-15.7 - JOINT_LIMIT_MARGIN<T>(), 15.7 + JOINT_LIMIT_MARGIN<T>()},  // FR_calf_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // RL_hip_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // RL_thigh_joint
            {-15.7 - JOINT_LIMIT_MARGIN<T>(), 15.7 + JOINT_LIMIT_MARGIN<T>()},  // RL_calf_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // RR_hip_joint
            {-30.1 - JOINT_LIMIT_MARGIN<T>(), 30.1 + JOINT_LIMIT_MARGIN<T>()},  // RR_thigh_joint
            {-15.7 - JOINT_LIMIT_MARGIN<T>(), 15.7 + JOINT_LIMIT_MARGIN<T>()}  // RR_calf_joint
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[12][2] = {
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // FL_hip_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // FL_thigh_joint
            {-45.43 - JOINT_LIMIT_MARGIN<T>(), 45.43 + JOINT_LIMIT_MARGIN<T>()},  // FL_calf_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // FR_hip_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // FR_thigh_joint
            {-45.43 - JOINT_LIMIT_MARGIN<T>(), 45.43 + JOINT_LIMIT_MARGIN<T>()},  // FR_calf_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // RL_hip_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // RL_thigh_joint
            {-45.43 - JOINT_LIMIT_MARGIN<T>(), 45.43 + JOINT_LIMIT_MARGIN<T>()},  // RL_calf_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // RR_hip_joint
            {-23.7 - JOINT_LIMIT_MARGIN<T>(), 23.7 + JOINT_LIMIT_MARGIN<T>()},  // RR_thigh_joint
            {-45.43 - JOINT_LIMIT_MARGIN<T>(), 45.43 + JOINT_LIMIT_MARGIN<T>()}  // RR_calf_joint
        };

}  // namespace plant
}  // namespace gato
