#pragma once
// Joint/velocity/effort limit tables for fr3, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[7][2] = {
            {-2.9007 - JOINT_LIMIT_MARGIN<T>(), 2.9007 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint1
            {-1.8361 - JOINT_LIMIT_MARGIN<T>(), 1.8361 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint2
            {-2.9007 - JOINT_LIMIT_MARGIN<T>(), 2.9007 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint3
            {-3.077 - JOINT_LIMIT_MARGIN<T>(), -0.1169 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint4
            {-2.8763 - JOINT_LIMIT_MARGIN<T>(), 2.8763 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint5
            {0.4398 - JOINT_LIMIT_MARGIN<T>(), 4.6216 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint6
            {-3.0508 - JOINT_LIMIT_MARGIN<T>(), 3.0508 + JOINT_LIMIT_MARGIN<T>()}  // fr3_joint7
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[7][2] = {
            {-2.62 - JOINT_LIMIT_MARGIN<T>(), 2.62 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint1
            {-2.62 - JOINT_LIMIT_MARGIN<T>(), 2.62 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint2
            {-2.62 - JOINT_LIMIT_MARGIN<T>(), 2.62 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint3
            {-2.62 - JOINT_LIMIT_MARGIN<T>(), 2.62 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint4
            {-5.26 - JOINT_LIMIT_MARGIN<T>(), 5.26 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint5
            {-4.18 - JOINT_LIMIT_MARGIN<T>(), 4.18 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint6
            {-5.26 - JOINT_LIMIT_MARGIN<T>(), 5.26 + JOINT_LIMIT_MARGIN<T>()}  // fr3_joint7
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[7][2] = {
            {-87.0 - JOINT_LIMIT_MARGIN<T>(), 87.0 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint1
            {-87.0 - JOINT_LIMIT_MARGIN<T>(), 87.0 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint2
            {-87.0 - JOINT_LIMIT_MARGIN<T>(), 87.0 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint3
            {-87.0 - JOINT_LIMIT_MARGIN<T>(), 87.0 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint4
            {-12.0 - JOINT_LIMIT_MARGIN<T>(), 12.0 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint5
            {-12.0 - JOINT_LIMIT_MARGIN<T>(), 12.0 + JOINT_LIMIT_MARGIN<T>()},  // fr3_joint6
            {-12.0 - JOINT_LIMIT_MARGIN<T>(), 12.0 + JOINT_LIMIT_MARGIN<T>()}  // fr3_joint7
        };

}  // namespace plant
}  // namespace gato
