#pragma once
// Joint/velocity/effort limit tables for indy7, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[6][2] = {
            {-3.0543261909900767 - JOINT_LIMIT_MARGIN<T>(), 3.0543261909900767 + JOINT_LIMIT_MARGIN<T>()},  // joint0
            {-3.0543261909900767 - JOINT_LIMIT_MARGIN<T>(), 3.0543261909900767 + JOINT_LIMIT_MARGIN<T>()},  // joint1
            {-3.0543261909900767 - JOINT_LIMIT_MARGIN<T>(), 3.0543261909900767 + JOINT_LIMIT_MARGIN<T>()},  // joint2
            {-3.0543261909900767 - JOINT_LIMIT_MARGIN<T>(), 3.0543261909900767 + JOINT_LIMIT_MARGIN<T>()},  // joint3
            {-3.0543261909900767 - JOINT_LIMIT_MARGIN<T>(), 3.0543261909900767 + JOINT_LIMIT_MARGIN<T>()},  // joint4
            {-3.7524578917878086 - JOINT_LIMIT_MARGIN<T>(), 3.7524578917878086 + JOINT_LIMIT_MARGIN<T>()}  // joint5
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[6][2] = {
            {-2.6179938779914944 - JOINT_LIMIT_MARGIN<T>(), 2.6179938779914944 + JOINT_LIMIT_MARGIN<T>()},  // joint0
            {-2.6179938779914944 - JOINT_LIMIT_MARGIN<T>(), 2.6179938779914944 + JOINT_LIMIT_MARGIN<T>()},  // joint1
            {-2.6179938779914944 - JOINT_LIMIT_MARGIN<T>(), 2.6179938779914944 + JOINT_LIMIT_MARGIN<T>()},  // joint2
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // joint3
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // joint4
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()}  // joint5
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[6][2] = {
            {-431.97 - JOINT_LIMIT_MARGIN<T>(), 431.97 + JOINT_LIMIT_MARGIN<T>()},  // joint0
            {-431.97 - JOINT_LIMIT_MARGIN<T>(), 431.97 + JOINT_LIMIT_MARGIN<T>()},  // joint1
            {-197.23 - JOINT_LIMIT_MARGIN<T>(), 197.23 + JOINT_LIMIT_MARGIN<T>()},  // joint2
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()},  // joint3
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()},  // joint4
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()}  // joint5
        };

}  // namespace plant
}  // namespace gato
