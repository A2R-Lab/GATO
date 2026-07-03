#pragma once
// Joint/velocity/effort limit tables for indy7, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[6][2] = {
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-3.0543 - JOINT_LIMIT_MARGIN<T>(), 3.0543 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-3.7520 - JOINT_LIMIT_MARGIN<T>(), 3.7520 + JOINT_LIMIT_MARGIN<T>()}   // joint 5
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[6][2] = {
            {-2.61 - JOINT_LIMIT_MARGIN<T>(), 2.61 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-2.61 - JOINT_LIMIT_MARGIN<T>(), 2.61 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-2.61 - JOINT_LIMIT_MARGIN<T>(), 2.61 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-3.14 - JOINT_LIMIT_MARGIN<T>(), 3.14 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-3.14 - JOINT_LIMIT_MARGIN<T>(), 3.14 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-3.14 - JOINT_LIMIT_MARGIN<T>(), 3.14 + JOINT_LIMIT_MARGIN<T>()}   // joint 5
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[6][2] = {
            {-431.97 - JOINT_LIMIT_MARGIN<T>(), 431.97 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-431.97 - JOINT_LIMIT_MARGIN<T>(), 431.97 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-197.23 - JOINT_LIMIT_MARGIN<T>(), 197.23 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-79.79 - JOINT_LIMIT_MARGIN<T>(), 79.79 + JOINT_LIMIT_MARGIN<T>()}   // joint 5
        };

}  // namespace plant
}  // namespace gato
