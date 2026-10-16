#pragma once
// Joint/velocity/effort limit tables for gen3, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[7][2] = {
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // joint_1
            {-2.24 - JOINT_LIMIT_MARGIN<T>(), 2.24 + JOINT_LIMIT_MARGIN<T>()},  // joint_2
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // joint_3
            {-2.57 - JOINT_LIMIT_MARGIN<T>(), 2.57 + JOINT_LIMIT_MARGIN<T>()},  // joint_4
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // joint_5
            {-2.09 - JOINT_LIMIT_MARGIN<T>(), 2.09 + JOINT_LIMIT_MARGIN<T>()},  // joint_6
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()}  // joint_7
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[7][2] = {
            {-1.3963 - JOINT_LIMIT_MARGIN<T>(), 1.3963 + JOINT_LIMIT_MARGIN<T>()},  // joint_1
            {-1.3963 - JOINT_LIMIT_MARGIN<T>(), 1.3963 + JOINT_LIMIT_MARGIN<T>()},  // joint_2
            {-1.3963 - JOINT_LIMIT_MARGIN<T>(), 1.3963 + JOINT_LIMIT_MARGIN<T>()},  // joint_3
            {-1.3963 - JOINT_LIMIT_MARGIN<T>(), 1.3963 + JOINT_LIMIT_MARGIN<T>()},  // joint_4
            {-1.2218 - JOINT_LIMIT_MARGIN<T>(), 1.2218 + JOINT_LIMIT_MARGIN<T>()},  // joint_5
            {-1.2218 - JOINT_LIMIT_MARGIN<T>(), 1.2218 + JOINT_LIMIT_MARGIN<T>()},  // joint_6
            {-1.2218 - JOINT_LIMIT_MARGIN<T>(), 1.2218 + JOINT_LIMIT_MARGIN<T>()}  // joint_7
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[7][2] = {
            {-39.0 - JOINT_LIMIT_MARGIN<T>(), 39.0 + JOINT_LIMIT_MARGIN<T>()},  // joint_1
            {-39.0 - JOINT_LIMIT_MARGIN<T>(), 39.0 + JOINT_LIMIT_MARGIN<T>()},  // joint_2
            {-39.0 - JOINT_LIMIT_MARGIN<T>(), 39.0 + JOINT_LIMIT_MARGIN<T>()},  // joint_3
            {-39.0 - JOINT_LIMIT_MARGIN<T>(), 39.0 + JOINT_LIMIT_MARGIN<T>()},  // joint_4
            {-9.0 - JOINT_LIMIT_MARGIN<T>(), 9.0 + JOINT_LIMIT_MARGIN<T>()},  // joint_5
            {-9.0 - JOINT_LIMIT_MARGIN<T>(), 9.0 + JOINT_LIMIT_MARGIN<T>()},  // joint_6
            {-9.0 - JOINT_LIMIT_MARGIN<T>(), 9.0 + JOINT_LIMIT_MARGIN<T>()}  // joint_7
        };

}  // namespace plant
}  // namespace gato
