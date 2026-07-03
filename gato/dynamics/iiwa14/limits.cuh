#pragma once
// Joint/velocity/effort limit tables for iiwa14, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[7][2] = {
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
            {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 0
            {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 1
            {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 2
            {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 3
            {-110.0 - JOINT_LIMIT_MARGIN<T>(), 110.0 + JOINT_LIMIT_MARGIN<T>()},  // joint 4
            {-40.0 - JOINT_LIMIT_MARGIN<T>(), 40.0 + JOINT_LIMIT_MARGIN<T>()},   // joint 5
            {-40.0 - JOINT_LIMIT_MARGIN<T>(), 40.0 + JOINT_LIMIT_MARGIN<T>()}   // joint 6
        };

}  // namespace plant
}  // namespace gato
