#pragma once
// Joint/velocity/effort limit tables for iiwa14, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[7][2] = {
            {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},  // A1
            {-2.0944 - JOINT_LIMIT_MARGIN<T>(), 2.0944 + JOINT_LIMIT_MARGIN<T>()},  // A2
            {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},  // A3
            {-2.0944 - JOINT_LIMIT_MARGIN<T>(), 2.0944 + JOINT_LIMIT_MARGIN<T>()},  // A4
            {-2.96706 - JOINT_LIMIT_MARGIN<T>(), 2.96706 + JOINT_LIMIT_MARGIN<T>()},  // A5
            {-2.0944 - JOINT_LIMIT_MARGIN<T>(), 2.0944 + JOINT_LIMIT_MARGIN<T>()},  // A6
            {-3.05433 - JOINT_LIMIT_MARGIN<T>(), 3.05433 + JOINT_LIMIT_MARGIN<T>()}  // A7
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[7][2] = {
            {-1.48353 - JOINT_LIMIT_MARGIN<T>(), 1.48353 + JOINT_LIMIT_MARGIN<T>()},  // A1
            {-1.48353 - JOINT_LIMIT_MARGIN<T>(), 1.48353 + JOINT_LIMIT_MARGIN<T>()},  // A2
            {-1.74533 - JOINT_LIMIT_MARGIN<T>(), 1.74533 + JOINT_LIMIT_MARGIN<T>()},  // A3
            {-1.309 - JOINT_LIMIT_MARGIN<T>(), 1.309 + JOINT_LIMIT_MARGIN<T>()},  // A4
            {-2.26893 - JOINT_LIMIT_MARGIN<T>(), 2.26893 + JOINT_LIMIT_MARGIN<T>()},  // A5
            {-2.35619 - JOINT_LIMIT_MARGIN<T>(), 2.35619 + JOINT_LIMIT_MARGIN<T>()},  // A6
            {-2.35619 - JOINT_LIMIT_MARGIN<T>(), 2.35619 + JOINT_LIMIT_MARGIN<T>()}  // A7
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[7][2] = {
            {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},  // A1
            {-320.0 - JOINT_LIMIT_MARGIN<T>(), 320.0 + JOINT_LIMIT_MARGIN<T>()},  // A2
            {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},  // A3
            {-176.0 - JOINT_LIMIT_MARGIN<T>(), 176.0 + JOINT_LIMIT_MARGIN<T>()},  // A4
            {-110.0 - JOINT_LIMIT_MARGIN<T>(), 110.0 + JOINT_LIMIT_MARGIN<T>()},  // A5
            {-40.0 - JOINT_LIMIT_MARGIN<T>(), 40.0 + JOINT_LIMIT_MARGIN<T>()},  // A6
            {-40.0 - JOINT_LIMIT_MARGIN<T>(), 40.0 + JOINT_LIMIT_MARGIN<T>()}  // A7
        };

}  // namespace plant
}  // namespace gato
