#pragma once
// Joint/velocity/effort limit tables for ur5e, from the URDF <limit> tags.
// Generated alongside grid.cuh (tools/regen_grid.py / gato.build); do not edit
// by hand. Included by gato/dynamics/plant.cuh only (needs JOINT_LIMIT_MARGIN).

namespace gato {
namespace plant {

        template<class T>
        __device__ constexpr T JOINT_LIMITS_DATA[6][2] = {
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // shoulder_pan_joint
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // shoulder_lift_joint
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // elbow_joint
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // wrist_1_joint
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()},  // wrist_2_joint
            {-6.283185307179586 - JOINT_LIMIT_MARGIN<T>(), 6.283185307179586 + JOINT_LIMIT_MARGIN<T>()}  // wrist_3_joint
        };

        template<class T>
        __device__ constexpr T VEL_LIMITS_DATA[6][2] = {
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // shoulder_pan_joint
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // shoulder_lift_joint
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // elbow_joint
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // wrist_1_joint
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()},  // wrist_2_joint
            {-3.141592653589793 - JOINT_LIMIT_MARGIN<T>(), 3.141592653589793 + JOINT_LIMIT_MARGIN<T>()}  // wrist_3_joint
        };

        template<class T>
        __device__ constexpr T CTRL_LIMITS_DATA[6][2] = {
            {-150.0 - JOINT_LIMIT_MARGIN<T>(), 150.0 + JOINT_LIMIT_MARGIN<T>()},  // shoulder_pan_joint
            {-150.0 - JOINT_LIMIT_MARGIN<T>(), 150.0 + JOINT_LIMIT_MARGIN<T>()},  // shoulder_lift_joint
            {-150.0 - JOINT_LIMIT_MARGIN<T>(), 150.0 + JOINT_LIMIT_MARGIN<T>()},  // elbow_joint
            {-28.0 - JOINT_LIMIT_MARGIN<T>(), 28.0 + JOINT_LIMIT_MARGIN<T>()},  // wrist_1_joint
            {-28.0 - JOINT_LIMIT_MARGIN<T>(), 28.0 + JOINT_LIMIT_MARGIN<T>()},  // wrist_2_joint
            {-28.0 - JOINT_LIMIT_MARGIN<T>(), 28.0 + JOINT_LIMIT_MARGIN<T>()}  // wrist_3_joint
        };

}  // namespace plant
}  // namespace gato
