# tiago_pro_description

`tiago_pro_right_arm.urdf` is the right arm of PAL Robotics' TIAGo Pro as a fixed-base
seven-joint chain, and `tiago_pro_right_arm_parameters.csv` holds the per-joint Coulomb
friction, viscous damping and reflected rotor inertia (armature) measured on the hardware.
Both come from Felix Muehlenberend, who attached them to
[issue #20](https://github.com/A2R-Lab/GATO/issues/20) on 2026-10-31; the URDF as attached has
sha256 `5fe89909dea6`. [Example 21](../21_rotor_armature.py) uses them.

The URDF is his reduction of PAL's TIAGo Pro description
([pal-robotics/tiago_pro_robot](https://github.com/pal-robotics/tiago_pro_robot), with the arm
from [pal-robotics/pal_sea_arm](https://github.com/pal-robotics/pal_sea_arm), both Apache-2.0;
the licence text is `LICENSE` in this folder and the file keeps PAL's notice and his statement
of changes): the root is `torso_lift_link`, the seven right-arm joints keep their native
coordinates (joint 6 turns about -z), the gripper is lumped into `arm_right_tool_link` at zero
gripper posture, and `arm_right_grasp_link` is a massless grasp frame. The joints' `<dynamics>`
tags carry his fitted friction and damping, not PAL's values.

GATO applies one edit to its copy:

- the attached file has no geometry. The collision meshes of the seven arm links are added as
  `<collision>` blocks at the link origin and vendored under `meshes/collision/` (0.83 MB),
  from `pal_sea_arm_description/meshes/arm_tiago_pro` at commit `6e05d0e01a7c` (links 5 to 7
  from `spherical-wrist`, the right arm's default wrist; scale 1, the right arm is not
  mirrored). The joint origins of the reduced model equal PAL's, so the link frames are the
  ones the meshes are drawn in. The lumped gripper has no collision geometry: the clearance
  rows cover the arm links only.

The armature is not representable in URDF; pass the CSV column to
`gato.register_robot(..., joint_armature=...)` in the model's velocity order (the example
matches the rows by joint name). With `joint_dynamics` off, the default, the `<dynamics>` tags
are not read and the model is loss-free, which is the setting of issue #20: friction is
compensated by an external controller. The joint limits are the file's. The end-effector frame
is the fixed joint `arm_right_grasp_joint`.
