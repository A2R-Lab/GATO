# ur5e_description

`ur5e.urdf` is the Universal Robots UR5e. The source is
[UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
at commit `22f055da2fa7` (2026-01-19, BSD-3-Clause), `urdf/ur.urdf.xacro` expanded with
`ur_type=ur5e name=ur5e` by xacrodoc 2.0.1 (the expansion has sha256 `adc09e693028`). GATO
applies these edits to its copy, by script, so the file can be regenerated from the source:

- every `<visual>` block is removed; the seven collision meshes (0.38 MB) are vendored under
  `meshes/collision/` and referenced by relative path, which is what the collision codegen
  spherizes;
- the `<dynamics>` tags are removed from the joints: GATO's plant is the rigid-body model
  without joint damping or friction (the source sets both to zero);
- the inertia tensors of `shoulder_link`, `wrist_1_link` and `wrist_2_link`, given in the
  source in an inertial frame rotated by a quarter turn about x, are rotated into the link
  frame (`<inertial><origin rpy>` zero, tensor I_link = R I R^T, centre of mass unchanged):
  an exact transformation, so pinocchio's dynamics are identical.
  `gato.builder.canonicalize_inertials` provides this optional normalization for any
  URDF. The pinned GRiD parser also accepts the original rotated form;
- the five fixed joints the ROS convention names `parent-child` are renamed without the
  hyphen (`flange-tool0` becomes `tool0_joint`, `wrist_3-flange` `flange_joint`,
  `wrist_3_link-ft_frame` `ft_frame_joint`, `base_link-base_link_inertia`
  `base_link_inertia_joint`, `base_link-base_fixed_joint` `base_fixed_joint`). These
  names remain for compatibility with GATO's registered model; the code generator
  also accepts the original hyphenated names and sanitizes its C++ identifiers.

The joint limits are the source's: plus or minus 2 pi on five joints and plus or minus pi on
the elbow, 3.14 rad/s on every joint, 150 N m on the first three joints and 28 N m on the
wrist. The end-effector frame is the fixed joint `tool0_joint` (the `tool0` frame of the
ROS convention). The limit tables are `gato/dynamics/ur5e/limits.cuh`.
