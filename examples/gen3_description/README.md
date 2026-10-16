# gen3_description

`gen3.urdf` is the Kinova Gen3 (7 dof, no gripper). The source is the copy vendored by
[RBDReference](https://github.com/A2R-Lab/RBDReference) (`robot_assets/gen3.urdf`, sha256
`0e7c89432f98`), itself the xacro expansion of
[Kinovarobotics/ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex)
`kortex_description/robots/gen3.xacro` with `dof=7` (BSD-3-Clause). GATO applies these edits
to its copy, by script, so the file can be regenerated from the source:

- joints 1, 3, 5 and 7 are `continuous` in the source (infinite-turn actuators). They are
  revolute here with position limits of plus or minus 2 pi: pinocchio represents a continuous
  joint by its cosine and sine, which breaks the nq = nv state layout GATO assumes, and the
  limit tables need finite position bounds. The box is a modelling choice, not a property of
  the hardware; an application that needs more turns can widen it (`set_row_group_bounds`).
  Velocity and effort limits are the source's;
- the `<ros2_control>` block and every `<visual>` block are removed; the eight collision meshes
  are the package's STL twins of the `.dae` files the source references (4.3 MB), vendored under
  `meshes/collision/` and referenced by relative path.

The end-effector frame is the fixed joint `tool_frame_joint` (the interface plate's tool
frame). The limit tables (`gato/dynamics/gen3/limits.cuh`) carry the bounds above.
