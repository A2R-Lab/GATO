# fr3_description

`fr3.urdf` is the Franka Research 3 with the Franka Hand. The source is the copy vendored by
[RBDReference](https://github.com/A2R-Lab/RBDReference) (`robot_assets/fr3.urdf`, sha256
`d0d99c0ca801`), itself the xacro expansion of
[frankarobotics/franka_description](https://github.com/frankarobotics/franka_description)
`robots/fr3/fr3.urdf.xacro` with the hand (Apache-2.0). GATO applies these edits to its copy,
by script, so the file can be regenerated from the source:

- the two prismatic finger joints are fixed, so the fingers are lumped into the hand (closed
  gripper) and the model is the seven arm joints, nq = nv = 7;
- every `<visual>` block is removed; the nine collision meshes (0.12 MB) are vendored under
  `meshes/collision/` and referenced by relative path, which is what the collision codegen
  spherizes;
- the `<dynamics>`, `<safety_controller>` and `<position_based_velocity_limits>` tags are
  removed from the joints: GATO's plant is the rigid-body model without joint damping or
  friction, and the limit tables come from the `<limit>` tags alone.

The end-effector frame is the fixed joint `fr3_hand_tcp_joint`, the hand's tool centre point.
The limit tables (`gato/dynamics/fr3/limits.cuh`) carry the URDF position, velocity and effort
limits of the seven joints.
