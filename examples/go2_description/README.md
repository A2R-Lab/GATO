# go2_description

`go2.urdf` is the Unitree Go2 description as vendored by
[GRiD](https://github.com/A2R-Lab/GRiD) (`config/robot_assets/go2.urdf`, GRiD
0.5.1; byte-identical to the file GATO's Go2 headers were generated from). The
released `grid-rbd` wheel does not ship robot assets, so GATO keeps its own copy.
Visual meshes are referenced by `package://` URI and are not included; collision
geometry is primitive, which is all the collision codegen uses.
