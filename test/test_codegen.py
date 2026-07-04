"""Codegen determinism: regenerating from the URDF must reproduce the vendored
grid.cuh + limits.cuh byte-for-byte (catches forgot-to-regen drift)."""
import pytest

from gato.builder import codegen

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("robot", ["indy7", "iiwa14"])
def test_regen_matches_vendored(robot, urdfs, repo_root, tmp_path):
    out = tmp_path / robot
    codegen(urdfs[robot], robot, ee_frame="EE", out_dir=out, register=False)
    vendored = repo_root / "gato" / "dynamics" / robot
    for fname in ("grid.cuh", "limits.cuh"):
        got = (out / fname).read_text()
        want = (vendored / fname).read_text()
        assert got == want, (
            f"{robot}/{fname} drifted from codegen output — re-run "
            f"tools/regen_grid.py and commit the result")


def test_unbounded_joint_rejected(tmp_path):
    """Continuous/unlimited joints must fail fast with an actionable error."""
    urdf = tmp_path / "cont.urdf"
    urdf.write_text("""<robot name="cont">
  <link name="world"/>
  <link name="l1"><inertial><mass value="1"/>
    <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l2"><inertial><mass value="1"/>
    <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l3"><inertial><mass value="1"/>
    <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="j0" type="fixed"><parent link="world"/><child link="l1"/></joint>
  <joint name="j1" type="continuous"><parent link="l1"/><child link="l2"/>
    <axis xyz="0 0 1"/><limit effort="10" velocity="1"/></joint>
  <joint name="EE" type="fixed"><parent link="l2"/><child link="l3"/></joint>
</robot>""")
    with pytest.raises(ValueError, match="finite.*limit|not supported"):
        codegen(urdf, "cont", ee_frame="EE", out_dir=tmp_path / "out", register=False)
