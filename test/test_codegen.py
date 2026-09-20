"""Codegen determinism: regenerating from the URDF must reproduce the vendored
grid.cuh + limits.cuh byte-for-byte (catches forgot-to-regen drift)."""
import pytest

from gato.builder import codegen, load_registry

pytestmark = pytest.mark.slow

# codegen kwargs per vendored robot — MUST mirror tools/regen_grid.py ROBOTS
# (go2: floating base + the four baked foot contact frames; N16-only module)
GO2_URDF = "external/GRiD/config/robot_assets/go2.urdf"
ROBOT_KW = {
    "indy7": dict(ee_frame="EE"),
    "iiwa14": dict(ee_frame="EE"),
    "go2": dict(ee_frame="imu_joint", floating_base=True,
                contact_frames=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]),
}


def _urdf(robot, urdfs, repo_root):
    return urdfs[robot] if robot in urdfs else repo_root / GO2_URDF


@pytest.mark.parametrize("robot", ["indy7", "iiwa14", "go2"])
def test_regen_matches_vendored(robot, urdfs, repo_root, tmp_path):
    """Regenerating from the URDF reproduces the vendored headers byte-for-byte
    AND the tracked registry entry (python/gato/_registry.json) — the registry
    is tracked, so its failure mode is STALENESS, not absence."""
    out = tmp_path / robot
    meta = codegen(_urdf(robot, urdfs, repo_root), robot, out_dir=out, register=False, **ROBOT_KW[robot])
    vendored = repo_root / "gato" / "dynamics" / robot
    for fname in ("grid.cuh", "limits.cuh"):
        got = (out / fname).read_text()
        want = (vendored / fname).read_text()
        assert got == want, (
            f"{robot}/{fname} drifted from codegen output — re-run "
            f"tools/regen_grid.py and commit the result")
    assert load_registry()[robot] == meta, (
        f"_registry.json[{robot}] is stale vs codegen metadata — re-run "
        f"tools/regen_grid.py --robot {robot} and commit the result")


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
