"""Codegen determinism: regenerating from the URDF must reproduce the vendored
grid.cuh + limits.cuh byte-for-byte (catches forgot-to-regen drift)."""
import subprocess

import pytest

from gato import _grid
from gato.builder import codegen, load_registry

pytestmark = pytest.mark.slow

# codegen kwargs per vendored robot — MUST mirror tools/regen_grid.py ROBOTS
# (go2: floating base + the four baked foot contact frames; N16-only module)
GO2_URDF = "examples/go2_description/go2.urdf"
REPO_GO2 = __import__("pathlib").Path(__file__).resolve().parents[1] / GO2_URDF
ROBOT_KW = {
    "indy7": dict(ee_frame="EE"),
    "iiwa14": dict(ee_frame="EE"),
    "go2": dict(ee_frame="imu_joint", floating_base=True,
                contact_frames=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]),
    "fr3": dict(ee_frame="fr3_hand_tcp_joint"),
    "gen3": dict(ee_frame="tool_frame_joint"),
    "ur5e": dict(ee_frame="tool0_joint"),
}


def _urdf(robot, urdfs, repo_root):
    return urdfs[robot] if robot in urdfs else repo_root / GO2_URDF


@pytest.mark.parametrize("robot", list(ROBOT_KW))
def test_regen_matches_vendored(robot, urdfs, repo_root, tmp_path):
    """Regenerating from the URDF reproduces the vendored headers byte-for-byte
    AND the tracked registry entry (python/gato/_registry.json) — the registry
    is tracked, so its failure mode is STALENESS, not absence."""
    # generate and compile with the same reviewed GLASS: the pinned wheel bundles it
    assert _grid.require()["peers"]["GLASS"]["commit"] == _grid.GLASS_PIN
    out = tmp_path / robot
    meta = codegen(_urdf(robot, urdfs, repo_root), robot, out_dir=out, register=False, **ROBOT_KW[robot])
    vendored = repo_root / "gato" / "dynamics" / robot
    for fname in ("grid.cuh", "limits.cuh"):
        got = (out / fname).read_text()
        want = (vendored / fname).read_text()
        assert got == want, (
            f"{robot}/{fname} drifted from codegen output — re-run "
            f"tools/regen_grid.py and commit the result")
    reg = load_registry()[robot]
    assert {k: reg.get(k) for k in meta} == meta, (
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


ROTATED_URDF = """<robot name="rot">
  <link name="world"/>
  <link name="l1"><inertial><mass value="1"/>
    <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l2"><inertial><origin xyz="0 0 0.1" rpy="1.5707963 0 0"/><mass value="2"/>
    <inertia ixx="0.03" iyy="0.01" izz="0.02" ixy="0.001" ixz="0.002" iyz="0.003"/></inertial></link>
  <link name="l3"><inertial><mass value="1"/>
    <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="j0" type="revolute"><parent link="world"/><child link="l1"/>
    <axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
  <joint name="j1" type="revolute"><parent link="l1"/><child link="l2"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
  <joint name="EE" type="fixed"><parent link="l2"/><child link="l3"/></joint>
</robot>"""


def test_rotated_inertial_origin_matches_canonicalized(tmp_path):
    """Raw and normalized inertia tensors parse to the same model; the optional
    canonicalizer also leaves pinocchio's dynamics unchanged."""
    import numpy as np
    from gato.builder import canonicalize_inertials, rotated_inertial_origins
    pin = pytest.importorskip("pinocchio")
    urdf = tmp_path / "rot.urdf"
    urdf.write_text(ROTATED_URDF)
    assert rotated_inertial_origins(urdf) == ["l2"]
    codegen(urdf, "rot", ee_frame="EE", out_dir=tmp_path / "out", register=False,
            collision_res=None)
    canon = tmp_path / "rot_canonical.urdf"
    assert canonicalize_inertials(urdf, canon) == ["l2"]
    assert rotated_inertial_origins(canon) == []
    m_a, m_b = pin.buildModelFromUrdf(str(urdf)), pin.buildModelFromUrdf(str(canon))
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, v, tau = rng.uniform(-2, 2, 2), rng.normal(size=2), rng.normal(size=2)
        np.testing.assert_allclose(pin.aba(m_a, m_a.createData(), q, v, tau),
                                   pin.aba(m_b, m_b.createData(), q, v, tau), rtol=1e-12, atol=1e-12)
    from gato.builder import _parse_urdf
    raw, normalized = _parse_urdf(urdf, "EE"), _parse_urdf(canon, "EE")
    assert raw.get_num_pos() == normalized.get_num_pos() == 2
    np.testing.assert_allclose(
        np.asarray(raw.get_Imats_ordered_by_id(), dtype=float),
        np.asarray(normalized.get_Imats_ordered_by_id(), dtype=float),
        rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_hyphenated_frame_header_compiles(tmp_path):
    """Use the original URDF frame name for lookup and valid emitted C++ symbols."""
    urdf = tmp_path / "hyphen.urdf"
    urdf.write_text(ROTATED_URDF.replace('name="EE"', 'name="flange-tool0"'))
    out = tmp_path / "generated"
    meta = codegen(urdf, "hyphen", ee_frame="flange-tool0", out_dir=out, register=False,
                   collision_res=None)
    assert meta["ee_frame"] == "flange-tool0"
    assert meta["contact_frames"] == ["flange-tool0"]
    result = subprocess.run(
        ["nvcc", "-std=c++17", "-DNDEBUG", "-x", "cu", "-c", str(out / "grid.cuh"),
         "-I", str(_grid.collision_dir()), "-o", str(tmp_path / "hyphen.o")],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_voxel_dependency_cannot_change_collision_geometry(monkeypatch, urdfs, tmp_path):
    """A lean install used to succeed with a different, fallback sphere set.
    Inject the underlying failure instead of uninstalling shared dependencies."""
    import trimesh

    def unavailable(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'scipy' (injected)")

    monkeypatch.setattr(trimesh.Trimesh, 'voxelized', unavailable)
    out = tmp_path / 'indy7'
    with pytest.raises(UserWarning, match='voxelization failed|SKIPPING'):
        codegen(urdfs['indy7'], 'indy7', out_dir=out, register=False)
    assert not (out / 'grid.cuh').exists()


def test_cost_frames_bake_a_second_frame(urdfs, tmp_path):
    """cost_frames=[joint] bakes that joint as cost frame 1 behind the end-effector
    target (frame 0): the header reports two frames and the registry metadata the
    list; a floating-base plant takes no cost frames (the menu is fixed-base only
    in grid-rbd 0.5.4); the codegen key moves with the list."""
    from gato.builder import _codegen_key
    out = tmp_path / "iiwa14_elbow"
    meta = codegen(urdfs["iiwa14"], "iiwa14_elbow", out_dir=out, register=False, ee_frame="EE", cost_frames=["A4"])
    header = (out / "grid.cuh").read_text()
    assert "const int GRID_PLANT_NUM_COST_FRAMES = 2;" in header
    assert "#define GRID_PLANT_HAS_COST_MENU 1" in header
    assert meta["cost_frames"] == ["EE", "A4"]
    base = codegen(urdfs["iiwa14"], "iiwa14_base", out_dir=tmp_path / "iiwa14_base", register=False, ee_frame="EE")
    assert base["cost_frames"] == ["EE"]
    k0 = _codegen_key(urdfs["iiwa14"], "EE", 0.15, None, False)
    k1 = _codegen_key(urdfs["iiwa14"], "EE", 0.15, None, False, cost_frames=["A4"])
    assert k0 != k1 and k0 == _codegen_key(urdfs["iiwa14"], "EE", 0.15, None, False, cost_frames=[])
    with pytest.raises(ValueError, match="fixed-base only"):
        codegen(REPO_GO2, "go2_frames", out_dir=tmp_path / "go2", register=False, ee_frame="imu_joint",
                floating_base=True, contact_frames=ROBOT_KW["go2"]["contact_frames"], cost_frames=["FL_hip_joint"])
