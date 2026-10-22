"""gato.register_robot: the pip-install path. Codegen into a per-host cache
(host tier), then the first solver for the robot compiles its module once and
later constructions hit the cache (gpu/slow tier). Every test uses its own
cache directory via GATO_CACHE_DIR."""
import json
import logging

import numpy as np
import pytest

import gato
from gato import _cache

NAME = "cachebot_indy7"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    _cache.toolchain_identity.cache_clear()
    _cache._native_source_id.cache_clear()
    return tmp_path / "cache"


def test_register_writes_headers_and_manifest(cache, urdfs):
    meta = gato.register_robot(NAME, urdfs["indy7"], N=[8, 16])
    assert meta["nq"] == meta["nv"] == 6 and meta["ee_frame"] == "EE"
    entry = _cache.robot_entry(NAME)
    dyn = cache / "robots" / entry["codegen_key"] / NAME
    assert (dyn / "grid.cuh").exists() and (dyn / "limits.cuh").exists()
    assert entry["N"] == [8, 16] and entry["urdf_sha256"]
    assert gato.robot_info(NAME)["codegen_key"] == entry["codegen_key"]
    # the same URDF + options re-registers without regenerating (content-keyed)
    stamp = (dyn / "grid.cuh").stat().st_mtime_ns
    gato.register_robot(NAME, urdfs["indy7"], N=[8])
    assert (dyn / "grid.cuh").stat().st_mtime_ns == stamp
    assert _cache.robot_entry(NAME)["N"] == [8]
    # nothing compiled yet: no module, and discovery does not list it
    assert _cache.find_module(NAME, 8) is None
    assert (NAME, 8) not in gato.available()


def test_register_rejects_bad_name_and_missing_urdf(cache, urdfs, tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        gato.register_robot("not a name", urdfs["indy7"])
    with pytest.raises(FileNotFoundError):
        gato.register_robot("ghost", tmp_path / "missing.urdf")


def test_unregistered_robot_raises_with_register_hint(cache, urdfs):
    with pytest.raises(ValueError, match="register_robot"):
        gato.BSQP(str(urdfs["indy7"]), batch_size=1, N=8, dt=0.01, plant_type="nobody_registered_this")


def test_module_key_tracks_inputs(cache, urdfs, monkeypatch):
    gato.register_robot(NAME, urdfs["indy7"], N=[8])
    entry = _cache.robot_entry(NAME)
    root = cache / "robots" / entry["codegen_key"]
    k1, inputs = _cache.module_key(entry, 8, "default", root, "120")
    assert set(inputs) == {"codegen_key", "N", "variant", "native_source_id", "toolchain", "schema"}
    assert _cache.module_key(entry, 8, "default", root, "120")[0] == k1
    assert _cache.module_key(entry, 16, "default", root, "120")[0] != k1      # horizon
    assert _cache.module_key(entry, 8, "fc", root, "120")[0] != k1           # variant
    assert _cache.module_key(entry, 8, "default", root, "89")[0] != k1       # GPU architecture
    other = dict(entry, codegen_key="0" * 16)
    assert _cache.module_key(other, 8, "default", root, "120")[0] != k1      # URDF / options / generator
    monkeypatch.setenv("GATO_MERIT_THREADS", "64")
    k64, inputs64 = _cache.module_key(entry, 8, "default", root, "120")
    assert k64 != k1 and inputs64["merit_threads"] == 64                    # merit block size (issue #19)
    monkeypatch.setenv("GATO_MERIT_THREADS", "48")
    with pytest.raises(ValueError, match="multiple of 32"):
        _cache.module_key(entry, 8, "default", root, "120")


@pytest.mark.gpu
@pytest.mark.slow
def test_first_solver_compiles_once_then_hits_cache(cache, urdfs, caplog):
    pytest.importorskip("pinocchio")
    urdf = str(urdfs["indy7"])
    gato.register_robot(NAME, urdf, N=[8])
    with caplog.at_level(logging.WARNING, logger="gato"):
        s = gato.BSQP(urdf, batch_size=2, N=8, dt=0.01, plant_type=NAME, max_sqp_iters=3)
    assert any("one-time" in r.getMessage() and f"bsqpN8_{NAME}" in r.getMessage() for r in caplog.records)
    so = _cache.find_module(NAME, 8)
    assert so is not None and so.parent.parent.parent == cache / "modules"
    meta = json.loads((so.parent / "meta.json").read_text())
    assert meta["inputs"]["N"] == 8 and meta["seconds"] > 0
    assert not (cache / "build" / NAME).exists() or not any((cache / "build" / NAME).iterdir())
    x = np.zeros((2, s.nx), np.float32)
    goals = np.zeros((2, 8 * 6), np.float32)
    goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
    assert np.isfinite(s.solve(x, goals).xu).all()
    assert (NAME, 8) in gato.available()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gato"):
        gato.BSQP(urdf, batch_size=1, N=8, dt=0.01, plant_type=NAME)
    assert not [r for r in caplog.records if "compiling" in r.getMessage()], "second construction must hit the cache"


@pytest.mark.gpu
@pytest.mark.slow
def test_vendored_robot_compiles_from_bundled_headers(cache, urdfs, caplog):
    """The robots GATO ships headers for need no register_robot call: a horizon
    with no package module compiles from the bundled headers into the cache."""
    N = 12   # not in the receipt profile, so never a package module
    assert ("indy7", N) not in gato.available()
    urdf = str(urdfs["indy7"])
    with caplog.at_level(logging.WARNING, logger="gato"):
        s = gato.BSQP(urdf, batch_size=1, N=N, dt=0.01, plant_type="indy7", max_sqp_iters=2)
    assert any(f"bsqpN{N}_indy7" in r.getMessage() for r in caplog.records)
    assert s.N == N and _cache.find_module("indy7", N) is not None
    assert ("indy7", N) in gato.available()
    x = np.zeros((1, s.nx), np.float32)
    goals = np.zeros((1, N * 6), np.float32)
    goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
    assert np.isfinite(s.solve(x, goals).xu).all()


@pytest.mark.gpu
@pytest.mark.slow
def test_registered_signed_and_mixed_axes_match_pinocchio(cache, tmp_path):
    """Signed, mixed-axis dynamics and linearization match Pinocchio in Release."""
    import pinocchio as pin

    urdf = tmp_path / "mixed_axes.urdf"
    links, joints = [], []
    # Cardinal axes use GRiD's helper-index/sign path rather than its skew-axis path.
    for i, axis in enumerate(("0 0 1", "0 0 -1", "1 0 0"), 1):
        links.append(f'''<link name="l{i}"><inertial>
          <origin xyz=".12 .03 .02"/><mass value="1"/>
          <inertia ixx=".02" iyy=".03" izz=".04" ixy="0" ixz="0" iyz="0"/>
        </inertial><collision><geometry><sphere radius=".04"/></geometry></collision></link>''')
        parent = "base" if i == 1 else f"l{i-1}"
        joints.append(f'''<joint name="j{i}" type="revolute">
          <parent link="{parent}"/><child link="l{i}"/>
          <origin xyz=".25 .02 .04"/><axis xyz="{axis}"/>
          <limit lower="-2" upper="2" effort="20" velocity="3"/></joint>''')
    urdf.write_text('<robot name="mixed_axes"><link name="base"/>'
                    + ''.join(links + joints)
                    + '<link name="tool"/><joint name="EE" type="fixed">'
                      '<parent link="l3"/><child link="tool"/>'
                      '<origin xyz=".2 0 0"/></joint></robot>')
    name, N, dt = "cachebot_mixed_axes", 8, .01
    gato.register_robot(name, urdf, N=[N])
    s = gato.BSQP(str(urdf), batch_size=1, N=N, dt=dt, plant_type=name)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q = np.array([.3, -.4, .2])
    v = np.array([.2, -.1, .15])
    u = np.array([.7, -.3, .5])
    x = np.r_[q, v]
    got = np.asarray(s.sim_forward(x[None], u[None], dt))[0]
    acceleration = pin.aba(model, data, q, v, u)
    expected = np.r_[q + dt * v + .5 * dt**2 * acceleration, v + dt * acceleration]
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=1e-4)

    # Compare the discrete-step A/B matrices with Pinocchio derivatives.
    xu = np.tile(np.r_[x, u], N)[:-3]
    # KKT setup requires a reference; its values do not affect A/B.
    goals = np.zeros((1, N * 6), dtype=np.float32)
    kkt = s.debug_setup_kkt(xu[None], x[None], goals)
    aq, av, au = pin.computeABADerivatives(model, data, q, v, u)
    eye = np.eye(3)
    expected_A = np.block([[eye + .5 * dt**2 * aq, dt * eye + .5 * dt**2 * av],
                           [dt * aq, eye + dt * av]])
    expected_B = np.vstack([.5 * dt**2 * au, dt * au])
    A = np.asarray(kkt["A"])[0].reshape(N, 6, 6)[0].T
    B = np.asarray(kkt["B"])[0].reshape(N, 3, 6)[0].T
    np.testing.assert_allclose(A, expected_A, atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(B, expected_B, atol=2e-5, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.slow
def test_merit_threads_knob_is_bit_identical(cache, urdfs, tmp_path):
    """GATO_MERIT_THREADS (issue #19): a smaller merit block compiles into its own
    cache entry and solves bit-identically to the default block size."""
    import os
    import subprocess
    import sys

    urdf = str(urdfs["indy7"])
    gato.register_robot(NAME, urdf, N=[8])
    x = np.zeros((2, 12), np.float32)
    x[0, :6] = [0.3, -0.4, 0.2, 0.1, -0.2, 0.15]
    x[1, 6:] = 0.2
    goals = np.zeros((2, 8 * 6), np.float32)
    goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "goals.npy", goals)
    script = (
        "import sys, numpy as np, gato\n"
        "urdf, name, out = sys.argv[1:4]\n"
        "x, goals = np.load(sys.argv[4]), np.load(sys.argv[5])\n"
        "s = gato.BSQP(urdf, batch_size=2, N=8, dt=0.01, plant_type=name)\n"
        "np.save(out, np.stack([np.asarray(s.solve(x, goals).xu).copy() for _ in range(4)]))\n")
    outs = {}
    for threads in (None, 64):   # separate processes: one module name per process
        env = dict(os.environ)
        env.pop("GATO_MERIT_THREADS", None)
        if threads:
            env["GATO_MERIT_THREADS"] = str(threads)
        out = tmp_path / f"xu_{threads}.npy"
        subprocess.run([sys.executable, "-c", script, urdf, NAME, str(out),
                        str(tmp_path / "x.npy"), str(tmp_path / "goals.npy")], check=True, env=env)
        outs[threads] = np.load(out)
    assert np.isfinite(outs[None]).all()
    assert np.array_equal(outs[64], outs[None])
    metas = sorted(str(json.loads(m.read_text())["inputs"].get("merit_threads"))
                   for m in (cache / "modules" / NAME).glob("*/meta.json"))
    assert metas == ["64", "None"]


@pytest.mark.gpu
@pytest.mark.slow
def test_registered_joint_dynamics_match_pinocchio(cache, tmp_path):
    """joint_dynamics=True (issue #20): the URDF's <dynamics damping> and smooth
    friction (tanh, friction_width) enter the step and its linearization; checked
    against Pinocchio with the joint torque applied to the input."""
    import pinocchio as pin

    damping = (0.5, 0.2, 0.8)
    friction = (0.3, 0.0, 0.6)
    width = 0.05
    urdf = tmp_path / "damped.urdf"
    links, joints = [], []
    for i, b in enumerate(damping, 1):
        links.append(f'''<link name="l{i}"><inertial>
          <origin xyz=".12 .03 .02"/><mass value="1"/>
          <inertia ixx=".02" iyy=".03" izz=".04" ixy="0" ixz="0" iyz="0"/>
        </inertial><collision><geometry><sphere radius=".04"/></geometry></collision></link>''')
        parent = "base" if i == 1 else f"l{i-1}"
        joints.append(f'''<joint name="j{i}" type="revolute">
          <parent link="{parent}"/><child link="l{i}"/>
          <origin xyz=".25 .02 .04"/><axis xyz="0 0 1"/><dynamics damping="{b}" friction="{friction[i - 1]}"/>
          <limit lower="-2" upper="2" effort="20" velocity="3"/></joint>''')
    urdf.write_text('<robot name="damped"><link name="base"/>'
                    + ''.join(links + joints)
                    + '<link name="tool"/><joint name="EE" type="fixed">'
                      '<parent link="l3"/><child link="tool"/>'
                      '<origin xyz=".2 0 0"/></joint></robot>')
    name, N, dt = "cachebot_damped", 8, .01
    gato.register_robot(name, urdf, N=[N], joint_dynamics=True, friction_width=width)
    entry = _cache.robot_entry(name)
    assert entry["options"]["joint_dynamics"] is True and entry["meta"]["joint_dynamics"] is True
    assert entry["options"]["friction_width"] == width and entry["meta"]["friction_model"] == "tanh"
    header = (cache / "robots" / entry["codegen_key"] / name / "grid.cuh").read_text()
    assert "JOINT_FRICTION_MODEL_TANH" in header
    s = gato.BSQP(str(urdf), batch_size=1, N=N, dt=dt, plant_type=name)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    D = np.diag(damping)
    C = np.asarray(friction)
    q = np.array([.3, -.4, .2])
    v = np.array([.2, -.1, .15])
    u = np.array([.7, -.3, .5])
    x = np.r_[q, v]
    got = np.asarray(s.sim_forward(x[None], u[None], dt))[0]
    tau = u - D @ v - C * np.tanh(v / width)
    acceleration = pin.aba(model, data, q, v, tau)
    expected = np.r_[q + dt * v + .5 * dt**2 * acceleration, v + dt * acceleration]
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=1e-4)
    undamped = pin.aba(model, data, q, v, u)
    assert not np.allclose(got, np.r_[q + dt * v + .5 * dt**2 * undamped, v + dt * undamped], atol=2e-5)

    xu = np.tile(np.r_[x, u], N)[:-3]
    goals = np.zeros((1, N * 6), dtype=np.float32)
    kkt = s.debug_setup_kkt(xu[None], x[None], goals)
    aq, av, au = pin.computeABADerivatives(model, data, q, v, tau)
    av = av - au @ (D + np.diag(C / width / np.cosh(v / width) ** 2))   # d(tau - D qd - C tanh(qd/w))/d qd through M^-1
    eye = np.eye(3)
    expected_A = np.block([[eye + .5 * dt**2 * aq, dt * eye + .5 * dt**2 * av],
                           [dt * aq, eye + dt * av]])
    expected_B = np.vstack([.5 * dt**2 * au, dt * au])
    A = np.asarray(kkt["A"])[0].reshape(N, 6, 6)[0].T
    B = np.asarray(kkt["B"])[0].reshape(N, 3, 6)[0].T
    np.testing.assert_allclose(A, expected_A, atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(B, expected_B, atol=2e-5, rtol=1e-4)


def test_joint_dynamics_keys_and_width_validation(cache, tmp_path):
    """joint_dynamics joins the codegen key only when on, the friction width is part
    of it, and a non-positive width is rejected before codegen."""
    from gato.builder import _codegen_key

    urdf = tmp_path / "fric.urdf"
    urdf.write_text('<robot name="fric"><link name="base"/>'
                    '<link name="l1"><inertial><origin xyz=".1 0 0"/><mass value="1"/>'
                    '<inertia ixx=".02" iyy=".03" izz=".04" ixy="0" ixz="0" iyz="0"/></inertial>'
                    '<collision><geometry><sphere radius=".04"/></geometry></collision></link>'
                    '<joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>'
                    '<axis xyz="0 0 1"/><dynamics damping="0.5" friction="0.3"/>'
                    '<limit lower="-2" upper="2" effort="20" velocity="3"/></joint>'
                    '<link name="tool"/><joint name="EE" type="fixed"><parent link="l1"/>'
                    '<child link="tool"/><origin xyz=".2 0 0"/></joint></robot>')
    with pytest.raises(ValueError, match="friction_width"):
        gato.register_robot("cachebot_fric", urdf, N=[8], joint_dynamics=True, friction_width=0.0)
    assert _codegen_key(urdf, "EE", 0.15, None, False) == _codegen_key(urdf, "EE", 0.15, None, False, False)
    assert _codegen_key(urdf, "EE", 0.15, None, False) != _codegen_key(urdf, "EE", 0.15, None, False, True)
    assert _codegen_key(urdf, "EE", 0.15, None, False, True, 0.03) != _codegen_key(urdf, "EE", 0.15, None, False, True, 0.05)
