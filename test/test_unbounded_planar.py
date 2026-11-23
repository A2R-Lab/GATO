"""Unbounded joints and the planar base (0.2.3, grid-rbd 0.7.2).

Host tier: gato.pinocchio_model gives pinocchio the module's coordinates, and
is checked against pinocchio's own continuous and planar joints. GPU tier: a
robot with a continuous joint, a one-sided revolute joint and a planar base is
registered, compiled and compared with pinocchio at the step, the
linearization and a solve, with the limit costs and the limit rows on."""
import numpy as np
import pytest

import gato

pin = pytest.importorskip("pinocchio")

DT, N = 0.01, 8


J2_OPEN, J2_FULL = '<limit lower="-2" effort="20"/>', '<limit lower="-2" upper="1e30" velocity="1e30" effort="20"/>'


def _urdf(tmp_path):
    """A base link, a continuous joint about z, a revolute joint with a lower
    bound only and no velocity limit, and a fixed tool frame."""
    link = '''<link name="{}"><inertial><origin xyz=".12 .03 .02"/><mass value="{}"/>
      <inertia ixx=".02" iyy=".03" izz=".04" ixy="0" ixz="0" iyz="0"/></inertial>
      <collision><geometry><sphere radius=".04"/></geometry></collision></link>'''
    path = tmp_path / "wheelarm.urdf"
    path.write_text(
        '<robot name="wheelarm">' + link.format("base", 4) + link.format("l1", 1) + link.format("l2", 1)
        + '''<joint name="j1" type="continuous"><parent link="base"/><child link="l1"/>
               <origin xyz=".25 .02 .04"/><axis xyz="0 0 1"/><limit effort="20" velocity="3"/></joint>
             <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
               <origin xyz=".25 .02 .04"/><axis xyz="0 1 0"/>''' + J2_OPEN + '''</joint>
             <link name="tool"/><joint name="EE" type="fixed"><parent link="l2"/><child link="tool"/>
               <origin xyz=".2 0 0"/></joint></robot>''')
    return path


def _symmetric(model, q):
    M = pin.crba(model, model.createData(), q)
    return np.triu(M) + np.triu(M, 1).T


def test_continuous_joint_is_one_angle_with_pinocchios_dynamics(tmp_path):
    urdf = _urdf(tmp_path)
    # pinocchio's own loader refuses a <limit> without a velocity, so it reads the completed tag
    mirror, native = gato.pinocchio_model(urdf), pin.buildModelFromXML(urdf.read_text().replace(J2_OPEN, J2_FULL))
    assert (mirror.nq, mirror.nv) == (2, 2) and (native.nq, native.nv) == (3, 2)
    q, v, u = np.array([40.3, -0.4]), np.array([0.2, -0.1]), np.array([0.7, -0.3])   # several turns
    q_native = np.array([np.cos(q[0]), np.sin(q[0]), q[1]])
    np.testing.assert_allclose(_symmetric(mirror, q), _symmetric(native, q_native), atol=1e-12)
    np.testing.assert_allclose(pin.aba(mirror, mirror.createData(), q, v, u),
                               pin.aba(native, native.createData(), q_native, v, u), atol=1e-11)
    np.testing.assert_array_equal(mirror.lowerPositionLimit, [-np.inf, -2.0])
    np.testing.assert_array_equal(mirror.upperPositionLimit, [np.inf, np.inf])
    np.testing.assert_array_equal(mirror.velocityLimit, [3.0, np.inf])
    np.testing.assert_array_equal(mirror.effortLimit, [20.0, 20.0])


def test_planar_base_coordinates_map_to_pinocchios_planar_joint(tmp_path):
    """q = [px, py, theta], world-frame linear velocity. With G = blockdiag(Rz(theta)^T, I):
    v_pin = G v, a_pin = G a + Gdot v, tau = G^T tau_pin, M = G^T M_pin G."""
    urdf = _urdf(tmp_path)
    mirror = gato.pinocchio_model(urdf, planar_base=True)
    native = pin.buildModelFromXML(urdf.read_text().replace(J2_OPEN, J2_FULL).replace('type="continuous"', 'type="revolute"')
                                   .replace('<limit effort="20" velocity="3"/>',
                                            '<limit lower="-1e30" upper="1e30" effort="20" velocity="3"/>'),
                                   pin.JointModelPlanar())
    assert (mirror.nq, mirror.nv) == (5, 5) and (native.nq, native.nv) == (6, 5)
    q = np.array([0.3, -0.2, 0.7, 40.3, -0.4])
    v = np.array([0.5, -0.3, 0.2, 0.2, -0.1])
    u = np.array([1.0, -2.0, 0.5, 0.7, -0.3])
    c, s = np.cos(q[2]), np.sin(q[2])
    G = np.eye(5)
    G[:2, :2] = [[c, s], [-s, c]]
    Gdot_v = np.r_[v[2] * np.array([-s * v[0] + c * v[1], -c * v[0] - s * v[1]]), 0.0, 0.0, 0.0]
    q_native = np.r_[q[:2], c, s, q[3:]]
    np.testing.assert_allclose(_symmetric(mirror, q), G.T @ _symmetric(native, q_native) @ G, atol=1e-12)
    a = pin.aba(mirror, mirror.createData(), q, v, u)
    a_native = pin.aba(native, native.createData(), q_native, G @ v, np.linalg.solve(G.T, u))
    np.testing.assert_allclose(G @ a + Gdot_v, a_native, atol=1e-10)
    # the analytic derivatives hold on this root (they do not on pinocchio's JointModelComposite)
    data = mirror.createData()   # the result is a view into data
    aq = pin.computeABADerivatives(mirror, data, q, v, u)[0]
    step = 1e-6 * np.eye(5)[2]
    difference = (pin.aba(mirror, mirror.createData(), q + step, v, u)
                  - pin.aba(mirror, mirror.createData(), q - step, v, u)) / 2e-6
    np.testing.assert_allclose(aq[:, 2], difference, atol=1e-6)
    assert np.isinf(mirror.lowerPositionLimit[:4]).all() and np.isinf(mirror.effortLimit[:3]).all()
    with pytest.raises(ValueError, match="separate base models"):
        gato.pinocchio_model(urdf, floating_base=True, planar_base=True)


def test_bounded_robots_keep_the_plain_pinocchio_model(urdfs):
    """No unbounded side: the model is pin.buildModelFromUrdf's, limits included."""
    for robot in ("indy7", "iiwa14"):
        mirror, plain = gato.pinocchio_model(urdfs[robot]), pin.buildModelFromUrdf(str(urdfs[robot]))
        assert mirror == plain


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from gato import _cache
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    _cache.toolchain_identity.cache_clear()
    _cache._native_source_id.cache_clear()
    return tmp_path / "cache"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("planar", [False, True])
def test_unbounded_robot_matches_pinocchio_and_solves(cache, tmp_path, planar):
    """The step and its linearization match pinocchio several turns away from
    zero, and the solve stays finite with the limit costs on (their unbounded
    sides contribute nothing) and with the limit rows installed."""
    from gato.controller import MPCController

    urdf = _urdf(tmp_path)
    name = "wheelarm_planar" if planar else "wheelarm"
    meta = gato.register_robot(name, urdf, N=[N], planar_base=planar)
    nq = 5 if planar else 2
    assert meta["nq"] == meta["nv"] == nq and bool(meta.get("planar_base", False)) == planar
    solver = gato.BSQP(str(urdf), batch_size=1, N=N, dt=DT, plant_type=name)
    assert solver.planar_base == planar and (solver.nq, solver.nv, solver.nu) == (nq, nq, nq)
    model = solver.model
    data = model.createData()
    q = np.r_[[0.3, -0.2, 0.7] if planar else [], 40.3, -0.4]
    v = np.r_[[0.5, -0.3, 0.2] if planar else [], 0.2, -0.1]
    u = np.r_[[1.0, -2.0, 0.5] if planar else [], 0.7, -0.3]
    x = np.r_[q, v]
    got = np.asarray(solver.sim_forward(x[None], u[None], DT))[0]
    acceleration = pin.aba(model, data, q, v, u)
    np.testing.assert_allclose(got, np.r_[q + DT * v + 0.5 * DT**2 * acceleration, v + DT * acceleration],
                               atol=2e-4, rtol=1e-5)   # fp32 at q = 40 rad

    xu = np.tile(np.r_[x, u], N)[:-nq]
    goals = np.zeros((1, N * 6), dtype=np.float32)
    kkt = solver.debug_setup_kkt(xu[None], x[None], goals)
    aq, av, au = pin.computeABADerivatives(model, data, q, v, u)
    eye = np.eye(nq)
    A = np.asarray(kkt["A"])[0].reshape(N, 2 * nq, 2 * nq)[0].T
    B = np.asarray(kkt["B"])[0].reshape(N, nq, 2 * nq)[0].T
    np.testing.assert_allclose(A, np.block([[eye + 0.5 * DT**2 * aq, DT * eye + 0.5 * DT**2 * av],
                                            [DT * aq, eye + DT * av]]), atol=2e-4, rtol=1e-4)
    np.testing.assert_allclose(B, np.vstack([0.5 * DT**2 * au, DT * au]), atol=2e-5, rtol=1e-4)
    for block in ("Q", "R", "q", "r"):   # the barrier folds are on by default
        assert np.isfinite(np.asarray(kkt[block])).all(), block

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    goal = data.oMf[model.getFrameId("tool")].translation + np.array([0.05, 0.03, 0.0])
    goals[..., 0::6], goals[..., 1::6], goals[..., 2::6] = goal
    rest = np.r_[q, np.zeros(nq)].astype(np.float32)
    result = solver.solve(rest[None], goals)
    assert np.isfinite(np.asarray(result.xu)).all()

    controller = MPCController(gato.BSQP(str(urdf), batch_size=1, N=N, dt=DT, plant_type=name), limits="admm")
    controller.reset(rest)
    tick = controller.step(rest[None], goals)
    assert np.isfinite(np.asarray(tick.u)).all() and tick.limits == "admm"
    for group in controller.solver.get_row_groups():
        lo, hi = np.asarray(group["lo"])[:nq], np.asarray(group["hi"])[:nq]
        assert not np.isnan(lo).any() and not np.isnan(hi).any() and (lo <= hi).all()
