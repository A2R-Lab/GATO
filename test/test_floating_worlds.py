"""CL-3 Wave B: go2 in the python MPC/world layer.

Gates the floating-base plumbing that does NOT need the solver's EE cost
(the closed-loop standing gate waits on GRiD BUG 7):

- ``common.state_difference`` (pure numpy, no scipy/pin at runtime) matches
  ``pin.difference`` on random free-flyer states;
- ``common.rk4`` scatters actuated-only torques onto the trailing dofs
  (== pin RK4 with the explicitly padded generalized force);
- MuJoCo-of-our-URDF is the SAME robot as the attested go2 fingerprint
  (contacts disabled; massless-link inertia pinned — MuJoCo synthesizes
  ~0.19 kg of phantom calflower mass from geometry otherwise);
- no phantom contacts (lifted robot over the plane => zero contacts);
- a standing drop under a joint-space PD settles onto four feet with total
  normal force ~mg, bit-deterministically run-to-run;
- the MPCController floating path: unit-quaternion validation + tangent
  pred_err.
"""
import importlib.util

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from gato.common import rk4, state_difference, check_floating_state
from gato.config import GO2_START_CONFIGS

URDF = "external/GRiD/config/robot_assets/go2.urdf"
NQ, NV, NU = 19, 18, 12
RNG = np.random.default_rng(3)


@pytest.fixture(scope="module")
def model():
    return pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())


def _standing_x():
    return np.concatenate([GO2_START_CONFIGS["standing"], np.zeros(NV)])


def _rand_x(model):
    q = np.zeros(NQ)
    q[:3] = RNG.standard_normal(3)
    quat = RNG.standard_normal(4)
    q[3:7] = quat / np.linalg.norm(quat)
    q[7:] = RNG.uniform(model.lowerPositionLimit[7:], model.upperPositionLimit[7:])
    return np.concatenate([q, RNG.normal(0, 0.5, NV)])


def _mujoco_world(**kw):
    pytest.importorskip("mujoco")
    from gato.worlds import MuJoCoWorld
    return MuJoCoWorld(URDF, floating=True, **kw)


def test_state_difference_matches_pinocchio(model):
    for _ in range(50):
        xa, xb = _rand_x(model), _rand_x(model)
        want = np.concatenate([pin.difference(model, xa[:NQ], xb[:NQ]),
                               xb[NQ:] - xa[NQ:]])
        got = state_difference(xa, xb, NQ, NV)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    # fixed-base path: plain subtraction
    a, b = RNG.standard_normal(14), RNG.standard_normal(14)
    np.testing.assert_array_equal(state_difference(a, b, 7, 7), b - a)


def test_check_floating_state(model):
    check_floating_state(_standing_x(), NQ, NV)
    with pytest.raises(ValueError):
        check_floating_state(np.zeros(NQ + NV), NQ, NV)
    check_floating_state(np.zeros(14), 7, 7)  # fixed base: no-op


def test_rk4_actuated_scatter_matches_padded(model):
    data = model.createData()
    x = _standing_x()
    u_act = RNG.normal(0, 3.0, NU)
    q1, v1 = rk4(model, data, x[:NQ], x[NQ:], u_act, 0.01)
    q2, v2 = rk4(model, data, x[:NQ], x[NQ:],
                 np.concatenate([np.zeros(6), u_act]), 0.01)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(v1, v2)
    assert np.isfinite(q1).all() and np.isfinite(v1).all()


def test_mujoco_go2_is_the_same_robot():
    mujoco = pytest.importorskip("mujoco")
    import gato.fingerprint as fp
    w = _mujoco_world()
    assert (w.nq, w.nv) == (NQ, NV)
    w.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    def mj_qdd(q, qd, u):
        d = w.data
        qm = np.array(q)
        qm[3] = q[6]
        qm[4:7] = q[3:6]  # xyzw -> wxyz (probe base velocities are zero)
        d.qpos[:] = qm
        d.qvel[:] = qd
        d.ctrl[:] = 0
        d.qfrc_applied[:] = u
        mujoco.mj_forward(w.model, d)
        return d.qacc.copy()

    res = fp.check(mj_qdd, "go2")
    assert res["ok"], fp.report(res)
    ok = np.isfinite(res["inertia_ratio"])
    np.testing.assert_allclose(res["inertia_ratio"][ok], 1.0, atol=0.05)


def test_mujoco_no_phantom_contacts():
    w = _mujoco_world(plane={"z": 0.0, "pos_xy": (0.0, 0.0), "size_xy": (1.0, 1.0)})
    x = _standing_x()
    x[2] = 1.0  # lifted well above the plane
    q, dq = w.step(x[:NQ], x[NQ:], np.zeros(NU), 1e-3)
    assert w.last_contact["ncon"] == 0
    assert np.isfinite(q).all() and np.isfinite(dq).all()


def _pd_standing_drop(substeps=600):
    w = _mujoco_world(plane={"z": 0.0, "pos_xy": (0.0, 0.0), "size_xy": (1.0, 1.0)})
    x = _standing_x()
    q, dq = x[:NQ].copy(), x[NQ:].copy()
    q[2] = 0.36  # slight drop onto the feet
    q_ref = x[7:NQ]
    for _ in range(substeps):
        u = np.clip(80.0 * (q_ref - q[7:]) - 3.0 * dq[6:], -20.0, 20.0)
        q, dq = w.step(q, dq, u, 1e-3)
    return w, q, dq


def test_mujoco_standing_drop_settles_on_four_feet(model):
    w, q, dq = _pd_standing_drop()
    mg = sum(i.mass for i in model.inertias) * 9.81
    c = w.last_contact
    assert c["ncon"] >= 4, c
    assert 0.7 * mg < c["fn"] < 1.3 * mg, (c, mg)
    assert 0.2 < q[2] < 0.4              # settled near the stance height
    assert np.linalg.norm(dq) < 0.5      # at rest
    assert abs(np.linalg.norm(q[3:7]) - 1.0) < 1e-6


def test_mujoco_standing_drop_deterministic():
    _, q1, dq1 = _pd_standing_drop(300)
    _, q2, dq2 = _pd_standing_drop(300)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(dq1, dq2)


@pytest.mark.gpu
def test_device_dynamics_match_go2_fingerprint():
    """Device qdd (via the exact SI-EULER sim identity) vs the committed
    table — the drift tripwire for the GRiD fold-in regen."""
    if importlib.util.find_spec("gato.bsqpN16_go2") is None:
        pytest.skip("bsqpN16_go2 module not built")
    import gato
    from gato import fingerprint
    s = gato.BSQP(model_path=URDF, batch_size=1, N=16, dt=0.01, plant_type="go2")
    res = fingerprint.check_solver(s, rtol=1e-4)
    assert res["ok"], fingerprint.report(res)


def _mpc_standing_run(model, steps=150):
    """MPC (go2 N16) in the loop on MuJoCo with ground contact — the CLOSED-LOOP
    PLUMBING smoke (controller + solver + world + conversions), fixed pacing
    (one solve per 10 ms of sim), torques saturated at the URDF effort limit.

    Scope note (2026-08-11): the solver has NO contact model, and a contactless
    free-flyer is WEIGHTLESS in its own frame (joint-space gravity terms vanish
    in free fall), so at the posture anchor the optimal torque is ~0 and the
    robot can only resist gravity through feedback stiffness — it settles in a
    deep crouch, not at the standing height. Holding a base-height band is
    physically out of reach for this model; the standing-at-height gate lands
    with the fc-on-feet wave (contact forces in the model). Known related
    behavior: a warm start with a knot0 discontinuity (measured state vs a
    stale tail) is a merit local minimum at mu=1 — the line search rejects
    every step (raising mu helps but does not cure; see the SSOT)."""
    import gato
    from gato.controller import MPCController
    N, DT = 16, 0.01
    s = gato.BSQP(model_path=URDF, batch_size=1, N=N, dt=DT, plant_type="go2",
                  q_cost=5.0, qd_cost=1e-1, u_cost=1e-4, N_cost=25.0,
                  q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
    x = _standing_x().astype(np.float32)
    s.set_q_nom(x[:NQ])
    s.set_q_pos_cost(50.0)
    # imu EE goal pinned at the STANDING pose's FK height (the BUG 7 unblock:
    # this cost only acts through a world-frame EE + nonzero base Jacobian)
    data = model.createData()
    pin.framesForwardKinematics(model, data, np.asarray(x[:NQ], dtype=np.float64))
    p = data.oMf[model.getFrameId("imu_joint")].translation
    goals = np.zeros(N * 6, dtype=np.float32)
    goals[0::6], goals[1::6], goals[2::6] = p[0], p[1], p[2]

    w = _mujoco_world(plane={"z": 0.0, "pos_xy": (0.0, 0.0), "size_xy": (1.0, 1.0)})
    ctrl = MPCController(s)
    ctrl.reset(x)
    q, dq = np.asarray(x[:NQ], np.float64).copy(), np.asarray(x[NQ:], np.float64).copy()
    q[2] = 0.36  # slight drop onto the feet, as the PD gate
    umax = 0.0
    for _ in range(steps):
        r = ctrl.step(np.concatenate([q, dq]).astype(np.float32), goals)
        u = np.clip(np.asarray(r.u, np.float64), -23.7, 23.7)  # go2 effort limit
        umax = max(umax, float(np.abs(u).max()))
        for _ in range(10):
            q, dq = w.step(q, dq, u, 1e-3)
    return w, q, dq, umax


@pytest.mark.gpu
@pytest.mark.slow
def test_mpc_closed_loop_settles_upright(model):
    if importlib.util.find_spec("gato.bsqpN16_go2") is None:
        pytest.skip("bsqpN16_go2 module not built")
    w, q, dq, umax = _mpc_standing_run(model)  # 1.5 s of sim
    mg = sum(i.mass for i in model.inertias) * 9.81
    c = w.last_contact
    assert np.isfinite(q).all() and np.isfinite(dq).all()
    assert c["ncon"] >= 4, c                 # resting on its legs, not tipped
    assert 0.5 * mg < c["fn"] < 1.5 * mg, (c, mg)
    assert 0.05 < q[2] < 0.45, q[2]          # settled (crouched: contactless model)
    assert np.linalg.norm(dq) < 1.0, dq      # at rest, not thrashing
    # upright: base rotation stays near identity (quat w component, xyzw)
    assert abs(q[6]) > 0.95, q[3:7]
    assert abs(np.linalg.norm(q[3:7]) - 1.0) < 1e-6
    # the anchor feedback path is LIVE (deflection produced restoring torque)
    assert umax > 1.0, umax


@pytest.mark.gpu
@pytest.mark.slow
def test_mpc_closed_loop_deterministic(model):
    if importlib.util.find_spec("gato.bsqpN16_go2") is None:
        pytest.skip("bsqpN16_go2 module not built")
    _, q1, dq1, _ = _mpc_standing_run(model, steps=60)
    _, q2, dq2, _ = _mpc_standing_run(model, steps=60)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(dq1, dq2)


@pytest.mark.gpu
def test_controller_floating_state_checks_and_pred_err(model):
    if importlib.util.find_spec("gato.bsqpN16_go2") is None:
        pytest.skip("bsqpN16_go2 module not built")
    import gato
    from gato.controller import MPCController
    s = gato.BSQP(model_path=URDF, batch_size=1, N=16, dt=0.01,
                  plant_type="go2", q_cost=1.0, qd_cost=1e-2, u_cost=1e-4,
                  N_cost=5.0, q_lim_cost=1e-3, vel_lim_cost=0.0,
                  ctrl_lim_cost=0.0)
    ctrl = MPCController(s)
    x0 = _standing_x().astype(np.float32)
    with pytest.raises(ValueError):
        ctrl.reset(np.zeros_like(x0))  # zero quaternion must fail loud
    ctrl.reset(x0)
    goals = np.zeros(16 * 6, dtype=np.float32)
    goals[2::6] = 0.35
    r1 = ctrl.step(x0, goals)
    assert np.isfinite(r1.u).all() and r1.u.shape == (NU,)
    assert np.isfinite(r1.pred_err)
    # tangent pred_err: a pure base yaw of 0.1 rad reads ~0.1, not a
    # quaternion-component artifact
    x_rot = x0.copy()
    x_rot[3:7] = [0.0, 0.0, np.sin(0.05), np.cos(0.05)]
    ctrl.reset(x0)
    ctrl.step(x0, goals)
    r2 = ctrl.step(x_rot, goals)
    assert np.isfinite(r2.pred_err)
    with pytest.raises(ValueError):
        ctrl.step(np.zeros_like(x0), goals)
