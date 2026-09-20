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

from conftest import TEST_PARAMS

pin = pytest.importorskip("pinocchio")

from gato.common import rk4, state_difference, check_floating_state
from gato.config import GO2_START_CONFIGS

from conftest import GO2_URDF, GO2_NQ as NQ, GO2_NV as NV, GO2_NU as NU, go2_standing_x as _standing_x, mujoco_world  # noqa: E402

URDF = str(GO2_URDF)
RNG = np.random.default_rng(3)


@pytest.fixture(scope="module")
def model(go2_model):
    return go2_model


def _rand_x(model):
    q = np.zeros(NQ)
    q[:3] = RNG.standard_normal(3)
    quat = RNG.standard_normal(4)
    q[3:7] = quat / np.linalg.norm(quat)
    q[7:] = RNG.uniform(model.lowerPositionLimit[7:], model.upperPositionLimit[7:])
    return np.concatenate([q, RNG.normal(0, 0.5, NV)])


def _mujoco_world(**kw):
    return mujoco_world(URDF, floating=True, **kw)


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
    s = gato.BSQP(model_path=URDF, batch_size=1, N=N, dt=DT, params=TEST_PARAMS.replace(q_cost=5.0, qd_cost=1e-1, u_cost=1e-4, N_cost=25.0, q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0), plant_type="go2")
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
    s = gato.BSQP(model_path=URDF, batch_size=1, N=16, dt=0.01, params=TEST_PARAMS.replace(q_cost=1.0, qd_cost=1e-2, u_cost=1e-4, N_cost=5.0, q_lim_cost=1e-3, vel_lim_cost=0.0, ctrl_lim_cost=0.0),
                  plant_type="go2")
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


# ---------------------------------------------------------------------------
# Wave F (fc-on-feet, 2026-09-20): the go2 fc module in closed loop on the
# MuJoCo ground plane — THE standing gate the contactless model physically
# cannot pass (a free-flyer with no contact model is weightless in its own
# frame; see _mpc_standing_run). S1 = all four feet in stance, no schedule:
# the solver explains gravity through the four foot wrenches (fc_ref = mg/4 up
# per foot, moment rows pinned), so the posture anchor + base-height goal are
# reachable at the stance height.
#
# Geometry trap (found 2026-09-20): the standing KEYFRAME (base z = 0.35) has
# the foot frames 8.5 cm above the ground — it is an in-the-air pose. Feet
# touch down at base z ≈ 0.287 (frame height minus the 2.2 cm foot sphere),
# so the stance pose/goal below are derived from FK, not the keyframe; an imu
# goal taken at the keyframe is 6 cm out of reach and keeps the loop fighting.
# Measured on the fc loop at the stance goal: base 0.287 ± 0.002 m, Σfn = mg,
# per foot 39.5 N (mg/4 = 39.45), solver fz explanation 39.5 N/foot; the
# one-step pred_err (~0.4-0.6) is joint-VELOCITY jitter from MuJoCo's soft
# contact (positions predict to 1e-4), so the controller's default
# reseed_threshold=None is used here (0.5 would re-seed every other tick).
# ---------------------------------------------------------------------------

from conftest import GO2_FEET, GO2_FC, go2_solver as _go2_solver  # noqa: E402

# MuJoCo's URDF import merges the fixed *_foot links into their calves, so the
# per-body contact split is keyed by the calf bodies
FOOT_BODY = {"FR_foot_joint": "FR_calf", "FL_foot_joint": "FL_calf",
             "RR_foot_joint": "RR_calf", "RL_foot_joint": "RL_calf"}
FOOT_RADIUS = 0.022   # go2.urdf *_foot collision sphere


def _mg(model):
    return sum(i.mass for i in model.inertias) * 9.81


def _stance_x(model):
    """The standing keyframe lowered so the foot spheres rest on the plane."""
    x = _standing_x().astype(np.float64)
    data = model.createData()
    pin.framesForwardKinematics(model, data, x[:NQ])
    foot_z = np.array([data.oMf[model.getFrameId(f)].translation[2] for f in GO2_FEET])
    x[2] -= foot_z.mean() - FOOT_RADIUS
    pin.framesForwardKinematics(model, data, x[:NQ])
    for f in GO2_FEET:
        assert abs(data.oMf[model.getFrameId(f)].translation[2] - FOOT_RADIUS) < 1e-3
    return x


def _imu_goals(model, x):
    data = model.createData()
    pin.framesForwardKinematics(model, data, np.asarray(x[:NQ], dtype=np.float64))
    p = data.oMf[model.getFrameId("imu_joint")].translation
    goals = np.zeros(16 * 6, dtype=np.float32)
    goals[0::6], goals[1::6], goals[2::6] = p[0], p[1], p[2]
    return goals


_STAND_PARAMS = dict(q_cost=5.0, qd_cost=1e-1, u_cost=1e-4, N_cost=25.0,
                     q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)


def _fc_standing_solver(model, fc_cost=1e-2, fn_ref=None, pin_moments=True, q_pos_cost=50.0):
    """go2 fc solver programmed for S1 standing: posture anchor at the stance
    pose, per-foot vertical reference fn_ref (default mg/4), moment rows
    pinned (AL). Returns (solver, x_stance, goals)."""
    s = _go2_solver(1, variant="fc", **_STAND_PARAMS)
    x = _stance_x(model).astype(np.float32)
    s.set_q_nom(x[:NQ])
    s.set_q_pos_cost(q_pos_cost)
    fn_ref = _mg(model) / 4 if fn_ref is None else fn_ref
    ref = np.zeros(GO2_FC, dtype=np.float32)
    for i in range(len(GO2_FEET)):
        ref[s.fc_slots(i, "f")[2]] = fn_ref
        if pin_moments:
            s.add_fc_box(0.0, 0.0, slots=s.fc_slots(i, "n"), mech="al")   # enforced, not telemetry
    s.set_fc_cost(fc_cost)
    s.set_fc_ref(ref)
    return s, x, _imu_goals(model, x)


def _closed_loop(s, x, goals, steps, drop=0.01, **ctrl_kw):
    """Fixed pacing (one solve per 10 ms of sim), torques saturated at the URDF
    effort limit; starts `drop` above x. Returns the world, final state, peak
    torque, the last StepResult and the base-height trace."""
    from gato.controller import MPCController
    w = _mujoco_world(plane={"z": 0.0, "pos_xy": (0.0, 0.0), "size_xy": (1.0, 1.0)})
    ctrl = MPCController(s, **ctrl_kw)
    ctrl.reset(x)
    q, dq = np.asarray(x[:NQ], np.float64).copy(), np.asarray(x[NQ:], np.float64).copy()
    q[2] += drop
    umax, z_hist, r = 0.0, [], None
    for _ in range(steps):
        r = ctrl.step(np.concatenate([q, dq]).astype(np.float32), goals)
        u = np.clip(np.asarray(r.u, np.float64), -23.7, 23.7)
        umax = max(umax, float(np.abs(u).max()))
        for _ in range(10):
            q, dq = w.step(q, dq, u, 1e-3)
        z_hist.append(q[2])
    return w, q, dq, umax, r, np.asarray(z_hist)


def _mpc_standing_run_fc(model, steps=150, **solver_kw):
    s, x, goals = _fc_standing_solver(model, **solver_kw)
    return (x[2],) + _closed_loop(s, x, goals, steps)


@pytest.mark.gpu
@pytest.mark.slow
def test_fc_mpc_stands_at_height(model):
    """S1 standing gate: with contact forces in the model the closed loop holds
    the STANCE height, upright on four feet, total normal force ~mg with each
    foot carrying a share, and the solver's knot-0 wrench explanation agrees
    with the measured contact (same sign, order and sum — fc is the model's
    contact explanation, so this is the "is the model live" check)."""
    z_stance, w, q, dq, umax, r, z = _mpc_standing_run_fc(model)   # 1.5 s of sim
    mg = _mg(model)
    c = w.last_contact
    assert np.isfinite(q).all() and np.isfinite(dq).all()
    assert c["ncon"] >= 4, c
    assert 0.9 * mg < c["fn"] < 1.1 * mg, (c["fn"], mg)
    per_foot = [c["fn_by_body"].get(FOOT_BODY[f], 0.0) for f in GO2_FEET]
    assert min(per_foot) > 0.6 * mg / 4 and max(per_foot) < 1.4 * mg / 4, (per_foot, mg / 4)
    # THE gate: settled AT the stance height (the contactless A/B below collapses)
    assert abs(q[2] - z_stance) < 0.015, (q[2], z_stance)
    assert np.ptp(z[-50:]) < 0.01, (z[-50:].min(), z[-50:].max())
    assert np.linalg.norm(dq) < 0.5, dq
    assert abs(q[6]) > 0.99 and abs(np.linalg.norm(q[3:7]) - 1.0) < 1e-6, q[3:7]
    assert umax > 1.0, umax
    # solver explanation vs world: vertical foot forces up, summing to ~mg
    fc0 = np.asarray(r.fc, np.float64)
    fz = fc0[5::6]
    assert np.all(fz > 0.0), fz
    assert 0.85 * mg < fz.sum() < 1.15 * mg, (fz.sum(), mg)
    assert np.abs(fc0[[j for i in range(4) for j in range(6 * i, 6 * i + 3)]]).max() < 0.5, fc0  # moments pinned


@pytest.mark.gpu
@pytest.mark.slow
def test_fc_mpc_holds_height_where_contactless_collapses(model):
    """The A/B that makes the gate above meaningful: the DEFAULT module (no
    contact model) from the same stance start, goal and params ends far below
    the stance height (measured: 0.076 m vs 0.287 m) — the fc model is what
    holds the robot up, not the posture anchor."""
    _, x, goals = _fc_standing_solver(model)
    z_stance = float(x[2])
    s0 = _go2_solver(1, **_STAND_PARAMS)
    s0.set_q_nom(x[:NQ])
    s0.set_q_pos_cost(50.0)
    _, q, _, _, _, _ = _closed_loop(s0, x, goals, steps=150)
    assert q[2] < z_stance - 0.1, (q[2], z_stance)


@pytest.mark.gpu
@pytest.mark.slow
def test_fc_mpc_standing_deterministic(model):
    _, _, q1, dq1, _, _, _ = _mpc_standing_run_fc(model, steps=60)
    _, _, q2, dq2, _, _, _ = _mpc_standing_run_fc(model, steps=60)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(dq1, dq2)
