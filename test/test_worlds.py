"""Simulation-world gates (gato.worlds, 2026-08-04).

The harness advances the plant through a WORLD object. Two invariants matter:

1. the default (PinocchioWorld) is bit-identical to the pre-worlds inline loop
   — guarded here by run-twice determinism of a fixed-pacing MPC run (the
   refactor itself was stash-verified hash-equal at landing);
2. MuJoCoWorld is THE SAME ROBOT as the solver/pinocchio model — guarded by
   the dynamics fingerprint on the contact-disabled model, plus the
   self-contact masks (MuJoCo does not parent-filter world-welded bodies, so
   the intersecting base/L1 meshes produce phantom contacts that corrupt the
   dynamics 100x at q=0 — the masks must keep robot-robot contact off).
"""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from gato.config import IIWA14_START_CONFIGS

URDF = "examples/iiwa_description/iiwa14.urdf"


from conftest import mujoco_world  # noqa: E402


def _mujoco_world(**kw):
    return mujoco_world(URDF, **kw)


def _ready_ee():
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    q0 = IIWA14_START_CONFIGS["ready"].copy()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    return model, data, q0, data.oMf[model.getFrameId("EE")].translation.copy()


def test_mujoco_is_the_same_robot():
    """MuJoCo-of-our-URDF matches the attested dynamics fingerprint (1e-7 class)."""
    mujoco = pytest.importorskip("mujoco")
    import gato.fingerprint as fp
    w = _mujoco_world()
    w.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    def mj_qdd(q, qd, u):
        d = w.data
        d.qpos[:] = q
        d.qvel[:] = qd
        d.qfrc_applied[:] = u
        mujoco.mj_forward(w.model, d)
        return d.qacc.copy()

    res = fp.check(mj_qdd, "iiwa14")
    assert res["ok"], fp.report(res)


def test_probe_tip_drop_finds_contact_point():
    """The EE-frame-origin-vs-tool-tip probe (the ~33 mm hidden-press-bias
    trap): the drop is positive, plausible, deterministic, and consistent —
    a table just below the probed tip is clear, just above it touches."""
    mujoco = pytest.importorskip("mujoco")
    from gato.worlds import MuJoCoWorld, probe_tip_drop
    model, data, q0, ee = _ready_ee()
    drop = probe_tip_drop(URDF, q0, ee)
    assert 0.005 < drop < 0.10, f"implausible tip drop {drop}"
    assert drop == probe_tip_drop(URDF, q0, ee)  # deterministic

    def touches(z):
        w = MuJoCoWorld(URDF, plane={"z": z, "pos_xy": (ee[0], ee[1])})
        w.data.qpos[:] = q0
        mujoco.mj_forward(w.model, w.data)
        return w.data.ncon > 0

    tip_z = ee[2] - drop
    assert not touches(tip_z - 1e-4) and touches(tip_z + 1e-4)


def test_mujoco_no_phantom_self_contacts():
    """q=0 is the config where base/L1 meshes interpenetrate: masks must hold."""
    w = _mujoco_world(plane={"z": -0.5})
    w.step(np.zeros(7), np.zeros(7), np.zeros(7), 1e-3)
    assert w.last_contact["ncon"] == 0


def test_mujoco_table_press_and_determinism():
    """A gravity-compensated press onto the table settles at a physical force,
    and the loop is bit-deterministic run-to-run (contact active)."""
    model, data, q0, ee = _ready_ee()
    FID = model.getFrameId("EE")
    J = pin.computeFrameJacobian(model, data, q0, FID,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
    dq_ik = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3),
                                  np.array([0.0, 0.0, -0.05]))
    q_tgt = q0 + dq_ik
    M0 = pin.crba(model, data, q0)
    Md = np.maximum(np.diag(np.triu(M0) + np.triu(M0, 1).T), 1e-2)
    KP, KD = 100.0 * Md, 20.0 * Md
    plane = {"z": ee[2] - 0.02, "pos_xy": (ee[0], ee[1])}

    def roll(n):
        w = _mujoco_world(plane=plane)
        q, dq = q0.copy(), np.zeros(7)
        tr, fn = [], 0.0
        for _ in range(n):
            g = pin.computeGeneralizedGravity(model, data, q)
            q, dq = w.step(q, dq, g + KP * (q_tgt - q) - KD * dq, 1e-3)
            tr.append(np.concatenate([q, dq]))
            fn = w.last_contact["fn"]
        return np.asarray(tr), fn

    tr1, fn1 = roll(600)
    tr2, _ = roll(600)
    assert np.array_equal(tr1, tr2), "MuJoCo world is not run-to-run deterministic"
    assert 1.0 < fn1 < 200.0, f"settled normal force {fn1} N is not physical"


@pytest.mark.gpu
def test_mpc_through_mujoco_world_reaches_goal():
    """Full closed loop: BSQP MPC driving the MuJoCo world (free space)."""
    pytest.importorskip("mujoco")
    from gato.mpc_gato import MPC_GATO
    from gato.worlds import MuJoCoWorld
    model = pin.buildModelFromUrdf(URDF)
    q0 = IIWA14_START_CONFIGS["ready"]
    x0 = np.concatenate([q0, np.zeros(7)])
    mpc = MPC_GATO(model, URDF, N=16, batch_size=1, world=MuJoCoWorld(URDF))
    _, s = mpc.run_mpc_goals(x0, [np.array([0.55, -0.1, 0.7])],
                             goal_timeout=3.0, pace_by_solve_time=False)
    assert s["goal_outcomes"] == ["reached"]


@pytest.mark.gpu
def test_default_world_fixed_pacing_deterministic():
    """The default PinocchioWorld path (pendulum + hypothesis batch) is
    bit-deterministic under fixed pacing — the invariant every pool rests on."""
    from gato.mpc_gato import MPC_GATO

    def run():
        model = pin.buildModelFromUrdf(URDF)
        mpc = MPC_GATO(model, URDF, N=16, batch_size=4,
                       pendulum_config={"mass": 5.0, "length": 0.5,
                                        "damping": 0.4, "initial_angle": 0.3})
        x0 = np.concatenate([IIWA14_START_CONFIGS["ready"], np.zeros(7)])
        _, s = mpc.run_mpc_goals(x0, [np.array([0.55, 0.2, 0.6])],
                                 goal_timeout=1.0, pace_by_solve_time=False)
        return np.concatenate([np.asarray(s["joint_positions"]).ravel(),
                               np.asarray(s["joint_velocities"]).ravel()])

    assert np.array_equal(run(), run())


@pytest.mark.gpu
def test_fig8_with_pendulum_keeps_augmented_state_out_of_solver():
    """Both task loops share valid sim initialization, but the solver sees only
    robot q/v (the spherical payload's four q / three v slots stay in the sim)."""
    from gato.mpc_gato import MPC_GATO
    model = pin.buildModelFromUrdf(URDF)
    x = np.concatenate([IIWA14_START_CONFIGS['ready'], np.zeros(model.nv)])
    mpc = MPC_GATO(model, URDF, N=16, batch_size=1, linsys='pcg',
                   pendulum_config=dict(mass=5.0, length=0.5, initial_angle=[0.2, 0, 0]))
    ref = np.tile(np.r_[mpc.solver.ee_pos(x[:model.nq]), np.zeros(3)], 200)
    _, stats = mpc.run_mpc_fig8(x, ref, sim_time=0.03, pace_by_solve_time=False)
    assert len(stats['timestamps']) > 0
    assert stats['joint_positions'].shape[1] == model.nq + 4
    assert stats['joint_velocities'].shape[1] == model.nv + 3
    assert np.isfinite(stats['joint_positions']).all()
    np.testing.assert_allclose(np.linalg.norm(stats['joint_positions'][:, model.nq:], axis=1),
                               1.0, atol=1e-12)


@pytest.mark.gpu
def test_custom_world_rejects_pendulum_and_fext():
    from gato.mpc_gato import MPC_GATO
    from gato.worlds import MuJoCoWorld
    pytest.importorskip("mujoco")
    model = pin.buildModelFromUrdf(URDF)
    with pytest.raises(ValueError, match="pendulum"):
        MPC_GATO(model, URDF, N=16, batch_size=1, world=MuJoCoWorld(URDF),
                 pendulum_config={"mass": 1.0, "length": 0.5, "damping": 0.4,
                                  "initial_angle": 0.0})
    model2 = pin.buildModelFromUrdf(URDF)
    with pytest.raises(ValueError, match="pendulum|f_ext"):
        MPC_GATO(model2, URDF, N=16, batch_size=1, world=MuJoCoWorld(URDF),
                 constant_f_ext=np.array([0, 0, -20.0, 0, 0, 0]))


@pytest.mark.gpu
def test_goal_dwell_gate_delays_reach_and_rejects_fly_through():
    """settle_time=0 is the paper's instantaneous gate; with a dwell the goal
    counts only after distance AND velocity have held that long, so the reach
    time moves later by at least the dwell and never earlier."""
    from gato.mpc_gato import MPC_GATO
    model = pin.buildModelFromUrdf(URDF)
    x0 = np.concatenate([IIWA14_START_CONFIGS["ready"], np.zeros(7)])
    goal = [np.array([0.55, -0.1, 0.7])]
    times = {}
    for dwell in (0.0, 0.2):
        mpc = MPC_GATO(model, URDF, N=16, batch_size=1)
        _, s = mpc.run_mpc_goals(x0, goal, goal_timeout=3.0, pace_by_solve_time=False,
                                 settle_time=dwell)
        assert s["goal_outcomes"] == ["reached"], (dwell, s["goal_events"])
        times[dwell] = s["goal_events"][-1]["time"]
        assert s["goal_events"][-1]["distance"] < 0.05 and s["goal_events"][-1]["velocity"] < 1.0
    assert times[0.2] >= times[0.0] + 0.2 - 1e-6


@pytest.mark.gpu
def test_goal_ramp_reference_delays_arrival():
    """goal_ramp > 0 feeds a minimum-jerk reference that reaches the goal only after the
    ramp time, so the arm cannot arrive before most of it has elapsed; goal_ramp=0 is the
    paper's step reference. Both reach the goal."""
    from gato.mpc_gato import MPC_GATO
    model = pin.buildModelFromUrdf(URDF)
    x0 = np.concatenate([IIWA14_START_CONFIGS["ready"], np.zeros(7)])
    goal = [np.array([0.55, -0.1, 0.7])]
    times = {}
    for ramp in (0.0, 1.0):
        mpc = MPC_GATO(model, URDF, N=16, batch_size=1)
        _, s = mpc.run_mpc_goals(x0, goal, goal_timeout=4.0, pace_by_solve_time=False,
                                 goal_ramp=ramp)
        assert s["goal_outcomes"] == ["reached"], (ramp, s["goal_events"])
        times[ramp] = s["goal_events"][-1]["time"]
    assert times[1.0] >= 0.6 and times[1.0] > times[0.0]
