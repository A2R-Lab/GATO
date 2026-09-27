"""CONTACT_POS rows (CL-4 §1.3): per-knot contact-frame POSITION residuals
g = p_f(q_k) - tgt_k over GRiD's contact_frame_positions surface (pin 5904dbd,
2026-09-26) — the stance/swing foot rows of the locomotion arc. Gates: the
device residual is pinocchio's (telemetry), AL/ADMM pin an arm's EE at every
knot, the per-knot mask and target table select (knot, row) slots, go2 keeps
its feet under a lateral base goal, GaitProgrammer.install_foot_rows programs
the same thing from a schedule."""
import numpy as np
import pytest

import gato
from conftest import (GO2_FC, go2_solver, go2_standing_x, go2_goals_at, GO2_URDF, ARM_START, arm_problem, go2_fc as _go2_fc,
                      go2_mg, q_at as _q_at, contact_frames_along as _frames, outer_solve as _outer, require_module)

pytestmark = pytest.mark.gpu


# The GENTLE arm scenario of test_row_masks (indy7 N8, receipt-profile module,
# u_cost 1e-4, goal 3 cm up): the SQP converges and the EE really moves — the
# paper problems' far goals leave the line search at its floor (and iiwa14 N8
# does not move at all for a 3 cm goal), so they cannot gate row enforcement.
GENTLE = ("indy7", 8)


def _gentle(make_solver):
    plant, N = GENTLE
    require_module("bsqpN8_indy7")
    s = make_solver(plant, N, u_cost=1e-4)
    q0 = np.asarray(ARM_START[plant], dtype=np.float32)
    x = np.concatenate([q0, np.zeros_like(q0)])[None]
    goals = np.zeros((1, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = s.ee_pos(q0.astype(np.float64)) + np.array([0.0, 0.0, 0.03])
    return s, x, goals


def _accepted(r):
    ss = np.asarray(r.stats.step_size)
    return int((ss[:, 0] > 0).sum()) if ss.size else 0


@_go2_fc
@pytest.mark.parametrize("mech", ["al", "barrier", "admm"])
def test_masked_contact_sanitizer_smoke(mech):
    """Bounded instrumentation gate: one SQP/ADMM iteration, changing targets
    and row masks. Full convergence and numerical parity are tested elsewhere.
    Small enough to run under compute-sanitizer racecheck on the shared box.
    """
    s = go2_solver(1, variant="fc", max_sqp_iters=1, rho=0.1)
    x = go2_standing_x().astype(np.float32)
    goals = go2_goals_at(s.model, x.astype(np.float64), 1)
    targets = np.tile(s.contact_positions(x[:s.nq]).reshape(1, -1), (s.N, 1))
    targets[s.N // 2:, 2] += 0.005
    gi = s.add_contact_pos_rows(targets=targets, mech=mech, rho=100.0, admm_iters=1)
    mask = np.ones((s.N, s.n_contact_rows), dtype=bool)
    mask[0] = False
    mask[1:s.N // 2, :3] = False
    s.set_row_group_mask(gi, mask)
    r = s.solve(x[None], goals)
    assert np.isfinite(r.xu).all()
    assert np.isfinite(r.stats.final_merit).all()


def test_contact_rows_telemetry_is_the_device_residual(make_solver, smallest_module):
    """Telemetry rows report max |p_f(q_0) - tgt| over the frame's xyz at knot 0
    (the measured state): the on-device contact-origin FK agrees with pinocchio,
    so the residual is exactly the programmed offset. The descriptor round-trips
    kind 6, three rows per frame and the per-knot target table."""
    plant, N = smallest_module
    x, goals = arm_problem(plant, N)
    s = make_solver(plant, N)
    s.enable_limit_telemetry()
    p0 = s.contact_positions(x[0, :s.nq])                    # (1, 3): the arm EE
    off = np.array([[0.01, -0.02, 0.03]])
    gi = s.add_contact_pos_rows(targets=p0 + off, mech="telemetry", knot_lo=0, knot_hi=1)
    r = s.solve(x, goals)
    assert abs(float(r.stats.row_max_violation[gi, 0]) - 0.03) < 2e-4, r.stats.row_max_violation
    grp = s.get_row_groups()[gi]
    assert grp["kind"] == 6 and grp["n_rows"] == 3 * len(s.contact_frames) == s.n_contact_rows
    np.testing.assert_allclose(grp["tgt"][0], (p0 + off).reshape(-1), atol=1e-6)
    tgt = np.tile((p0 + off).reshape(1, -1), (N, 1))
    tgt[0, 2] = p0[0, 2] - 0.05                              # knot 0's own target moves
    s.set_row_group_targets(gi, tgt)
    r = s.solve(x, goals)
    assert abs(float(r.stats.row_max_violation[gi, 0]) - 0.05) < 2e-4
    with pytest.raises(ValueError):
        s.set_row_group_targets(0, tgt)                       # the limit box is not a CONTACT_POS group


# Mechanism regime (measured 2026-09-26 on the gentle indy7 scenario, tracking =
# stay, row target 2 cm up): the AL outer loop converges/freezes after the first
# solve on these problems, so the rows act as PENALTIES — |p - tgt| 2.3 cm at rho
# 10, 1.3 cm at rho 100, 0.8 mm at rho 1e3 (the terminal EE_POS row does not move
# the EE at all here: KKT-converged at start at every rho). ADMM projects per inner
# iteration and reaches the same at rho ~1e2. On the go2 standing costs the foot
# rows need AL rho 1e3 solver-level (GaitProgrammer defaults to the loop-stable 100).
ROW_RHO = {"al": 1000.0, "admm": 100.0}


@pytest.mark.parametrize("mech", ["al", "admm"])
def test_contact_rows_move_the_arm_ee_to_a_violated_target(make_solver, mech):
    """Rows on knots 4..N-1 with the target 2 cm above the start while the tracking
    goal says STAY: the free solve leaves the EE near the start (it drifts a few
    mm), the rows move it to the target within millimetres at every active knot
    (per-knot residual rows enforce against the tracking cost; knots 1..3 stay
    outside the window)."""
    s, x, goals = _gentle(make_solver)
    N = s.N
    p0 = s.contact_positions(x[0, :s.nq])
    goals = np.zeros_like(goals)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = p0[0]                       # stay
    tgt = p0 + np.array([[0.0, 0.0, 0.02]])
    free = _gentle(make_solver)[0]
    assert np.linalg.norm(_frames(free, free.solve(x, goals))[-1] - tgt) > 1e-2   # tracking alone does not reach the target
    s.enable_limit_telemetry()
    gi = s.add_contact_pos_rows(targets=tgt, mech=mech, rho=ROW_RHO[mech], knot_lo=4)
    r = _outer(s, x, goals, 6 if mech == "al" else 3)
    f = _frames(s, r)
    err = np.linalg.norm(f[4:] - tgt, axis=2)
    assert err.max() < 3e-3, err
    assert float(r.stats.row_max_violation[gi, 0]) < 3e-3
    assert np.linalg.norm(f[1] - p0) < 5e-3                                      # knot 1 outside the window: not pulled


def test_contact_rows_mask_selects_knots(make_solver):
    """The same 2 cm target on ALL knots >= 1 but MASKED to the second half of the
    horizon: enforced there, the first half stays with the tracking goal."""
    s, x, goals = _gentle(make_solver)
    N = s.N
    p0 = s.contact_positions(x[0, :s.nq])
    goals = np.zeros_like(goals)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = p0[0]
    tgt = p0 + np.array([[0.0, 0.0, 0.02]])
    s.enable_limit_telemetry()
    gi = s.add_contact_pos_rows(targets=tgt, mech="al", rho=ROW_RHO["al"], knot_lo=1)
    mask = np.zeros((N, s.n_contact_rows), bool)
    mask[N // 2:] = True
    s.set_row_group_mask(gi, mask)
    r = _outer(s, x, goals, 6)
    d = np.linalg.norm(_frames(s, r) - tgt, axis=2)[:, 0]
    assert d[N // 2:].max() < 3e-3, d
    assert d[1] > 1.5e-2, d                                                      # knot 1 masked off: stays (2 cm from tgt)


@_go2_fc
def test_go2_contact_rows_keep_the_feet_under_a_lateral_goal():
    """go2 fc: the four feet pinned at their current positions (AL) while the imu
    goal moves 4 cm to +y — the free solve drags the feet by millimetres, the
    rows hold them below 1 mm at every knot."""
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    mg = go2_mg(model)
    x = go2_standing_x().astype(np.float32)
    goals = go2_goals_at(model, x.astype(np.float64), 1)
    goals[:, 1::6] += 0.04

    def make():
        s = go2_solver(1, variant="fc", q_cost=5.0, N_cost=25.0)
        ref = np.zeros(GO2_FC, np.float32)
        ref[5::6] = mg / 4
        s.set_fc_ref(ref)
        s.enable_limit_telemetry()
        return s

    free = make()
    p0 = free.contact_positions(x[:free.nq].astype(np.float64))     # (4, 3)
    drift_free = np.linalg.norm(_frames(free, _outer(free, x[None], goals, 4))[1:] - p0, axis=2).max()
    s = make()
    gi = s.add_contact_pos_rows(targets=p0, mech="al", rho=ROW_RHO["al"], knot_lo=1)
    r = _outer(s, x[None], goals, 8)
    drift = np.linalg.norm(_frames(s, r)[1:] - p0, axis=2).max()
    assert drift < 1e-3, (drift, drift_free)
    assert drift_free > 3e-3, (drift, drift_free)
    assert float(r.stats.row_max_violation[gi, 0]) < 1e-3


@_go2_fc
def test_go2_gait_programmer_installs_foot_rows():
    """GaitProgrammer.install_foot_rows on a 'stand' schedule: apply(t, q) writes
    the stance targets (= the current foot positions), masks knot 0 off and every
    later knot on, the swing group stays fully masked; a solve keeps the feet."""
    import pinocchio as pin
    from gato.gait import GaitSchedule, GaitProgrammer
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    mg = go2_mg(model)
    x = go2_standing_x().astype(np.float32)
    goals = go2_goals_at(model, x.astype(np.float64), 1)
    s = go2_solver(1, variant="fc", q_cost=5.0, N_cost=25.0)
    s.enable_limit_telemetry()
    prog = GaitProgrammer(s, GaitSchedule(gait="stand", period=1.0, dt=s.dt, N=s.N), mg)
    g_st, g_sw = prog.install_foot_rows()
    with pytest.raises(ValueError):
        prog.apply(0.0)                                        # foot rows need q
    assert prog.apply(0.0, x[:s.nq]).all()
    groups = s.get_row_groups()
    p0 = s.contact_positions(x[:s.nq].astype(np.float64))
    np.testing.assert_allclose(groups[g_st]["tgt"], np.tile(p0.reshape(1, -1), (s.N, 1)), atol=1e-6)
    act = groups[g_st]["active"]
    assert int(act[0]) == 0 and all(int(w) == (1 << s.n_contact_rows) - 1 for w in act[1:])
    assert all(int(w) == 0 for w in groups[g_sw]["active"])
    r = _outer(s, x[None], goals, 6)
    assert np.linalg.norm(_frames(s, r)[1:] - p0, axis=2).max() < 1e-3


@_go2_fc
def test_go2_contact_rows_lift_a_foot_in_the_horizon():
    """Solver-only S3 (CL-4): from the stance pose with the S1 standing costs,
    FR swings from knot 4 — its wrench pinned to zero there, fn reference mg/3
    on the other three, stance rows hold FL/RR/RL at their footholds and a swing
    row group asks FR for +3 cm from knot 4 on. Measured 2026-09-26: AL rho 1e3
    lifts FR by ~3 cm at knot 12 with the stance feet held below 1 cm (rho 100:
    0.9 cm, ADMM 100: 1.3 cm)."""
    import pinocchio as pin
    import test_floating_worlds as fw
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    x = fw._stance_x(model).astype(np.float32)
    goals = fw._imu_goals(model, x)[None]
    mg = fw._mg(model)
    s = go2_solver(1, variant="fc", **fw._STAND_PARAMS)
    s.set_linsys("bdsv")
    s.set_q_nom(x[:s.nq])
    s.set_q_pos_cost(50.0)
    ref = np.zeros((s.N, GO2_FC), np.float32)
    ref[:, 5::6] = mg / 4
    ref[4:, 5] = 0.0
    ref[4:, 11::6] = mg / 3
    s.set_fc_ref(ref)
    s.set_fc_cost(1e-2)
    s.add_fc_box(0.0, 0.0, slots=[j for f in range(4) for j in s.fc_slots(f, "n")], mech="al")
    g_pin = s.add_fc_box(0.0, 0.0, slots=s.fc_slots(0), mech="al", rho=10.0)
    m = np.zeros((s.N, 6), bool); m[4:] = True
    s.set_row_group_mask(g_pin, m)
    p0 = s.contact_positions(x[:s.nq].astype(np.float64))
    g_st = s.add_contact_pos_rows(targets=p0, mech="al", rho=1000.0)
    m = np.ones((s.N, 12), bool); m[0] = False; m[4:, 0:3] = False
    s.set_row_group_mask(g_st, m)
    tgt = np.tile(p0.reshape(1, -1), (s.N, 1)); tgt[4:, 2] += 0.03
    g_sw = s.add_contact_pos_rows(targets=tgt, mech="al", rho=1000.0)
    m = np.zeros((s.N, 12), bool); m[4:, 0:3] = True
    s.set_row_group_mask(g_sw, m)
    r = _outer(s, x[None], goals, 8)
    f = _frames(s, r)
    assert f[12, 0, 2] - p0[0, 2] > 0.02, f[:, 0, 2] - p0[0, 2]                 # FR lifted at knot 12
    assert np.linalg.norm(f[1:, 1:] - p0[1:], axis=2).max() < 0.01              # stance feet held
    assert abs(_q_at(s, r, s.N - 1)[2] - x[2]) < 0.01                          # base height kept
    assert np.abs(r.fc_traj(0)[4:, 5]).max() < 1.0                              # FR wrench pinned in swing
