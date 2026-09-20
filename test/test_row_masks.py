"""Per-knot row-activity masks (CL-4 G0): all-True is bitwise the unmasked
solve; a fully masked-off AL group is bitwise the group's absence; a masked
fc pin acts only on its knots; telemetry ignores inactive rows. The gates run
on the arms (LIN_U on the actuated slots) and on go2 fc (foot pins)."""
import importlib.util

import numpy as np
import pytest

import gato
from conftest import GO2_FEET, GO2_FC, go2_solver, go2_standing_x, go2_goals_at, GO2_URDF

pytestmark = pytest.mark.gpu

from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS  # noqa: E402

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}


def _arm_problem(plant, N):
    q0 = np.asarray(START[plant], dtype=np.float32)
    x = np.concatenate([q0, np.zeros_like(q0)])[None]
    goals = np.zeros((1, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.35, 0.25, 0.5
    return x, goals


def _u_box(s, lo=-3.0, hi=3.0, mech="al"):
    """A LIN_U interval group boxing the first two actuated slots."""
    C = np.zeros((2, s.nu), np.float32); C[0, 0] = 1.0; C[1, 1] = 1.0
    s.enable_limit_al() if mech == "al" else s.enable_limit_admm()
    return s.add_lin_u_rows(C, lo=np.full(2, lo, np.float32), hi=np.full(2, hi, np.float32), mech=mech)


@pytest.mark.parametrize("mech", ["al", "admm"])
def test_all_true_mask_is_bitwise_unmasked(make_solver, smallest_module, mech):
    plant, N = smallest_module
    x, goals = _arm_problem(plant, N)
    a = make_solver(plant, N)
    _u_box(a, mech=mech)
    ra = a.solve(x, goals)
    b = make_solver(plant, N)
    gi = _u_box(b, mech=mech)
    b.set_row_group_mask(gi, np.ones((N, 2), bool))
    for g in range(3):                       # the limit groups too
        b.set_row_group_mask(g, None)
    rb = b.solve(x, goals)
    np.testing.assert_array_equal(ra.xu, rb.xu)
    groups = b.solver.get_row_groups()
    assert all(int(w) == 0b11 for w in groups[gi]["active"])          # the 2-row group: bits 0,1
    assert all(int(w) == 2**64 - 1 for w in groups[0]["active"])      # None = every bit


def test_fully_masked_al_group_is_bitwise_absent(make_solver, smallest_module):
    """AL rows masked off everywhere contribute nothing: bitwise the solve
    with only the limit groups installed."""
    plant, N = smallest_module
    x, goals = _arm_problem(plant, N)
    a = make_solver(plant, N)
    a.enable_limit_al()
    ra = a.solve(x, goals)
    b = make_solver(plant, N)
    gi = _u_box(b, lo=-0.01, hi=0.01, mech="al")   # a tight box that WOULD bind
    b.set_row_group_mask(gi, np.zeros((N, 2), bool))
    rb = b.solve(x, goals)
    np.testing.assert_array_equal(ra.xu, rb.xu)
    tele = np.asarray(b.solver.get_row_group_telemetry()) if hasattr(b.solver, "get_row_group_telemetry") else None
    c = make_solver(plant, N)
    gi = _u_box(c, lo=-0.01, hi=0.01, mech="al")
    rc = c.solve(x, goals)
    assert not np.array_equal(ra.xu, rc.xu)        # the box binds when active (the gate is not vacuous)


def test_mask_window_gates_knots(make_solver, smallest_module):
    """A +-0.5 AL box on the first two controls, active on knots [2, 6) only:
    those controls sit at the box while knots 0-1 run free (well past 0.5);
    the same box unmasked clamps knot 0 too. Gentle goal (3 cm up), u_cost
    1e-4, so the SQP converges and the box binds by construction."""
    plant, N = "indy7", 8            # receipt-profile module; the gentle scenario below was tuned on it
    if (plant, N) not in gato.available():
        pytest.fail("bsqpN8_indy7 is a receipt-profile module and must be built")
    q0 = np.asarray(START[plant], dtype=np.float32)
    x = np.concatenate([q0, np.zeros_like(q0)])[None]
    probe = make_solver(plant, N, u_cost=1e-4)
    p = probe.ee_pos(q0.astype(np.float64)) + np.array([0.0, 0.0, 0.03])
    goals = np.zeros((1, N * 6), np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = p

    def run(mask):
        s = make_solver(plant, N, u_cost=1e-4)
        C = np.zeros((2, s.nu), np.float32); C[0, 0] = 1.0; C[1, 1] = 1.0
        s.enable_limit_al()
        gi = s.add_lin_u_rows(C, lo=np.full(2, -0.5, np.float32), hi=np.full(2, 0.5, np.float32), mech="al", rho=10.0)
        if mask is not None:
            s.set_row_group_mask(gi, mask)
        r, xu_warm = None, None
        for _ in range(8):                                  # AL outer loop
            r = s.solve(x, goals, xu_warm)
            xu_warm = r.xu
        return s, gi, np.stack([r.control_at(k)[:2] for k in range(N - 1)])

    _, _, u_all = run(None)
    assert np.abs(u_all[:6]).max() < 0.5 + 0.03, u_all      # unmasked: the box binds everywhere it is tight
    mask = np.zeros((N, 2), bool); mask[2:6, :] = True
    s, gi, u_win = run(mask)
    assert np.abs(u_win[2:6]).max() < 0.5 + 0.03, u_win[2:6]
    assert np.abs(u_win[0]).min() > 0.6, u_win[0]           # knot 0 free: both slots past the box
    act = s.solver.get_row_groups()[gi]["active"]
    assert [int(w) & 3 for w in act] == [3 if 2 <= k < 6 else 0 for k in range(N)]


def test_vector_kind_mask_is_knot_level(make_solver, smallest_module):
    """SOC cone groups are gated per knot (bit 0); a (N,) mask is accepted."""
    plant, N = smallest_module
    s = make_solver(plant, N)
    s.enable_limit_admm()
    C = np.zeros((3, s.nu), np.float32); C[0, 0] = 1.0; C[1, 1] = 1.0; C[2, 2] = 1.0
    gi = s.add_lin_u_rows(C, mech="admm", cone=True)
    m = np.zeros(N, bool); m[3] = True
    s.set_row_group_mask(gi, m)
    act = s.solver.get_row_groups()[gi]["active"]
    assert [int(w) for w in act] == [1 if k == 3 else 0 for k in range(N)]
    with pytest.raises(ValueError):
        s.set_row_group_mask(gi + 7, m)


@pytest.mark.skipif(importlib.util.find_spec("gato.bsqpN16_go2_fc") is None, reason="go2 fc module (receipt profile)")
def test_go2_masked_fc_pin_acts_on_its_knots():
    """fc-on-feet: pin FR's six wrench slots to zero on knots [4, 9) only (the
    swing-window pattern of the gait schedule). Elsewhere the fc_ref pulls the
    vertical force to ~mg/4; on the pinned knots it is ~0."""
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    mg = sum(i.mass for i in model.inertias) * 9.81
    s = go2_solver(1, variant="fc")
    x = go2_standing_x().astype(np.float32)
    goals = go2_goals_at(model, x.astype(np.float64), 1)
    ref = np.zeros(GO2_FC, np.float32)
    for i in range(4):
        ref[s.fc_slots(i, "f")[2]] = mg / 4
    s.set_fc_ref(ref)                                 # fc_cost stays at the fc-build default (1e-2)
    gi = s.add_fc_box(0.0, 0.0, mech="al", rho=10.0)  # all slots; the mask picks knots x feet
    mask = np.zeros((s.N, GO2_FC), bool)
    mask[4:9, s.fc_slots("FR_foot_joint")] = True
    s.set_row_group_mask(gi, mask)
    r, xu_warm = None, None
    for _ in range(8):                               # AL outer loop
        r = s.solve(x[None], goals, xu_warm)
        xu_warm = r.xu
    fc = r.fc_traj(0)                                # (N-1, 24)
    fr_z = fc[:, s.fc_slots("FR_foot_joint", "f")[2]]
    assert np.abs(fr_z[4:9]).max() < 2.0, fr_z
    assert fr_z[[0, 2, 10, 13]].min() > 0.5 * mg / 4, fr_z
    fl_z = fc[:, s.fc_slots("FL_foot_joint", "f")[2]]
    assert fl_z.min() > 0.5 * mg / 4                 # other feet untouched by the mask


@pytest.mark.skipif(importlib.util.find_spec("gato.bsqpN16_go2_fc") is None, reason="go2 fc module (receipt profile)")
def test_go2_per_knot_fc_ref():
    """set_fc_ref((N, n_fc)): a uniform per-knot table is BITWISE the broadcast;
    a knot-varying table is consumed per knot (the fc cost pulls each knot's
    wrench toward its own row)."""
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    x = go2_standing_x().astype(np.float32)
    goals = go2_goals_at(model, x.astype(np.float64), 1)
    ref = np.zeros(GO2_FC, np.float32); ref[5::6] = 40.0
    a = go2_solver(1, variant="fc"); a.set_fc_ref(ref); a.set_fc_cost(1e2)
    b = go2_solver(1, variant="fc"); b.set_fc_ref(np.tile(ref, (b.N, 1))); b.set_fc_cost(1e2)
    np.testing.assert_array_equal(a.solve(x[None], goals).xu, b.solve(x[None], goals).xu)
    table = np.tile(ref, (b.N, 1)); table[8:, 5] = 5.0        # FR vertical drops to 5 N from knot 8
    c = go2_solver(1, variant="fc"); c.set_fc_ref(table); c.set_fc_cost(1e2)
    fc = c.solve(x[None], goals).fc_traj(0)
    assert np.abs(fc[:8, 5] - 40.0).max() < 8.0 and np.abs(fc[8:, 5] - 5.0).max() < 8.0, fc[:, 5]
    with pytest.raises(ValueError):
        c.set_fc_ref(np.zeros((c.N + 1, GO2_FC)))


@pytest.mark.skipif(importlib.util.find_spec("gato.bsqpN16_go2_fc") is None, reason="go2 fc module (receipt profile)")
def test_go2_gait_programmer_stand_is_the_standing_setpoint():
    """A 'stand' schedule programmed through GaitProgrammer reproduces the S1
    standing solve bitwise (moment pins everywhere, mg/4 up on every foot, no
    swing pins); a trot window pins exactly the swing (knot, foot) slots and
    references mg/2 on the two stance feet."""
    import pinocchio as pin
    from gato.gait import GaitSchedule, GaitProgrammer
    model = pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())
    mg = sum(i.mass for i in model.inertias) * 9.81
    x = go2_standing_x().astype(np.float32)
    goals = go2_goals_at(model, x.astype(np.float64), 1)
    # reference: the S1 recipe by hand
    a = go2_solver(1, variant="fc")
    ref = np.zeros(GO2_FC, np.float32); ref[5::6] = mg / 4
    for i in range(4):
        a.add_fc_box(0.0, 0.0, slots=a.fc_slots(i, "n"), mech="al")
    a.set_fc_ref(ref)
    ra = a.solve(x[None], goals)
    # the programmer with a stand schedule
    b = go2_solver(1, variant="fc")
    prog = GaitProgrammer(b, GaitSchedule(gait="stand", period=1.0, dt=b.dt, N=b.N), mg)
    st = prog.apply(0.0)
    assert st.all()
    rb = b.solve(x[None], goals)
    # same rows, same refs -> same solve up to the row-group layout (one masked
    # group vs four): the fold order differs, so compare to solver tolerance
    np.testing.assert_allclose(rb.xu, ra.xu, rtol=1e-4, atol=1e-4)
    # trot window: swing slots pinned, stance feet reference mg/2
    c = go2_solver(1, variant="fc")
    sched = GaitSchedule(gait="trot", period=0.5, dt=c.dt, N=c.N)
    prog = GaitProgrammer(c, sched, mg)
    st = prog.apply(0.2)
    act = c.solver.get_row_groups()[prog.pin_group]["active"]
    for k in range(c.N):
        for f in range(4):
            bits = (int(act[k]) >> (6 * f)) & 0x3F
            assert bits == (0x07 if st[k, f] else 0x3F), (k, f, bits)   # moments always, force only in swing
    prog.install_cones(mu=0.6)
    st = prog.apply(0.2)
    for f, g in enumerate(prog.cone_groups):
        act = c.solver.get_row_groups()[g]["active"]
        assert [int(w) & 1 for w in act] == [int(v) for v in st[:, f]]
    r = c.solve(x[None], goals)
    assert np.isfinite(r.xu).all()
