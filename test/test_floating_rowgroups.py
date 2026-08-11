"""CL-3 Wave A: constraint row-groups on the floating base (go2, N=16).

Selection rows are ACTUATED-ONLY with explicit stored-read / tangent-fold
slot maps (rowgroups.cuh stored_*_index / tangent_*_index). Gates:

- telemetry stays off the solver path (bit-parity with groups disabled);
- installed groups are 12-row actuated boxes whose bounds match the URDF
  actuated <limit> tags (the free-flyer base has no limit rows);
- telemetry matches a numpy oracle over the ACTUATED stored slots, and an
  infeasible actuated start produces nonzero BOX_Q violation (teeth);
- an extreme BASE pose contributes NOTHING to the box telemetry (the
  mis-mapping detector: the old tangent-offset read would count base slots);
- barrier / ADMM / AL reduce the violation, stay finite, deterministic;
- the EE terminal telemetry row equals |ee(q_N) - target| from a pinocchio
  FK oracle (stored-q read on the cooperative path);
- collision clearance rows: a far obstacle reports exactly zero violation,
  a bubble intersecting the trunk reports positive; deterministic.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pin = pytest.importorskip("pinocchio")
if importlib.util.find_spec("gato.bsqpN16_go2") is None:
    pytest.skip("bsqpN16_go2 module not built", allow_module_level=True)

import gato

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "external" / "GRiD" / "config" / "robot_assets" / "go2.urdf"

N = 16
DT = 0.01
NQ, NV, NU = 19, 18, 12
NX = NQ + NV
XU_STRIDE = NX + NU

JOINT_LIMIT_MARGIN = -0.1  # plant.cuh: limits TIGHTENED by |margin|
KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U = 0, 1, 2
BLOCK_X, BLOCK_U = 0, 1


@pytest.fixture(scope="module")
def model():
    return pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())


def _standing_q():
    q = np.zeros(NQ)
    q[2] = 0.35
    q[6] = 1.0  # quat w (xyzw)
    q[7:] = np.tile([0.0, 0.9, -1.8], 4)
    return q


def _standing_x(**base):
    x = np.zeros(NX)
    x[:NQ] = _standing_q()
    for k, v in base.items():
        x[{"px": 0, "py": 1, "pz": 2}[k]] = v
    return x


def _solver(B, **kw):
    # plant barrier costs ZERO: the row-group mechanisms are the only limit
    # terms, so violation deltas are attributable
    params = dict(q_cost=1.0, qd_cost=1e-2, u_cost=1e-4, N_cost=5.0,
                  q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)
    params.update(kw)
    return gato.BSQP(model_path=str(URDF), batch_size=B, N=N, dt=DT,
                     plant_type="go2", **params)


def _goals_at(model, x, B):
    """EE (imu) goal pinned at the CURRENT pose so the solve stays quiet."""
    data = model.createData()
    pin.framesForwardKinematics(model, data, x[:NQ])
    p = data.oMf[model.getFrameId("imu_joint")].translation
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = p[0], p[1], p[2]
    return goals


def _oracle_violations(xu, groups):
    """numpy {max, sum} true violation per BOX group from the STORED
    trajectory's ACTUATED slots (f32 like the kernel)."""
    out = []
    for grp in groups:
        lo = np.asarray(grp["lo"], dtype=np.float32)
        hi = np.asarray(grp["hi"], dtype=np.float32)
        viols = []
        for k in range(grp["knot_lo"], grp["knot_hi"]):
            base = k * XU_STRIDE
            if grp["kind"] == KIND_BOX_Q:
                g = xu[base + 7:base + 7 + NU]
            elif grp["kind"] == KIND_BOX_QD:
                g = xu[base + NQ + 6:base + NQ + 6 + NU]
            else:
                g = xu[base + NX:base + NX + NU]
            g = g.astype(np.float32)
            viols.append(np.maximum(0, g - hi) + np.maximum(0, lo - g))
        v = np.concatenate(viols)
        out.append((v.max(), v.sum(dtype=np.float64)))
    return out


def _infeasible_x():
    """all four calves pushed past the tightened lower joint limit."""
    x = _standing_x()
    for c in (9, 12, 15, 18):  # FL/FR/RL/RR calf stored slots
        x[c] = -2.75  # tightened lo = -2.7227 + 0.1 = -2.6227
    return x


def test_telemetry_off_the_solver_path(model):
    B = 2
    X = np.tile(_standing_x(), (B, 1)).astype(np.float32)
    goals = _goals_at(model, X[0], B)
    ref = _solver(B).solve(X, goals)
    s = _solver(B)
    s.enable_limit_telemetry()
    got = s.solve(X, goals)
    np.testing.assert_array_equal(ref.xu, got.xu)
    assert got.stats.row_max_violation.shape == (3, B)


def test_limit_groups_actuated(model):
    s = _solver(1)
    s.enable_limit_telemetry()
    groups = s.get_row_groups()
    assert [g["kind"] for g in groups] == [KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U]
    assert [g["block"] for g in groups] == [BLOCK_X, BLOCK_X, BLOCK_U]
    assert [g["n_rows"] for g in groups] == [NU, NU, NU]

    m = JOINT_LIMIT_MARGIN
    np.testing.assert_allclose(groups[0]["lo"], model.lowerPositionLimit[7:] - m, rtol=1e-6)
    np.testing.assert_allclose(groups[0]["hi"], model.upperPositionLimit[7:] + m, rtol=1e-6)
    np.testing.assert_allclose(groups[1]["lo"], -model.velocityLimit[6:] - m, rtol=1e-6)
    np.testing.assert_allclose(groups[1]["hi"], model.velocityLimit[6:] + m, rtol=1e-6)
    np.testing.assert_allclose(groups[2]["lo"], -model.effortLimit[6:] - m, rtol=1e-6)
    np.testing.assert_allclose(groups[2]["hi"], model.effortLimit[6:] + m, rtol=1e-6)
    assert groups[0]["knot_lo"] == 1 and groups[0]["knot_hi"] == N
    assert groups[2]["knot_lo"] == 0 and groups[2]["knot_hi"] == N - 1


def test_telemetry_matches_numpy_oracle(model):
    B = 4
    rng = np.random.default_rng(5)
    X = np.tile(_infeasible_x(), (B, 1)).astype(np.float32)
    X[:, 7:NQ] += rng.normal(0, 0.01, (B, NU)).astype(np.float32)
    goals = _goals_at(model, X[0], B)
    s = _solver(B)
    s.enable_limit_telemetry()
    res = s.solve(X, goals)
    groups = s.get_row_groups()
    assert np.isfinite(res.xu).all()

    for b in range(B):
        oracle = _oracle_violations(res.xu[b], groups)
        for g, (vmax, vsum) in enumerate(oracle):
            np.testing.assert_allclose(res.stats.row_max_violation[g, b], vmax,
                                       rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(res.stats.row_sum_violation[g, b], vsum,
                                       rtol=1e-5, atol=1e-6)
    # teeth: the calf start is outside the tightened limit
    assert res.stats.row_max_violation[0].min() > 0.05


def test_base_slots_never_counted(model):
    """Extreme base pose, actuated state well inside every limit: the box
    telemetry must be ZERO. The pre-mapping bug read the base slots (pz =
    5 m vs a 1.05 rad hip bound would report >= 3.9)."""
    B = 2
    x = _standing_x(px=3.0, pz=5.0)
    X = np.tile(x, (B, 1)).astype(np.float32)
    goals = _goals_at(model, x, B)
    s = _solver(B)
    s.enable_limit_telemetry()
    res = s.solve(X, goals)
    assert np.isfinite(res.xu).all()
    assert res.stats.row_max_violation[0].max() == 0.0
    assert res.stats.row_max_violation[1].max() == 0.0


def _mech_violation(model, enable, B=2):
    X = np.tile(_infeasible_x(), (B, 1)).astype(np.float32)
    goals = _goals_at(model, X[0], B)
    s = _solver(B)
    enable(s)
    r1 = s.solve(X, goals)
    assert np.isfinite(r1.xu).all()
    # base retraction stays sane under the mechanism (unit quaternion)
    qN = r1.xu[0, (N - 1) * XU_STRIDE:(N - 1) * XU_STRIDE + NQ]
    assert abs(np.linalg.norm(qN[3:7]) - 1.0) < 1e-4
    s2 = _solver(B)
    enable(s2)
    r2 = s2.solve(X, goals)
    np.testing.assert_array_equal(r1.xu, r2.xu)
    return r1.stats.row_max_violation[0].max()


def test_mechanisms_reduce_violation(model):
    ref = _mech_violation(model, lambda s: s.enable_limit_telemetry())
    assert ref > 0.05
    bar = _mech_violation(model, lambda s: s.enable_limit_barrier(mu=1e-1, delta=0.1))
    adm = _mech_violation(model, lambda s: s.enable_limit_admm(rho=10.0, iters=10))
    al = _mech_violation(model, lambda s: s.enable_limit_al(rho=100.0))
    assert bar < ref
    assert adm < ref
    assert al < ref


@pytest.mark.xfail(strict=True, reason=(
    "GRiD BUG 7 (asks_from_gato_2026-08-09.md): the generated EE pose inner's "
    "ping-pong parity drops the base transform on chains that reach the root "
    "early — go2 imu returns the BASE-RELATIVE offset (and J == 0). strict: "
    "this flips to a hard error when the regen lands, re-activating the gate."))
def test_ee_terminal_telemetry_matches_pin_fk(model):
    B = 2
    x = _standing_x()
    X = np.tile(x, (B, 1)).astype(np.float32)
    goals = _goals_at(model, x, B)
    target = np.asarray(goals[0, :3], dtype=np.float64) + np.array([0.1, 0.0, 0.05])
    s = _solver(B)
    s.enable_limit_telemetry()
    s.solver.enable_ee_terminal_equality(target.astype(np.float32), 1.0)
    res = s.solve(X, goals)
    data = model.createData()
    fid = model.getFrameId("imu_joint")
    for b in range(B):
        qN = res.xu[b, (N - 1) * XU_STRIDE:(N - 1) * XU_STRIDE + NQ].astype(np.float64)
        pin.framesForwardKinematics(model, data, qN)
        diff = np.abs(data.oMf[fid].translation - target)
        np.testing.assert_allclose(res.stats.row_max_violation[3, b], diff.max(),
                                   rtol=1e-4, atol=2e-4)


def test_collision_telemetry(model):
    B = 2
    x = _standing_x()
    X = np.tile(x, (B, 1)).astype(np.float32)
    goals = _goals_at(model, x, B)

    def run(spheres):
        s = _solver(B)
        s.enable_limit_telemetry()
        s.set_collision_environment(spheres=spheres)
        s.enable_collision(mech="telemetry", margin=0.02)
        res = s.solve(X, goals)
        assert np.isfinite(res.xu).all()
        return res.stats.row_max_violation[3]

    far = run([(100.0, 0.0, 0.0, 0.1)])
    assert far.max() == 0.0
    near = run([(0.0, 0.0, 0.35, 0.4)])  # bubble through the trunk
    assert near.min() > 0.1
    near2 = run([(0.0, 0.0, 0.35, 0.4)])
    np.testing.assert_array_equal(near, near2)
