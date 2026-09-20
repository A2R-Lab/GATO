"""GaitSchedule oracle gates (CL-4 G0, CPU-only): the stance tables of the
named gaits, duty factors, the rolling window's continuity, the fc pin / fn
reference windows, and the Raibert-lite geometry against numpy references."""
import numpy as np
import pytest

from gato.gait import GAITS, GaitSchedule, base_reference


@pytest.mark.parametrize("gait", sorted(GAITS))
def test_duty_factor_and_period(gait):
    g = GaitSchedule(gait=gait, period=0.5, dt=0.005, N=100)
    ts = np.arange(0, 0.5, 0.0005)
    st = g.stance(ts)
    beta = GAITS[gait]["beta"]
    np.testing.assert_allclose(st.mean(axis=0), beta, atol=0.01)     # each foot stands beta of the period
    assert np.array_equal(g.stance(ts), g.stance(ts + 0.5))          # periodic
    if gait != "stand":
        assert np.all(st.sum(axis=1) >= 1)                           # never airborne for these gaits


def test_trot_pairs_and_bound_pairs():
    trot = GaitSchedule(gait="trot", period=0.4, dt=0.01, N=16)
    ts = np.linspace(0, 0.4, 41)
    st = trot.stance(ts)                                          # FR, FL, RR, RL
    assert np.array_equal(st[:, 0], st[:, 3]) and np.array_equal(st[:, 1], st[:, 2])   # diagonal pairs
    assert np.array_equal(st[:, 0], ~st[:, 1])                    # the pairs alternate
    bound = GaitSchedule(gait="bound", period=0.4, dt=0.01, N=16)
    sb = bound.stance(ts)
    assert np.array_equal(sb[:, 0], sb[:, 1]) and np.array_equal(sb[:, 2], sb[:, 3])   # front / rear pairs


def test_window_is_a_rolling_slice():
    g = GaitSchedule(gait="trot", period=0.5, dt=0.01, N=16)
    w0 = g.window(0.123)
    w1 = g.window(0.123 + g.dt)
    assert w0.shape == (16, 4) and w0.dtype == bool
    np.testing.assert_array_equal(w0[1:], w1[:-1])                # shift by one knot
    assert g.window(0.0).sum() < 64                               # a 160 ms window sees a phase change at trot


def test_swing_phase_progress():
    g = GaitSchedule(gait="trot", period=0.5, dt=0.01, N=16)
    # foot 0: stance for [0, 0.25), swing for [0.25, 0.5)
    assert g.swing_phase(0.1)[0] == 0.0
    s = g.swing_phase(np.array([0.25, 0.375, 0.4999]))[:, 0]
    assert s[0] == pytest.approx(0.0, abs=1e-9) and s[1] == pytest.approx(0.5, abs=1e-6) and s[2] < 1.0
    assert g.stance_duration(0) == pytest.approx(0.25) and g.swing_duration(0) == pytest.approx(0.25)


def test_fc_pin_mask_and_fn_ref_windows():
    g = GaitSchedule(gait="trot", period=0.5, dt=0.01, N=16)
    mg = 100.0
    st = g.window(0.0)
    pins = g.fc_pin_mask(0.0)
    ref = g.fn_ref_window(0.0, mg)
    assert pins.shape == (16, 24) and ref.shape == (16, 24)
    for k in range(16):
        for f in range(4):
            assert np.all(pins[k, 6 * f:6 * f + 6] == (not st[k, f]))
            assert ref[k, 6 * f + 5] == pytest.approx(mg / st[k].sum() if st[k, f] else 0.0)
            assert np.all(ref[k, 6 * f:6 * f + 5] == 0.0)
    np.testing.assert_allclose(ref[:, 5::6].sum(axis=1), mg)     # every knot carries the weight
    stand = GaitSchedule(gait="stand", period=1.0, dt=0.01, N=16)
    assert not stand.fc_pin_mask(0.3).any()
    np.testing.assert_allclose(stand.fn_ref_window(0.3, mg)[:, 5::6], mg / 4)


def test_swing_curve_and_foothold():
    g = GaitSchedule(gait="trot", period=0.5, dt=0.01, N=16, swing_height=0.06)
    p0, p1 = np.array([0.0, 0.0, 0.02]), np.array([0.1, 0.0, 0.02])
    np.testing.assert_allclose(g.swing_curve(p0, p1, 0.0), p0)
    np.testing.assert_allclose(g.swing_curve(p0, p1, 1.0), p1, atol=1e-12)
    apex = g.swing_curve(p0, p1, 0.5)
    assert apex[2] == pytest.approx(0.08) and apex[0] == pytest.approx(0.05)
    fh = g.foothold(hip_xy=[0.19, -0.14], v_base_xy=[0.3, 0.0], v_cmd_xy=[0.3, 0.0], f=0)
    np.testing.assert_allclose(fh, [0.19 + 0.5 * 0.25 * 0.3, -0.14])
    fh2 = g.foothold([0.0, 0.0], [0.0, 0.0], [0.4, 0.0], f=0, k_gain=0.1)
    np.testing.assert_allclose(fh2, [0.5 * 0.25 * 0.4 - 0.04, 0.0])


def test_base_reference_rotates_command():
    ref = base_reference([1.0, 2.0], np.pi / 2, [0.3, 0.0], 0.0, 0.01, 4)
    np.testing.assert_allclose(ref[0], [1.0, 2.0])
    np.testing.assert_allclose(ref[3], [1.0, 2.0 + 3 * 0.01 * 0.3], atol=1e-12)


def test_validation():
    with pytest.raises(ValueError):
        GaitSchedule(gait=None, period=0.5)
    with pytest.raises(ValueError):
        GaitSchedule(gait="trot", beta=1.5)
    with pytest.raises(KeyError):
        GaitSchedule(gait="gallop")
