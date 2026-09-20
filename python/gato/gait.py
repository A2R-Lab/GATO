"""Gait ORACLE for the fixed-gait locomotion arc (CL-4, 2026-09-20).

The solver never discovers a gait: a ``GaitSchedule`` is data the controller
feeds it every tick — a rolling per-knot STANCE MASK for the contact frames,
the swing phase of each swinging foot, and (Raibert-lite) foothold and base
references. Pure numpy, no solver dependency; the solver-side consumers are
``BSQP.set_row_group_mask`` (stance/swing row activity), the per-knot
``set_fc_ref`` and the per-knot EE goals.

Conventions
-----------
- feet are indexed in the plant's ``contact_frames`` order (go2: FR, FL, RR, RL);
- a gait is (period T_g, per-foot duty factor beta_f = stance fraction, per-foot
  phase offset phi_f in [0, 1)): foot f is in STANCE at absolute time t iff
  ((t / T_g + phi_f) mod 1) < beta_f;
- ``window(t)`` samples knots k = 0..N-1 at t + k*dt (the knot spacing of the
  horizon), so a 160 ms horizon sees the phase changes inside it.
"""
from dataclasses import dataclass, field

import numpy as np

# standard quadruped gaits (phase offsets in contact_frames order FR, FL, RR, RL)
GAITS = {
    "stand": dict(beta=1.0, phase=(0.0, 0.0, 0.0, 0.0)),
    "trot":  dict(beta=0.5, phase=(0.0, 0.5, 0.5, 0.0)),    # diagonal pairs (FR+RL, FL+RR)
    "bound": dict(beta=0.5, phase=(0.0, 0.0, 0.5, 0.5)),    # front pair, rear pair
    "pace":  dict(beta=0.5, phase=(0.0, 0.5, 0.0, 0.5)),    # lateral pairs
    "walk":  dict(beta=0.75, phase=(0.0, 0.5, 0.75, 0.25)), # lateral-sequence walk
}


@dataclass
class GaitSchedule:
    """Rolling stance/swing schedule for ``n_feet`` contact frames.

    Args:
        gait: name in GAITS, or None to pass ``beta``/``phase`` explicitly.
        period: gait period T_g [s].
        dt: knot spacing of the horizon [s]; N: horizon knots.
        beta: stance fraction per foot (scalar broadcasts).
        phase: per-foot phase offsets in [0, 1).
        swing_height: apex height of the swing curve [m] (foothold helper).
    """
    gait: str | None = "trot"
    period: float = 0.5
    dt: float = 0.01
    N: int = 16
    n_feet: int = 4
    beta: float | tuple = None
    phase: tuple = None
    swing_height: float = 0.06
    _beta: np.ndarray = field(init=False, repr=False)
    _phase: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.gait is not None:
            spec = GAITS[self.gait]
            beta = spec["beta"] if self.beta is None else self.beta
            phase = spec["phase"] if self.phase is None else self.phase
        else:
            if self.beta is None or self.phase is None:
                raise ValueError("gait=None needs explicit beta and phase")
            beta, phase = self.beta, self.phase
        self._beta = np.broadcast_to(np.asarray(beta, dtype=float), (self.n_feet,)).copy()
        self._phase = np.asarray(phase, dtype=float).reshape(self.n_feet) % 1.0
        if not (self.period > 0 and self.dt > 0 and self.N >= 1):
            raise ValueError("period, dt must be > 0 and N >= 1")
        if np.any(self._beta <= 0) or np.any(self._beta > 1):
            raise ValueError("duty factors must be in (0, 1]")

    # ---- phase geometry -------------------------------------------------
    def local_phase(self, t):
        """Per-foot gait phase in [0, 1) at absolute time(s) t (shape (..., n_feet))."""
        t = np.asarray(t, dtype=float)[..., None]
        return (t / self.period + self._phase) % 1.0

    def stance(self, t):
        """Per-foot stance flags at absolute time(s) t (bool, shape (..., n_feet))."""
        return self.local_phase(t) < self._beta

    def swing_phase(self, t):
        """Progress through the swing arc in [0, 1) per foot (0 on stance feet)."""
        p = self.local_phase(t)
        s = (p - self._beta) / (1.0 - self._beta + 1e-12)
        return np.where(p < self._beta, 0.0, s)

    def stance_duration(self, f=0):
        return float(self._beta[f] * self.period)

    def swing_duration(self, f=0):
        return float((1.0 - self._beta[f]) * self.period)

    # ---- the rolling horizon window -----------------------------------------
    def window(self, t):
        """(N, n_feet) stance mask over the horizon knots t, t+dt, ..., t+(N-1)dt."""
        ts = t + self.dt * np.arange(self.N)
        return self.stance(ts)

    def swing_window(self, t):
        """(N, n_feet) swing progress per knot (0 on stance knots)."""
        ts = t + self.dt * np.arange(self.N)
        return self.swing_phase(ts)

    def fc_pin_mask(self, t):
        """(N, 6*n_feet) row mask for an all-slot ``add_fc_box(0, 0)`` group:
        True = pinned (foot in SWING at that knot). Feed to set_row_group_mask."""
        st = self.window(t)                     # (N, n_feet) stance
        return np.repeat(~st, 6, axis=1)

    def fn_ref_window(self, t, mg):
        """(N, 6*n_feet) fc reference: mg / n_stance up (world z) on stance feet,
        0 on swing feet, moments 0 — the standing setpoint generalised."""
        st = self.window(t)
        n_st = np.maximum(st.sum(axis=1, keepdims=True), 1)
        ref = np.zeros((self.N, 6 * self.n_feet))
        ref[:, 5::6] = np.where(st, mg / n_st, 0.0)
        return ref

    # ---- Raibert-lite references -----------------------------------------------
    def swing_curve(self, p_lift, p_land, s):
        """Point on the swing arc at progress s in [0, 1]: linear in the plane,
        a raised cosine in z peaking at swing_height above the higher end."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)[..., None]
        p = (1.0 - s) * p_lift + s * p_land
        z_base = (1.0 - s[..., 0]) * p_lift[..., 2] + s[..., 0] * p_land[..., 2]
        p[..., 2] = z_base + self.swing_height * np.sin(np.pi * s[..., 0])
        return p

    def foothold(self, hip_xy, v_base_xy, v_cmd_xy, f, k_gain=0.1):
        """Raibert-lite foothold for foot f: hip projection + half the stance
        time at the commanded velocity + a velocity-error term."""
        t_st = self.stance_duration(f)
        hip_xy = np.asarray(hip_xy, dtype=float)
        v_base_xy = np.asarray(v_base_xy, dtype=float)
        v_cmd_xy = np.asarray(v_cmd_xy, dtype=float)
        return hip_xy + 0.5 * t_st * v_cmd_xy + k_gain * (v_base_xy - v_cmd_xy)


def base_reference(p0_xy, yaw, v_cmd_body_xy, t, dt, N):
    """(N, 2) planar base reference: p0 integrated at the commanded body-frame
    velocity (rotated by yaw) over the horizon knots starting at time t (the
    caller passes the CURRENT base position as p0 each tick)."""
    c, s = np.cos(yaw), np.sin(yaw)
    v_world = np.array([c * v_cmd_body_xy[0] - s * v_cmd_body_xy[1],
                        s * v_cmd_body_xy[0] + c * v_cmd_body_xy[1]])
    ks = np.arange(N)[:, None]
    return np.asarray(p0_xy, dtype=float)[None, :] + ks * dt * v_world[None, :]


class GaitProgrammer:
    """Writes a GaitSchedule into an fc-build BSQP every tick: the swing-foot fc
    pins (one all-slot ``add_fc_box(0, 0)`` group, masked per knot), the per-knot
    fn reference (mg / n_stance up on stance feet), and, when cone groups were
    installed through ``install_cones``, the stance-only cone masks. The
    stance-foot POSITION rows (CL-4 §1.3) attach here once the contact-frame
    position surface lands. Everything is data on the solver; no solve is
    issued. ``pin_mech``/``cone_mech`` default to AL / ADMM-SOC.
    """

    def __init__(self, solver, schedule, mg, pin_mech="al", pin_moments=True):
        if solver.n_fc != 6 * schedule.n_feet:
            raise ValueError(f"solver has {solver.n_fc} fc slots, schedule has {schedule.n_feet} feet")
        if solver.N != schedule.N:
            raise ValueError(f"solver N={solver.N} != schedule N={schedule.N}")
        self.solver, self.schedule, self.mg = solver, schedule, float(mg)
        self.pin_moments = pin_moments
        # all-slot pin group; the per-tick mask selects (knot, foot) swing slots
        self.pin_group = solver.add_fc_box(0.0, 0.0, mech=pin_mech)
        self.cone_groups = []
        self.t = None

    def install_cones(self, mu, mech="al", rho=10.0, **kw):
        """One SOC friction cone per foot on the fc force slots (stance-masked per
        tick). Default = AL conic PHR at rho 10: on the go2 standing loop it holds
        the stance (0.286 m, 3.8 ms/solve); the relaxed barrier also holds (8.5 ms);
        per-foot ADMM SOC cones SAG the stance (0.08-0.22 m) at 16-57 ms/solve —
        an open item in the CL-4 plan, do not default to them."""
        self.cone_groups = [self.solver.add_fc_cone(f, mu, mech=mech, rho=rho, **kw)
                            for f in range(self.schedule.n_feet)]
        return self.cone_groups

    def apply(self, t):
        """Program the horizon starting at time t. Returns the (N, n_feet) stance mask."""
        sched, s = self.schedule, self.solver
        stance = sched.window(t)                              # (N, n_feet)
        pins = sched.fc_pin_mask(t)                           # (N, 6 n_feet): swing slots pinned
        if self.pin_moments:                                  # point contacts: moments pinned always
            for f in range(sched.n_feet):
                pins[:, 6 * f:6 * f + 3] = True
        s.set_row_group_mask(self.pin_group, pins)
        s.set_fc_ref(sched.fn_ref_window(t, self.mg).astype(np.float32))
        for f, g in enumerate(self.cone_groups):
            s.set_row_group_mask(g, stance[:, f])
        self.t = t
        return stance
