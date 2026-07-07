"""ArmTrackEnv: a gymnasium environment for torque-controlled arm EE tracking.

The env owns everything a controller should not: pinocchio RK4 integration,
the ground-truth (unmodeled) external wrench, time, and the reward/termination
logic. Pair it with gato.MPCPolicy for batch-robust MPC-as-policy, or drive it
with any other torque policy.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for gato.envs; install it with: pip install -e '.[examples]'"
    ) from e

from .common import _require_pin, rk4, world_wrench_to_joint_local


class ArmTrackEnv(gym.Env):
    """Torque-controlled arm with RK4 substeps and an optional unmodeled EE wrench.

    Observation: the state [q, dq] (Box, (nq+nv,)).
    Action: joint torques (Box bounded by the URDF effort limits), applied
        zero-order-hold for `ctrl_dt` (int(ctrl_dt/sim_dt) RK4 substeps).
    Reward: -||ee_pos(q) - reference.window(t)[0:3]|| (0 if no reference).
    terminated: non-finite state. truncated: t >= horizon_s or reference.done(t).
    info: {"t", "ee_pos", "ref0", "tracking_err"}.
    """

    metadata = {"render_modes": []}

    def __init__(self, urdf_path, *, ctrl_dt=0.01, sim_dt=1e-3,
                 f_ext_world=None, reference=None, x0=None,
                 horizon_s=5.0, ee_frame="EE", render_mode=None):
        pin = _require_pin()
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        self.nq, self.nv = self.model.nq, self.model.nv

        self.ctrl_dt = float(ctrl_dt)
        self.sim_dt = float(sim_dt)
        self.substeps = max(1, int(round(self.ctrl_dt / self.sim_dt)))
        self.horizon_s = float(horizon_s)
        self.reference = reference
        self.render_mode = render_mode

        # ground-truth disturbance: constant world-frame EE wrench, unmodeled by
        # the policy unless it hypothesizes it (the fig-5 mechanism). Re-expressed
        # into the EE parent joint's LOCAL frame every step (pin.aba fext is joint-
        # local; applying world coords directly makes the force rotate with the wrist).
        self._f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self._f_ext.append(pin.Force.Zero())
        self._f_ext_world = (np.asarray(f_ext_world, dtype=float).reshape(6)
                             if f_ext_world is not None else None)

        # EE metric frame (the same frame the solver optimizes)
        self._ee_frame_id = (self.model.getFrameId(ee_frame)
                             if self.model.existFrame(ee_frame) else None)
        if self._f_ext_world is not None and self._ee_frame_id is None:
            raise ValueError(f"f_ext_world needs EE frame {ee_frame!r} in the URDF")

        self._x0 = (np.asarray(x0, dtype=np.float64).reshape(self.nq + self.nv)
                    if x0 is not None else np.zeros(self.nq + self.nv))

        effort = np.asarray(self.model.effortLimit, dtype=np.float32)
        effort = np.where(np.isfinite(effort) & (effort > 0), effort, 1e3).astype(np.float32)
        self.action_space = spaces.Box(-effort, effort, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.nq + self.nv,), dtype=np.float32)

        self.q = self._x0[:self.nq].copy()
        self.dq = self._x0[self.nq:].copy()
        self.t = 0.0

    def ee_pos(self, q):
        pin = _require_pin()
        pin.forwardKinematics(self.model, self.data, q)
        if self._ee_frame_id is not None:
            pin.updateFramePlacement(self.model, self.data, self._ee_frame_id)
            return np.asarray(self.data.oMf[self._ee_frame_id].translation)
        return np.asarray(self.data.oMi[self.model.njoints - 1].translation)

    def _obs(self):
        return np.concatenate([self.q, self.dq]).astype(np.float32)

    def _info(self):
        ee = self.ee_pos(self.q)
        ref0 = (np.asarray(self.reference.window(self.t)[:3])
                if self.reference is not None else None)
        err = float(np.linalg.norm(ee - ref0)) if ref0 is not None else 0.0
        return {"t": self.t, "ee_pos": ee, "ref0": ref0, "tracking_err": err}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.q = self._x0[:self.nq].copy()
        self.dq = self._x0[self.nq:].copy()
        self.t = 0.0
        return self._obs(), self._info()

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64).reshape(self.nv),
                    self.action_space.low, self.action_space.high)
        for _ in range(self.substeps):
            if self._f_ext_world is not None:
                jid, Fj = world_wrench_to_joint_local(
                    self.model, self.data, self.q, self._f_ext_world, self._ee_frame_id)
                self._f_ext[jid] = Fj
            self.q, self.dq = rk4(self.model, self.data, self.q, self.dq, u,
                                  self.sim_dt, self._f_ext)
        self.t += self.substeps * self.sim_dt

        info = self._info()
        reward = -info["tracking_err"]
        terminated = not (np.isfinite(self.q).all() and np.isfinite(self.dq).all())
        truncated = self.t >= self.horizon_s or (
            self.reference is not None and self.reference.done(self.t))
        return self._obs(), reward, terminated, truncated, info
