"""MPCPolicy: an obs -> action adapter usable in any gymnasium env (or plain loop).

Pure numpy — no gymnasium import here. The policy holds its own clock and slides
a reference window over it; the controller does the solving.
"""
import numpy as np


class TrajectoryReference:
    """Sliding (6N,) window over a flat [x, y, z, 0, 0, 0]*T trajectory."""

    def __init__(self, traj_flat, dt_knot, N):
        self.traj = np.asarray(traj_flat, dtype=np.float32).reshape(-1)
        if self.traj.size % 6:
            raise ValueError("trajectory must be flat [x,y,z,0,0,0]*T (multiple of 6)")
        self.dt_knot = float(dt_knot)
        self.N = int(N)
        if self.traj.size < 6 * self.N:
            raise ValueError(f"trajectory shorter than one horizon (need >= {6*self.N} values)")

    def window(self, t):
        """The (6N,) goal window starting at time t (clamped to the trajectory end)."""
        last = self.traj.size // 6 - self.N
        off = min(int(t / self.dt_knot), last)
        return self.traj[6 * off: 6 * (off + self.N)]

    def done(self, t):
        """True once the window start would run past the end of the trajectory."""
        return int(t / self.dt_knot) >= self.traj.size / 6 - 6 * self.N


class GoalReference:
    """Constant window: hold a single 3D EE goal over the whole horizon."""

    def __init__(self, goal_xyz, N):
        goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
        self._window = np.tile(np.concatenate([goal_xyz, np.zeros(3, dtype=np.float32)]), int(N))

    def window(self, t):
        return self._window

    def done(self, t):
        return False


class MPCPolicy:
    """Callable obs -> action wrapping an MPCController.

    Args:
        controller: gato.MPCController.
        reference: object with .window(t) -> (6N,) (TrajectoryReference/GoalReference).
        dt_step: the env's control period (how long each returned action is applied);
            also used as the hypothesis-scoring dt.
        obs_to_state: maps the env observation to the (nx,) solver state
            (default: identity/asarray — for envs whose obs IS [q, dq]).
    """

    def __init__(self, controller, reference, dt_step, obs_to_state=None):
        self.controller = controller
        self.reference = reference
        self.dt_step = float(dt_step)
        self.obs_to_state = obs_to_state or (lambda o: np.asarray(o, dtype=np.float32))
        self.t = 0.0
        self.last = None  # StepResult of the most recent __call__

    def reset(self, obs):
        """Reset the controller and run a warm-up solve from the initial observation."""
        x0 = self.obs_to_state(obs)
        self.controller.reset(x0)
        self.controller.warmup(x0, self.reference.window(0.0))
        self.t = 0.0
        self.last = None

    def __call__(self, obs):
        x = self.obs_to_state(obs)
        r = self.controller.step(x, self.reference.window(self.t), dt_elapsed=self.dt_step)
        self.t += self.dt_step
        self.last = r
        return r.u
