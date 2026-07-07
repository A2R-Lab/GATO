"""Hypothesis batches: run B parallel model/solver hypotheses through the batched
solver each MPC tick and let reality pick the winner.

This is the "batch-as-identity" API: the batch dimension carries *hypotheses*
(disturbance guesses, solver-parameter spreads, ...) rather than independent
problems. Lifecycle per tick:

    hypotheses.apply(solver, x_measured)   # before solve: program the batch
    result = solver.solve(...)
    best = hypotheses.select(solver, result, x_prev, u_prev, x_measured, dt)

At batch_size == 1 both hooks are no-ops (select returns 0), so a controller
built with hypotheses=None or B=1 behaves as a plain single-solve MPC.
"""
from abc import ABC, abstractmethod

import numpy as np

from .common import _require_pin


class HypothesisBatch(ABC):
    """B parallel hypotheses evaluated by the batched solver each MPC tick."""

    batch_size: int

    @abstractmethod
    def apply(self, solver, x):
        """Sample/refresh B hypotheses and program them into the solver
        (set_f_ext_B, set_rho_penalty_batch, set_mu_batch, ...)."""

    def select(self, solver, result, x_prev, u_prev, x_meas, dt):
        """Pick the hypothesis that best explains reality; return its index.

        Default scorer: roll the PREVIOUS state/control one step under each
        hypothesis' dynamics on the GPU and compare to the measured next state:

            x_next_B = solver.sim_forward(x_prev, u_prev, dt)   # per-hypothesis f_ext
            best     = argmin_b || x_next_B[b] - x_meas ||

        Hypothesis types that don't alter the dynamics (e.g. per-batch rho/mu
        spreads) should override select() and score from ``result.stats``
        (e.g. argmin final_merit) instead.
        """
        x_next_batch = solver.sim_forward(np.asarray(x_prev, dtype=np.float32),
                                          np.asarray(u_prev, dtype=np.float32), dt)
        errors = np.linalg.norm(x_next_batch - np.asarray(x_meas)[None, :], axis=1)
        best_id = int(np.argmin(errors))
        self.update(best_id, errors)
        return best_id

    @abstractmethod
    def update(self, best_id, errors):
        """Feed the selection outcome back into the hypothesis sampler."""

    def reset(self):
        pass

    def get_stats(self):
        return {}


class ForceHypothesisBatch(HypothesisBatch):
    """Wrench-disturbance hypotheses driven by a ForceEstimator/CEMForceEstimator.

    ``estimator`` is duck-typed: generate_batch() -> (B, 6) WORLD-frame wrenches,
    update(best, errors, batch_used), reset(), get_stats(). The adapter owns the
    world->GATO frame transform (URDF ``ee_frame`` -> parent-joint local wrench)
    and uploads the transformed batch via solver.set_f_ext_B.
    """

    def __init__(self, estimator, model, ee_frame="EE"):
        pin = _require_pin()
        self.estimator = estimator
        self.batch_size = estimator.batch_size
        self.model = model                      # solver model (no pendulum augmentation)
        self._data = model.createData()         # one Data, reused every tick
        self._ee_frame_id = model.getFrameId(ee_frame)
        if self._ee_frame_id >= model.nframes:
            raise ValueError(f"URDF has no frame named {ee_frame!r}")
        self._jid_ee = model.frames[self._ee_frame_id].parentJoint
        self._last_world_batch = None

    def apply(self, solver, x):
        world = self.estimator.generate_batch()          # (B, 6) world frame
        self._last_world_batch = world
        q = np.asarray(x)[: self.model.nq]
        gato_frame = np.stack([self._world_to_gato(q, w) for w in world])
        solver.set_f_ext_B(gato_frame.astype(np.float32))

    def update(self, best_id, errors):
        # estimator state lives in the WORLD frame -> pass the world-frame batch
        self.estimator.update(best_id, errors, batch_used=self._last_world_batch)

    def reset(self):
        self.estimator.reset()
        self._last_world_batch = None

    def get_stats(self):
        return self.estimator.get_stats()

    def _world_to_gato(self, q, f_world):
        """World-axes wrench at the EE frame origin -> GRiD's last-body f_ext slot.

        GRiD's slot is the last body's spatial force in the LAST JOINT's local frame
        about the joint origin, Featherstone-ordered [angular(3); linear(3)] — verified
        against pin.aba to 1e-8 (fext_frame_probe 2026-07-07). The old version here got
        both the frame chain and the [linear; angular] ordering wrong, which is why a
        seeded-truth hypothesis LOST to zero-wrench rollouts (~45% win rate)."""
        from gato.common import world_wrench_to_joint_local
        _, Fj = world_wrench_to_joint_local(self.model, self._data, q, f_world,
                                            self._ee_frame_id)
        return np.concatenate([Fj.angular, Fj.linear])
