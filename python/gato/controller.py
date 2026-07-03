"""Task-agnostic batched MPC controller: one `step(x, reference)` per tick.

The controller owns ONLY the solver interaction — warm-start buffer, hypothesis
batch hooks, best-of-batch selection, and the real-time-iteration time shift.
It never integrates dynamics, never sleeps, and never paces itself: simulation
and timing belong to the caller (a gym env, a hardware loop, or the legacy
MPC_GATO driver).
"""
from dataclasses import dataclass

import numpy as np

from .common import initialize_warm_start
from .interface import SolveResult


@dataclass(frozen=True)
class StepResult:
    u: np.ndarray            # (nu,) first control of the winning trajectory
    best_id: int             # winning hypothesis index (0 when B==1 / no hypotheses)
    xu_best: np.ndarray      # full winning trajectory (flat) — drivers that play
                             # multiple knots per solve need all of it
    solve: SolveResult       # solver result/stats pass-through
    hypo_stats: dict | None  # hypotheses.get_stats() snapshot, or None


class MPCController:
    """Batched-SQP MPC controller around a gato.BSQP solver.

    Args:
        solver: a constructed gato.BSQP.
        hypotheses: optional gato.hypotheses.HypothesisBatch whose batch_size
            matches the solver's (each batch entry solves under its own
            hypothesis; reality picks the winner each tick).
        warm_start: "shift" (default) rolls the winning trajectory forward one
            stage each tick, duplicating the final stage (real-time iteration);
            "hold" re-seeds every batch entry with the winning trajectory as-is.
        reset_rho_each_step: restore the rho penalty to its configured default
            before each solve (line-search adaptation is per-solve state).
    """

    def __init__(self, solver, *, hypotheses=None, warm_start="shift",
                 reset_rho_each_step=True):
        if warm_start not in ("shift", "hold"):
            raise ValueError(f"warm_start must be 'shift' or 'hold', got {warm_start!r}")
        if hypotheses is not None and hypotheses.batch_size != solver.batch_size:
            raise ValueError(
                f"hypotheses.batch_size ({hypotheses.batch_size}) != "
                f"solver batch_size ({solver.batch_size})"
            )
        self.solver = solver
        self.hypotheses = hypotheses
        self.warm_start = warm_start
        self.reset_rho_each_step = reset_rho_each_step
        self.nx, self.nu, self.N = solver.nx, solver.nu, solver.N
        self.batch_size = solver.batch_size
        self._XU = np.zeros((self.batch_size, self.N * (self.nx + self.nu) - self.nu),
                            dtype=np.float32)
        self._x_prev = None
        self._u_prev = None

    def reset(self, x0):
        """Reset solver state + warm start to a hold at x0."""
        x0 = np.asarray(x0, dtype=np.float32)
        self.solver.reset_dual()
        self.solver.reset_rho()
        xu0 = initialize_warm_start(x0, self.N, self.nx, self.nu).astype(np.float32)
        self._XU[:, :] = xu0[None, :]
        if self.hypotheses is not None:
            self.hypotheses.reset()
        self._x_prev = None
        self._u_prev = None

    def warmup(self, x_measured, reference):
        """Warm-up solve: seed the warm start with each batch entry's own solution.

        Unlike step(), this neither selects a winner nor time-shifts — every row
        keeps its own solved trajectory (each hypothesis' solution warm-starts
        itself on the next step). Returns the SolveResult.
        """
        x = np.asarray(x_measured, dtype=np.float32).reshape(self.nx)
        ref = np.asarray(reference, dtype=np.float32).reshape(-1)
        if self.hypotheses is not None:
            self.hypotheses.apply(self.solver, x)
        self._XU[:, : self.nx] = x
        res = self.solver.solve(np.tile(x, (self.batch_size, 1)),
                                np.tile(ref, (self.batch_size, 1)), self._XU)
        self._XU[:, :] = res.xu
        self._x_prev = x
        self._u_prev = res.xu[0, self.nx : self.nx + self.nu].copy()
        return res

    def step(self, x_measured, reference, dt_elapsed=None):
        """One MPC tick: solve from the measured state, return the control.

        Args:
            x_measured: (nx,) measured state.
            reference: flat (6N,) EE-goal window (or (N, 6), flattened).
            dt_elapsed: how long the PREVIOUS control was actually applied
                (used only for hypothesis scoring); defaults to solver.dt.
                Pass the REALIZED integration time, not wall time.
        """
        x = np.asarray(x_measured, dtype=np.float32).reshape(self.nx)
        ref = np.asarray(reference, dtype=np.float32).reshape(-1)

        # program the hypothesis batch, then solve from the measured state
        if self.hypotheses is not None:
            self.hypotheses.apply(self.solver, x)
        if self.reset_rho_each_step:
            self.solver.reset_rho()
        self._XU[:, : self.nx] = x

        res = self.solver.solve(np.tile(x, (self.batch_size, 1)),
                                np.tile(ref, (self.batch_size, 1)), self._XU)

        # reality picks the winning hypothesis (0 on the first tick / B==1)
        best_id = 0
        if self.hypotheses is not None and self._x_prev is not None:
            best_id = self.hypotheses.select(
                self.solver, res, self._x_prev, self._u_prev, x,
                dt_elapsed if dt_elapsed is not None else self.solver.dt)

        xu_best = res.xu[best_id].copy()

        # warm start for the next tick
        if self.warm_start == "shift":
            stride = self.nx + self.nu
            shifted = np.concatenate([xu_best[stride:], xu_best[-stride:]])
            self._XU[:, :] = shifted[None, :]
        else:  # "hold"
            self._XU[:, :] = xu_best[None, :]

        u = xu_best[self.nx : self.nx + self.nu].copy()
        self._x_prev = x
        self._u_prev = u
        return StepResult(
            u=u, best_id=best_id, xu_best=xu_best, solve=res,
            hypo_stats=self.hypotheses.get_stats() if self.hypotheses is not None else None,
        )
