"""Task-agnostic batched MPC controller: one `step(x, reference)` per tick.

The controller owns ONLY the solver interaction — warm-start buffer, hypothesis
batch hooks, best-of-batch selection, and the real-time-iteration time shift.
It never integrates dynamics, never sleeps, and never paces itself: simulation
and timing belong to the caller (a gym env, a hardware loop, or the legacy
MPC_GATO driver).
"""
from dataclasses import dataclass

import numpy as np

from .common import check_floating_state, initialize_warm_start, state_difference
from .interface import SolveResult
from .linsys_autotune import resolve_linsys


@dataclass(frozen=True)
class StepResult:
    u: np.ndarray            # (nu,) first control of the winning trajectory
    best_id: int             # winning hypothesis index (0 when B==1 / no hypotheses)
    xu_best: np.ndarray      # full winning trajectory (flat) — drivers that play
                             # multiple knots per solve need all of it
    solve: SolveResult       # solver result/stats pass-through
    hypo_stats: dict | None  # hypotheses.get_stats() snapshot, or None
    pred_err: float = 0.0    # ‖x_measured − x_predicted‖ (warm-startedness; drives linsys="auto")
    reseeded: bool = False   # this step's warm start was re-seeded to hold-at-x


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
        linsys: S·λ = γ path policy. "pcg"/"bdsv"/"bdsv_first" pin the solver's
            path; "auto" picks per step from warm-startedness: pred_err =
            ‖x_measured − x_predicted‖ (the shifted warm start IS the model's
            one-step prediction) — above bdsv_threshold, or after a PCG
            cap-out, the step is treated as cold and solved with "bdsv_first"
            (exact solve on the fresh linearization, PCG warm-started after);
            otherwise "pcg".
            None (default) resolves to the wired per-base default: "auto" for
            fixed-base solvers (the 08-12 CDF study: auto captures warm pcg's
            fast body AND clips the cold tail at bdsv's flat cost — matches or
            beats pure pcg at every percentile on indy7/iiwa14), "bdsv" for
            floating-base solvers (w36 sweep: wins every batch size, no
            crossover). BREAK vs pre-08-12: None used to leave the solver's
            configured path untouched — pass the solver's mode explicitly to
            pin it.
        bdsv_threshold: pred_err threshold for linsys="auto"; defaults to 0.1
            under auto (the CDF study's [0.08, 0.17] deadline band — low τ
            trades a tighter max for a fatter p90). Calibrate per task from
            StepResult.pred_err (see tools/autotune_linsys.py).
        task_tag: optional workload tag into the autotuned linsys table
            (python/gato/linsys_tuning.json, written by
            tools/autotune_linsys.py; $GATO_LINSYS_TUNING overrides the
            location). With linsys=None, a tuned (plant, N, task_tag) entry
            overrides the wired default; an explicit linsys arg always wins.
        reseed_threshold: optional pred_err threshold above which the warm
            start is RE-SEEDED to a hold at the measured state (and the
            solver duals reset) before solving. None (default) = off. A warm
            start whose tail is inconsistent with the measured state can be a
            merit LOCAL MINIMUM — the line search rejects every step and the
            gap ratchets (measured on floating-base MPC, 2026-08-11); the
            hold seed is always consistent, if conservative. Uses the same
            pred_err signal as linsys="auto"; a re-seeded step is treated as
            cold there. Floating tasks should set this; set it above
            bdsv_threshold (re-seeding is the bigger hammer).
    """

    def __init__(self, solver, *, hypotheses=None, warm_start="shift",
                 reset_rho_each_step=True, linsys=None, bdsv_threshold=None,
                 task_tag=None, reseed_threshold=None):
        if warm_start not in ("shift", "hold"):
            raise ValueError(f"warm_start must be 'shift' or 'hold', got {warm_start!r}")
        if linsys not in (None, "pcg", "bdsv", "bdsv_first", "auto"):
            raise ValueError(f"linsys must be None, 'pcg', 'bdsv', 'bdsv_first' or 'auto', got {linsys!r}")
        if hypotheses is not None and hypotheses.batch_size != solver.batch_size:
            raise ValueError(
                f"hypotheses.batch_size ({hypotheses.batch_size}) != "
                f"solver batch_size ({solver.batch_size})"
            )
        self.solver = solver
        self.hypotheses = hypotheses
        self.warm_start = warm_start
        self.reset_rho_each_step = reset_rho_each_step
        self.floating_base = bool(getattr(solver, "floating_base", False))
        linsys, bdsv_threshold = resolve_linsys(
            self.floating_base, linsys, bdsv_threshold,
            plant=getattr(solver, "plant_type", None), N=solver.N,
            task_tag=task_tag)
        self.linsys = linsys
        self.bdsv_threshold = bdsv_threshold
        self.reseed_threshold = reseed_threshold
        if linsys != "auto":
            solver.set_linsys(linsys)
        self.nx, self.nu, self.N = solver.nx, solver.nu, solver.N
        self.nq, self.nv = solver.nq, solver.nv
        self.batch_size = solver.batch_size
        self._XU = np.zeros((self.batch_size, self.N * (self.nx + self.nu) - self.nu),
                            dtype=np.float32)
        self._x_prev = None
        self._u_prev = None
        self._prev_capout = False

    def reset(self, x0):
        """Reset solver state + warm start to a hold at x0."""
        x0 = np.asarray(x0, dtype=np.float32)
        check_floating_state(x0, self.nq, self.nv, "reset x0")
        self.solver.reset_dual()
        self.solver.reset_rho()
        xu0 = initialize_warm_start(x0, self.N, self.nx, self.nu).astype(np.float32)
        self._XU[:, :] = xu0[None, :]
        if self.hypotheses is not None:
            self.hypotheses.reset()
        self._x_prev = None
        self._u_prev = None
        self._prev_capout = False

    def warmup(self, x_measured, reference):
        """Warm-up solve: seed the warm start with each batch entry's own solution.

        Unlike step(), this neither selects a winner nor time-shifts — every row
        keeps its own solved trajectory (each hypothesis' solution warm-starts
        itself on the next step). Returns the SolveResult.
        """
        x = np.asarray(x_measured, dtype=np.float32).reshape(self.nx)
        ref = np.asarray(reference, dtype=np.float32).reshape(-1)
        check_floating_state(x, self.nq, self.nv, "x_measured")
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
        check_floating_state(x, self.nq, self.nv, "x_measured")

        # warm-startedness signal: after last tick's shift, _XU[0,:nx] is the model's
        # one-step-ahead prediction of the current state — read it BEFORE the
        # measured-state overwrite below. Floating base: tangent norm
        # ‖x ⊟ x_pred‖ (a Euclidean norm on the stored layout would mix
        # quaternion components into the metric).
        x_pred = self._XU[0, : self.nx]
        if self.floating_base and abs(np.linalg.norm(x_pred[3:7]) - 1.0) < 1e-3:
            pred_err = float(np.linalg.norm(
                state_difference(x_pred, x, self.nq, self.nv)))
        else:
            pred_err = float(np.linalg.norm(x - x_pred))
        # stale-tail recovery: hold-at-x is always a consistent seed; the duals
        # rode the same staleness, so they reset with it (see reseed_threshold)
        reseeded = (self.reseed_threshold is not None
                    and pred_err > self.reseed_threshold)
        if reseeded:
            xu0 = initialize_warm_start(x, self.N, self.nx, self.nu).astype(np.float32)
            self._XU[:, :] = xu0[None, :]
            self.solver.reset_dual()
        if self.linsys == "auto":
            cold = pred_err > self.bdsv_threshold or self._prev_capout or reseeded
            self.solver.set_linsys("bdsv_first" if cold else "pcg")

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
        # a PCG cap-out means the last linearization was ill-conditioned for the
        # iterative path — "auto" forces the direct solve on the next step
        self._prev_capout = bool(
            res.stats.pcg_iters.size
            and int(res.stats.pcg_iters.max()) >= self.solver.max_pcg_iters)
        return StepResult(
            u=u, best_id=best_id, xu_best=xu_best, solve=res,
            hypo_stats=self.hypotheses.get_stats() if self.hypotheses is not None else None,
            pred_err=pred_err, reseeded=reseeded,
        )
