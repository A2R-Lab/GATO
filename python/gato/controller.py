"""Task-agnostic batched MPC controller: one `step(x, reference)` per tick.

The controller owns ONLY the solver interaction — warm-start buffer, hypothesis
batch hooks, best-of-batch selection, and the real-time-iteration time shift.
It never integrates dynamics, never sleeps, and never paces itself: simulation
and timing belong to the caller (a gym env, a hardware loop, or the legacy
MPC_GATO driver).
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .common import check_floating_state, initialize_warm_start, state_difference
from .interface import SolveResult
from .linsys_autotune import resolve_admm_linsys, resolve_linsys


@dataclass(frozen=True)
class StepResult:
    """One MPC tick's outcome: the ACTUATED control to apply (``u``), the
    winning trajectory, the raw SolveResult and the warm-startedness signal."""
    u: np.ndarray            # (n_actuated,) first ACTUATED control of the winner — what you apply
    best_id: int             # winning hypothesis index (0 when B==1 / no hypotheses)
    xu_best: np.ndarray      # full winning trajectory (flat) — drivers that play
                             # multiple knots per solve need all of it
    solve: SolveResult       # solver result/stats pass-through
    hypo_stats: dict | None  # hypotheses.get_stats() snapshot, or None
    pred_err: float = 0.0    # ‖x_measured − x_predicted‖ (warm-startedness; drives linsys="auto")
    reseeded: bool = False   # this step's warm start was re-seeded to hold-at-x
    fc: np.ndarray = None    # (n_fc,) the winner's first contact-wrench slots (fc variants; else empty)
    iterations: int = 0      # SQP iterations the solve ran (the batch runs them in lockstep)
    budget_hit: bool = False # the solve stopped on step(budget_s=...) rather than on max_sqp_iters/convergence
    wall_s: float = 0.0      # wall time of this step() call, entry to return (host staging + solve + shift)
    admm_linsys: str = ""    # ADMM inner linsys this step ran ("pcg" | "bdsv_factor"; "" when ADMM is off)


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
            fixed-base solvers and "bdsv" for floating-base solvers. This is
            a workload-dependent policy, not a speed guarantee. Pass the
            solver's mode explicitly to keep it pinned.
        bdsv_threshold: pred_err threshold for linsys="auto"; defaults to 0.1
            under auto (the CDF study's [0.08, 0.17] deadline band — low τ
            trades a tighter max for a fatter p90). Calibrate per task from
            StepResult.pred_err (see tools/autotune_linsys.py).
        task_tag: optional workload tag into the autotuned linsys table
            (linsys_tuning.json (written at runtime next to _registry.json, gitignored; $GATO_LINSYS_TUNING overrides the path), written by
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
        admm_linsys: the ADMM inner loop's linear solver policy, read only
            while a row group runs the ADMM mechanism (solver.admm_active).
            "pcg" / "bdsv_factor" pin solver.set_admm_linsys; "auto" picks per
            step from the SAME warm-startedness signal as linsys="auto": a cold
            step (pred_err above admm_threshold, a PCG cap-out on the previous
            step, or a re-seed) factors the Schur system once and back-solves
            per inner iteration, a warm step runs warm PCG per inner
            iteration (the inner loop then starts near its fixed point and
            the residual exit of solver.set_admm_exit stops it after a few
            iterations). None (default) resolves "auto", with a tuned
            (plant, N, task_tag) entry's admm_policy/admm_tau when present.
        admm_threshold: pred_err threshold for admm_linsys="auto"; defaults
            to bdsv_threshold (one signal, two switches) until a fitted value
            exists (tools/autotune_linsys.py --admm).
    """

    def __init__(self, solver, *, hypotheses=None, warm_start="shift",
                 reset_rho_each_step=True, linsys=None, bdsv_threshold=None,
                 task_tag=None, reseed_threshold=None, admm_linsys=None,
                 admm_threshold=None):
        if warm_start not in ("shift", "hold"):
            raise ValueError(f"warm_start must be 'shift' or 'hold', got {warm_start!r}")
        if linsys not in (None, "pcg", "bdsv", "bdsv_first", "auto"):
            raise ValueError(f"linsys must be None, 'pcg', 'bdsv', 'bdsv_first' or 'auto', got {linsys!r}")
        if admm_linsys not in (None, "pcg", "bdsv_factor", "auto"):
            raise ValueError(f"admm_linsys must be None, 'pcg', 'bdsv_factor' or 'auto', got {admm_linsys!r}")
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
        admm_linsys, admm_threshold = resolve_admm_linsys(
            admm_linsys, admm_threshold, fallback_tau=bdsv_threshold,
            plant=getattr(solver, "plant_type", None), N=solver.N, task_tag=task_tag)
        self.admm_linsys = admm_linsys
        self.admm_threshold = admm_threshold
        if admm_linsys != "auto" and hasattr(solver, "set_admm_linsys"):
            solver.set_admm_linsys(admm_linsys)
        self.nx, self.nu, self.N = solver.nx, solver.nu, solver.N
        self.n_actuated, self.n_fc = solver.n_actuated, solver.n_fc
        self.nq, self.nv = solver.nq, solver.nv
        self.batch_size = solver.batch_size
        self._XU = np.zeros((self.batch_size, self.N * (self.nx + self.nu) - self.nu),
                            dtype=np.float32)
        self._x_prev = None
        self._u_prev = None
        self._prev_capout = False

    def reset(self, x0, *, xu_warm=None):
        """Reset solver state and seed the trajectory.

        By default every knot holds x0 with zero control. An explicit
        ``xu_warm`` has shape (xu_size,) (broadcast) or (B, xu_size); its
        first state is replaced by x0. This permits reproducible non-default
        initial guesses without reaching into controller internals. Inputs
        are copied. Floating guesses must have unit base quaternions.
        """
        x0 = np.asarray(x0, dtype=np.float32).reshape(self.nx)
        if not np.isfinite(x0).all():
            raise ValueError("reset x0 must be finite")
        check_floating_state(x0, self.nq, self.nv, "reset x0")
        if xu_warm is None:
            seed = initialize_warm_start(x0, self.N, self.nx, self.nu)
        else:
            seed = np.asarray(xu_warm, dtype=np.float32)
            if seed.shape not in ((self._XU.shape[1],), self._XU.shape):
                raise ValueError("xu_warm must have shape (xu_size,) or (B, xu_size)")
            if not np.isfinite(seed).all():
                raise ValueError("xu_warm must be finite")
            if self.floating_base:
                for row in seed.reshape(-1, self._XU.shape[1]):
                    for k in range(1, self.N):
                        offset = k * (self.nx + self.nu)
                        check_floating_state(row[offset:offset + self.nx],
                                             self.nq, self.nv, "xu_warm")
        self.solver.reset_dual()
        self.solver.reset_rho()
        self._XU[:, :] = seed
        self._XU[:, :self.nx] = x0
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

    def step(self, x_measured, reference, dt_elapsed=None, *, budget_s=None):
        """One MPC tick: solve from the measured state, return the control.

        Args:
            x_measured: (nx,) measured state.
            reference: flat (6N,) EE-goal window (or (N, 6), flattened).
            dt_elapsed: how long the PREVIOUS control was actually applied
                (used only for hypothesis scoring); defaults to solver.dt.
                Pass the REALIZED integration time, not wall time.
            budget_s: wall-clock budget for this step, in seconds, measured
                from entry. The solver runs SQP iterations while the time
                spent plus the next iteration's predicted cost (its measured
                cost in the previous budgeted solve) still fits the remainder,
                then returns the best plan so far (always at least one
                iteration; the cut is at an iteration boundary, so a budget is
                exceeded by at most one iteration).
                None (default): ``max_sqp_iters`` iterations, the path
                unchanged by this option. ``StepResult.iterations``,
                ``budget_hit`` and ``wall_s`` report what happened.

        ``x_measured`` / ``reference`` may be numpy, torch or JAX arrays on any
        device; the controller runs the solver's host path (its warm-start
        buffer is host memory) and returns ``u`` in the caller's kind.
        """
        t0 = perf_counter()
        if budget_s is not None and not float(budget_s) > 0.0:
            raise ValueError(f"budget_s must be positive (None = no budget), got {budget_s!r}")
        from . import _arrays
        in_kind = _arrays.kind(x_measured)
        in_device = getattr(x_measured, "device", None) if in_kind == "torch" else None
        x = _arrays.to_numpy(x_measured).reshape(self.nx)
        ref = _arrays.to_numpy(reference).reshape(-1)
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
        admm_linsys = ""
        if getattr(self.solver, "admm_active", False):
            if self.admm_linsys == "auto":
                cold = pred_err > self.admm_threshold or self._prev_capout or reseeded
                self.solver.set_admm_linsys("bdsv_factor" if cold else "pcg")
            admm_linsys = self.solver.admm_linsys

        # program the hypothesis batch, then solve from the measured state
        if self.hypotheses is not None:
            self.hypotheses.apply(self.solver, x)
        if self.reset_rho_each_step:
            self.solver.reset_rho()
        self._XU[:, : self.nx] = x

        # the solver's budget is what is left of the step's: host staging above
        # counts; a budget already spent still buys the one iteration that always runs
        remaining = None if budget_s is None else max(float(budget_s) - (perf_counter() - t0), 1e-9)
        res = self.solver.solve(np.tile(x, (self.batch_size, 1)),
                                np.tile(ref, (self.batch_size, 1)), self._XU, budget_s=remaining)

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

        u_full = xu_best[self.nx : self.nx + self.nu]
        u = u_full[: self.n_actuated].copy()          # torques to apply
        fc = u_full[self.n_actuated :].copy()         # the solver's contact explanation (fc variants)
        self._x_prev = x
        self._u_prev = u_full.copy()                  # hypothesis scoring uses the full control
        # a PCG cap-out means the last linearization was ill-conditioned for the
        # iterative path — "auto" forces the direct solve on the next step
        self._prev_capout = bool(
            res.stats.pcg_iters.size
            and int(res.stats.pcg_iters.max()) >= self.solver.max_pcg_iters)
        if in_kind != "numpy":
            u = _arrays.like(u, in_kind, device=in_device)
        return StepResult(
            u=u, best_id=best_id, fc=fc, xu_best=xu_best, solve=res,
            hypo_stats=self.hypotheses.get_stats() if self.hypotheses is not None else None,
            pred_err=pred_err, reseeded=reseeded,
            iterations=int(res.stats.sqp_iters.max()), budget_hit=res.stats.budget_hit,
            wall_s=perf_counter() - t0, admm_linsys=admm_linsys,
        )
