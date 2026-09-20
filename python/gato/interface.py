import glob
import importlib
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .common import _require_pin, _pin_available, initialize_warm_start
from .config import SolverParams, COST_FIELDS
from .linsys_autotune import resolve_linsys


VARIANTS = ("default", "fc", "eh")   # module ABI variants: base / contact forces / exact Hessian
_MODULE_RE = re.compile(r"bsqpN(\d+)_([A-Za-z0-9]+?)(?:_(fc|eh))?\.")


def module_name(plant, N, variant=None):
    """Import name of a built solver module: bsqpN{N}_{plant}[_{variant}]."""
    variant = variant or "default"
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return f"bsqpN{N}_{plant}" + ("" if variant == "default" else f"_{variant}")


def available(variant="default"):
    """Discover built solver modules in this package.

    variant="default" (the plain solver), "fc" (contact-force controls) or
    "eh" (exact Hessian) -> {(plant, N): filename} for that variant;
    variant="all" -> {(plant, N, variant): filename}."""
    here = os.path.dirname(os.path.abspath(__file__))
    found = {}
    for so in sorted(glob.glob(os.path.join(here, "bsqpN*_*.so"))):
        m = _MODULE_RE.match(os.path.basename(so))
        if not m:
            continue
        N, plant, var = int(m.group(1)), m.group(2), m.group(3) or "default"
        if variant == "all":
            found[(plant, N, var)] = os.path.basename(so)
        elif var == variant:
            found[(plant, N)] = os.path.basename(so)
    return found


def robot_info(plant_type):
    """Registry metadata for a plant ({nq, nv, ee_frame, urdf}), or {} if unregistered.

    The registry (_registry.json) is written by gato.build / tools/regen_grid.py."""
    from .builder import load_registry
    return load_registry().get(plant_type, {})


# S·λ = γ linear-system paths (see gato/bsqp/kernels/{pcg,bdsv}.cuh):
# pcg = iterative (warm-start friendly), bdsv = direct block-Cholesky (exact,
# iteration-count free), bdsv_first = direct on SQP iteration 0 then pcg.
LINSYS_MODES = {"pcg": 0, "bdsv": 1, "bdsv_first": 2}
_LINSYS_NAMES = {v: k for k, v in LINSYS_MODES.items()}


@dataclass(frozen=True)
class SolverStats:
    """Per-solve solver statistics (batch-shaped numpy arrays)."""
    solve_time_us: int
    sqp_iters: np.ndarray       # (B,) int32
    kkt_converged: np.ndarray   # (B,) int32
    final_merit: np.ndarray     # (B,) float32 — merit of the returned trajectory
    initial_merit: np.ndarray   # (B,) float32 — merit before the first iteration
    ls_num_iters: int           # SQP iterations that reached the line search
    pcg_iters: np.ndarray       # (sqp_iters_run, B) int32
    pcg_times_us: np.ndarray    # (sqp_iters_run,) float32
    min_merit: np.ndarray       # (ls_num_iters, B) float32 — accepted merit per iter
    step_size: np.ndarray       # (ls_num_iters, B) float32 — -1 marks a line-search failure
    # linear-system path used for this solve. NOTE pcg_iters semantics on the
    # bdsv path: 0 still means "converged at start" (same guard as pcg), a
    # direct solve reports 1, and 2 marks a SKIPPED update (f32 Cholesky hit a
    # non-PD pivot — barely-regularized costs; λ kept its warm start and rho
    # adaptation retries). Only == 0 carries meaning downstream.
    linsys: str = "pcg"         # "pcg" | "bdsv" | "bdsv_first"
    # constraint row-group telemetry (None unless enable_limit_telemetry()):
    # (n_groups, B) true violation of the RETURNED trajectory per row-group
    # (group order: BOX_Q, BOX_QD, BOX_U)
    row_max_violation: np.ndarray = None
    row_sum_violation: np.ndarray = None
    # (B,) last-iteration ADMM residuals (None unless enable_limit_admm())
    admm_r_prim: np.ndarray = None
    admm_r_dual: np.ndarray = None


@dataclass(frozen=True)
class SolveResult:
    """Result of one batched solve.

    ``xu`` is the flat batch of trajectories, one row per solve:
    ``[x_0, u_0, x_1, u_1, ..., x_{N-1}]`` with row length ``N*(nx+nu) - nu``.
    """
    xu: np.ndarray              # (B, N*(nx+nu)-nu) float32
    solve_time_us: int
    stats: SolverStats
    nx: int
    nu: int                     # full control width (== n_actuated + n_fc)
    N: int
    # On GATO_CONTACT_FORCES builds the control is [tau; fc] and only the first
    # n_actuated entries are torques an MPC applies; fc is the solver's contact
    # explanation. 0 = legacy default (all of nu is actuated).
    n_actuated: int = 0

    @property
    def batch_size(self):
        return self.xu.shape[0]

    @property
    def _na(self):
        return self.n_actuated or self.nu

    @property
    def n_fc(self):
        return self.nu - self._na

    def xu_b(self, b=0):
        """Trajectory of batch entry b (flat)."""
        return self.xu[b]

    def u0(self, b=0):
        """First ACTUATED control of batch entry b — what an MPC applies
        (contact-force slots, if any, are excluded)."""
        return self.xu[b, self.nx:self.nx + self._na]

    def control_at(self, k, b=0):
        """Actuated control at knot k (clamped to the last control knot) of batch
        entry b (contact-force slots, if any, are excluded)."""
        k = min(int(k), self.N - 2)  # last knot has no control
        start = self.nx + (self.nx + self.nu) * k
        return self.xu[b, start:start + self._na]

    def fc_at(self, k, b=0):
        """Contact-wrench slots at knot k of batch entry b ((n_fc,); empty on
        non-contact-force builds)."""
        k = min(int(k), self.N - 2)
        start = self.nx + (self.nx + self.nu) * k + self._na
        return self.xu[b, start:start + self.n_fc]

    def fc_traj(self, b=0):
        """(N-1, n_fc) contact-wrench trajectory of batch entry b."""
        return np.stack([self.fc_at(k, b) for k in range(self.N - 1)])

    @property
    def diverged(self):
        """(B,) bool — non-finite merit or trajectory."""
        bad_merit = ~np.isfinite(self.stats.final_merit)
        bad_xu = ~np.isfinite(self.xu).all(axis=1)
        return bad_merit | bad_xu


class BSQP:
    """Batched SQP trajectory optimizer over one compiled solver module.

    Args:
        model_path: robot URDF (must be the one the module was codegen'd from).
        batch_size: number of parallel solves B (runtime; any B >= 1).
        N: horizon length — selects the module bsqpN{N}_{plant}[_{variant}].
        dt: knot spacing [s].
        params: gato.SolverParams (defaults = the paper/MPC set); keyword
            overrides of its fields may be passed directly (``BSQP(..., mu=5.0)``).
        plant_type: plant name; None auto-detects from model_path against the
            built modules / registry.
        variant: "default" | "fc" (contact-force controls appended to u) |
            "eh" (exact-Hessian SO-SQP). Variant modules are separate ABIs.
        rho_batch / mu_batch / pcg_tol_batch: optional per-solve (B,) overrides.

    Construction needs numpy and the module only. pinocchio is loaded lazily
    by the FK helpers (ee_pos) and the sim/estimator layers (``[test]`` extra).

    Warm start: ``solve(x, ref, xu_warm=None)`` is STATELESS w.r.t. the
    trajectory — None seeds a hold at x; pass the previous ``SolveResult.xu``
    (or let MPCController own the shifted warm start) to warm-start. The
    solver IS stateful across solves in rho adaptation, duals and ADMM state
    (reset_rho / reset_dual / reset).

    Cost precedence: SolverParams scalars < set_cost_weights (scalars) <
    set_cost_weights_per_knot (N,3) [ee, qd, u]; per-joint vectors
    (set_q_pos_cost / set_u_cost_vec) and the fc cost are orthogonal terms.

    Reference layout: ``ref`` is (B, 6N): per knot [x, y, z, 0, 0, 0] — the
    EE position target (orientation slots are reserved, currently unused).
    Trajectory layout: ``xu`` is (B, N*(nx+nu) - nu) =
    [x_0, u_0, x_1, u_1, ..., x_{N-1}] with x = [q; qd].
    """

    def __init__(self, model_path, batch_size, N, dt, params=None, *,
                 plant_type=None, variant=None,
                 rho_batch=None, mu_batch=None, pcg_tol_batch=None, **overrides):
        params = SolverParams.from_mapping(params)
        if overrides:
            params = params.replace(**overrides)   # unknown names raise here
        self.params = params
        model_path = str(model_path)

        # Auto-detect plant type from model_path if not explicitly specified.
        # Unknown robots are a hard error (a wrong plant silently runs the wrong
        # dynamics with mismatched state size).
        if plant_type is None:
            from .builder import load_registry
            plants = sorted({p for p, _ in available(variant or "default")} | set(load_registry()))
            low = model_path.lower()
            # match the plant name or its alpha prefix (iiwa14 -> "iiwa") in the path
            matches = [p for p in plants
                       if p in low or re.sub(r"\d+$", "", p) in os.path.basename(low)]
            if len(matches) != 1:
                raise ValueError(
                    f"Could not auto-detect plant from model_path={model_path!r}; "
                    f"pass plant_type explicitly. Built plants: {plants or 'none'}"
                )
            plant_type = matches[0]

        self.variant = variant or "default"
        mod = module_name(plant_type, N, self.variant)
        try:
            base = importlib.import_module("gato." + mod)
        except ImportError as e:
            flag = {"fc": " contact_forces=True", "eh": " exact_hessian=True", "default": ""}[self.variant]
            raise ValueError(
                f"No compiled module for plant={plant_type!r}, N={N}, variant={self.variant!r} "
                f"(could not import gato.{mod}): {e}\n"
                f"Built modules: {sorted(available('all')) or 'none'} — build with, e.g.:\n"
                f"  python -c \"import gato; gato.build('{model_path}', name='{plant_type}', N=[{N}],{flag})\"\n"
                f"  or: cmake -S . -B build -DMODULES={plant_type}:{N} && cmake --build build --parallel 2"
            )

        # batch_size is a runtime constructor argument (one class per precision)
        class_name = "BSQP_float"
        if not hasattr(base, class_name):
            raise ValueError(
                f"Module {mod} does not export {class_name} — rebuild the "
                f"solver modules (old per-batch-size builds are incompatible)"
            )
        self.lib = base
        self.solver_class = getattr(base, class_name)
        self.plant_type = plant_type
        # Body-major external-force buffer width is 6*NUM_BODIES per solve (the GPU
        # d_f_ext_batch_ buffer + set_f_ext_batch upload are sized to this).
        self.n_bodies = int(getattr(base, "NUM_BODIES", 0))

        self._cost_weights = {k: float(getattr(params, k)) for k in COST_FIELDS}
        self.solver = self.solver_class(
            batch_size,
            int(params.max_sqp_iters),
            int(params.max_pcg_iters), float(params.pcg_tol), float(params.solve_ratio), float(params.mu),
            float(params.q_cost), float(params.qd_cost), float(params.u_cost), float(params.N_cost),
            float(params.q_lim_cost), float(params.vel_lim_cost), float(params.ctrl_lim_cost),
            float(params.rho),
        )
        # Dimensions come from the MODULE (NQ/NV/CONTROL_SIZE/... are baked at
        # compile time) — pinocchio is NOT needed to construct or run a solver.
        # The pinocchio model is built lazily (self.model / self.data) by the
        # FK helpers (ee_pos), the sim/estimator layers and the tests; when it
        # IS available the module/URDF dimension cross-check below runs.
        self.floating_base = bool(getattr(base, "FLOATING_BASE", False))
        self.model_path = model_path
        self._model = None
        self._data = None
        self._ee_frame_id = None
        if hasattr(base, "NQ"):
            self.nq, self.nv = int(base.NQ), int(base.NV)
        else:  # pre-CL-3 modules did not export the state layout: fall back to pinocchio
            self.nq, self.nv = self.model.nq, self.model.nv
        if _pin_available():
            if (self.nq, self.nv) != (self.model.nq, self.model.nv):
                raise ValueError(
                    f"module {mod} has (nq, nv) = ({self.nq}, {self.nv}) but the "
                    f"pinocchio model from {model_path!r} has ({self.model.nq}, {self.model.nv}) — "
                    f"wrong URDF or a fixed/floating mismatch")
        # The solver/grid.cuh optimizes the EE-position cost in the frame the module
        # was codegen'd with (fixed_target_name; recorded in the registry, "EE" for
        # the vendored robots). The frame id is resolved lazily with the model.
        self.ee_frame = robot_info(plant_type).get("ee_frame", "EE")
        self.batch_size = batch_size
        self.N = N
        self.dt = dt
        self.f_ext_B = np.zeros((self.batch_size, 6), dtype=np.float32)
        self.set_f_ext_B(self.f_ext_B)

        self.nx = self.nq + self.nv
        # Control width comes from the MODULE: on fc variants CONTROL_SIZE =
        # ACTUATED_SIZE + FC_SIZE (contact-wrench slots appended after the
        # torques), so nu > nv and every xu stride follows it.
        self.n_actuated = int(getattr(base, "ACTUATED_SIZE", self.nv))
        self.n_fc = int(getattr(base, "FC_SIZE", 0))
        self.nu = int(getattr(base, "CONTROL_SIZE", self.nv))
        self.xu_size = self.N * (self.nx + self.nu) - self.nu

        # Optional batched hyperparameters
        if rho_batch is not None:
            rho_batch = np.asarray(rho_batch, dtype=np.float32).reshape(self.batch_size)
            self.solver.set_rho_penalty_batch(rho_batch, True)
        self.solver.set_rho_adaptation(bool(params.adapt_rho))
        if mu_batch is not None:
            mu_batch = np.asarray(mu_batch, dtype=np.float32).reshape(self.batch_size)
            self.solver.set_mu_batch(mu_batch)
        if pcg_tol_batch is not None:
            pcg_tol_batch = np.asarray(pcg_tol_batch, dtype=np.float32).reshape(self.batch_size)
            self.solver.set_pcg_tol_batch(pcg_tol_batch)
        self.max_pcg_iters = int(params.max_pcg_iters)
        self.linsys = "pcg"  # the C++ default; set_linsys only calls into the module on change
        # ONE resolver for both the raw solver and the controller
        # (linsys_autotune.resolve_linsys): the solver takes the STATIC arm —
        # "auto" (a per-step controller policy) maps to its warm-body path, pcg.
        mode, _ = resolve_linsys(self.floating_base, params.linsys, None,
                                 plant=plant_type, N=N, task_tag=None)
        self.set_linsys("pcg" if mode == "auto" else mode)
        self._row_mech = None  # active enable_limit_* mode (add_lin_u_rows mech=None default)
        self._n_appended_groups = 0
        self.exact_hessian = False  # the C++ default
        if params.exact_hessian:
            self.set_exact_hessian(True)

    def set_linsys(self, mode):
        """Pick the S·λ = γ path for subsequent solves: "pcg" | "bdsv" | "bdsv_first".

        Host-side and zero-cost — an MPC loop can switch it per step.
        """
        if mode not in LINSYS_MODES:
            raise ValueError(f"linsys must be one of {sorted(LINSYS_MODES)}, got {mode!r}")
        if mode != self.linsys:
            self.solver.set_linsys_mode(LINSYS_MODES[mode])
            self.linsys = mode
        self.params = self.params.replace(linsys=mode)

    def set_admm_linsys(self, mode):
        """ADMM inner-loop linear solver: "pcg" (default) | "bdsv_factor".

        Measured defaults, rulings and provenance: docs/constraints.md (`set_admm_linsys`)."""
        if mode not in ("bdsv_factor", "pcg"):
            raise ValueError(f'admm_linsys must be "bdsv_factor" or "pcg", got {mode!r}')
        self.solver.set_admm_linsys_pcg(mode == "pcg")

    def set_collect_stats(self, on):
        """Collect per-iteration solver stats (PCG timings via cudaEvents,
        line-search merits). On by default; MPC loops that do not read
        SolverStats can switch it off (two event records per SQP iteration)."""
        self.solver.set_collect_stats(bool(on))

    def set_drho_batch(self, drho_batch, set_as_reset_default=True):
        """Per-solve trust-region rho ADAPTATION factor (B,) — the multiplier the
        line search applies on reject/accept (reset_rho restores the default)."""
        d = np.ascontiguousarray(np.asarray(drho_batch, dtype=np.float32).reshape(self.batch_size))
        self.solver.set_drho_batch(d, bool(set_as_reset_default))

    def get_lambda(self):
        """(B, N+2, nx) lagged KKT multipliers λ from the last solve (padded layout)."""
        return np.asarray(self.solver.get_lambda()).reshape(self.batch_size, self.N + 2, self.nx)

    def debug_setup_kkt(self, xu_B, xcur_B, eepos_goals_B):
        """Run setup_kkt ONLY on a trajectory and return the assembled KKT blocks
        {Q, R, q, r, A, B, c} (B, ...) — the way the anchor/exact-Hessian gates
        pin cost claims directly instead of arguing from solve outcomes."""
        xu = np.ascontiguousarray(np.asarray(xu_B, dtype=np.float32).reshape(self.batch_size, self.xu_size))
        x = np.ascontiguousarray(np.asarray(xcur_B, dtype=np.float32).reshape(self.batch_size, self.nx))
        g = np.ascontiguousarray(np.asarray(eepos_goals_B, dtype=np.float32).reshape(self.batch_size, -1))
        return self.solver.debug_setup_kkt(xu, self.dt, x, g)

    def exact_hessian_available(self):
        """True if the loaded module was compiled with -DGATO_EXACT_HESSIAN=ON."""
        return bool(getattr(self.lib, "EXACT_HESSIAN_AVAILABLE", False))

    def set_exact_hessian(self, on):
        """Toggle the SO-SQP stage-Hessian PSD projection for subsequent solves.

        Measured defaults, rulings and provenance: docs/constraints.md (`set_exact_hessian`)."""
        on = bool(on)
        if on and not self.exact_hessian_available():
            raise RuntimeError(
                f"module {self.lib.__name__} compiled without USE_EXACT_HESSIAN — "
                "rebuild with cmake -DGATO_EXACT_HESSIAN=ON"
            )
        if on != self.exact_hessian:
            self.solver.set_exact_hessian(on)
            self.exact_hessian = on
        self.params = self.params.replace(exact_hessian=bool(on))

    def _heal_floating_warm_start(self, XU_B, xcur_B):
        """Replace warm-start knots whose base quaternion is degenerate by the
        measured state (controls kept). The constructed XU_B is all-zero, and a
        zero quaternion is not a point on the manifold — the device solve
        cannot step off it (state_difference/retract are undefined there), so
        a cold floating solve would silently return the zero tail unchanged.
        Valid warm starts pass through untouched (bit-identical)."""
        step = self.nx + self.nu
        for k in range(self.N):
            quat = XU_B[:, k * step + 3:k * step + 7]
            bad = np.linalg.norm(quat, axis=1) < 0.5
            if bad.any():
                XU_B[bad, k * step:k * step + self.nx] = xcur_B[bad]

    def solve(self, xcur_B, eepos_goals_B, xu_warm=None):
        """Solve the batch from states ``xcur_B`` (B, nx) toward the EE reference
        ``eepos_goals_B`` (B, 6N) — see the class docstring for both layouts.

        ``xu_warm`` (B, N*(nx+nu)-nu): the warm-start trajectory. None seeds a
        hold at ``xcur_B`` (a cold start; NOT the previous solution — the solver
        keeps no trajectory state; pass the last ``SolveResult.xu`` to
        warm-start, or use MPCController). The caller's array is never
        modified; ``SolveResult.xu`` is a fresh array.
        """
        xcur_B = np.asarray(xcur_B, dtype=np.float32).reshape(self.batch_size, self.nx)
        eepos_goals_B = np.asarray(eepos_goals_B, dtype=np.float32).reshape(self.batch_size, -1)
        if xu_warm is None:
            XU_B = np.stack([initialize_warm_start(xcur_B[b], self.N, self.nx, self.nu)
                             for b in range(self.batch_size)]).astype(np.float32)
        else:
            XU_B = np.array(xu_warm, dtype=np.float32, copy=True).reshape(self.batch_size, self.xu_size)
        XU_B[:, : self.nx] = xcur_B
        if self.floating_base:
            self._heal_floating_warm_start(XU_B, xcur_B)

        raw = self.solver.solve(XU_B, self.dt, xcur_B, eepos_goals_B)

        xu = np.array(raw["XU"], dtype=np.float32, copy=True)
        B = self.batch_size
        stats = SolverStats(
            solve_time_us=int(raw["sqp_time_us"]),
            sqp_iters=np.asarray(raw["sqp_iters"], dtype=np.int32).reshape(B),
            kkt_converged=np.asarray(raw["kkt_converged"], dtype=np.int32).reshape(B),
            final_merit=np.asarray(raw["final_merit"], dtype=np.float32).reshape(B),
            initial_merit=np.asarray(raw["initial_merit"], dtype=np.float32).reshape(B),
            ls_num_iters=int(raw["ls_num_iters"]),
            pcg_iters=np.asarray(raw["pcg_iters"], dtype=np.int32).reshape(-1, B),
            pcg_times_us=np.asarray(raw["pcg_times_us"], dtype=np.float32).reshape(-1),
            min_merit=np.asarray(raw["ls_min_merit"], dtype=np.float32).reshape(-1, B),
            step_size=np.asarray(raw["ls_step_size"], dtype=np.float32).reshape(-1, B),
            linsys=_LINSYS_NAMES[int(raw.get("linsys_mode", 0))],
            row_max_violation=(np.asarray(raw["row_max_violation"], dtype=np.float32)
                               if "row_max_violation" in raw else None),
            row_sum_violation=(np.asarray(raw["row_sum_violation"], dtype=np.float32)
                               if "row_sum_violation" in raw else None),
            admm_r_prim=(np.asarray(raw["admm_r_prim"], dtype=np.float32)
                         if "admm_r_prim" in raw else None),
            admm_r_dual=(np.asarray(raw["admm_r_dual"], dtype=np.float32)
                         if "admm_r_dual" in raw else None),
        )
        return SolveResult(xu=xu, solve_time_us=stats.solve_time_us,
                           stats=stats, nx=self.nx, nu=self.nu, N=self.N,
                           n_actuated=self.n_actuated)

    # rows::Mechanism enum values (rowgroups.cuh)
    _MECHS = {"telemetry": 0, "barrier": 1, "admm": 2, "al": 3}

    def enable_limit_telemetry(self):
        """Install the canonical limit row-groups (position/velocity/torque boxes
        from the URDF limit tables) in TELEMETRY mode: every solve() reports each
        group's true violation of the returned trajectory in
        ``stats.row_{max,sum}_violation`` (group order BOX_Q, BOX_QD, BOX_U).
        Telemetry never touches the solver path — trajectories are bit-identical
        with it on or off. Part of the constraint row-group layer (CL-0)."""
        self._check_no_appended_groups("enable_limit_telemetry")
        self.solver.enable_limit_telemetry()
        self._row_mech = "telemetry"

    def enable_limit_barrier(self, mu=3e-3, delta=0.05):
        """Bind the limit row-groups to the RELAXED log-barrier mechanism: a
        C² barrier with bounded Hessian (quadratic extension within ``delta``
        of a bound) folded into the KKT cost and merit — infeasible-start safe,
        the constraint layer's soft prior mode. Additive to grid_plant's own
        clamped log barriers; zero q_lim/vel_lim/ctrl_lim_cost for a clean
        comparison. Telemetry (stats.row_*_violation) stays on."""
        self._check_no_appended_groups("enable_limit_barrier")
        self.solver.enable_limit_barrier(float(mu), float(delta))
        self._row_mech = "barrier"

    def enable_limit_admm(self, rho=0.01, iters=10):
        """Bind the limit row-groups to the ADMM-projection mechanism: an
        OSQP-style fixed-budget inner loop per SQP iteration on a REUSED
        direct (bdsv) factorization — the constraint layer's
        "approximately hard" mode. ``rho`` is the ADMM penalty (fixed within
        a solve; adapt it between solves), ``iters`` the fixed budget.
        R1 default rho=0.01: the penalty must ride the COST-HESSIAN scale —
        rho >= 1 swamps the u-block (natural scale u_cost=1e-6), freezing
        controls at the warm start (closed-loop MPC parks); the measured
        pocket is ~0.005-0.02 (r1_report_2026-07-11.md).
        iters=10 BOUND by R2 (r2_report_2026-07-30.md): 2/5 park the feasible
        cone cell; box cells saturate by 10 (20 = marginal viol gains at 2x
        inner cost).
        Duals warm-start across solves (reset_dual() reinitializes) —
        EXCEPT equality rows (lo == hi, e.g. enable_ee_terminal_equality),
        whose (z, y) reinit every solve: a warm-started dual on a row the
        primal may not reach is an unbounded violation integrator (measured).
        stats gain admm_r_prim/admm_r_dual; telemetry stays on."""
        self._check_no_appended_groups("enable_limit_admm")
        self.solver.enable_limit_admm(float(rho), int(iters))
        self._row_mech = "admm"

    def enable_limit_al(self, rho=1.0):
        """Bind the limit row-groups to the PHR augmented-Lagrangian mechanism:
        hinge-activated grad/GN-Hessian and C¹ AL value folded into the KKT
        cost and merit, with the outer dual update
        ``lam <- max(0, lam + rho*violation)`` run ONCE per solve on the final
        trajectory (equality rows ``lo == hi`` always active) — warm-started
        repeat solves are the outer loop, so feasibility converges across MPC
        steps. The update is gated on TRUE-violation acceptance (feasible or
        strictly improved), so a stalled primal freezes the duals instead of
        drifting. ``rho`` is fixed per enable; duals persist across solves
        (reset_dual() zeroes them). Violation is honestly telemetry-reported;
        ``get_row_duals()`` exposes the multipliers. While active, solves use
        the direct (bdsv) linear solver and freeze trust-region rho
        adaptation — both required for outer convergence (measured; see
        bsqp.cuh dispatch comments). REQUIRES the trust-region floor
        (constructor rho > 0, the default): f32 bdsv on an unregularized
        Schur system returns garbage steps (R1). R1 default rho=1.0 — the
        fold lands rho on ACTIVE rows whose natural Hessian scale is tiny
        (qd rows ~1e-4): rho >= 10 makes the f32 factor error large enough
        that closed-loop MPC destabilizes on tight-limit plants (measured:
        iiwa14 pickplace spins at 100 rad/s at rho=100, final 5mm at
        rho=1). Higher rho = tighter transients — raise it only within the
        f32 ceiling (rho ~ 1e4 x the block's natural Hessian scale)."""
        self._check_no_appended_groups("enable_limit_al")
        self.solver.enable_limit_al(float(rho))
        self._row_mech = "al"

    def enable_ee_terminal_equality(self, target, rho=10.0):
        """Append an EE terminal-position equality row-group: the returned
        trajectory's final-knot EE position is constrained to ``target`` (xyz,
        ``lo == hi``). The first non-selection row kind — evaluated by
        on-device FK, in the SOLVER's EE frame (``ee_pos(q, frame="solver")``
        — the same frame the tracking cost optimizes; see ee_pos for the
        frame-offset caveat). Mechanism follows the current mode: AL when
        enable_limit_al() is active (always-active equality, signed
        multiplier in lam_hi), ADMM when enable_limit_admm() is active
        (linearized inner-loop projection: z pins to target, y accumulates
        the equality multiplier), telemetry-only reporting otherwise. Call
        AFTER enable_limit_* — mechanism enables reinstall the canonical
        groups and drop appended ones. R1 binding ruling: ADMM binding measured
        best for closed-loop MPC (2mm finals at rho=10); AL binding works at
        SOFT rho (al rho=1, ee rho=1: ~5mm finals) — at rho=100 the equality
        multiplier winds up through the f32 factor error and diverges."""
        self.solver.enable_ee_terminal_equality(
            np.asarray(target, dtype=np.float32).reshape(3), float(rho))
        self._n_appended_groups += 1

    def disable_row_groups(self):
        """Remove all constraint row-groups (stats lose the row_* fields)."""
        self.solver.disable_row_groups()
        self._row_mech = None
        self._n_appended_groups = 0

    def _check_no_appended_groups(self, what):
        # Mechanism enables reinstall the canonical 3 limit groups and would
        # silently DROP appended groups (EE rows, LIN_U rows, collision) — the
        # ordering contract is enforced instead of documented: enables first.
        if self._n_appended_groups:
            raise RuntimeError(
                f"{what}() after {self._n_appended_groups} appended row-group(s) would drop them; "
                "call enable_limit_* BEFORE add_lin_u_rows / enable_ee_terminal_equality / "
                "enable_collision (or disable_row_groups() first)")

    def get_row_groups(self):
        """List of installed row-group descriptors (dicts with kind/block/mech,
        knot mask, and per-row lo/hi bounds)."""
        return self.solver.get_row_groups()

    def get_row_duals(self):
        """AL multipliers dict {lam_hi, lam_lo}, each shaped
        (B, MAX_ROW_GROUPS, N, MAX_ROWS_PER_GROUP) in the dense row-state
        layout (group gi's active slots are [:, gi, knot_lo:knot_hi, :n_rows]).
        Equality rows carry their signed multiplier in the lam_hi slot."""
        return self.solver.get_row_duals()

    def get_admm_state(self):
        """ADMM state dict {z, y} (auxiliary/dual), same dense layout as
        get_row_duals(). y is the interval-constraint multiplier estimate."""
        return self.solver.get_admm_state()

    def set_row_group_bounds(self, g, lo, hi):
        """Override group ``g``'s interval bounds (arrays of n_rows each).
        ``lo == hi`` rows become always-active equalities under AL. ADMM's
        auxiliary state reinitializes on the next solve (re-clip)."""
        self.solver.set_row_group_bounds(int(g),
                                         np.asarray(lo, dtype=np.float32),
                                         np.asarray(hi, dtype=np.float32))

    def set_row_group_soft(self, g, sigma):
        """Soft/slack toggle (TurboMPC delta_xi) for group ``g``: sigma > 0
        makes its rows ELASTIC — transient violation is traded against the
        elastic weight instead of forced to zero. AL: L1 slack — the
        effective multiplier saturates at sigma (the outer update caps
        |lam| <= sigma; the principled lambda-cap for conflict regimes).
        ADMM: quadratic slack — smoothed z-projection (slope
        rho/(rho+sigma) past a bound; sigma -> inf recovers the hard clamp).
        sigma = 0 restores the exact hard path. Telemetry always reports
        the TRUE violation, slack notwithstanding."""
        self.solver.set_row_group_soft(int(g), float(sigma))

    def set_admm_merit(self, on=True):
        """R1 ablation toggle: include the AL-form ADMM constraint value
y'(g - z) + (rho/2)|g - z|^2 (current row state) in the line-search
merit. v1 ADMM's merit is tracking-only, so the line search rejects
steps that trade tracking for feasibility (measured: closed-loop MPC
parks in conservative basins). Off by default — the exact v1
semantics; only read while ADMM mode is active.

        Measured defaults, rulings and provenance: docs/constraints.md (`set_admm_merit`)."""
        self.solver.set_admm_merit(bool(on))

    def set_admm_rho_adaptation(self, on=True):
        """OSQP-style ADMM rho adaptation (opt-in; default OFF = bitwise
pre-adaptation path). One per-solve SCALAR multiplier on top of every
ADMM group's rho baseline (the bound per-group ratios — cone u-block
0.01 vs collision Q-block 1.0+ — are preserved), updated once per SQP
iteration from the inner loop's final residuals: adapt when
r_prim/r_dual is imbalanced by >5x, step by sqrt(ratio), clamp to
[1e-2, 1e2] (OSQP's rule). The dual form is unscaled, so rho changes
need no y-rescaling; the rho*G'G fold refreshes each SQP iteration.
The scale persists across solves (warm rho, like the (z, y) dual warm
start); toggling resets it to 1. Telemetry: get_admm_rho_scale().

        Measured defaults, rulings and provenance: docs/constraints.md (`set_admm_rho_adaptation`)."""
        self.solver.set_admm_rho_adaptation(bool(on))

    def get_admm_rho_scale(self):
        """Per-solve adapted rho scale, shape (B,). Effective ADMM rho of a
        group = its rho baseline * this scale (1.0 = unadapted)."""
        return np.asarray(self.solver.get_admm_rho_scale())

    def add_lin_u_rows(self, C, d=None, lo=None, hi=None, mech=None, rho=None,
                       delta=0.05, sigma=0.0, cone=False, knot_lo=0,
                       knot_hi=None, admm_iters=0, equilibrate=False,
                       normalize=True):
        """Append a LIN_U row-group: m rows ``g = C @ u + d`` on the control
block (C shape (m, nu), FROZEN at a host-chosen configuration — the
cross-term audit's contact-frame rule for config-dependent maps).

        Measured defaults, rulings and provenance: docs/constraints.md (`add_lin_u_rows`)."""
        C = np.ascontiguousarray(np.asarray(C, dtype=np.float32))
        if C.ndim != 2 or C.shape[1] != self.nu:
            raise ValueError(f"C must be (m, {self.nu}); got {C.shape}")
        m = C.shape[0]
        if mech is None:
            mech = self._row_mech or "telemetry"
        if mech not in self._MECHS:
            raise ValueError(f"mech must be one of {sorted(self._MECHS)}, got {mech!r}")
        if rho is None:
            rho = {"telemetry": 0.0, "barrier": 3e-3, "admm": 0.01, "al": 1.0}[mech]
        # ascontiguousarray: the binding consumes raw .ptr buffers — a strided view
        # (e.g. np.broadcast_to) must be densified here (belt; the binding also
        # forces c_style since 2026-08-02)
        d = np.ascontiguousarray(np.asarray([] if d is None else d,
                                            dtype=np.float32).reshape(-1))
        if cone:
            if normalize:
                s = float(np.linalg.norm(C.astype(np.float64), 2))
                if s > 0.0:
                    C = np.ascontiguousarray((C.astype(np.float64) / s).astype(np.float32))
                    d = np.ascontiguousarray((d.astype(np.float64) / s).astype(np.float32))
            lo_a = np.asarray([], dtype=np.float32)
            hi_a = np.asarray([], dtype=np.float32)
        else:
            if lo is None or hi is None:
                raise ValueError("interval LIN_U rows need lo and hi (length m)")
            lo_a = np.ascontiguousarray(np.asarray(lo, dtype=np.float32).reshape(m))
            hi_a = np.ascontiguousarray(np.asarray(hi, dtype=np.float32).reshape(m))
        if knot_hi is None:
            knot_hi = self.N - 1  # no terminal control
        self.solver.add_lin_u_group(self._MECHS[mech], C, d, lo_a, hi_a,
                                    bool(cone), float(rho), float(delta),
                                    float(sigma), int(knot_lo), int(knot_hi),
                                    int(admm_iters), bool(equilibrate))
        self._n_appended_groups += 1

    def add_fc_box(self, lo, hi, slots=None, **kw):
        """Box rows on contact-force slots (GATO_CONTACT_FORCES builds only):
        selection LIN_U rows on control columns n_actuated+slots. ``slots``
        indexes into the fc block (default: all n_fc slots); lo/hi broadcast.
        Pin the wrench torque rows of a point contact with
        ``add_fc_box(0, 0, slots=range(3))`` (wrench layout is [n; f]).
        Extra kwargs go to add_lin_u_rows (mech/rho/knot range/...)."""
        if self.n_fc == 0:
            raise RuntimeError("add_fc_box needs a GATO_CONTACT_FORCES build "
                               "(this module has no fc slots)")
        slots = list(range(self.n_fc)) if slots is None else list(slots)
        if any(s < 0 or s >= self.n_fc for s in slots):
            raise ValueError(f"fc slots must be in [0, {self.n_fc}); got {slots}")
        m = len(slots)
        C = np.zeros((m, self.nu), dtype=np.float32)
        for i, s in enumerate(slots):
            C[i, self.n_actuated + s] = 1.0
        lo_a = np.full(m, lo, dtype=np.float32) if np.ndim(lo) == 0 else np.asarray(lo, dtype=np.float32)
        hi_a = np.full(m, hi, dtype=np.float32) if np.ndim(hi) == 0 else np.asarray(hi, dtype=np.float32)
        return self.add_lin_u_rows(C, lo=lo_a, hi=hi_a, **kw)

    def enable_u_cone(self, C, d=None, mech=None, rho=None, form="soc",
                      facets=8, facet_scale="inscribed", **kw):
        """Cone constraint on a mapped control quantity g = C @ u + d
(CL-2 demo surface: e.g. an EE contact-force friction cone with
C = S @ pinv(J(q).T), rows [mu*f_n; f_t1; f_t2], frozen at q).

        Measured defaults, rulings and provenance: docs/constraints.md (`enable_u_cone`)."""
        C = np.asarray(C, dtype=np.float64)
        m = C.shape[0]
        d = np.zeros(m) if d is None else np.asarray(d, dtype=np.float64).reshape(m)
        gi = len(self.get_row_groups())
        if form == "soc":
            self.add_lin_u_rows(C, d, mech=mech, rho=rho, cone=True, **kw)
        elif form == "pyramid":
            if m != 3:
                raise ValueError("pyramid form supports 3-row cones (t, x, y)")
            s = np.cos(np.pi / facets) if facet_scale == "inscribed" else 1.0
            th = 2.0 * np.pi * np.arange(facets) / facets
            F = np.stack([np.cos(th), np.sin(th), -s * np.ones(facets)], axis=1)  # rows: [c, s, -s0] on (g1, g2, g0)
            P = F @ C[[1, 2, 0], :]           # facet map on u
            pd = F @ d[[1, 2, 0]]             # facet offsets
            self.add_lin_u_rows(P, pd, lo=np.full(facets, -np.inf),
                                hi=np.zeros(facets), mech=mech, rho=rho, **kw)
        else:
            raise ValueError(f"form must be 'soc' or 'pyramid', got {form!r}")
        return gi

    def set_collision_environment(self, spheres=None, capsules=None,
                                  cuboids=None, planes=None):
        """Upload the runtime obstacle set for the COLLISION clearance rows
(CL-2). Lists of tuples, one per obstacle (all in world frame, meters):

        Measured defaults, rulings and provenance: docs/constraints.md (`set_collision_environment`)."""
        def arr(x, w):
            a = np.ascontiguousarray(np.asarray([] if x is None else x, dtype=np.float32))
            return a.reshape(-1, w) if a.size else np.zeros((0, w), dtype=np.float32)
        self.solver.set_collision_environment(arr(spheres, 4), arr(capsules, 7),
                                              arr(cuboids, 15), arr(planes, 4))

    def enable_collision(self, mech=None, margin=0.02, rho=None, delta=0.05,
                         sigma=0.0, knot_lo=1, admm_iters=0):
        """Append THE collision clearance group (one max): per-sphere rows
d_i(q_k) >= margin over knots [knot_lo, N] — d_i = signed clearance of
collision sphere i (baked at codegen, ``collision_res``) to the
nearest obstacle from set_collision_environment (call that FIRST).
The covering spheres are already conservative (inflated by the
spherizer), so margin is extra safety on top.

        Measured defaults, rulings and provenance: docs/constraints.md (`enable_collision`)."""
        if mech is None:
            mech = self._row_mech or "telemetry"
        if mech not in self._MECHS:
            raise ValueError(f"mech must be one of {sorted(self._MECHS)}, got {mech!r}")
        if rho is None:
            # admm 1.0 (NOT the cone/box 0.01): clearance rows fold onto the
            # Q block — see the docstring's 2b binding paragraph
            rho = {"telemetry": 0.0, "barrier": 3e-3, "admm": 1.0, "al": 1.0}[mech]
        self.solver.enable_collision(self._MECHS[mech], float(margin), float(rho),
                                     float(delta), float(sigma), int(knot_lo),
                                     int(admm_iters))
        self._n_appended_groups += 1

    def get_collision_row_duals(self):
        """COLLISION-band AL duals: dict of (B, N, n_spheres) arrays
        (lam_hi unused — the rows are one-sided; lam_lo is the hinge dual)."""
        return self.solver.get_collision_row_duals()

    def get_collision_admm_state(self):
        """COLLISION-band ADMM (z, y): dict of (B, N, n_spheres) arrays."""
        return self.solver.get_collision_admm_state()

    def set_cost_weights(self, q_cost=None, qd_cost=None, u_cost=None, N_cost=None,
                         q_lim_cost=None, vel_lim_cost=None, ctrl_lim_cost=None):
        """Update scalar cost weights at runtime (None keeps the current value)."""
        w = self._cost_weights
        for name, val in (("q_cost", q_cost), ("qd_cost", qd_cost), ("u_cost", u_cost),
                          ("N_cost", N_cost), ("q_lim_cost", q_lim_cost),
                          ("vel_lim_cost", vel_lim_cost), ("ctrl_lim_cost", ctrl_lim_cost)):
            if val is not None:
                w[name] = float(val)
        self.solver.set_cost_weights(w["q_cost"], w["qd_cost"], w["u_cost"], w["N_cost"],
                                     w["q_lim_cost"], w["vel_lim_cost"], w["ctrl_lim_cost"])
        self.params = self.params.replace(**w)   # self.params stays truthful

    def set_q_pos_cost(self, weight):
        """Joint-posture nullspace anchor: adds 0.5*weight*||q - q_nom||^2 as a RUNNING
        cost on the q-block (default 0 = the historic EE-only cost, bitwise-off).

        The EE-position cost leaves a nullspace on redundant arms (iiwa14: wrist roll
        j7 has a ~zero EE-position Jacobian column); by default only the q-limit
        barrier anchors it. Where the operating posture sits near/outside the
        margin-shrunk limits (barrier limits = URDF limits - 0.1 rad, plant.cuh
        JOINT_LIMIT_MARGIN), set q_lim_cost=0 and use a small q_pos_cost toward
        set_q_nom(x0[:nq]) instead — anchors without the barrier field.

        ``weight`` may be a scalar or a length-nq array (per-joint anchor
        stiffness — tune light joints independently; see set_u_cost_vec for the
        matching effort-side knob and the closed-loop rate-limit story).
        """
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim == 0:
            self.solver.set_q_pos_cost_vec(np.empty(0, dtype=np.float32))  # back to scalar
            self.solver.set_q_pos_cost(float(w))
        else:
            self.solver.set_q_pos_cost_vec(np.ascontiguousarray(w.reshape(-1)))

    def set_u_cost_vec(self, weights=None):
        """Per-joint control effort weights (length n_actuated; None resets to the
        scalar u_cost). The per-joint knob that makes a joint's commanded
        correction respect a discrete control loop: with effort nearly free
        (u_cost=1e-6 default) the optimal posture/velocity correction is
        deadbeat-aggressive, and on a near-massless joint (iiwa14 j7,
        J_eff ~ 1e-3 kg m^2) a plant-model inertia mismatch turns the 100 Hz
        loop into a growing Nyquist oscillation (PDDP round-5, 2026-08-02).
        Raising ONLY that joint's effort weight (e.g. [1e-6]*6 + [3e-3]) softens
        its channel without degrading the arm's tracking."""
        if weights is None:
            self.solver.set_u_cost_vec(np.empty(0, dtype=np.float32))
        else:
            w = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).reshape(-1))
            self.solver.set_u_cost_vec(w)

    def set_q_nom(self, q_nom=None):
        """Posture target for set_q_pos_cost (length-nq array; None resets to zeros)."""
        if q_nom is None:
            self.solver.set_q_nom(np.empty(0, dtype=np.float32))
        else:
            q = np.ascontiguousarray(np.asarray(q_nom, dtype=np.float32).ravel())
            self.solver.set_q_nom(q)

    def set_fc_cost(self, weight):
        """Contact-force regularization 0.5*weight*||f_c||^2 per knot (GATO_CONTACT_FORCES
        modules only — the fc slots appended to every control; inert on default builds).
        Default on fc builds is 1e-2, NOT 0: unregularized fc is a free 6-DoF wrench
        actuator and the solve destabilizes (1e-3 is already marginal — measured
        2026-08-02). Use large weights (~1e6) to effectively pin fc to zero, or 0
        only together with fc box rows (add_fc_box)."""
        self.solver.set_fc_cost(float(weight))

    def set_fc_ref(self, ref=None):
        """Contact-wrench reference for the fc regularization (GATO_CONTACT_FORCES
        modules only): the fc cost becomes 0.5*fc_cost*||f_c - ref||^2 per running
        knot. ``ref`` has n_fc entries (wrench layout [n; f] per contact frame,
        world-aligned at the baked frame), shared across knots and batch rows.
        None/empty resets to zeros — bitwise the historic pure regularization.

        This is how a force SETPOINT enters the solve: ref = the desired contact
        wrench (e.g. [0,0,0, 0,0,+F] for an F-newton press reaction), fc_cost =
        the force-tracking weight. Pair with cone rows on the fc columns
        (add_lin_u_rows / enable_u_cone with a selection C) for friction limits."""
        if self.n_fc == 0:
            raise RuntimeError("set_fc_ref needs a GATO_CONTACT_FORCES build "
                               "(this module has no fc slots)")
        if ref is None:
            self.solver.set_fc_ref(np.empty(0, dtype=np.float32))
            return
        r = np.ascontiguousarray(np.asarray(ref, dtype=np.float32).ravel())
        if r.size != self.n_fc:
            raise ValueError(f"fc_ref must have n_fc = {self.n_fc} entries, got {r.size}")
        self.solver.set_fc_ref(r)

    def set_cost_weights_per_knot(self, knot_weights):
        """Per-knot [ee, qd, u] weight triples, shape (N, 3): overrides the scalar
        q/qd/u/N weights (terminal EE weight = row N-1's ee entry). Enables
        via-points, terminal-only goals, and horizon masking at runtime.
        ``None`` clears the table (back to the scalar weights)."""
        if knot_weights is None:
            self.solver.clear_cost_weights_per_knot()
            return
        w = np.ascontiguousarray(np.asarray(knot_weights, dtype=np.float32)).reshape(self.N, 3)
        self.solver.set_cost_weights_per_knot(w)

    # ---- lazy pinocchio model (FK helpers / sim / estimators / tests) ----
    @property
    def model(self):
        """pinocchio Model for model_path (built on first use; needs the [test] extra)."""
        if self._model is None:
            pin = _require_pin()
            if self.floating_base:
                # floating-base modules (CL-3): the stored state carries the
                # free-flyer q [p; quat xyzw] — mirror it in the pin model so
                # nq/nv/nx and every oracle computation match the module.
                self._model = pin.buildModelFromUrdf(self.model_path, pin.JointModelFreeFlyer())
            else:
                self._model = pin.buildModelFromUrdf(self.model_path)
        return self._model

    @property
    def data(self):
        if self._data is None:
            self._data = self.model.createData()
        return self._data

    @property
    def ee_frame_id(self):
        """Frame id of the codegen'd EE target frame, or None if the URDF lacks it."""
        if self._ee_frame_id is None:
            m = self.model
            self._ee_frame_id = m.getFrameId(self.ee_frame) if m.existFrame(self.ee_frame) else -1
        return None if self._ee_frame_id == -1 else self._ee_frame_id

    def ee_pos(self, q, frame="ee"):
        """EE position via pinocchio FK.

        frame="ee": the URDF ee_frame (fixed-joint child, e.g. tcp). Since
        GRiD e31f7bd the device FK (tracking cost AND EE row-groups) uses the
        named-target ``*_EE`` codegen, which INCLUDES the terminal fixed-joint
        origin — device == this frame to f32 precision (~1e-7).
        frame="solver": historical alias for the device frame; now identical
        to "ee" (the old dropped-origin convention is gone upstream).
        """
        pin = _require_pin()
        pin.forwardKinematics(self.model, self.data, q)
        if self.ee_frame_id is None:
            return np.array(self.data.oMi[self.model.njoints - 1].translation)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return np.array(self.data.oMf[self.ee_frame_id].translation)

    def reset(self):
        """Clear all solver state carried across solves: duals (AL/ADMM), the
        adapted trust-region rho, and the external-force buffer. (Trajectory
        warm starts are the caller's — see solve().)"""
        self.reset_dual()
        self.reset_rho()  # adapted rho is solver state -> a full reset must clear it too
        self.set_f_ext_B(np.zeros((self.batch_size, 6)))

    def sim_forward(self, xk, uk, sim_dt):
        """One plant step on the DEVICE integrator for every batch entry:
        (B, nx) x (B, nu) -> (B, nx) (knot-0 external wrench applies)."""
        xk = np.asarray(xk, dtype=np.float32)
        uk = np.asarray(uk, dtype=np.float32)
        return self.solver.sim_forward(xk, uk, sim_dt)

    def set_f_ext_B(self, f_ext_B):
        """Upload the external-wrench band the dynamics see.

        The GPU buffer is body-major: 6*NUM_BODIES per (solve, knot); each 6-slot is
        that body's spatial force in its JOINT-LOCAL frame about the joint origin,
        Featherstone-ordered [angular(3); linear(3)] (verified vs pin.aba 2026-07-07).
        World wrenches go through gato.common.world_wrench_to_joint_local and are
        reordered — see hypotheses.world_wrench_to_gato_slot.

        Accepted shapes (N = knot count; wrench k applies to interval [k, k+1]):
          (B, 6)               per-solve EE wrench, broadcast over knots
          (B, 6*NUM_BODIES)    per-solve body-major, broadcast over knots
          (B, N, 6)            per-knot EE wrench (scattered into the EE body slot)
          (B, N, 6*NUM_BODIES) per-knot body-major
        A correctly-sized contiguous buffer is always uploaded (a short buffer
        would make the device copy over-read -> garbage wrench -> NaN dynamics)."""
        f_ext_B = np.asarray(f_ext_B, dtype=np.float32)
        nb = self.n_bodies or self.nv
        per_knot = (f_ext_B.ndim == 3)
        if per_knot:
            if f_ext_B.shape[:2] != (self.batch_size, self.N):
                raise ValueError(
                    f"per-knot f_ext_B must be (batch, N, ...) = ({self.batch_size}, "
                    f"{self.N}, ...); got {f_ext_B.shape}"
                )
            width = f_ext_B.shape[2]
            body_major = np.zeros((self.batch_size, self.N, 6 * nb), dtype=np.float32)
        else:
            f_ext_B = f_ext_B.reshape(self.batch_size, -1)
            width = f_ext_B.shape[1]
            body_major = np.zeros((self.batch_size, 6 * nb), dtype=np.float32)
        self.f_ext_B = f_ext_B
        if width == 6:
            ee = 6 * (nb - 1)  # end-effector body slot
            body_major[..., ee:ee + 6] = f_ext_B
        elif width == 6 * nb:
            body_major[...] = f_ext_B
        else:
            raise ValueError(
                f"f_ext_B must have width 6 (EE wrench) or {6 * nb} (body-major "
                f"6*NUM_BODIES); got {width}"
            )
        if per_knot:
            self.solver.set_f_ext_knot_batch(np.ascontiguousarray(body_major))
        else:
            self.solver.set_f_ext_batch(np.ascontiguousarray(body_major))
        
    def reset_rho(self):
        self.solver.reset_rho()

    def reset_dual(self):
        self.solver.reset_dual()

