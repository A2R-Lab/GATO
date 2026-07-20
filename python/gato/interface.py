import glob
import importlib
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .common import _require_pin


def available():
    """Discover built solver modules in this package: {(plant, N): filename}."""
    here = os.path.dirname(os.path.abspath(__file__))
    found = {}
    for so in sorted(glob.glob(os.path.join(here, "bsqpN*_*.so"))):
        m = re.match(r"bsqpN(\d+)_([A-Za-z0-9_]+?)\.", os.path.basename(so))
        if m:
            found[(m.group(2), int(m.group(1)))] = os.path.basename(so)
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
    nu: int
    N: int

    @property
    def batch_size(self):
        return self.xu.shape[0]

    def xu_b(self, b=0):
        """Trajectory of batch entry b (flat)."""
        return self.xu[b]

    def u0(self, b=0):
        """First control of batch entry b — what an MPC applies."""
        return self.xu[b, self.nx:self.nx + self.nu]

    def control_at(self, k, b=0):
        """Control at knot k (clamped to the last control knot) of batch entry b."""
        k = min(int(k), self.N - 2)  # last knot has no control
        start = self.nx + (self.nx + self.nu) * k
        return self.xu[b, start:start + self.nu]

    @property
    def diverged(self):
        """(B,) bool — non-finite merit or trajectory."""
        bad_merit = ~np.isfinite(self.stats.final_merit)
        bad_xu = ~np.isfinite(self.xu).all(axis=1)
        return bad_merit | bad_xu


class BSQP:
    def __init__(
        self,
        model_path,
        batch_size,
        N,
        dt,
        max_sqp_iters=10,
        kkt_tol=1e-4,
        max_pcg_iters=100,
        pcg_tol=1e-4,
        solve_ratio=1.0,
        mu=1.0,
        q_cost=2.0,
        qd_cost=1e-4,
        u_cost=1e-6,
        N_cost=50.0,
        q_lim_cost=1e-3,
        vel_lim_cost=0.0,
        ctrl_lim_cost=0.0,
        rho=1e-3,  # trust-region floor: f32 bdsv paths return garbage steps at rho=0 (R1)
        rho_batch=None,
        mu_batch=None,
        pcg_tol_batch=None,
        adapt_rho=True,
        linsys="pcg",  # "pcg" | "bdsv" | "bdsv_first" (see LINSYS_MODES)
        plant_type='indy7',  # 'indy7' or 'iiwa14'
    ):
        # Dynamically import the correct bsqp_N* module and get the solver class
        # The modules should be named like 'bsqpN{N}_{plant_type}', e.g., 'bsqpN32_indy7'
        
        # Auto-detect plant type from model_path if not explicitly specified.
        # Unknown robots are a hard error (a wrong plant silently runs the wrong
        # dynamics with mismatched state size).
        if plant_type is None:
            from .builder import load_registry
            plants = sorted({p for p, _ in available()} | set(load_registry()))
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

        # Build the module name for the given N and plant
        module_name = f"gato.bsqpN{N}_{plant_type}"
        try:
            base = importlib.import_module(module_name)
        except ImportError as e:
            raise ValueError(
                f"No compiled module for plant={plant_type!r}, N={N} "
                f"(could not import {module_name}): {e}\n"
                f"Built modules: {sorted(available()) or 'none'} — build with, e.g.:\n"
                f"  cmake -S . -B build -DPLANT={plant_type} -DKNOTS={N} && "
                f"cmake --build build --parallel 4"
            )

        # batch_size is a runtime constructor argument (one class per precision)
        class_name = "BSQP_float"
        if not hasattr(base, class_name):
            raise ValueError(
                f"Module {module_name} does not export {class_name} — rebuild the "
                f"solver modules (old per-batch-size builds are incompatible)"
            )
        self.lib = base
        self.solver_class = getattr(base, class_name)
        self.plant_type = plant_type
        # Body-major external-force buffer width is 6*NUM_BODIES per solve (the GPU
        # d_f_ext_batch_ buffer + set_f_ext_batch upload are sized to this). Exposed
        # by the module; fall back to the pinocchio nv (== NUM_BODIES for a fixed
        # serial chain) for older modules that don't export it.
        self.n_bodies = int(getattr(base, "NUM_BODIES", 0))

        self._cost_weights = dict(q_cost=q_cost, qd_cost=qd_cost, u_cost=u_cost,
                                  N_cost=N_cost, q_lim_cost=q_lim_cost,
                                  vel_lim_cost=vel_lim_cost, ctrl_lim_cost=ctrl_lim_cost)
        self.solver = self.solver_class(
            batch_size,
            dt,
            max_sqp_iters,
            kkt_tol,
            max_pcg_iters,
            pcg_tol,
            solve_ratio,
            mu,
            q_cost,
            qd_cost,
            u_cost,
            N_cost,
            q_lim_cost,
            vel_lim_cost,
            ctrl_lim_cost,
            rho,  # rho
        )
        pin = _require_pin()
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
        # The solver/grid.cuh optimizes the EE-position cost in the frame the module
        # was codegen'd with (fixed_target_name; recorded in the registry, "EE" for
        # the vendored robots). The last JOINT origin (oMi[njoints-1]) can sit several
        # cm short of it, so the success metric MUST use the same frame or it reports
        # spurious tracking error. Resolve the frame id once; fall back to the last
        # joint only if the URDF lacks the frame.
        self.ee_frame = robot_info(plant_type).get("ee_frame", "EE")
        if self.model.existFrame(self.ee_frame):
            self.ee_frame_id = self.model.getFrameId(self.ee_frame)
        else:
            self.ee_frame_id = None
        self.batch_size = batch_size
        self.N = N
        self.dt = dt
        self.f_ext_B = np.zeros((self.batch_size, 6), dtype=np.float32)
        self.set_f_ext_B(self.f_ext_B)

        self.nx = self.model.nq + self.model.nv
        self.nu = self.model.nv
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.XU_B = np.zeros(
            (self.batch_size, self.N * (self.nx + self.nu) - self.nu),
            dtype=np.float32,
        )

        # Optional batched hyperparameters
        if rho_batch is not None:
            rho_batch = np.asarray(rho_batch, dtype=np.float32).reshape(self.batch_size)
            self.solver.set_rho_penalty_batch(rho_batch, True)
        # Control whether line-search adapts rho or keeps per-batch rho fixed
        self.solver.set_rho_adaptation(bool(adapt_rho))
        if mu_batch is not None:
            mu_batch = np.asarray(mu_batch, dtype=np.float32).reshape(self.batch_size)
            self.solver.set_mu_batch(mu_batch)
        if pcg_tol_batch is not None:
            pcg_tol_batch = np.asarray(pcg_tol_batch, dtype=np.float32).reshape(self.batch_size)
            self.solver.set_pcg_tol_batch(pcg_tol_batch)
        self.max_pcg_iters = int(max_pcg_iters)
        self.linsys = "pcg"  # the C++ default; set_linsys only calls into the module on change
        self.set_linsys(linsys)
        self._row_mech = None  # active enable_limit_* mode (add_lin_u_rows mech=None default)

    def set_linsys(self, mode):
        """Pick the S·λ = γ path for subsequent solves: "pcg" | "bdsv" | "bdsv_first".

        Host-side and zero-cost — an MPC loop can switch it per step.
        """
        if mode not in LINSYS_MODES:
            raise ValueError(f"linsys must be one of {sorted(LINSYS_MODES)}, got {mode!r}")
        if mode != self.linsys:
            self.solver.set_linsys_mode(LINSYS_MODES[mode])
            self.linsys = mode

    def solve(self, xcur_B, eepos_goals_B, XU_B=None):
        """Solve the batch; returns a SolveResult (also stores xu as the next warm start)."""
        xcur_B = np.asarray(xcur_B, dtype=np.float32)
        eepos_goals_B = np.asarray(eepos_goals_B, dtype=np.float32)
        if XU_B is None:
            XU_B = self.XU_B
        else:
            XU_B = np.asarray(XU_B, dtype=np.float32)
        XU_B[:, : self.nx] = xcur_B

        raw = self.solver.solve(XU_B, self.dt, xcur_B, eepos_goals_B)

        self.XU_B = np.asarray(raw["XU"], dtype=np.float32)
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
        return SolveResult(xu=self.XU_B, solve_time_us=stats.solve_time_us,
                           stats=stats, nx=self.nx, nu=self.nu, N=self.N)

    # rows::Mechanism enum values (rowgroups.cuh)
    _MECHS = {"telemetry": 0, "barrier": 1, "admm": 2, "al": 3}

    def enable_limit_telemetry(self):
        """Install the canonical limit row-groups (position/velocity/torque boxes
        from the URDF limit tables) in TELEMETRY mode: every solve() reports each
        group's true violation of the returned trajectory in
        ``stats.row_{max,sum}_violation`` (group order BOX_Q, BOX_QD, BOX_U).
        Telemetry never touches the solver path — trajectories are bit-identical
        with it on or off. Part of the constraint row-group layer (CL-0)."""
        self.solver.enable_limit_telemetry()
        self._row_mech = "telemetry"

    def enable_limit_barrier(self, mu=3e-3, delta=0.05):
        """Bind the limit row-groups to the RELAXED log-barrier mechanism: a
        C² barrier with bounded Hessian (quadratic extension within ``delta``
        of a bound) folded into the KKT cost and merit — infeasible-start safe,
        the constraint layer's soft prior mode. Additive to grid_plant's own
        clamped log barriers; zero q_lim/vel_lim/ctrl_lim_cost for a clean
        comparison. Telemetry (stats.row_*_violation) stays on."""
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
        Duals warm-start across solves (reset_dual() reinitializes) —
        EXCEPT equality rows (lo == hi, e.g. enable_ee_terminal_equality),
        whose (z, y) reinit every solve: a warm-started dual on a row the
        primal may not reach is an unbounded violation integrator (measured).
        stats gain admm_r_prim/admm_r_dual; telemetry stays on."""
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

    def disable_row_groups(self):
        """Remove all constraint row-groups (stats lose the row_* fields)."""
        self.solver.disable_row_groups()
        self._row_mech = None

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
        semantics; only read while ADMM mode is active."""
        self.solver.set_admm_merit(bool(on))

    def add_lin_u_rows(self, C, d=None, lo=None, hi=None, mech=None, rho=None,
                       delta=0.05, sigma=0.0, cone=False, knot_lo=0,
                       knot_hi=None, admm_iters=0):
        """Append a LIN_U row-group: m rows ``g = C @ u + d`` on the control
        block (C shape (m, nu), FROZEN at a host-chosen configuration — the
        cross-term audit's contact-frame rule for config-dependent maps).

        ``cone=True`` binds SECOND-ORDER-CONE semantics to the row vector
        (row 0 = axis t, rows 1.. = x-bar; feasible iff ||x-bar|| <= t;
        lo/hi unused): ADMM z-update = SOC projection (``admm_soc``), AL =
        conic PHR (dual vector projected onto K each outer update; hard-only),
        barrier = relaxed barrier on the margin t - ||x-bar||. ``cone=False``
        keeps interval semantics on the mapped rows (lo/hi required) — the
        pyramid-facet path.

        ``mech`` is "telemetry" | "barrier" | "admm" | "al" (None = follow the
        active enable_limit_* mode). Mixing mechanisms across groups composes
        (e.g. AL boxes + ADMM cone). Call AFTER enable_limit_* — mechanism
        enables reinstall the canonical groups and drop appended ones.
        ``rho`` defaults per mechanism (provisional until the R2 round binds
        them: admm 0.01, al 1.0, barrier 3e-3) — the rho-scale law applies:
        the fold lands rho * C^T C on the R block, so scale rho DOWN by
        ||C||^2 when the map is large. Telemetry reports the cone margin
        violation max(0, ||x-bar|| - t) (interval rows: interval violation)."""
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
        d = np.asarray([] if d is None else d, dtype=np.float32).reshape(-1)
        if cone:
            lo_a = np.asarray([], dtype=np.float32)
            hi_a = np.asarray([], dtype=np.float32)
        else:
            if lo is None or hi is None:
                raise ValueError("interval LIN_U rows need lo and hi (length m)")
            lo_a = np.asarray(lo, dtype=np.float32).reshape(m)
            hi_a = np.asarray(hi, dtype=np.float32).reshape(m)
        if knot_hi is None:
            knot_hi = self.N - 1  # no terminal control
        self.solver.add_lin_u_group(self._MECHS[mech], C, d, lo_a, hi_a,
                                    bool(cone), float(rho), float(delta),
                                    float(sigma), int(knot_lo), int(knot_hi),
                                    int(admm_iters))

    def enable_u_cone(self, C, d=None, mech=None, rho=None, form="soc",
                      facets=8, facet_scale="inscribed", **kw):
        """Cone constraint on a mapped control quantity g = C @ u + d
        (CL-2 demo surface: e.g. an EE contact-force friction cone with
        C = S @ pinv(J(q).T), rows [mu*f_n; f_t1; f_t2], frozen at q).

        form="soc": exact second-order cone via add_lin_u_rows(cone=True).
        form="pyramid": m must be 3; the cone is replaced by ``facets``
        one-sided linear rows h_j = cos(th_j) g1 + sin(th_j) g2 - s*g0 <= 0
        riding the ordinary interval machinery (any mechanism, slack toggle
        included). facet_scale="inscribed" (s = cos(pi/facets), conservative:
        facet-feasible => cone-feasible) or "circumscribed" (s = 1, outer
        approximation). Returns the appended group index."""
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

    def set_cost_weights_per_knot(self, knot_weights):
        """Per-knot [ee, qd, u] weight triples, shape (N, 3): overrides the scalar
        q/qd/u/N weights (terminal EE weight = row N-1's ee entry). Enables
        via-points, terminal-only goals, and horizon masking at runtime."""
        w = np.ascontiguousarray(np.asarray(knot_weights, dtype=np.float32)).reshape(self.N, 3)
        self.solver.set_cost_weights_per_knot(w)

    def clear_cost_weights_per_knot(self):
        self.solver.clear_cost_weights_per_knot()

    def ee_pos(self, q, frame="ee"):
        """EE position via pinocchio FK.

        frame="ee": the URDF ee_frame (fixed-joint child, e.g. tcp).
        frame="solver": the LAST MOVING JOINT origin — the frame the device
        FK (tracking cost AND EE row-groups) actually evaluates. MEASURED:
        the generated grid.cuh end_effector_pose drops the terminal
        fixed-joint origin (indy7: 6 cm z, iiwa14: 4 cm z), so these frames
        differ by that constant transform; device == "solver" to f32
        precision (~1e-7). Upstream fix (GCG named-target alias) pending —
        until then EE equality targets must be given in the solver frame.
        """
        pin = _require_pin()
        pin.forwardKinematics(self.model, self.data, q)
        if frame == "solver" or self.ee_frame_id is None:
            return np.array(self.data.oMi[self.model.njoints - 1].translation)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return np.array(self.data.oMf[self.ee_frame_id].translation)

    def reset(self):
        self.reset_dual()
        self.reset_rho()  # adapted rho is solver state -> a full reset must clear it too
        self.set_f_ext_B(np.zeros((self.batch_size, 6)))
        self.XU_B = np.zeros((self.batch_size, self.N * (self.nx + self.nu) - self.nu))

    def sim_forward(self, xk, uk, sim_dt):
        xk = np.asarray(xk, dtype=np.float32)
        uk = np.asarray(uk, dtype=np.float32)
        return self.solver.sim_forward(xk, uk, sim_dt)

    def set_f_ext_B(self, f_ext_B):
        # The GPU wrench buffer is body-major: 6*NUM_BODIES per solve. Each 6-slot is
        # that body's spatial force in its JOINT-LOCAL frame about the joint origin,
        # Featherstone-ordered [angular(3); linear(3)] (verified vs pin.aba 2026-07-07).
        # World wrenches must go through gato.common.world_wrench_to_joint_local and
        # be reordered — see hypotheses.ForceHypothesisBatch._world_to_gato. Accept either
        # a per-solve 6-vector EE wrench (scattered into the end-effector body slot)
        # or a full body-major (batch, 6*NUM_BODIES) array, and always upload a
        # correctly-sized contiguous buffer (a short buffer makes set_f_ext_batch's
        # cudaMemcpy over-read host memory -> garbage wrench -> NaN dynamics).
        f_ext_B = np.asarray(f_ext_B, dtype=np.float32).reshape(self.batch_size, -1)
        self.f_ext_B = f_ext_B
        nb = self.n_bodies or self.model.nv
        body_major = np.zeros((self.batch_size, 6 * nb), dtype=np.float32)
        if f_ext_B.shape[1] == 6:
            ee = 6 * (nb - 1)  # end-effector body slot
            body_major[:, ee:ee + 6] = f_ext_B
        elif f_ext_B.shape[1] == 6 * nb:
            body_major[:] = f_ext_B
        else:
            raise ValueError(
                f"f_ext_B must have width 6 (EE wrench) or {6 * nb} (body-major "
                f"6*NUM_BODIES); got {f_ext_B.shape[1]}"
            )
        self.solver.set_f_ext_batch(np.ascontiguousarray(body_major))
        
    def reset_rho(self):
        self.solver.reset_rho()

    def reset_dual(self):
        self.solver.reset_dual()

