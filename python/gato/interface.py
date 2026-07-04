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
        rho=0.0,
        rho_batch=None,
        mu_batch=None,
        pcg_tol_batch=None,
        adapt_rho=True,
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
        )
        return SolveResult(xu=self.XU_B, solve_time_us=stats.solve_time_us,
                           stats=stats, nx=self.nx, nu=self.nu, N=self.N)

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

    def ee_pos(self, q):
        # Measure the SAME frame the solver optimizes (URDF "EE"), not the last
        # joint origin, so the success/tracking metric matches the cost the solver
        # actually minimizes (see ee_frame_id resolution in __init__).
        pin = _require_pin()
        pin.forwardKinematics(self.model, self.data, q)
        if self.ee_frame_id is not None:
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            return self.data.oMf[self.ee_frame_id].translation
        return self.data.oMi[self.model.njoints - 1].translation

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
        # The GPU wrench buffer is body-major: 6*NUM_BODIES per solve. Accept either
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

