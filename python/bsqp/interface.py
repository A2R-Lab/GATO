import os
import sys
import importlib
import numpy as np
import torch
import pinocchio as pin

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
        solve_ratio=0.875,
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
    ):
        # Dynamically import the correct bsqp_N* module and get the solver class
        # The modules should be named like 'bsqp_N16', etc., and contain classes like 'BSQP_4', etc.

        # Build the module name for the given N
        module_name = f"bsqp.bsqpN{N}"
        try:
            base = importlib.import_module(module_name)
        except ImportError as e:
            raise ValueError(
                f"Number of knots {N} not supported (could not import {module_name}): {e}"
            )

        # Build the class name for the given batch_size
        class_name = f"BSQP_{batch_size}_float"
        if not hasattr(base, class_name):
            raise ValueError(
                f"Batch size {batch_size} not supported in module {module_name}"
            )
        self.lib = base
        self.solver_class = getattr(base, class_name)

        self.solver = self.solver_class(
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
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
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

        self.stats = {
            "sqp_time_us": np.array([]),
            "sqp_iters": np.array([]),
            "kkt_converged": np.array([]),
            "pcg_iters": np.array([np.zeros(self.batch_size)]),
            "pcg_times_us": np.array([np.zeros(self.batch_size)]),
            "min_merit": np.array([np.zeros(self.batch_size)]),
            "step_size": np.array([np.zeros(self.batch_size)]),
            "initial_merit": np.array([]),
            "best_initial_merit": np.array([]),
        }

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
        # Ensure float32 inputs for CUDA bindings
        xcur_B = np.asarray(xcur_B, dtype=np.float32)
        eepos_goals_B = np.asarray(eepos_goals_B, dtype=np.float32)
        if XU_B is None:
            XU_B = self.XU_B
        else:
            XU_B = np.asarray(XU_B, dtype=np.float32)
        XU_B[:, : self.nx] = xcur_B

        result = self.solver.solve(XU_B, self.dt, xcur_B, eepos_goals_B)

        # Copy raw results
        self.XU_B = np.asarray(result["XU"], dtype=np.float32)
        self.stats["sqp_time_us"] = int(result["sqp_time_us"]) if not isinstance(result["sqp_time_us"], np.ndarray) else int(result["sqp_time_us"])  # scalar us
        self.stats["sqp_iters"] = np.asarray(result["sqp_iters"], dtype=np.int32).reshape(self.batch_size)
        self.stats["kkt_converged"] = np.asarray(result["kkt_converged"], dtype=np.int32).reshape(self.batch_size)
        self.stats["final_merit"] = np.asarray(result["final_merit"], dtype=np.float32).reshape(self.batch_size)
        if "initial_merit" in result:
            self.stats["initial_merit"] = np.asarray(result["initial_merit"], dtype=np.float32).reshape(self.batch_size)
            # For consistency with the "best across batch" curve, derive the baseline as the best (min) initial merit across the batch
            if self.stats["initial_merit"].size:
                self.stats["best_initial_merit"] = float(np.min(self.stats["initial_merit"]))
            else:
                self.stats["best_initial_merit"] = np.array([], dtype=np.float32)
        self.stats["ls_num_iters"] = int(result.get("ls_num_iters", 0))

        # Normalize shapes for per-iteration stats
        def _to_np(a, dtype=None):
            arr = np.asarray(a)
            if arr.dtype == object:
                try:
                    arr = np.asarray([np.asarray(v).reshape(-1) for v in arr.tolist()])
                except Exception:
                    arr = np.array([], dtype=(np.float32 if dtype is None else dtype))
            if dtype is not None and arr.dtype != dtype:
                arr = arr.astype(dtype)
            return arr

        num_iters = self.stats["ls_num_iters"]
        min_merit = _to_np(result.get("ls_min_merit", np.array([])), dtype=np.float32)
        step_size = _to_np(result.get("ls_step_size", np.array([])), dtype=np.float32)
        pcg_iters = _to_np(result.get("pcg_iters", np.array([])), dtype=np.int32)
        pcg_times = _to_np(result.get("pcg_times_us", np.array([])), dtype=np.float32)

        # Ensure shapes: (iters, B) for min_merit/step_size/pcg_iters; (iters,) for pcg_times
        if min_merit.size:
            if min_merit.ndim == 1 and self.batch_size == 1 and num_iters == min_merit.shape[0]:
                min_merit = min_merit.reshape(num_iters, 1)
            elif min_merit.ndim == 2 and min_merit.shape[0] != num_iters and min_merit.size == num_iters * self.batch_size:
                min_merit = min_merit.reshape(num_iters, self.batch_size)
        if step_size.size:
            if step_size.ndim == 1 and self.batch_size == 1 and num_iters == step_size.shape[0]:
                step_size = step_size.reshape(num_iters, 1)
            elif step_size.ndim == 2 and step_size.shape[0] != num_iters and step_size.size == num_iters * self.batch_size:
                step_size = step_size.reshape(num_iters, self.batch_size)
        if pcg_iters.size:
            if pcg_iters.ndim == 1 and self.batch_size == 1 and num_iters == pcg_iters.shape[0]:
                pcg_iters = pcg_iters.reshape(num_iters, 1)
            elif pcg_iters.ndim == 2 and pcg_iters.shape[0] != num_iters and pcg_iters.size == num_iters * self.batch_size:
                pcg_iters = pcg_iters.reshape(num_iters, self.batch_size)

        self.stats["pcg_iters"] = pcg_iters
        self.stats["pcg_times_us"] = pcg_times
        self.stats["min_merit"] = min_merit
        self.stats["step_size"] = step_size

        # Derive per-iteration aggregates based on actual returned shapes
        ls_np = self.stats.get("min_merit", np.array([]))
        if isinstance(ls_np, list):
            ls_np = np.asarray(ls_np, dtype=np.float32)
        if isinstance(ls_np, np.ndarray) and ls_np.size and ls_np.ndim == 2:
            best_per_iter = np.min(ls_np.astype(np.float32), axis=1)
            self.stats["best_merit_per_iter"] = best_per_iter
            self.stats["best_merit_iter1"] = float(best_per_iter[0]) if best_per_iter.size > 0 else float("nan")
        else:
            self.stats["best_merit_per_iter"] = np.array([], dtype=np.float32)
            self.stats["best_merit_iter1"] = float("nan")

        # If initial baseline is present, provide a normalized curve convenience entry
        if np.size(self.stats.get("best_initial_merit", [])):
            denom = float(self.stats["best_initial_merit"]) if self.stats["best_initial_merit"] else None
            if denom and denom != 0 and self.stats["best_merit_per_iter"].size:
                self.stats["best_merit_per_iter_normalized"] = self.stats["best_merit_per_iter"] / denom
            else:
                self.stats["best_merit_per_iter_normalized"] = self.stats["best_merit_per_iter"]
        else:
            self.stats["best_merit_per_iter_normalized"] = self.stats["best_merit_per_iter"]

        return self.XU_B, result["sqp_time_us"]

    def get_best_idx(self, x_last, u_last, x_curr, dt):
        """Simple evaluation of best trajectory based on forward simulation.
        
        Note: This is kept for backward compatibility but is typically not used
        when the force estimator is integrated at the MPC level.
        """
        if self.batch_size == 1:
            return 0

        best_err, best_id = np.inf, 0
        x_next_B = self.sim_forward(x_last, u_last, dt)
        for i in range(self.batch_size):
            err = np.linalg.norm(x_next_B[i, :] - x_curr)
            if err < best_err:
                best_err = err
                best_id = i

        return best_id

    def ee_pos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[self.model.njoints - 1].translation

    def reset(self):
        self.reset_dual()
        self.set_f_ext_B(np.zeros((self.batch_size, 6)))
        self.XU_B = np.zeros((self.batch_size, self.N * (self.nx + self.nu) - self.nu))

    def sim_forward(self, xk, uk, sim_dt):
        xk = np.asarray(xk, dtype=np.float32)
        uk = np.asarray(uk, dtype=np.float32)
        return self.solver.sim_forward(xk, uk, sim_dt)

    def set_f_ext_B(self, f_ext_B):
        self.f_ext_B = np.asarray(f_ext_B, dtype=np.float32)
        self.solver.set_f_ext_batch(self.f_ext_B)

    def reset_dual(self):
        self.solver.reset_dual()

    def get_stats(self):
        return self.stats

    # Hyperparameter setters
    def set_rho_penalty_batch(self, rho_batch, set_as_reset_default=True):
        rho_batch = np.asarray(rho_batch, dtype=np.float32).reshape(self.batch_size)
        self.solver.set_rho_penalty_batch(rho_batch, set_as_reset_default)

    def set_drho_batch(self, drho_batch, set_as_reset_default=True):
        drho_batch = np.asarray(drho_batch, dtype=np.float32).reshape(self.batch_size)
        self.solver.set_drho_batch(drho_batch, set_as_reset_default)

    def set_mu_batch(self, mu_batch):
        mu_batch = np.asarray(mu_batch, dtype=np.float32).reshape(self.batch_size)
        self.solver.set_mu_batch(mu_batch)

    def set_pcg_tol_batch(self, tol_batch):
        tol_batch = np.asarray(tol_batch, dtype=np.float32).reshape(self.batch_size)
        self.solver.set_pcg_tol_batch(tol_batch)
