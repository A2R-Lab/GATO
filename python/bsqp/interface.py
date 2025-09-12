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
        use_force_estimator: bool = True,
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
        # Ensure estimator is off for batch_size == 1
        if self.batch_size == 1:
            use_force_estimator = False
        # Inform backend
        try:
            self.solver.set_use_force_estimator(bool(use_force_estimator))
        except AttributeError:
            # Older builds may not have this method; ignore silently
            pass
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
        }

    def solve(self, xcur, eepos_goals, XU=None):
        """Solve using single-trajectory inputs. Internally replicated on device.

        Returns the stats dict with timings/iters; trajectory selection happens via `select_best_and_copy`.
        """
        xcur = np.asarray(xcur, dtype=np.float32)
        eepos_goals = np.asarray(eepos_goals, dtype=np.float32)
        if XU is None or (hasattr(XU, 'ndim') and np.asarray(XU).ndim == 2):
            XU = np.zeros(self.N * (self.nx + self.nu) - self.nu, dtype=np.float32)
            for i in range(self.N):
                k0 = i * (self.nx + self.nu)
                XU[k0:k0 + self.nx] = xcur
        else:
            XU = np.asarray(XU, dtype=np.float32)

        XU[: self.nx] = xcur
        result = self.solver.solve_single(XU, self.dt, xcur, eepos_goals)
        # self.stats["sqp_time_us"] = result["sqp_time_us"]
        self.stats["sqp_iters"] = result["sqp_iters"]
        # self.stats["kkt_converged"] = result["kkt_converged"]
        # self.stats["pcg_iters"] = result["pcg_iters"]
        # self.stats["pcg_times_us"] = result["pcg_times_us"]
        # self.stats["min_merit"] = result["ls_min_merit"]
        # self.stats["step_size"] = result["ls_step_size"]
        return result

#     def solve_existing(self, xcur):
#         xcur = np.asarray(xcur, dtype=np.float32)
#         result = self.solver.solve_existing(xcur, self.dt)
#         self.stats["sqp_time_us"] = result["sqp_time_us"]
#         self.stats["sqp_iters"] = result["sqp_iters"]
#         self.stats["kkt_converged"] = result["kkt_converged"]
#         return result

    def solve_existing_with_ref(self, xcur, eepos_goals):
        xcur = np.asarray(xcur, dtype=np.float32)
        eepos_goals = np.asarray(eepos_goals, dtype=np.float32)
        result = self.solver.solve_existing_with_ref(xcur, eepos_goals, self.dt)
        # self.stats["sqp_time_us"] = result["sqp_time_us"]
        self.stats["sqp_iters"] = result["sqp_iters"]
        # self.stats["kkt_converged"] = result["kkt_converged"]
        return result

#     def get_best_idx(self, x_last, u_last, x_curr, dt):
#         """Simple evaluation of best trajectory based on forward simulation.
        
#         Note: This is kept for backward compatibility but is typically not used
#         when the force estimator is integrated at the MPC level.
#         """
#         if self.batch_size == 1:
#             return 0

#         best_err, best_id = np.inf, 0
#         x_next_B = self.sim_forward(x_last, u_last, dt)
#         for i in range(self.batch_size):
#             err = np.linalg.norm(x_next_B[i, :] - x_curr)
#             if err < best_err:
#                 best_err = err
#                 best_id = i

#         return best_id

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

    def select_best_and_copy(self, x_last, u_last, x_curr, dt):
        """Post-solve: compute best hypothesis on device, replicate best trajectory across batch, and return it."""
        x_last = np.asarray(x_last, dtype=np.float32)
        u_last = np.asarray(u_last, dtype=np.float32)
        x_curr = np.asarray(x_curr, dtype=np.float32)
        xu_best = self.solver.post_solve_select_and_copy(x_last, u_last, x_curr, dt)
        return xu_best

    def set_f_ext_B(self, f_ext_B):
        self.f_ext_B = np.asarray(f_ext_B, dtype=np.float32)
        self.solver.set_f_ext_batch(self.f_ext_B)

    def set_use_force_estimator(self, enabled: bool):
        try:
            self.solver.set_use_force_estimator(bool(enabled))
        except AttributeError:
            raise RuntimeError("Backend does not support toggling force estimator. Rebuild required.")

#     def set_wrench_transform(self, A6):
#         A6 = np.asarray(A6, dtype=np.float32).reshape(6, 6)
#         self.solver.set_wrench_transform(A6)

    def reset_dual(self):
        self.solver.reset_dual()

    def get_stats(self):
        return self.stats
