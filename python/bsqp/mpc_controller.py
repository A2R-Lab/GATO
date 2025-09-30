import sys
import time
import numpy as np
import pinocchio as pin
from .interface import BSQP
from .common import rk4, get_ee_position

# Import force estimator if available
sys.path.append('./examples')
try:
    from force_estimator import ForceEstimator
except ImportError:
    ForceEstimator = None


class MPC_GATO:

    def __init__(
        self,
        model,
        model_path,
        N=32,
        dt=0.03125,
        batch_size=1,
        constant_f_ext=None,
        track_full_stats=False,
        plant_type='indy7',
    ):
        """
        Initialize MPC controller.
        
        Args:
            model: Pinocchio model
            N: Prediction horizon (knot points)
            dt: Time step
            batch_size: Number of parallel trajectories
            constant_f_ext: Constant external force/torque (optional)
            track_full_stats: If True, track all stats; if False, only essential ones
            plant_type: Plant identifier used for selecting dynamics (e.g., 'indy7', 'iiwa14')
        """
        self.model = model
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = model.createData()
        
        # Initialize solver with standard parameters
        self.solver = BSQP(
            model_path=model_path,
            batch_size=batch_size,
            N=N,
            dt=dt,
            max_sqp_iters=1,
            kkt_tol=0.001,
            max_pcg_iters=200,
            pcg_tol=1e-4,
            solve_ratio=1.0,
            mu=10.0,
            q_cost=2.0,
            qd_cost=1e-2,
            u_cost=2e-6,
            N_cost=50.0,
            q_lim_cost=0.01,
            rho=0.01,
            plant_type=plant_type,
        )
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.model.nv
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        self.track_full_stats = track_full_stats
        
        # Setup external forces if provided
        self.setup_external_forces(constant_f_ext)
        
        # Setup force estimator for batch > 1
        self.setup_force_estimator()
        
    def setup_external_forces(self, constant_f_ext):
        """Setup external forces for simulation."""
        self.constant_f_ext_world = constant_f_ext if constant_f_ext is not None else np.zeros(6)
        
        # Create force vector for Pinocchio simulation
        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())
            
        if constant_f_ext is not None:
            self.actual_f_ext[-1] = pin.Force(constant_f_ext[:3], constant_f_ext[3:])
    
    def setup_force_estimator(self):
        """Initialize force estimator for batch processing."""
        if self.batch_size > 1 and ForceEstimator is not None:
            self.force_estimator = ForceEstimator(
                batch_size=self.batch_size,
                initial_radius=5.0,
                min_radius=0.5,
                max_radius=10.0,
                smoothing_factor=0.9
            )
        else:
            self.force_estimator = None
            
    def run_mpc_fig8(self, x_start, fig8_traj, sim_dt=0.001, sim_time=5.0):
        """
        Run MPC controller tracking figure-8 trajectory.
        
        Returns only essential statistics for visualization and analysis.
        """
        # Initialize essential statistics
        stats = {
            'timestamps': [],
            'solve_times': [],        # GPU solve time in ms
            'goal_distances': [],     # Tracking error in meters
            'ee_actual': [],          # Actual end-effector positions
            'joint_positions': [],    # Joint positions over time
            'joint_velocities': [],   # Joint velocities over time
        }
        
        # Add SQP iterations only if tracking full stats
        if self.track_full_stats:
            stats['sqp_iters'] = []
            
        # Initialize simulation
        total_sim_time = 0.0
        accumulated_time = 0.0
        
        x_curr = x_start
        q = x_start[:self.nq]
        dq = x_start[self.nq:self.nx]
        
        # Prepare batch inputs
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        ee_g = fig8_traj[:6*self.N]
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        
        # Initialize warm start
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        XU_batch = np.tile(XU, (self.batch_size, 1))
        
        # Reset solver
        self.solver.reset_dual()
        
        # Warm up solve
        self.update_force_batch(q)
        XU_batch, _ = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        XU_best = XU_batch[0, :]
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, time={sim_time}s")
        if np.any(self.constant_f_ext_world):
            print(f"External force: {self.constant_f_ext_world[:3]}")
            
        # Main control loop
        solve_time = self.dt
        while total_sim_time < sim_time:
            
            # Store state for force estimation
            x_last = x_curr
            u_last = XU_best[self.nx:self.nx+self.nu]
            
            # Simulate forward with current control
            timestep = solve_time
            nsteps = int(timestep/sim_dt)
            
            for i in range(nsteps):
                offset = int(i/(self.dt/sim_dt))
                u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                u = XU_best[u_idx:u_idx+self.nu]
                q, dq = rk4(self.model, self.data, q, dq, u, sim_dt, self.actual_f_ext)
                total_sim_time += sim_dt
                
            # Handle residual time
            if timestep % sim_dt > 1e-5:
                accumulated_time += timestep % sim_dt
                if accumulated_time >= sim_dt:
                    accumulated_time -= sim_dt
                    offset = int(nsteps/(self.dt/sim_dt))
                    u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                    u = XU_best[u_idx:u_idx+self.nu]
                    q, dq = rk4(self.model, self.data, q, dq, u, sim_dt, self.actual_f_ext)
                    total_sim_time += sim_dt
                    
            x_curr = np.concatenate([q, dq])
            
            # Check if trajectory is complete
            eepos_offset = int(total_sim_time / self.dt)
            if eepos_offset >= len(fig8_traj)/6 - 6*self.N:
                break
                
            # Prepare next optimization
            x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
            ee_g = fig8_traj[6*eepos_offset:6*(eepos_offset+self.N)]
            ee_g_batch[:, :] = ee_g
            XU_batch[:, :self.nx] = x_curr
            
            # Update forces and solve
            self.update_force_batch(q)
            
            start = time.time()
            XU_batch_new, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
            solve_time = time.time() - start
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, sim_dt)
            XU_best = XU_batch_new[best_id, :]
            XU_batch[:, :] = XU_best
            
            # Collect essential statistics
            ee_pos = get_ee_position(self.model, self.data, q)
            goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])
            
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Convert to ms
            stats['goal_distances'].append(goal_dist)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q.copy())
            stats['joint_velocities'].append(dq.copy())
            
            if self.track_full_stats:
                solver_stats = self.solver.get_stats()
                stats['sqp_iters'].append(solver_stats['sqp_iters'])
                
        # Convert to numpy arrays
        for key in stats:
            if stats[key]:
                stats[key] = np.array(stats[key])
                
        # Print summary
        print(f"Avg error: {np.mean(stats['goal_distances']):.4f}m")
        print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
        
        return None, stats  # Return None for trajectory (not needed)
    
    def update_force_batch(self, q):
        """Update force hypotheses for batch solving."""
        if self.batch_size == 1 or self.force_estimator is None:
            return
            
        # Generate force batch
        force_batch = self.force_estimator.generate_batch()
        
        # Transform to GATO frame
        transformed_batch = np.zeros_like(force_batch)
        for i in range(self.batch_size):
            transformed_batch[i, :] = self.transform_force_to_gato_frame(q, force_batch[i, :])
            
        self.solver.set_f_ext_B(transformed_batch)
        
    def evaluate_best_trajectory(self, x_last, u_last, x_curr, dt):
        """Evaluate which trajectory best matches reality."""
        if self.batch_size == 1 or self.force_estimator is None:
            return 0
            
        # Simulate all hypotheses
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        # Calculate errors
        errors = np.linalg.norm(x_next_batch - x_curr[None, :], axis=1)
        best_id = np.argmin(errors)
        
        # Update estimator
        self.force_estimator.update(best_id, errors, alpha=0.4, beta=0.5)
        
        return best_id
        
    def transform_force_to_gato_frame(self, q, f_world):
        """Transform force from world frame to GATO frame."""
        data = self.model.createData()
        q_robot = q[:self.nq]
        
        pin.forwardKinematics(self.model, data, q_robot)
        pin.updateFramePlacements(self.model, data)
        
        # Joint indices
        jid_5_pin = 6
        jid_ee_pin = self.model.njoints - 1
        
        # Get transformations
        transform_world_to_j5 = data.oMi[jid_5_pin]
        transform_world_to_ee = data.oMi[jid_ee_pin]
        transform_j5_to_ee = transform_world_to_j5.inverse() * transform_world_to_ee
        
        # Transform force
        force_ee_world = pin.Force(f_world[:3], f_world[3:])
        force_ee_local = transform_world_to_ee.actInv(force_ee_world)
        wrench_j5_local = transform_j5_to_ee.actInv(force_ee_local)
        
        result = np.zeros(6)
        result[:3] = wrench_j5_local.linear
        result[3:] = wrench_j5_local.angular
        
        return result
