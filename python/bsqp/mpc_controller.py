import sys
import time
import numpy as np
import pinocchio as pin
from .interface import BSQP
from .common import rk4
from .config import DEFAULT_SOLVER_PARAMS

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
        pendulum_config=None,
        solver_params=None,
    ):
        """
        Initialize MPC controller.
        
        Args:
            model: Pinocchio model
            model_path: Path to URDF file
            N: Prediction horizon (knot points)
            dt: Time step
            batch_size: Number of parallel trajectories
            constant_f_ext: Constant external force/torque (optional)
            track_full_stats: If True, track all stats; if False, only essential ones
            plant_type: Plant identifier used for selecting dynamics (e.g., 'indy7', 'iiwa14')
            pendulum_config: Optional dict with keys: mass, length, damping, initial_angle
        """
        # Store original model for solver (without pendulum)
        self.solver_model = model
        
        # Add pendulum to simulation model if configured
        if pendulum_config is not None:
            self.model = self._add_pendulum_to_model(model.copy(), pendulum_config)
            self.pendulum_config = pendulum_config
            self.has_pendulum = True
            # Store dimensions
            self.nq_robot = self.solver_model.nq
            self.nv_robot = self.solver_model.nv
        else:
            self.model = model
            self.pendulum_config = None
            self.has_pendulum = False
            self.nq_robot = model.nq
            self.nv_robot = model.nv
            
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        
        # Initialize solver with configurable parameters
        solver_cfg = DEFAULT_SOLVER_PARAMS.copy()
        if solver_params is not None:
            solver_cfg.update(solver_params)

        self.solver = BSQP(
            model_path=model_path,
            batch_size=batch_size,
            N=N,
            dt=dt,
            plant_type=plant_type,
            max_sqp_iters=solver_cfg['max_sqp_iters'],
            kkt_tol=solver_cfg['kkt_tol'],
            max_pcg_iters=solver_cfg['max_pcg_iters'],
            pcg_tol=solver_cfg['pcg_tol'],
            solve_ratio=solver_cfg['solve_ratio'],
            mu=solver_cfg['mu'],
            q_cost=solver_cfg['q_cost'],
            qd_cost=solver_cfg['qd_cost'],
            u_cost=solver_cfg['u_cost'],
            N_cost=solver_cfg['N_cost'],
            q_lim_cost=solver_cfg['q_lim_cost'],
            vel_lim_cost=solver_cfg['vel_lim_cost'],
            ctrl_lim_cost=solver_cfg['ctrl_lim_cost'],
            rho=solver_cfg['rho'],
        )

        self.solver_params = solver_cfg
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq_robot + self.nv_robot  # Solver state dimension (robot only)
        self.nu = self.solver_model.nv  # Control dimension (robot only)
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
                min_radius=2.0,
                max_radius=20.0,
                smoothing_factor=0.5
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
                    accumulated_time = 0.0
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
            self.solver.reset_rho()
            
            start = time.time()
            XU_batch_new, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
            solve_time = time.time() - start
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, max(sim_dt, round(timestep / sim_dt) * sim_dt))
            XU_best = XU_batch_new[best_id, :]
            XU_batch[:, :] = XU_best
            
            # Collect essential statistics
            ee_pos = self.solver.ee_pos(q)
            goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])
            
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Convert to ms
            stats['goal_distances'].append(goal_dist)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q.copy())
            stats['joint_velocities'].append(dq.copy())
            
            if self.track_full_stats:
                solver_stats = self.solver.get_stats()
                # Get first element from batch for sqp_iters
                sqp_iters = solver_stats['sqp_iters']
                if isinstance(sqp_iters, np.ndarray):
                    stats['sqp_iters'].append(int(sqp_iters[0]))
                else:
                    stats['sqp_iters'].append(int(sqp_iters))
                
        # Convert to numpy arrays
        for key in stats:
            if stats[key]:
                try:
                    stats[key] = np.array(stats[key])
                except (ValueError, TypeError):
                    # Keep as list if conversion fails
                    pass
                
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
        self.force_estimator.update(best_id, errors, alpha=0.6, beta=0.5)
        
        return best_id
        
    def transform_force_to_gato_frame(self, q, f_world):
        """Transform force from world frame to GATO frame."""
        data = self.solver_model.createData()
        q_robot = q[:self.nq_robot]
        
        pin.forwardKinematics(self.solver_model, data, q_robot)
        pin.updateFramePlacements(self.solver_model, data)
        
        # Joint indices - get end-effector parent joint
        jid_ee_fin = self.solver_model.getFrameId("EE")
        jid_ee_pin = self.solver_model.frames[jid_ee_fin].parentJoint
        jid_eep_pin = jid_ee_pin - 1  # End-effector parent joint
        
        # Get transformations
        transform_world_to_ee = data.oMi[jid_ee_pin]
        transform_world_to_jeep = data.oMi[jid_eep_pin]
        transform_jeep_to_ee = transform_world_to_jeep.inverse() * transform_world_to_ee
        
        # Transform force
        force_ee_world = pin.Force(f_world[:3], f_world[3:])
        force_ee_local = transform_world_to_ee.actInv(force_ee_world)
        wrench_jeep_local = transform_jeep_to_ee.actInv(force_ee_local)
        
        result = np.zeros(6)
        result[:3] = wrench_jeep_local.linear
        result[3:] = wrench_jeep_local.angular
        
        return result
    
    def _add_pendulum_to_model(self, model, config):
        """Add a 3D pendulum (spherical joint) to the end-effector."""
        mass = config.get('mass', 15.0)
        length = config.get('length', 0.3)
        
        ee_joint_id = model.njoints - 1  # Last joint is EE
        pendulum_joint_id = model.addJoint(
            ee_joint_id,
            pin.JointModelSpherical(),
            pin.SE3.Identity(),
            "pendulum_joint"
        )
        
        # Create inertia for pendulum bob (point mass at distance)
        com = np.array([0.0, 0.0, -length])  # Center of mass along -Z
        inertia_matrix = np.diag([0.001, 0.001, 0.001])  # Small inertia at COM
        pendulum_inertia = pin.Inertia(mass, com, inertia_matrix)
        model.appendBodyToJoint(pendulum_joint_id, pendulum_inertia, pin.SE3.Identity())
        
        return model
    
    def run_mpc_goals(
        self, 
        x_start, 
        goals, 
        sim_dt=0.001, 
        goal_timeout=5.0,
        goal_threshold=0.05,
        velocity_threshold=1.0
    ):
        """
        Run MPC controller tracking discrete goal positions.
        
        Args:
            x_start: Initial state (robot only, no pendulum)
            goals: List of 3D goal positions np.array([x, y, z])
            sim_dt: Simulation timestep
            goal_timeout: Max time per goal before timeout
            goal_threshold: Distance threshold for goal reached (m)
            velocity_threshold: Velocity threshold for goal reached (rad/s L1 norm)
            
        Returns:
            q_traj: Joint trajectory (robot only)
            stats: Dictionary with tracking statistics including:
                - goal_outcomes: ['reached', 'timeout', ...] per goal
                - goal_reached_times: [time1, time2, ...] or None
                - time_to_all_reached: total time if all succeeded, else None
                - standard tracking stats (solve_times, timestamps, etc.)
        """
        # Initialize statistics
        stats = {
            'timestamps': [],
            'solve_times': [],
            'goal_distances': [],
            'ee_actual': [],
            'joint_positions': [],
            'joint_velocities': [],
            'best_trajectory_id': []
        }
        
        if self.track_full_stats:
            stats['sqp_iters'] = []
            stats['pcg_iters'] = []
            
        stats['goal_outcomes'] = ['not_reached'] * len(goals)
        stats['goal_reached_times'] = [None] * len(goals)
        stats['time_to_all_reached'] = None
        
        # Initialize simulation state
        total_sim_time = 0.0
        accumulated_time = 0.0
        
        # Initialize augmented state with pendulum if configured
        if self.has_pendulum:
            x_start_aug = np.zeros(self.nq + self.nv)
            x_start_aug[:self.nx] = x_start  # Robot state
            # Pendulum initial angle
            pendulum_init = self.pendulum_config.get('initial_angle', np.array([0.3, 0.0, 0.0]))
            x_start_aug[self.nq_robot:self.nq_robot+3] = pendulum_init
            q = x_start_aug[:self.nq]
            dq = x_start_aug[self.nq:]
        else:
            q = x_start[:self.nq]
            dq = x_start[self.nq:]
        
        # Solver uses robot-only state
        x_curr = x_start
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        
        # Initialize first goal
        current_goal_idx = 0
        current_goal = goals[current_goal_idx]
        ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        
        # Initialize warm start
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        self.solver.reset_dual()
        XU_batch = np.tile(XU, (self.batch_size, 1))
        
        # Warm up solve
        self.update_force_batch(q[:self.nq_robot] if self.has_pendulum else q)
        XU_batch, _ = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        XU_best = XU_batch[0, :]
        
        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, {len(goals)} goals")
        if self.has_pendulum:
            print(f"Pendulum: mass={self.pendulum_config['mass']}kg, length={self.pendulum_config['length']}m")
        
        # Start timing for the current goal
        goal_start_time = total_sim_time
        solve_time = self.dt
        
        # Main control loop
        while total_sim_time < goal_timeout * len(goals):
            
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
                
                # Augment control with robot torques and pendulum damping if needed
                if self.has_pendulum:
                    damping = self.pendulum_config.get('damping', 0.4)
                    u_aug = np.zeros(self.nv)
                    u_aug[:self.nu] = u
                    u_aug[self.nu:] = -damping * dq[self.nv_robot:]
                else:
                    u_aug = u
                    
                q, dq = rk4(self.model, self.data, q, dq, u_aug, sim_dt, self.actual_f_ext)
                total_sim_time += sim_dt
                
            # Handle residual time
            if timestep % sim_dt > 1e-5:
                accumulated_time += timestep % sim_dt
                if accumulated_time >= sim_dt:
                    accumulated_time = 0.0
                    offset = int(nsteps/(self.dt/sim_dt))
                    u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                    u = XU_best[u_idx:u_idx+self.nu]
                    
                    if self.has_pendulum:
                        damping = self.pendulum_config.get('damping', 0.4)
                        u_aug = np.zeros(self.nv)
                        u_aug[:self.nu] = u
                        u_aug[self.nv_robot:] = -damping * dq[self.nv_robot:]
                    else:
                        u_aug = u
                        
                    q, dq = rk4(self.model, self.data, q, dq, u_aug, sim_dt, self.actual_f_ext)
                    total_sim_time += sim_dt
                    
            # Update solver state (robot only)
            q_robot = q[:self.nq_robot] if self.has_pendulum else q
            dq_robot = dq[:self.nv_robot] if self.has_pendulum else dq
            x_curr = np.concatenate([q_robot, dq_robot])
            
            # Check goal reached or timeout
            ee_pos = self.solver.ee_pos(q_robot)
            current_dist = np.linalg.norm(ee_pos - current_goal)
            current_vel = np.linalg.norm(dq_robot, ord=1)
            reached = (current_dist < goal_threshold) and (current_vel < velocity_threshold)
            timeout = (total_sim_time - goal_start_time) >= goal_timeout
            
            if reached or timeout:
                if reached:
                    stats['goal_outcomes'][current_goal_idx] = 'reached'
                    stats['goal_reached_times'][current_goal_idx] = total_sim_time
                else:
                    stats['goal_outcomes'][current_goal_idx] = 'timeout'
                    
                current_goal_idx += 1
                if current_goal_idx >= len(goals):
                    break
                    
                current_goal = goals[current_goal_idx]
                ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
                goal_start_time = total_sim_time
                self.solver.reset_rho()
            
            # Prepare next optimization
            x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
            ee_g_batch[:, :] = ee_g
            XU_batch[:, :self.nx] = x_curr
            
            # Update forces and solve
            self.update_force_batch(q_robot)
            self.solver.reset_rho()
            
            start = time.time()
            XU_batch_new, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
            solve_time = time.time() - start
            
            # Select best trajectory
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, max(sim_dt, round(timestep / sim_dt) * sim_dt))
            XU_best = XU_batch_new[best_id, :]
            XU_batch[:, :] = XU_best
            
            # Collect statistics
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(gpu_solve_time/1000.0)  # Convert to ms
            stats['goal_distances'].append(current_dist)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q_robot.copy())
            stats['joint_velocities'].append(dq_robot.copy())
            stats['best_trajectory_id'].append(best_id)
            
            if self.track_full_stats:
                solver_stats = self.solver.get_stats()
                # Get first element from batch for sqp_iters and pcg_iters
                sqp_iters = solver_stats['sqp_iters']
                if isinstance(sqp_iters, np.ndarray):
                    stats['sqp_iters'].append(int(sqp_iters[0]))
                else:
                    stats['sqp_iters'].append(int(sqp_iters))
                    
                pcg_iters = solver_stats.get('pcg_iters', [])
                if len(pcg_iters) > 0:
                    if isinstance(pcg_iters, np.ndarray) and pcg_iters.ndim == 2:
                        stats['pcg_iters'].append(int(pcg_iters[0, 0]))
                    else:
                        stats['pcg_iters'].append(int(pcg_iters[0]))
                else:
                    stats['pcg_iters'].append(0)
        
        # Convert to numpy arrays
        for key in stats:
            if isinstance(stats[key], list) and len(stats[key]) > 0:
                if key not in ['goal_outcomes', 'goal_reached_times', 'time_to_all_reached']:
                    try:
                        stats[key] = np.array(stats[key])
                    except (ValueError, TypeError):
                        # Keep as list if conversion fails (e.g., inhomogeneous shapes)
                        pass
        
        # Compute time to all-goals-reached if all were reached
        if all([o == 'reached' for o in stats['goal_outcomes']]):
            reached_times = [t for t in stats['goal_reached_times'] if t is not None]
            if len(reached_times) == len(goals):
                stats['time_to_all_reached'] = float(np.max(reached_times))
        
        # Print summary
        goals_reached = sum(1 for o in stats['goal_outcomes'] if o == 'reached')
        print(f"Goals reached: {goals_reached}/{len(goals)}")
        if len(stats['solve_times']) > 0:
            print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
        
        return None, stats  # Return None for trajectory (not needed)
