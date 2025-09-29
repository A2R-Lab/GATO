"""
MPC controller for GATO trajectory optimization.
"""

import sys
import time
import numpy as np
import pinocchio as pin
from .interface import BSQP
from .common import rk4, get_ee_position

# Import force estimator if available
sys.path.append('./examples')
try:
    from force_estimator import ImprovedForceEstimator
except ImportError:
    ImprovedForceEstimator = None


class MPC_GATO:
    
    def __init__(self, model, N=32, dt=0.03125, batch_size=1, constant_f_ext=None):
        self.model = model
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = model.createData()
        
        self.solver = BSQP(
            model_path="examples/indy7_description/indy7.urdf",
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
            rho=0.01 
        )
        
        self.q_traj = []  # trajectory for visualization
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.model.nv
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        
        # Store the original world-frame force for reference
        self.constant_f_ext_world = constant_f_ext if constant_f_ext is not None else np.zeros(6)
        
        # Compute the transformed force for GATO (at joint 5 in local frame)
        self.constant_f_ext_gato = None
        if constant_f_ext is not None:
            # We'll compute this after forward kinematics in the simulation
            self.constant_f_ext_gato = np.zeros(6)  # Will be updated
        
        # Initialize improved force estimator with smoother parameters
        if batch_size > 1 and ImprovedForceEstimator is not None:
            self.force_estimator = ImprovedForceEstimator(
                batch_size=batch_size,
                initial_radius=5.0,  
                min_radius=0.5,     
                max_radius=10.0,     
                smoothing_factor=0.9 
            )
            self.current_force_batch = None
        else:
            self.force_estimator = None
            self.current_force_batch = None
        
        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())
            
        if constant_f_ext is not None:
            self.actual_f_ext[-1] = pin.Force(constant_f_ext[:3], constant_f_ext[3:])

    def transform_force_to_gato_frame(self, q, f_world):
        """
        Transform a force from world frame at end-effector to local frame at joint 5.
        Uses only robot joints (first nq_robot elements of q).
        """
        # Create data for solver model to do kinematics
        data = self.model.createData()
        
        # Use only robot configuration
        q_robot = q[:self.nq]
        
        # Update kinematics on solver model
        pin.forwardKinematics(self.model, data, q_robot)
        pin.updateFramePlacements(self.model, data)
        
        # Joint indices
        jid_5_pin = 6  # Joint 5 in GATO = Joint 6 in Pinocchio
        jid_ee_pin = self.model.njoints - 1  # End-effector
        
        # Get transformations
        transform_world_to_j5 = data.oMi[jid_5_pin]
        transform_world_to_ee = data.oMi[jid_ee_pin]
        
        # Compute transformation from Joint 5 to End-Effector
        transform_j5_to_ee = transform_world_to_j5.inverse() * transform_world_to_ee
        
        # Create force at end-effector in world frame
        force_ee_world = pin.Force(f_world[:3], f_world[3:])
        
        # Transform to Joint 5 local frame
        force_ee_local = transform_world_to_ee.actInv(force_ee_world)
        wrench_j5_local = transform_j5_to_ee.actInv(force_ee_local)
        
        result = np.zeros(6)
        result[:3] = wrench_j5_local.linear
        result[3:] = wrench_j5_local.angular
        
        return result
    
    def update_force_batch(self, q):
        """Generate and set force batch BEFORE solving (called pre-solve)."""
        
        # No external force hypothesis for single batch
        if self.batch_size == 1:  
            return
        
        if self.force_estimator is None:
            return
            
        self.current_force_batch = self.force_estimator.generate_batch()
        # Transform each force hypothesis to GATO frame
        transformed_batch = np.zeros_like(self.current_force_batch)
        for i in range(self.batch_size):
            # Each hypothesis is in world frame, transform to GATO frame
            transformed_batch[i, :] = self.transform_force_to_gato_frame(q, self.current_force_batch[i, :])
        
        self.solver.set_f_ext_B(transformed_batch)
    
    def evaluate_best_trajectory(self, x_last, u_last, x_curr, dt):
        """Evaluate which trajectory best matches reality (called post-solve)."""
        if self.batch_size == 1:
            return 0
        
        if self.force_estimator is None:
            return 0
            
        # Simulate all hypotheses with their corresponding forces
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        # Calculate errors for all hypotheses
        errors = np.linalg.norm(x_next_batch - x_curr[None, :], axis=1)
        best_id = np.argmin(errors)
        
        # Update estimator with results
        self.force_estimator.update(best_id, errors, alpha=0.4, beta=0.5)
        
        return best_id
                
    def run_mpc_fig8(self, x_start, fig8_traj, sim_dt=0.001, sim_time=5):        
            
        stats = {
            'solve_times': [],
            'goal_distances': [],
            'ee_goal': [],
            'ee_actual': [],  # Actual end-effector positions
            'ee_velocity': [],  # End-effector velocities
            'controls': [],  # Control inputs (torques)
            'joint_positions': [],  # All joint positions
            'joint_velocities': [],  # All joint velocities
            'timestamps': [],  # Time stamps for each step
            'sqp_iters': [],  # SQP iterations
            'pcg_iters': [],  # PCG iterations
            'force_estimates': [],  # Force estimates (if batch)
            'force_estimates_gato': [],  # Force estimates in GATO frame
            'force_radius': [],  # Force estimator search radius
            'force_confidence': [],  # Force estimator confidence
            'best_trajectory_id': []  # Which trajectory was selected
        }
        
        total_sim_time = 0.0
        steps = 0
        accumulated_time = 0.0
        
        x_curr, q, dq = x_start, x_start[:self.nq], x_start[self.nq:self.nx]
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        ee_g = fig8_traj[:6*self.N]
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        self.solver.reset_dual()
        XU_batch = np.tile(XU, (self.batch_size, 1))
        
        # Warm up run with initial force batch
        self.update_force_batch(q)
        solve_start = time.time()
        XU_batch, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        solve_time = time.time() - solve_start
        XU_best = XU_batch[0, :]
        
        print(f"\n========== Running MPC for {sim_time} seconds with N={self.N} and batch size={self.batch_size} ==========")
        if self.constant_f_ext_world is not None:
            print(f"External force at EE (world frame): {self.constant_f_ext_world[:3]}")
                
        while total_sim_time < sim_time:
            steps += 1
            
            timestep = solve_time
            
            x_last = x_curr
            u_last = XU_best[self.nx:self.nx+self.nu]
            
            # ----- Step Simulation -----
            
            nsteps = int(timestep/sim_dt)
            for i in range(nsteps):
                offset = int(i/(self.dt/sim_dt))  # get correct control input
                u = XU_best[self.nx+(self.nx+self.nu)*offset:(self.nx+self.nu)*(offset+1)]
                q, dq = rk4(self.model, self.data, q, dq, u, sim_dt, self.actual_f_ext)
                total_sim_time += sim_dt
                self.q_traj.append(q)
                
            if timestep%sim_dt > 1e-5: # 
                accumulated_time += timestep%sim_dt
                
            if accumulated_time - sim_dt > 0.0:
                accumulated_time = 0.0
                
                offset = int(nsteps/(self.dt/sim_dt))
                u = XU_best[self.nx+(self.nx+self.nu)*offset:(self.nx+self.nu)*(offset+1)]
                q, dq = rk4(self.model, self.data, q, dq, u, sim_dt, self.actual_f_ext)
                total_sim_time += sim_dt
                self.q_traj.append(q)
                
            x_curr = np.concatenate([q, dq])
            
            # ----- Optimize trajectory -----
            
            # shift eepos goal
            eepos_offset = int(total_sim_time / self.dt)
            if eepos_offset >= len(fig8_traj)/6 - 6*self.N: 
                print("End of trajectory")
                break
            
            x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
            ee_g = self.get_ee_g_traj(fig8_traj, eepos_offset)
            ee_g_batch[:, :] = ee_g
            XU_batch[:, :self.nx] = x_curr
            
            self.update_force_batch(q)
            
            solve_start = time.time()
            XU_batch_new, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
            solve_time = time.time() - solve_start
            
            best_id = self.evaluate_best_trajectory(x_last, u_last, x_curr, sim_dt)

            XU_best = XU_batch_new[best_id, :]
            XU_batch[:, :] = XU_best
            # -----
            
            ee_pos = self.eepos(q)
            pin.forwardKinematics(self.model, self.data, q, dq)
            ee_vel = pin.getFrameVelocity(self.model, self.data, 6, pin.LOCAL_WORLD_ALIGNED).linear
            
            stats['timestamps'].append(total_sim_time)
            # Store GPU time for statistics (in ms)
            stats['solve_times'].append(float(round(gpu_solve_time/1e3, 5)))
            goaldist = np.sqrt(np.sum((ee_pos[:3] - ee_g[6:9])**2))
            stats['goal_distances'].append(float(round(goaldist, 5)))
            stats['ee_goal'].append(ee_g[6:9].copy())
            stats['ee_actual'].append(ee_pos.copy())
            stats['ee_velocity'].append(ee_vel.copy())
            stats['controls'].append(u_last.copy())
            stats['joint_positions'].append(q.copy())
            stats['joint_velocities'].append(dq.copy())
            stats['best_trajectory_id'].append(best_id)
            
            # Get solver statistics
            solver_stats = self.solver.get_stats()
            stats['sqp_iters'].append(solver_stats['sqp_iters'])
            stats['pcg_iters'].append(solver_stats['pcg_iters'][0] if len(solver_stats['pcg_iters']) > 0 else 0)
            
            # Force estimator statistics (if batch)
            if self.force_estimator:
                est_stats = self.force_estimator.get_stats()
                stats['force_estimates'].append(est_stats['smoothed_estimate'].copy())
                # Also store the GATO-frame equivalent
                gato_force = self.transform_force_to_gato_frame(q, est_stats['smoothed_estimate'])
                stats['force_estimates_gato'].append(gato_force.copy())
                stats['force_radius'].append(est_stats['radius'])
                stats['force_confidence'].append(est_stats['confidence'])
            else:
                stats['force_estimates'].append(self.constant_f_ext_world.copy() if self.constant_f_ext_world is not None else np.zeros(6))
                stats['force_estimates_gato'].append(self.constant_f_ext_gato.copy() if self.constant_f_ext_gato is not None else np.zeros(6))
                stats['force_radius'].append(0.0)
                stats['force_confidence'].append(1.0)

        print(f"avg err: {np.mean(stats['goal_distances']):4.3f}")
        print(f"avg t_sqp: {np.mean(stats['solve_times']):4.3f}ms")
        print(f"========== MPC finished ==========")
        
        # Convert lists to numpy arrays for easier processing
        for key in stats:
            if stats[key]:
                stats[key] = np.array(stats[key])

        return self.q_traj, stats
    
    def get_ee_g_traj(self, traj, offset):
        if offset >= len(traj)/6 - 6*self.N:
            print("=> end of trajectory, wrapping around")
            offset %= len(traj)/6
        return traj[6*offset:6*(offset+self.N)]
    
    def eepos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[6].translation
