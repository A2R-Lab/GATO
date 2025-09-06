import sys
import os
import time
import numpy as np
import pinocchio as pin

# Add the bsqp interface to path
sys.path.append('./python/bsqp')
sys.path.append('./python')
sys.path.append('../build/python')

from bsqp.interface import BSQP
from force_estimator import ImprovedForceEstimator

np.set_printoptions(linewidth=99999999)
np.random.seed(42)


class MPC_GATO:
    def __init__(self, model, N=32, dt=0.03125, batch_size=1, constant_f_ext=None):
        # Store original model for solver (without pendulum)
        self.solver_model = model
        
        # Create augmented model for simulation (with pendulum)
        self.model = self.add_pendulum_to_model(model.copy())
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        
        # Initialize BSQP solver with original model (no pendulum)
        self.solver = BSQP(
            model_path="iiwa-mpc/description/iiwa.urdf",
            batch_size=batch_size,
            N=N,
            dt=dt,
            max_sqp_iters=1,
            kkt_tol=0.0,
            max_pcg_iters=100,
            pcg_tol=1e-6,
            solve_ratio=1.0,
            mu=10.0,
            q_cost=100.0,
            qd_cost=1e-2,
            u_cost=0e-7,
            N_cost=0.0,
            q_lim_cost=0.01,
            rho=0.1 
        )
        
        self.q_traj = []  # trajectory for visualization (robot only)
        self.q_traj_full = []  # full trajectory including pendulum
        
        # Dimensions - note model has extra DOFs from pendulum
        self.nq_robot = self.solver_model.nq
        self.nv_robot = self.solver_model.nv
        self.nq = self.model.nq  # Robot + pendulum
        self.nv = self.model.nv  # Robot + pendulum
        self.nx = self.nq_robot + self.nv_robot  # Solver state dimension (robot only)
        self.nu = self.solver_model.nv  # Control dimension (robot only)
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        

        if batch_size > 1:
            self.force_estimator = ImprovedForceEstimator(
                batch_size=batch_size,
                initial_radius=5.0,  
                min_radius=2.0,      
                max_radius=20.0,     
                smoothing_factor=0.5 
            )
            self.current_force_batch = None
        else:
            self.force_estimator = None
            self.current_force_batch = None
        
        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())
    
    def add_pendulum_to_model(self, model):
        """Add a 3D pendulum (spherical joint) to the end-effector."""
        # Get end-effector joint
        ee_joint_id = model.njoints - 1  # Last joint is EE
        
        # Add spherical joint for pendulum
        pendulum_joint_id = model.addJoint(
            ee_joint_id,
            pin.JointModelSpherical(),
            pin.SE3.Identity(),
            "pendulum_joint"
        )
        
        # Pendulum parameters
        mass = 1.0  # kg
        length = 0.5  # m
        
        # Create inertia for pendulum bob (point mass at distance)
        com = np.array([0.0, 0.0, -length])  # Center of mass along -Z
        # For a point mass at distance L: Ixx = Iyy = m*L^2, Izz ≈ 0
        inertia_matrix = np.diag([0.001, 0.001, 0.001])  # Small inertia at COM
        # Create inertia using the constructor
        pendulum_inertia = pin.Inertia(mass, com, inertia_matrix)
        
        # Add pendulum body
        model.appendBodyToJoint(pendulum_joint_id, pendulum_inertia, pin.SE3.Identity())
        
        # Add joint limits 
        # model.upperPositionLimit = np.concatenate([model.upperPositionLimit[:7], np.array([np.pi, np.pi, np.pi])])
        # model.lowerPositionLimit = np.concatenate([model.lowerPositionLimit[:7], np.array([-np.pi, -np.pi, -np.pi])])
        
        return model
    
    def transform_force_to_gato_frame(self, q, f_world):
        """
        Transform a force from world frame at end-effector to local frame at joint 5.
        Uses only robot joints (first nq_robot elements of q).
        """
        # Create data for solver model to do kinematics
        solver_data = self.solver_model.createData()
        
        # Use only robot configuration
        q_robot = q[:self.nq_robot]
        
        # Update kinematics on solver model
        pin.forwardKinematics(self.solver_model, solver_data, q_robot)
        pin.updateFramePlacements(self.solver_model, solver_data)
        
        # Joint indices
        jid_ee_fin = self.solver_model.getFrameId("EE") # self.solver_model.njoints - 1  # End-effector
        jid_ee_pin = self.solver_model.frames[jid_ee_fin].parent # end-effector parent joint
        jid_eep_pin = jid_ee_pin - 1 # End-effector parent joint
        
        print(f"jid_ee_pin: {jid_ee_pin}, jid_eep_pin: {jid_eep_pin}")
        
        # Get transformations
        transform_world_to_ee = solver_data.oMi[jid_eep_pin]
        transform_world_to_jeep = solver_data.oMi[jid_eep_pin-1]
        
        # Compute transformation from last parent joint to End-Effector
        transform_jeep_to_ee = transform_world_to_jeep.inverse() * transform_world_to_ee

        # Create force at end-effector in world frame
        force_ee_world = pin.Force(f_world[:3], f_world[3:])
        
        # Transform to ee parent joint local frame
        force_ee_local = transform_world_to_ee.actInv(force_ee_world)
        wrench_jeep_local = transform_jeep_to_ee.actInv(force_ee_local)

        result = np.zeros(6)
        result[:3] = wrench_jeep_local.linear
        result[3:] = wrench_jeep_local.angular

        return result
    
    def update_force_batch(self, q):
        
        # No external force hypothesis for single batch
        if self.batch_size == 1:  
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
        
        # Simulate all hypotheses with their corresponding forces
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        # Calculate errors for all hypotheses
        errors = np.linalg.norm(x_next_batch - x_curr[None, :], axis=1)
        best_id = np.argmin(errors)
        
        # Update estimator with results
        self.force_estimator.update(best_id, errors, alpha=0.4, beta=0.5)
        
        return best_id
                
    def run_mpc(self, x_start, goals, sim_dt=0.001, sim_time=5):        
            
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
        
        stats['goal_outcomes_by_idx'] = ['not_reached'] * len(goals)
        
        total_sim_time = 0.0
        steps = 0
        accumulated_time = 0.0
        
        # Initialize augmented state with pendulum at rest
        x_start_aug = np.zeros(self.nq + self.nv)
        x_start_aug[:self.nx] = x_start  # Robot state
        # Pendulum starts with small initial angle
        x_start_aug[self.nq_robot:self.nq_robot+3] = np.array([0.3, 0.0, 0.0])  # Small rotation
        
        q = x_start_aug[:self.nq]
        dq = x_start_aug[self.nq:]
        
        # Solver uses robot-only state
        x_curr = x_start
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))

        # Check for NaN or Inf in x_curr before using it
        if np.any(np.isnan(x_curr)) or np.any(np.isinf(x_curr)):
            print("WARNING: x_curr contains NaN or Inf! Values:", x_curr)

        print(f"Size of x_curr: {x_curr.shape}")

        # Initialize first goal
        current_goal_idx = 0
        current_goal = goals[current_goal_idx]
        ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
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
        
        # Start timing for the current goal
        goal_start_time = total_sim_time
                
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
                
                # Augment control with zero torques for pendulum
                u_aug = np.zeros(self.nv)
                u_aug[:self.nu] = u
                
                q, dq = self.rk4(q, dq, u_aug, sim_dt)
                total_sim_time += sim_dt
                self.q_traj.append(q[:self.nq_robot])  # Store only robot joints
                self.q_traj_full.append(q.copy())  # Store full state
                
            if timestep%sim_dt > 1e-5: 
                accumulated_time += timestep%sim_dt
                
            if accumulated_time - sim_dt > 0.0:
                accumulated_time = 0.0
                
                offset = int(nsteps/(self.dt/sim_dt))
                u = XU_best[self.nx+(self.nx+self.nu)*offset:(self.nx+self.nu)*(offset+1)]
                
                # Augment control with zero torques for pendulum
                u_aug = np.zeros(self.nv)
                u_aug[:self.nu] = u
                
                q, dq = self.rk4(q, dq, u_aug, sim_dt)
                total_sim_time += sim_dt
                self.q_traj.append(q[:self.nq_robot])  # Store only robot joints
                self.q_traj_full.append(q.copy())  # Store full state
                
            # Update solver state (robot only)
            x_curr = np.concatenate([q[:self.nq_robot], dq[:self.nv_robot]])
            
            # ----- Optimize trajectory toward current goal -----
            
            current_dist = np.linalg.norm(self.eepos(q[:self.nq_robot]) - current_goal)
            current_vel = np.linalg.norm(dq[:self.nv_robot], ord=1)
            reached = (current_dist < 5e-2) and (current_vel < 1.0)
            timeout = (total_sim_time - goal_start_time) >= 8.0
            if reached or timeout:
                if reached:
                    stats['goal_outcomes_by_idx'][current_goal_idx] = 'reached'
                else:
                    stats['goal_outcomes_by_idx'][current_goal_idx] = 'timeout'
                current_goal_idx += 1
                if current_goal_idx >= len(goals):
                    print("All goals processed")
                    break
                current_goal = goals[current_goal_idx]
                ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
                goal_start_time = total_sim_time
            
            x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
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
            
            ee_pos = self.eepos(q[:self.nq_robot])
            pin.forwardKinematics(self.solver_model, self.solver_model.createData(), q[:self.nq_robot], dq[:self.nv_robot])
            ee_vel = pin.getFrameVelocity(self.solver_model, self.solver_model.createData(), 6, pin.LOCAL_WORLD_ALIGNED).linear
            
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(float(round(gpu_solve_time/1e3, 5)))
            goaldist = np.sqrt(np.sum((ee_pos[:3] - current_goal)**2))
            stats['goal_distances'].append(float(round(goaldist, 5)))
            stats['ee_goal'].append(current_goal.copy())
            stats['ee_actual'].append(ee_pos.copy())
            stats['ee_velocity'].append(ee_vel.copy())
            stats['controls'].append(u_last.copy())
            stats['joint_positions'].append(q[:self.nq_robot].copy())
            stats['joint_velocities'].append(dq[:self.nv_robot].copy())
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
                # Calculate actual pendulum force reaction for comparison
                pin.forwardKinematics(self.model, self.data, q)
                pendulum_com_acc = pin.getFrameClassicalAcceleration(self.model, self.data, self.model.njoints-1, pin.LOCAL_WORLD_ALIGNED).linear
                pendulum_mass = 10.0
                pendulum_force = -pendulum_mass * (pendulum_com_acc - self.model.gravity.linear)
                
                stats['force_estimates'].append(np.concatenate([pendulum_force, np.zeros(3)]))
                stats['force_estimates_gato'].append(np.zeros(6))
                stats['force_radius'].append(0.0)
                stats['force_confidence'].append(1.0)
            
            if steps % 512 == 0:
                # Also print force estimate for monitoring
                if self.force_estimator:
                    est_stats = self.force_estimator.get_stats()
                    smoothed = est_stats['smoothed_estimate']
                    gato_force = stats['force_estimates_gato'][-1]
                    print(f"err=\033[91m{goaldist:4.3f}\033[0m | t_sqp=\033[92m{gpu_solve_time/1e3:4.3f}\033[0m ms | id={best_id} | f_world=[{smoothed[0]:5.1f}, {smoothed[1]:5.1f}, {smoothed[2]:5.1f}] | f_gato=[{gato_force[0]:5.1f}, {gato_force[1]:5.1f}, {gato_force[2]:5.1f}] | t={total_sim_time:4.3f}s")
                else:
                    print(f"err=\033[91m{goaldist:4.3f}\033[0m | t_sqp=\033[92m{gpu_solve_time/1e3:4.3f}\033[0m ms | id={best_id} | t={total_sim_time:4.3f}s")

        print(f"avg err: {np.mean(stats['goal_distances']):4.3f}")
        print(f"avg t_sqp: {np.mean(stats['solve_times']):4.3f}ms")
        print(f"========== MPC finished ==========")
        
        # Convert lists to numpy arrays for easier processing
        for key in stats:
            if stats[key]:
                stats[key] = np.array(stats[key])

        return self.q_traj, stats
    

    
    def rk4(self, q, dq, u, dt):
        k1q = dq
        k1v = pin.aba(self.model, self.data, q, dq, u, self.actual_f_ext)
        q2 = pin.integrate(self.model, q, k1q * dt / 2)
        k2q = dq + k1v * dt/2
        k2v = pin.aba(self.model, self.data, q2, k2q, u, self.actual_f_ext)
        q3 = pin.integrate(self.model, q, k2q * dt / 2)
        k3q = dq + k2v * dt/2
        k3v = pin.aba(self.model, self.data, q3, k3q, u, self.actual_f_ext)
        q4 = pin.integrate(self.model, q, k3q * dt)
        k4q = dq + k3v * dt
        k4v = pin.aba(self.model, self.data, q4, k4q, u, self.actual_f_ext)
        dq_next = dq + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        avg_dq = (k1q + 2*k2q + 2*k3q + k4q) / 6
        q_next = pin.integrate(self.model, q, avg_dq * dt)
        return q_next, dq_next
            
    def eepos(self, q):
        """Get end-effector position using solver model (robot only)."""
        solver_data = self.solver_model.createData()
        pin.forwardKinematics(self.solver_model, solver_data, q)
        jid_ee_pin = self.solver_model.getFrameId("EE") # self.solver_model.njoints - 1  # End-effector
        jid_eep_pin = self.solver_model.frames[jid_ee_pin].parent
        return solver_data.oMi[jid_eep_pin].translation
        # return solver_data.oMi[6].translation


# Use class MPC_GATO as main
if __name__ == "__main__":
    # Load robot model
    model_path = "iiwa-mpc/description/iiwa.urdf"
    model = pin.buildModelFromUrdf(model_path)
    
    # Create MPC_GATO instance
    mpc = MPC_GATO(model, N=32, dt=0.03125, batch_size=16)
    
    # Initial state (home position)
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dq0 = np.zeros(7)
    x0 = np.concatenate([q0, dq0])
    
    # Define goals in 3D space
    goals = [
        np.array([0.4, 0.2, 0.4]),
        np.array([0.4, -0.2, 0.4]),
        np.array([0.6, 0.0, 0.6]),
        np.array([0.5, 0.3, 0.5]),
        np.array([0.5, -0.3, 0.5])
    ]
    
    # Compute end effector position at start
    ee_start = mpc.eepos(q0)
    print(f"End-effector starts at: {ee_start}")

    # Validate transform_force_to_gato_frame
    f_world = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])  # Gravity in world frame
    f_gato = mpc.transform_force_to_gato_frame(q0, f_world)
    print(f"Transformed force (GATO frame): {f_gato}")

    # Simulate all hypotheses with their corresponding forces
    x_last = x0
    u_last = np.zeros(mpc.nu)
    dt = 0.03125
    x_next_batch = mpc.solver.sim_forward(x_last, u_last, dt)

    print(f"Size of x_last: {x_last.shape}")
    print(f"Size of u_last: {u_last.shape}")
    print(f"x_next_batch: {x_next_batch}")
    print(f"Size of x_next_batch: {x_next_batch.shape}")