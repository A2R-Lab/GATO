import time
import math
import numpy as np
import pinocchio as pin

import sys 
import contextlib
import io
import os

import traceback

from neuromeka import EtherCAT
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from bsqp.interface import BSQP
from force_estimator import ImprovedForceEstimator

# set seed
np.random.seed(123)
np.set_printoptions(precision=2, suppress=True, formatter={'float': '{:6.2f}'.format})

FIG8 = False

def figure8():
    xamplitude = 0.25 # X goes from -xamplitude to xamplitude
    zamplitude = 0.4 # Z goes from -zamplitude/2 to zamplitude/2
    period = 6 # seconds
    dt = 0.01 # seconds
    x = lambda t:  0.5 # + 0.1 * np.sin(2*(t + np.pi/4))
    y = lambda t: xamplitude * np.sin(t)
    z = lambda t: 0.37 + zamplitude * np.sin(2*t)/2 + zamplitude/2
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    points = np.array([[x(t), y(t), z(t)] for t in timesteps]).reshape(-1)
    return points

class TorqueCalculator():
    def __init__(self):

        # safety factors
        servo_torque_limit_factor = 1.0
        self.applied_torque_factor = 1.0
        self.fext_factor = 1.0

        self.ip = '160.39.102.105'
        # >>> TODO: Initialize EtherCAT
        # self.ecat = EtherCAT(self.ip)
        
        self.servos = [
            {"index": 0, "direction": -1, "gear_ratio": 121, "ppr": 65536, "max_ecat_torque": 48.0, "rated_torque": 0.08839, "version":  "", "correction_rad": -0.054279739737023644, "torque_constant": 0.2228457},
            {"index": 1, "direction": -1, "gear_ratio": 121, "ppr": 65536, "max_ecat_torque": 48.0, "rated_torque": 0.0839705, "version":  "", "correction_rad": -0.013264502315156903, "torque_constant": 0.2228457},
            {"index": 2, "direction": 1, "gear_ratio": 121, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.0891443, "version":  "", "correction_rad": 2.794970264143719, "torque_constant": 0.10965625},
            {"index": 3, "direction": -1, "gear_ratio": 101, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.05798, "version":  "", "correction_rad": -0.0054105206811824215, "torque_constant": 0.061004},
            {"index": 4, "direction": -1, "gear_ratio": 101, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.055081, "version":  "", "correction_rad": 2.7930504019665254, "torque_constant": 0.061004},
            {"index": 5, "direction": -1, "gear_ratio": 101, "ppr": 65536, "max_ecat_torque": 96.0, "rated_torque": 0.05798, "version":  "", "correction_rad": -0.03490658503988659, "torque_constant": 0.061004}
        ]
        self.directions = np.array([self.servos[i]["direction"] for i in range(6)])
        self.torque_constants = np.array([self.servos[i]["torque_constant"] for i in range(6)])
        self.servo_min_torques = servo_torque_limit_factor * np.array([-431.97, -431.97, -197.23, -79.79, -79.79, -79.79], dtype=int)
        self.servo_max_torques = servo_torque_limit_factor * np.array([431.97, 431.97, 197.23, 79.79, 79.79, 79.79], dtype=int)
        self.pos_limit_lower_deg = np.array([-175.0, -175.0, -175.0, -175.0, -175.0, -215.0])
        self.pos_limit_upper_deg = np.array([175.0, 175.0, 175.0, 175.0, 175.0, 215.0])
        self.vel_limit_lower_deg_per_s = np.array([-159.0, -159.0, -159.0, -189.0, -189.0, -189.0])
        self.vel_limit_upper_deg_per_s = np.array([159.0, 159.0, 159.0, 189.0, 189.0, 189.0])
        self.pos_limit_lower = self.pos_limit_lower_deg * np.pi / 180.0
        self.pos_limit_upper = self.pos_limit_upper_deg * np.pi / 180.0
        self.vel_limit_lower = self.vel_limit_lower_deg_per_s * np.pi / 180.0
        self.vel_limit_upper = self.vel_limit_upper_deg_per_s * np.pi / 180.0
        
        # >>> TODO: Enable all servos
        # self.ecat.set_servo(0, True)
        # self.ecat.set_servo(1, True)
        # self.ecat.set_servo(2, True)
        # self.ecat.set_servo(3, True)
        # self.ecat.set_servo(4, True)
        # self.ecat.set_servo(5, True)
      
        urdf_filename = "indy7-mpc/description/indy7.urdf"
        
        self.batch_size = 1
        self.num_threads = self.batch_size
        self.dt = 0.01
        N = 16
        self.fext_timesteps = N
        max_qp_iters = 5
        num_threads = self.batch_size
        if FIG8:
            Q_cost = 9.0
            dQ_cost = 1e-3
            R_cost = 1e-6
            QN_cost = 3 * Q_cost
            Qpos_cost = 0.001
            Qvel_cost = 0.00
            Qacc_cost = 0.001
            orient_cost = 0.01
        else:
            Q_cost = 2.0
            dQ_cost = 1e-2
            R_cost = 1e-6
            QN_cost = 20.0
            Qpos_cost = 1e-5
            Qvel_cost = 4.0
            Qacc_cost = 5e-4
            orient_cost = 0.0
        self.resample_fext = 1 and (self.batch_size > 1)
        self.file_prefix = f'batchctrl_{self.batch_size}_fig8_cut'

        self.config = {
            'file_prefix': self.file_prefix,
            'urdf_filename': urdf_filename,
            'batch_size': self.batch_size,
            'N': N,
            'dt': self.dt,
            'max_qp_iters': max_qp_iters,
            'num_threads': num_threads,
            'fext_timesteps': self.fext_timesteps,
            'Q_cost': Q_cost,
            'dQ_cost': dQ_cost,
            'R_cost': R_cost,
            'QN_cost': QN_cost,
            'Qpos_cost': Qpos_cost,
            'Qvel_cost': Qvel_cost,
            'Qacc_cost': Qacc_cost,
            'orient_cost': orient_cost,
            'resample_fext': self.resample_fext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }

        self.solver = BSQP(
            model_path="indy7-mpc/description/indy7.urdf",
            batch_size=self.batch_size,
            N=N,
            dt=self.dt,
            max_sqp_iters=1,
            kkt_tol=0.0,
            max_pcg_iters=100,
            pcg_tol=1e-6,
            solve_ratio=1.0,
            mu=10.0,
            q_cost=Q_cost,
            qd_cost=dQ_cost,
            u_cost=R_cost,
            N_cost=QN_cost,
            q_lim_cost=0.01,
            rho=0.05 
        )

        # Store model:
        model_dir = "indy7-mpc/description/"
        model, visual_model, collision_model = pin.buildModelsFromUrdf(urdf_filename, model_dir)
        self.solver_model = model
        # Create augmented model for simulation (with pendulum???)
        self.model = model

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
        
        # >>>> GATO - Python-based >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.batch_size > 1:
            self.force_estimator = ImprovedForceEstimator(
                batch_size=self.batch_size,
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

        # >>>> SQPCPU
        self.last_state_msg_time = None
        self.start_time = None

        self.fig8 = figure8()
        self.fig8_offset = 0
        self.goal_trace = self.fig8[:3*self.solver.N].copy()
        self.last_goal_update = time.monotonic()

        self.xs = np.zeros(self.solver.nx)
        if FIG8:
            self.eepos_g = self.fig8[:3*self.solver.N].copy()
        else:
            self.eepos_g = np.tile(np.array([0.5, -.1865, 0.5]), self.solver.N)
            
            self.goals = [
                np.array([0.5, -.1865, 0.5]),
                np.array([0.5, 0.3, 0.2]),
                np.array([0.3, 0.3, 0.8]),
                np.array([0.6, -0.5, 0.2]),
                np.array([0., -0.5, 0.8])
            ]
            
            self.eepos_g = np.tile(self.goals[0], self.solver.N)
            self.goal_idx = 0
            self.last_goal_switch_time = time.monotonic()
        
        self.last_xs = None
        self.last_u = np.zeros(6)
        self.last_commanded = np.zeros(6)
        self.last_rotation = np.eye(3)
        self.last_joint_state_time = time.monotonic()
    
        # >>>> SQPCPU Stats
        self.spin_ct = 0

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
        jid_5_pin = 6  # Joint 5 in GATO = Joint 6 in Pinocchio
        jid_ee_pin = self.solver_model.njoints - 1  # End-effector
        
        # Get transformations
        transform_world_to_j5 = solver_data.oMi[jid_5_pin]
        transform_world_to_ee = solver_data.oMi[jid_ee_pin]
        
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
    
    def eepos(self, q):
        """Get end-effector position using solver model (robot only)."""
        solver_data = self.solver_model.createData()
        pin.forwardKinematics(self.solver_model, solver_data, q)
        return solver_data.oMi[6].translation
    
    def joint_callback(self):

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
        
        stats['goal_outcomes_by_idx'] = ['not_reached'] * len(self.goals)

        if self.start_time is None:
            self.start_time = time.monotonic()
        # 
        self.spin_ct += 1

        self.xs = np.zeros(self.solver.nx)
        self.torques_tx = np.zeros(6)
        for i in range(6):
            ppr = self.servos[i]["ppr"]
            gear_ratio = self.servos[i]["gear_ratio"]
            # pos, vel, tor = self.ecat.get_servo_tx(i)[2:5] # get position, velocity, torque from EtherCAT
            pos, vel, tor = 0, 0, 0
            pos_rad = ((2 * math.pi * pos / gear_ratio / ppr) + self.servos[i]["correction_rad"]) * self.servos[i]["direction"]
            vel_rad = 2 * math.pi * vel / gear_ratio / ppr * self.servos[i]["direction"]
            self.xs[i] = pos_rad
            self.xs[i+6] = vel_rad
            self.torques_tx[i] = tor
            if pos_rad < self.pos_limit_lower[i] or pos_rad > self.pos_limit_upper[i]:
                print(f'servo {i} pos out of range: {pos_rad}')
                return 1
            if vel_rad < self.vel_limit_lower[i] or vel_rad > self.vel_limit_upper[i]:
                print(f'servo {i} vel out of range: {vel_rad}')
                return 1
        # >>> sqpcpu (commented)
        # print(f'xs: {self.xs.round(2)}')
        self.msg_time = time.monotonic()
        self.torques_tx *= self.directions
        self.torques_tx *= self.torque_constants

        # ====================================================================================================
        # GATO SOLVER - Optimization toward current goal
        # Update solver state (robot only)
        # Simulation uses q and q_dot from step simulation
        # Experiment uses q and q_dot from robot recorded/acquired data
        x_curr = self.xs
        current_goal = self.goals[self.goal_idx]
        print(f"Current goal: {current_goal}")
        print(f"nq_robot: {self.nq_robot}, nx: {self.nx}")

        current_dist = np.linalg.norm(self.eepos(x_curr[:self.nq_robot]) - current_goal)
        current_vel = np.linalg.norm(x_curr[self.nq_robot:self.nx], ord=1)
        reached = (current_dist < 5e-2) and (current_vel < 1.0)
        # timeout = (total_sim_time - goal_start_time) >= 8.0

        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        # 
        ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))
        # 
        if self.last_state_msg_time is None:
            print("Initializing XU_batch")
            XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
            for i in range(self.N):
                start_idx = i * (self.nx + self.nu)
                XU[start_idx:start_idx+self.nx] = x_curr
            self.solver.reset_dual()
            self.XU_batch = np.tile(XU, (self.batch_size, 1))

        self.XU_batch[:, :self.nx] = x_curr

        self.update_force_batch(x_curr[:self.nq_robot])

        # ====================================================================================================
        solve_start = time.monotonic()
        XU_batch_new, gpu_solve_time = self.solver.solve(x_curr_batch, ee_g_batch, self.XU_batch)
        solve_end = time.monotonic()
        solve_time = solve_end - solve_start
        # ====================================================================================================        
        # # >>> SQPCPU        
        # all_updated = self.solver.sqp(self.xs, self.eepos_g) # run batch sqp
        # all_updated = None
        # e = time.monotonic()
        # if not all_updated:
        #     print("batch sqp failed")
        #     return 1
        # ====================================================================================================        

        if self.last_state_msg_time is not None:
            step_duration = self.msg_time - self.last_state_msg_time

            torque_mean = (self.torques_tx + self.last_u) / 2

            best_id = self.evaluate_best_trajectory(self.last_xs, torque_mean, self.xs, step_duration)

            XU_best = XU_batch_new[best_id, :]
            print(f"XU_best size: {XU_best.size}")
            print(f"XU_best shape: {XU_best.shape}")
            print(f"Time N: {self.N}, batch size: {self.batch_size}")
            self.XU_batch[:, :] = XU_best
        
        #   # >>> sqpcpu this is for a test
        
        #   # >>> sqpcpu
        #   best_result = (self.solver.get_results() * error_weights.reshape(-1,1)).sum(axis=0)
            best_result = self.XU_batch[0, :]

        else:
        #   # >>> sqpcpu
        #   best_result = self.solver.get_results()[best_tracker_idx]
            best_id = 0

            XU_best = XU_batch_new[best_id, :]
            self.XU_batch[:, :] = XU_best        
            best_result = self.XU_batch[0, :]

        print(f"\tBatch SQP time: {np.round(1000 * (solve_time), 1)} ms\n")

        print(f"Solver nx: {self.solver.nx}")

        # >>> sqpcpu
        # torques_smoothed = best_result[self.solver.nx:self.solver.nx+6] # + 0.2 * self.last_commanded
        
        torques_smoothed = best_result[self.solver.nx:self.solver.nx+self.solver.nu]
        torques_nm_applied = self.applied_torque_factor * torques_smoothed.clip(min=self.servo_min_torques, max=self.servo_max_torques)
        servo_torques = np.round(torques_nm_applied / self.torque_constants).astype(int) * self.directions
        self.last_commanded = torques_nm_applied
        # print(f'torques: {torques_nm_applied.round(2)}')
        
        # Redirect stdout to suppress print statements
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(6):
                # TODO: Uncomment
                # self.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, servo_torques[i])

                # update xs
                ppr = self.servos[i]["ppr"]
                gear_ratio = self.servos[i]["gear_ratio"]
                # TODO: Uncomment
                # pos, vel, tor = self.ecat.get_servo_tx(i)[2:5]
                pos, vel, tor = 0, 0, 0
                pos_rad = ((2 * math.pi * pos / gear_ratio / ppr) + self.servos[i]["correction_rad"]) * self.servos[i]["direction"]
                vel_rad = 2 * math.pi * vel / gear_ratio / ppr * self.servos[i]["direction"]
                self.xs[i] = pos_rad
                self.xs[i+6] = vel_rad
                self.last_u[i] = tor

        # >>> sqpgpu 
        # self.last_u = torques_nm_applied
        
        self.last_u *= self.directions
        self.last_u *= self.torque_constants

        self.last_xs = self.xs
        self.last_state_msg_time = time.monotonic()

        print("HERE")

        # Record stats
        eepos = self.eepos(self.xs[0:self.solver.nq])
        print(f"End-effector position: {eepos.round(2)}\n")
        pin.forwardKinematics(self.solver_model, self.solver_model.createData(), self.xs[0:self.solver.nq], self.xs[self.solver.nq:self.solver.nx])
        ee_vel = pin.getFrameVelocity(self.solver_model, self.solver_model.createData(), 6, pin.LOCAL_WORLD_ALIGNED).linear
        
        stats['solve_times'].append(float(round(gpu_solve_time/1e3, 5)))
        goaldist = np.sqrt(np.sum((eepos[:3] - current_goal)**2))
        stats['goal_distances'].append(float(round(goaldist, 5)))
        stats['ee_goal'].append(current_goal.copy())
        stats['ee_actual'].append(eepos.copy())
        stats['ee_velocity'].append(ee_vel.copy())
        stats['controls'].append(self.last_u.copy())
        stats['joint_positions'].append(self.xs[:self.nq_robot].copy())
        stats['joint_velocities'].append(self.xs[self.nq_robot:self.nx].copy())
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
        # else:
        #     # Calculate actual pendulum force reaction for comparison
        #     pin.forwardKinematics(self.model, self.data, q)
        #     pendulum_com_acc = pin.getFrameClassicalAcceleration(self.model, self.data, self.model.njoints-1, pin.LOCAL_WORLD_ALIGNED).linear
        #     pendulum_mass = 10.0
        #     pendulum_force = -pendulum_mass * (pendulum_com_acc - self.model.gravity.linear)
            
        #     stats['force_estimates'].append(np.concatenate([pendulum_force, np.zeros(3)]))
        #     stats['force_estimates_gato'].append(np.zeros(6))
        #     stats['force_radius'].append(0.0)
        #     stats['force_confidence'].append(1.0)
        
        print("HERE2")
        
        # Exit cond
        print(f"\nTime: {time.monotonic() - self.start_time:.2f} seconds", end='')
        if time.monotonic() - self.start_time > 30.0:
            # Save tracking err to a file
            # >>>> SQPCPU - 
            # np.save(f'data/tracking_errs_{self.config["file_prefix"]}.npy', np.array(self.tracking_errs))
            # np.save(f'data/positions_{self.config["file_prefix"]}.npy', np.array(self.positions))
            # np.save(f'data/state_transitions_{self.config["file_prefix"]}.npy', np.array(self.state_transitions))
            print(f"Experiment Ended after: {time.monotonic() - self.start_time} seconds")
            return 1

        # Shift the goal trace
        if FIG8 and (time.monotonic() - self.last_goal_update > self.dt):
            self.goal_trace[:-3] = self.goal_trace[3:]
            self.goal_trace[-3:] = self.fig8[self.fig8_offset:self.fig8_offset+3]
            self.fig8_offset += 3
            self.fig8_offset %= len(self.fig8)
            self.last_goal_update = time.monotonic()
            self.eepos_g = self.goal_trace
        elif not FIG8 and ((np.linalg.norm(self.eepos_g[:3] - eepos) < 0.05 
                      and np.linalg.norm(self.xs[6:]) < 1.0) 
                      or (time.monotonic() - self.last_goal_switch_time > 6.0)):
            if (np.linalg.norm(self.eepos_g[:3] - eepos) < 0.05 and np.linalg.norm(self.xs[6:]) < 1.0):
                print(f'\n Goal reached\n')
            else:
                print(f'\n Goal failed\n')
            
            self.goal_idx += 1
            if self.goal_idx == len(self.goals):
                return 1

            self.eepos_g = np.tile(self.goals[self.goal_idx], self.solver.N)
            self.last_goal_switch_time = time.monotonic()
        
        print(f"\n Goal idx: {self.goal_idx}")
        print("\n HERE3")
        print()

def main(args=None):
    try:
        controller = TorqueCalculator()
        while 1:
            if controller.joint_callback():
                break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down gracefully...")
        # for i in range(6):
        #     controller.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
        #     time.sleep(0.02)
        #     controller.ecat.set_servo(i, False)
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        print("\nShutting down gracefully...")
        # for i in range(6):
        #     controller.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
        #     time.sleep(0.02)
        #     controller.ecat.set_servo(i, False)
    finally:
        # for i in range(6):
        #     controller.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
        #     time.sleep(0.02)
        #     controller.ecat.set_servo(i, False)
        print("filename: ", controller.file_prefix)
    

if __name__ == '__main__':
    main()
