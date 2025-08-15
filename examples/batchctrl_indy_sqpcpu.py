import time
import math
import numpy as np
from neuromeka import EtherCAT
import sys 
import contextlib
import io
import os
from collections import deque
import traceback
current_dir = os.getcwd()
build_dir = os.path.join(current_dir, 'build')
sys.path.append(build_dir)
import pysqpcpu

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
        self.ecat = EtherCAT(self.ip)
        
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
        
        # enable all servos
        self.ecat.set_servo(0, True)
        self.ecat.set_servo(1, True)
        self.ecat.set_servo(2, True)
        self.ecat.set_servo(3, True)
        self.ecat.set_servo(4, True)
        self.ecat.set_servo(5, True)
      

        urdf_filename = "urdfs/indy7_limited.urdf"

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
            Q_cost = 170.0
            dQ_cost = 0.4
            R_cost = 1e-5
            QN_cost = 4 * Q_cost
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

        self.solver = pysqpcpu.BatchThneed(urdf_filename=urdf_filename, 
                                           eepos_frame_name="end_effector", 
                                           batch_size=self.batch_size, 
                                           num_threads=self.num_threads, 
                                           N=N, 
                                           dt=self.dt, 
                                           max_qp_iters=max_qp_iters, 
                                           fext_timesteps=self.fext_timesteps, 
                                           Q_cost=Q_cost, 
                                           dQ_cost=dQ_cost, 
                                           R_cost=R_cost, 
                                           QN_cost=QN_cost, 
                                           Qpos_cost=Qpos_cost, 
                                           Qvel_cost=Qvel_cost, 
                                           Qacc_cost=Qacc_cost, 
                                           orient_cost=orient_cost)

        # facing right
        self.solver.update_goal_orientation(np.array([[ 0., -0.,  1.],
                                                [ 0.,  1.,  0.],
                                                [-1.,  0.,  0.]]))

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
            # self.eepos_g = np.tile(np.array([-.1865, 0., 1.328]), self.solver.N)
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
        

        self.fext_sigma = 5.0
        self.fext_deque = deque(maxlen=1)
        self.state_transition_deque = deque(maxlen=1)
        self.fext_mask = np.zeros((self.batch_size, 6, self.solver.nq+1))
        self.fext_mask[:,0,self.solver.nq] = 1
        self.fext_mask[:,1,self.solver.nq] = 1
        self.fext_mask[:,2,self.solver.nq] = 1
        self.fext_batch = np.zeros_like(self.fext_mask)
        self.best_fext = np.zeros_like(self.fext_mask[0])
        
        # self.fext_batch[1:] = np.random.normal(self.fext_batch[1:], self.fext_sigma)
        self.fext_batch *= self.fext_mask
        self.solver.batch_set_fext(self.fext_batch)
        self.last_xs = None
        self.last_u = np.zeros(6)
        self.last_commanded = np.zeros(6)
        self.last_rotation = np.eye(3)
        self.last_joint_state_time = time.monotonic()
    
        # stats
        self.tracking_errs = []
        self.positions = []
        self.state_transitions = []
        self.spin_ct = 0

    def set_normal_rotation(self, point):
        point[2] -= 0.3 # offset z to center around higher point
        # Normalize input vector (z-axis in new frame)
        z_axis = point / np.linalg.norm(point)

        # Find perpendicular vector (y-axis in new frame)
        ref = np.array([0., 1., 0.]) if not np.isclose(np.abs(z_axis.dot(np.array([0., 1., 0.]))), 1.0) else np.array([1., 0., 0.])
        y_axis = np.cross(z_axis, ref)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Find third perpendicular vector (x-axis in new frame)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Construct rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        # self.last_rotation = 0.6 * rotation_matrix + 0.4 * self.last_rotation
        self.solver.update_goal_orientation(rotation_matrix)

    def joint_callback(self):
        if self.start_time is None:
            self.start_time = time.monotonic()
        # self.last_joint_state_time = time.monotonic()
        self.spin_ct += 1

        self.xs = np.zeros(self.solver.nx)
        self.torques_tx = np.zeros(6)
        for i in range(6):
            ppr = self.servos[i]["ppr"]
            gear_ratio = self.servos[i]["gear_ratio"]
            pos, vel, tor = self.ecat.get_servo_tx(i)[2:5]
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
        # print(f'xs: {self.xs.round(2)}')
        self.msg_time = time.monotonic()
        self.torques_tx *= self.directions
        self.torques_tx *= self.torque_constants

        s = time.monotonic()
        # if FIG8:
        #     self.set_normal_rotation(self.eepos_g[3:6]) # sets goal orientation to normal
        all_updated = self.solver.sqp(self.xs, self.eepos_g) # run batch sqp
        e = time.monotonic()
        if not all_updated:
            print("batch sqp failed")
            return 1

        if self.last_state_msg_time is not None:
            step_duration = self.msg_time - self.last_state_msg_time

            torque_mean = (self.torques_tx + self.last_u) / 2

            transition = np.hstack([self.last_xs, torque_mean, self.xs, step_duration, self.msg_time - self.start_time, self.best_fext[:3,6].flatten()])
            self.state_transition_deque.append(transition)

            # this is for a test
            self.state_transitions.append(transition)

            # get prediction based on last applied control


            errors = np.zeros(self.batch_size)

            for i in range(len(self.state_transition_deque)):
                state = self.state_transition_deque[i][:12]
                ctrl = self.state_transition_deque[i][12:18]
                nextstate = self.state_transition_deque[i][18:30]
                dt = self.state_transition_deque[i][30]
                predictions = self.solver.predict_fwd(state, ctrl, dt)
                for i, result in enumerate(predictions):
                    # get expected state for each result
                    errors[i] += np.linalg.norm(result[self.solver.nq:] - nextstate[self.solver.nq:])
                    # errors[i] += 1e-3 * np.linalg.norm(self.fext_batch[i]) # regularize

            best_tracker_idx = np.argmin(errors)
            error_weights = 1.0 / errors
            error_weights /= np.sum(error_weights)

            # resample fexts around the best result
            if self.resample_fext:
                weighted_fext = (self.fext_batch * error_weights.reshape(-1,1,1)).sum(axis=0) * self.fext_factor
                self.fext_deque.append(weighted_fext)
                self.best_fext = np.mean(self.fext_deque, axis=0)
                self.fext_batch[:] = self.best_fext
                self.fext_batch[0,0,6] -= self.fext_sigma
                self.fext_batch[0,1,6] -= self.fext_sigma
                self.fext_batch[0,2,6] -= self.fext_sigma
                self.fext_batch[1,0,6] += self.fext_sigma
                self.fext_batch[1,1,6] -= self.fext_sigma
                self.fext_batch[1,2,6] -= self.fext_sigma
                self.fext_batch[2,0,6] -= self.fext_sigma
                self.fext_batch[2,1,6] += self.fext_sigma
                self.fext_batch[2,2,6] -= self.fext_sigma
                self.fext_batch[3,0,6] += self.fext_sigma
                self.fext_batch[3,1,6] += self.fext_sigma
                self.fext_batch[3,2,6] -= self.fext_sigma
                self.fext_batch[4,0,6] -= self.fext_sigma
                self.fext_batch[4,1,6] -= self.fext_sigma
                self.fext_batch[4,2,6] += self.fext_sigma
                self.fext_batch[5,0,6] += self.fext_sigma
                self.fext_batch[5,1,6] -= self.fext_sigma
                self.fext_batch[5,2,6] += self.fext_sigma
                self.fext_batch[6,0,6] -= self.fext_sigma
                self.fext_batch[6,1,6] += self.fext_sigma
                self.fext_batch[6,2,6] += self.fext_sigma
                self.fext_batch[7,0,6] += self.fext_sigma
                self.fext_batch[7,1,6] += self.fext_sigma
                self.fext_batch[7,2,6] += self.fext_sigma
                self.fext_batch *= self.fext_mask
                self.solver.batch_set_fext(self.fext_batch)
                print(np.round(self.best_fext[:3,self.solver.nq],2).T, sep='\t', end='')
            best_result = (self.solver.get_results() * error_weights.reshape(-1,1)).sum(axis=0)

        else:
            best_tracker_idx = 0
            best_result = self.solver.get_results()[best_tracker_idx]

        print(f"\tbatch sqp time: {np.round(1000 * (e - s), 1)} ms", end='')
        
        torques_smoothed = best_result[self.solver.nx:self.solver.nx+6] # + 0.2 * self.last_commanded
        torques_nm_applied = self.applied_torque_factor * torques_smoothed.clip(min=self.servo_min_torques, max=self.servo_max_torques)
        servo_torques = np.round(torques_nm_applied / self.torque_constants).astype(int) * self.directions
        self.last_commanded = torques_nm_applied
        # print(f'torques: {torques_nm_applied.round(2)}')
        
        # redirect stdout to suppress print statements
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(6):
                self.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, servo_torques[i])

                # update xs
                ppr = self.servos[i]["ppr"]
                gear_ratio = self.servos[i]["gear_ratio"]
                pos, vel, tor = self.ecat.get_servo_tx(i)[2:5]
                pos_rad = ((2 * math.pi * pos / gear_ratio / ppr) + self.servos[i]["correction_rad"]) * self.servos[i]["direction"]
                vel_rad = 2 * math.pi * vel / gear_ratio / ppr * self.servos[i]["direction"]
                self.xs[i] = pos_rad
                self.xs[i+6] = vel_rad
                self.last_u[i] = tor

        # self.last_u = torques_nm_applied
        self.last_u *= self.directions
        self.last_u *= self.torque_constants

        self.last_xs = self.xs
        self.last_state_msg_time = time.monotonic()

        # record stats
        eepos = self.solver.eepos(self.xs[0:self.solver.nq])
        if not FIG8:
            print(np.linalg.norm(self.eepos_g[:3] - eepos).round(2), np.linalg.norm(self.xs[6:]).round(2), end='')
        self.positions.append(eepos)
        self.tracking_errs.append(np.linalg.norm(eepos - self.goal_trace[:3]))
        # if self.spin_ct % 1000 == 0:
        # exit cond
        if time.monotonic() - self.start_time > 30.0:
            # save tracking err to a file
            np.save(f'data/tracking_errs_{self.config["file_prefix"]}.npy', np.array(self.tracking_errs))
            np.save(f'data/positions_{self.config["file_prefix"]}.npy', np.array(self.positions))
            np.save(f'data/state_transitions_{self.config["file_prefix"]}.npy', np.array(self.state_transitions))
            return 1

        # shift the goal trace
        if FIG8 and (time.monotonic() - self.last_goal_update > self.dt):
            self.goal_trace[:-3] = self.goal_trace[3:]
            self.goal_trace[-3:] = self.fig8[self.fig8_offset:self.fig8_offset+3]
            self.fig8_offset += 3
            self.fig8_offset %= len(self.fig8)
            self.last_goal_update = time.monotonic()
            self.eepos_g = self.goal_trace
        elif not FIG8 and ((np.linalg.norm(self.eepos_g[:3] - eepos) < 0.05 and np.linalg.norm(self.xs[6:]) < 1.0) or (time.monotonic() - self.last_goal_switch_time > 6.0)):
            if (np.linalg.norm(self.eepos_g[:3] - eepos) < 0.05 and np.linalg.norm(self.xs[6:]) < 1.0):
                print(f'\ngoal reached\n')
            else:
                print(f'\ngoal failed\n')
            
            self.goal_idx += 1
            if self.goal_idx == len(self.goals):
                return 1

            self.eepos_g = np.tile(self.goals[self.goal_idx], self.solver.N)
            self.last_goal_switch_time = time.monotonic()
        print()

def main(args=None):
    try:
        controller = TorqueCalculator()
        while 1:
            if controller.joint_callback():
                break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down gracefully...")
        for i in range(6):
            controller.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
            time.sleep(0.02)
            controller.ecat.set_servo(i, False)
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        print("\nShutting down gracefully...")
        for i in range(6):
            controller.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
            time.sleep(0.02)
            controller.ecat.set_servo(i, False)
    finally:
        for i in range(6):
            controller.ecat.set_servo_rx(i, 0x0f, 0x0a, 0, 0, 0)
            time.sleep(0.02)
            controller.ecat.set_servo(i, False)
        print("filename: ", controller.file_prefix)
    

if __name__ == '__main__':
    main()
