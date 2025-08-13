import time
import math
import numpy as np
import pinocchio as pin
import sys
import os
from tqdm import tqdm
import pickle

# Add the bsqp interface to path
sys.path.append('./python/bsqp')
sys.path.append('./python')
from bsqp.interface import BSQP

np.set_printoptions(precision=3)
np.set_printoptions(linewidth=990)


class PinocchioBenchmark:
    def __init__(self, file_prefix='', batch_size=1, knot_points=16, use_f_ext=False):
        # Configuration parameters
        self.file_prefix = file_prefix
        self.batch_size = batch_size
        self.N = knot_points
        self.dt = 0.01
        self.use_f_ext = use_f_ext
        self.realtime = True
        self.resample_f_ext = False
        
        # Solver parameters
        max_sqp_iters = 5
        kkt_tol = 0.0
        max_pcg_iters = 100
        pcg_tol = 1e-6
        Q_cost = 2.0
        dQ_cost = 1e-3
        R_cost = 1e-8 * self.N
        QN_cost = 20.0
        Qpos_cost = 0.0
        Qvel_cost = 0.0
        Qacc_cost = 0.0
        rho = 1e-1
        
        # Save configuration
        config = {
            'file_prefix': file_prefix,
            'urdf_filename': 'examples/indy7-mpc/description/indy7.urdf',
            'batch_size': batch_size,
            'N': self.N,
            'dt': self.dt,
            'max_qp_iters': max_sqp_iters,
            'Q_cost': Q_cost,
            'dQ_cost': dQ_cost,
            'R_cost': R_cost,
            'QN_cost': QN_cost,
            'Qpos_cost': Qpos_cost,
            'Qvel_cost': Qvel_cost,
            'Qacc_cost': Qacc_cost,
            'realtime': self.realtime,
            'resample_fext': self.resample_f_ext,
            'usefext': self.use_f_ext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        pickle.dump(config, open(f'data/benchmark_stats{file_prefix}_benchmark_config.pkl', 'wb'))
        
        # Initialize BSQP solver
        f_ext_B_std = 0.0 if not use_f_ext else 1.0
        f_ext_resample_std = 0.0 if not self.resample_f_ext else 0.1
        
        self.solver = BSQP(
            model_path="examples/indy7-mpc/description/indy7.urdf",
            batch_size=batch_size,
            N=self.N,
            dt=self.dt,
            max_sqp_iters=max_sqp_iters,
            kkt_tol=kkt_tol,
            max_pcg_iters=max_pcg_iters,
            pcg_tol=pcg_tol,
            solve_ratio=1.0,
            mu=10.0,
            q_cost=Q_cost,
            qd_cost=dQ_cost,
            u_cost=R_cost,
            N_cost=QN_cost,
            q_lim_cost=Qpos_cost,
            rho=0.1,
            f_ext_B_std=f_ext_B_std if use_f_ext else None,
            f_ext_resample_std=f_ext_resample_std if self.resample_f_ext else None
        )
        
        # Load Pinocchio model
        self.model = pin.buildModelFromUrdf("examples/indy7-mpc/description/indy7.urdf")
        self.data = self.model.createData()
        
        # Get dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nv
        self.nx = self.nq + self.nv
        
        # Load benchmark points
        self.points = np.load('examples/points1000.npy')[1:]
        
        # Home position (end-effector position at zero configuration)
        self.ee_pos_zero = self.ee_pos(np.zeros(self.nq))
        
        # Initialize state variables
        self.xs = np.zeros(self.nx)
        self.XU = np.zeros(self.solver.N * (self.nx + self.nu) - self.nu)
        self.last_control = np.zeros(self.nu)
        
        # Setup external forces
        self.real_f_ext = np.array([0.0, 0.0, -5.0]) * use_f_ext
        self.real_f_ext_generator = np.random.default_rng(123)
        self.f_ext_generator = np.random.default_rng(321)
        
        # Pinocchio external forces structure
        self.f_ext_pin = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.f_ext_pin.append(pin.Force.Zero())
        
        if use_f_ext:
            # Apply force to end-effector (last joint)
            self.f_ext_pin[-1] = pin.Force(self.real_f_ext, np.zeros(3))
    
    def ee_pos(self, q):
        """Get end-effector position"""
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[6].translation
    
    def dist_to_goal(self, q, goal_point):
        """Calculate distance from end-effector to goal"""
        ee_position = self.ee_pos(q)
        return np.linalg.norm(ee_position - goal_point[:3])
    
    def rk4(self, q, dq, u, dt):
        """RK4 integrator with Pinocchio dynamics"""
        # k1
        k1q = dq
        k1v = pin.aba(self.model, self.data, q, dq, u, self.f_ext_pin)
        
        # k2
        q2 = pin.integrate(self.model, q, k1q * dt / 2)
        k2q = dq + k1v * dt / 2
        k2v = pin.aba(self.model, self.data, q2, k2q, u, self.f_ext_pin)
        
        # k3
        q3 = pin.integrate(self.model, q, k2q * dt / 2)
        k3q = dq + k2v * dt / 2
        k3v = pin.aba(self.model, self.data, q3, k3q, u, self.f_ext_pin)
        
        # k4
        q4 = pin.integrate(self.model, q, k3q * dt)
        k4q = dq + k3v * dt
        k4v = pin.aba(self.model, self.data, q4, k4q, u, self.f_ext_pin)
        
        # Combine
        dq_next = dq + (dt / 6) * (k1v + 2*k2v + 2*k3v + k4v)
        avg_dq = (k1q + 2*k2q + 2*k3q + k4q) / 6
        q_next = pin.integrate(self.model, q, avg_dq * dt)
        
        return q_next, dq_next
    
    def get_f_ext(self):
        """Update external force with random perturbation"""
        if self.use_f_ext:
            self.real_f_ext = np.clip(
                self.real_f_ext_generator.normal(self.real_f_ext, 2.0),
                -50.0, 50.0
            )
            # Update Pinocchio force structure
            self.f_ext_pin[-1] = pin.Force(self.real_f_ext, np.zeros(3))
        return self.real_f_ext
    
    def reset_solver(self):
        """Reset solver primals and duals"""
        self.solver.reset_dual()
    
    def runMPC(self, goal_point):
        """Run MPC to navigate to goal point"""
        sim_steps = 0
        solves = 0
        total_cost = 0
        total_dist = 0
        total_ctrl = 0.0
        total_vel = 0.0
        best_cost = np.inf
        avg_solve_time = 0
        max_solve_time = 0
        
        # Initialize trajectory
        self.XU = np.zeros(self.solver.N * (self.nx + self.nu) - self.nu)
        XU_batch = np.tile(self.XU, (self.batch_size, 1))
        
        # Start with home as goal, then switch to actual goal
        goal_trace = np.tile(np.concatenate([self.ee_pos_zero, np.zeros(3)]), (self.solver.N, 1))
        goal_trace_batch = np.tile(goal_trace, (self.batch_size, 1))
        goal_set = False
        
        # Initialize state (start from current self.xs)
        q = self.xs[:self.nq]
        dq = self.xs[self.nq:]
        
        while sim_steps < 2000:
            # Check if reached goal
            current_dist = self.dist_to_goal(q, goal_point)
            current_vel = np.linalg.norm(dq, ord=1)
            
            if goal_set and current_dist < 5e-2 and current_vel < 1.0:
                print(f'Got to goal in {sim_steps} steps')
                break
            
            # Switch to actual goal when close to home
            if not goal_set and self.dist_to_goal(q, self.ee_pos_zero) < 0.4:
                print(f'goal set')
                goal_trace = np.tile(np.concatenate([goal_point, np.zeros(3)]), (self.solver.N, 1))
                goal_trace_batch = np.tile(goal_trace, (self.batch_size, 1))
                goal_set = True
            
            # Get current state
            self.xs = np.hstack((q, dq))
            xs_batch = np.tile(self.xs, (self.batch_size, 1))
            
            # Solve
            solve_start = time.monotonic()
            XU_batch_new, gpu_solve_time = self.solver.solve(xs_batch, goal_trace_batch, XU_batch)
            solve_time = time.monotonic() - solve_start
            
            # Check for NaN or Inf
            if np.any(np.isnan(XU_batch_new)) or np.any(np.isinf(XU_batch_new)):
                print("solve returned nan or inf")
                self.reset_solver()
                XU_batch = np.tile(np.zeros(self.solver.N * (self.nx + self.nu) - self.nu), (self.batch_size, 1))
                continue
            
            # Simulate forward for solve_time duration (real-time constraint)
            sim_time = solve_time
            while sim_time > 0:
                total_ctrl += np.linalg.norm(self.last_control)
                total_vel += np.linalg.norm(dq, ord=1)
                total_dist += self.dist_to_goal(q, goal_trace[0, :3])
                
                # Apply external force
                self.get_f_ext()
                
                # Step simulation
                q, dq = self.rk4(q, dq, self.last_control, self.dt)
                sim_time -= self.dt
                sim_steps += 1
                
                # Print stats periodically
                if sim_steps % 100 == 0:
                    linear_dist = self.dist_to_goal(q, goal_point)
                    abs_vel = np.linalg.norm(dq, ord=1)
                    print(f"{np.round(linear_dist, 3)}\t{np.round(0, 3)}\t{np.round(abs_vel, 3)}\t{np.round(1000 * solve_time, 0)}")
                
                if not self.realtime:
                    break
            
            x_next = np.hstack((q, dq))
            
            # Get best control from batch
            predictions = self.solver.sim_forward(
                self.xs, 
                self.last_control,
                self.dt * math.ceil(solve_time / self.dt)
            )
            
            best_tracker = None
            best_error = np.inf
            for i, result in enumerate(predictions):
                error = np.linalg.norm(result - x_next)
                if error < best_error:
                    best_error = error
                    best_tracker = i
            
            if best_tracker is not None:
                best_ctrl = XU_batch_new[best_tracker][self.nx:self.nx + self.nu]
            else:
                print("solve failed")
                best_ctrl = np.zeros(self.nu)
                self.XU = np.zeros(self.solver.N * (self.nx + self.nu) - self.nu)
                self.reset_solver()
                XU_batch = np.tile(self.XU, (self.batch_size, 1))
            
            # Set control for next step (85% new, 0% old - matching original)
            self.last_control = best_ctrl * 0.85
            XU_batch[:] = XU_batch_new[best_tracker]
            
            # Update stats
            avg_solve_time += solve_time
            max_solve_time = max(max_solve_time, solve_time)
            solves += 1
        
        # Store final state for next leg
        self.xs = np.hstack((q, dq))
        
        stats = {
            'failed': sim_steps >= 2000,
            'cumulative_dist': total_dist,
            'cumulative_cost': total_cost,
            'best_cost': best_cost,
            'avg_solve_time': avg_solve_time / solves if solves > 0 else 0,
            'max_solve_time': max_solve_time,
            'steps': sim_steps,
            'solves': solves,
            'total_ctrl': total_ctrl,
            'total_vel': total_vel
        }
        
        print(f'average vel: {total_vel / sim_steps if sim_steps > 0 else 0}')
        print(f'average ctrl: {total_ctrl / sim_steps if sim_steps > 0 else 0}')
        
        return stats
    
    def runBench(self):
        """Run full benchmark over all points"""
        allstats = {
            'failed': [],
            'cumulative_cost': [],
            'cumulative_dist': [],
            'best_cost': [],
            'avg_solve_time': [],
            'max_solve_time': [],
            'steps': [],
            'total_ctrl': [],
            'total_vel': []
        }
        
        for i in tqdm(range(min(len(self.points) - 1, 50))):  # Limited to 50 like original
            print(f'Point{i}: {self.points[i]}, {self.points[i+1]}')
            
            # Reset to zero configuration
            self.xs = np.zeros(self.nx)
            self.last_control = np.zeros(self.nu)
            
            # Leg 1: Go to first point
            p1 = self.points[i]
            self.reset_solver()
            leg1 = self.runMPC(p1)
            
            # Leg 2: Go to second point
            p2 = self.points[i+1]
            self.reset_solver()
            leg2 = self.runMPC(p2)
            
            # Leg 3: Return to home
            self.reset_solver()
            leg3 = self.runMPC(self.ee_pos_zero)
            
            # Aggregate statistics
            failed = leg1['failed'] or leg2['failed'] or leg3['failed']
            color = '\\033[92m' if not failed else '\\033[91m'
            end_color = '\\033[0m'
            print(f'Failed: {color}{failed}{end_color}')
            
            allstats['failed'].append(failed)
            allstats['cumulative_cost'].append(
                leg1['cumulative_cost'] + leg2['cumulative_cost'] + leg3['cumulative_cost']
            )
            allstats['cumulative_dist'].append(
                leg1['cumulative_dist'] + leg2['cumulative_dist'] + leg3['cumulative_dist']
            )
            allstats['best_cost'].append(
                min(leg1['best_cost'], leg2['best_cost'], leg3['best_cost'])
            )
            
            total_steps = leg1['steps'] + leg2['steps'] + leg3['steps']
            if total_steps > 0:
                allstats['avg_solve_time'].append(
                    (leg1['avg_solve_time'] * leg1['steps'] + 
                     leg2['avg_solve_time'] * leg2['steps'] + 
                     leg3['avg_solve_time'] * leg3['steps']) / total_steps
                )
            else:
                allstats['avg_solve_time'].append(0)
            
            allstats['max_solve_time'].append(
                max(leg1['max_solve_time'], leg2['max_solve_time'], leg3['max_solve_time'])
            )
            allstats['steps'].append(total_steps)
            allstats['total_ctrl'].append(
                leg1['total_ctrl'] + leg2['total_ctrl'] + leg3['total_ctrl']
            )
            allstats['total_vel'].append(
                leg1['total_vel'] + leg2['total_vel'] + leg3['total_vel']
            )
        
        # Save final statistics
        pickle.dump(allstats, open(f'data/benchmark_stats{self.file_prefix}_stats_final.pkl', 'wb'))
        
        return allstats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Pinocchio-based GATO benchmark')
    parser.add_argument('--prefix', type=str, default='pinocchio', help='File prefix for output')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--knot_points', type=int, default=16, help='Number of knot points')
    parser.add_argument('--use_f_ext', action='store_true', help='Use external forces')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Run benchmark
    benchmark = PinocchioBenchmark(
        file_prefix=f'{args.prefix}_batch{args.batch_size}_N{args.knot_points}',
        batch_size=args.batch_size,
        knot_points=args.knot_points,
        use_f_ext=args.use_f_ext
    )
    
    print(f"Running benchmark with batch_size={args.batch_size}, N={args.knot_points}, f_ext={args.use_f_ext}")
    benchmark.runBench()
