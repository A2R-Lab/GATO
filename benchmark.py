import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import sys 
import os
from tqdm import tqdm
import pickle
import importlib
sys.path.append('./python/bsqp')

np.set_printoptions(precision=3)
np.set_printoptions(linewidth=990)

class GATO:
    def __init__(self, N, dt, batch_size, f_ext_std, max_sqp_iters=8, kkt_tol=0.005, max_pcg_iters=50, pcg_tol=1e-3, Q_cost=1.0, dQ_cost=1e-2, u_cost=1e-6, QN_cost=20.0, Qlim_cost=0.0, Qvel_cost=0.0, Qacc_cost=0.0, rho=0.0):
        module_name = f"bsqpN{N}" 
        try: lib = importlib.import_module(module_name)
        except ImportError as e: raise ValueError(f"Number of knots {N} not supported (could not import {module_name}): {e}")
        
        class_name = f"BSQP_{batch_size}_float" 
        if not hasattr(lib, class_name): raise ValueError(f"Batch size {batch_size} not supported in module {module_name}")
        
        self.solver = getattr(lib, class_name)(dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, 1.0, 10.0, Q_cost, dQ_cost, u_cost, QN_cost, Qlim_cost, Qvel_cost, Qacc_cost, rho)
        
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        if f_ext_std == 0.0:
            self.f_ext_batch = np.zeros((self.batch_size, 6))
        else:
            self.f_ext_batch = np.random.normal(0, f_ext_std, (self.batch_size, 6))
            self.f_ext_batch[:, 3:] = 0.0
            self.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        self.solver.set_f_ext_batch(self.f_ext_batch)
        
        self.stats = {
            'solve_time': {'values': [], 'unit': 'us', 'multiplier': 1},
            'pcg_iters': {'values': [], 'unit': '', 'multiplier': 1},
            "step_size": {"values": [], "unit": "", "multiplier": 1},
            'sqp_iters': {'values': [], 'unit': '', 'multiplier': 1}
        }
        
    def solve(self, x_curr_batch, eepos_goals_batch, XU_batch):
        # self.reset_dual()
        # self.reset_rho()
        result = self.solver.solve(XU_batch, self.dt, x_curr_batch, eepos_goals_batch)
        self.stats['solve_time']['values'].append(result["sqp_time_us"])
        self.stats['sqp_iters']['values'].append(result["sqp_iters"])
        self.stats['pcg_iters']['values'].append(result["pcg_iters"])
        self.stats["step_size"]["values"].append(result["ls_step_size"])
        return result["XU"], result["sqp_time_us"]
    
    def sim_forward(self, xk, uk, sim_dt):
        x_next_batch = self.solver.sim_forward(xk, uk, sim_dt)
        return x_next_batch # [batch size x nx]
    
    # [batch size x 6]
    def set_f_ext_batch(self, f_ext_batch):
        self.solver.set_f_ext_batch(f_ext_batch)
        
    def reset_dual(self):
        self.solver.reset_dual()
        
    def reset_rho(self):
        self.solver.reset_rho()
        
    def get_stats(self):
        return self.stats
    

class Benchmark():
    def __init__(self, file_prefix='', batch_size=1, usefext=False):
        # xml_filename = "urdfs/frankapanda/mjx_panda.xml"
        urdf_filename = "urdfs/indy7.urdf"
        N = 16
        self.N = N
        dt = 0.01
        max_qp_iters = 5
        num_threads = batch_size
        fext_timesteps = 8
        Q_cost = 1.0
        dQ_cost = 5e-2
        R_cost = 1e-7
        QN_cost = 20.0
        Qpos_cost = 0.0
        Qvel_cost = 0.0
        Qacc_cost = 0.0
        rho = 1e-5
        # orient_cost = 0.0
        kkt_tol = 1e-9
        max_pcg_iters = 500
        pcg_tol = 1e-8
        self.realtime = False
        self.resample_fext = 0 and (batch_size > 1)
        self.usefext = usefext
        self.file_prefix = file_prefix
        self.batch_size = batch_size

        config = {
            'file_prefix': file_prefix,
            'urdf_filename': urdf_filename,
            'batch_size': batch_size,
            'N': N,
            'dt': dt,
            'max_qp_iters': max_qp_iters,
            'num_threads': num_threads,
            'fext_timesteps': fext_timesteps,
            'Q_cost': Q_cost,
            'dQ_cost': dQ_cost,
            'R_cost': R_cost,
            'QN_cost': QN_cost,
            'Qpos_cost': Qpos_cost,
            'Qvel_cost': Qvel_cost,
            'Qacc_cost': Qacc_cost,
            # 'orient_cost': orient_cost,
            'realtime': self.realtime,
            'resample_fext': self.resample_fext,
            'usefext': self.usefext,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        pickle.dump(config, open(f'benchmark_stats{file_prefix}_benchmark_config.pkl', 'wb'))

        # solver
        self.solver = GATO(N=N, dt=dt, batch_size=batch_size, 
                           f_ext_std=0.0, max_sqp_iters=max_qp_iters, 
                           kkt_tol=kkt_tol, max_pcg_iters=max_pcg_iters, pcg_tol=pcg_tol,
                           Q_cost=Q_cost, dQ_cost=dQ_cost, u_cost=R_cost, QN_cost=QN_cost, Qlim_cost=Qpos_cost, Qvel_cost=Qvel_cost, Qacc_cost=Qacc_cost, rho=rho)

        # mujoco
        # self.model = mujoco.MjModel.from_xml_path(xml_filename)
        self.model = mujoco.MjModel.from_xml_path("urdfs/mujocomodels/indy7.xml")
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    

        # points
        self.points = np.load('examples/points1000.npy')[1:]
        # self.configs = np.load('points/configs10.npy')[1:]

        self.realfext_generator = np.random.default_rng(123)
        self.fext_generator = np.random.default_rng(321)

        # constants
        self.nq = 6
        self.nv = 6
        self.nu = 6
        self.nx = 12
        self.eepos_zero = np.array([0.0, -.1865,  1.3275])

        # tmp instance variables
        self.xs = np.zeros(self.nx)
        self.XU = np.zeros(self.solver.N*(self.nx+self.nu)-self.nu)
        self.update_batches()
        self.update_goal_trace_batch()

        self.dist_to_goal = lambda goal_point: np.linalg.norm(self.data.site_xpos[self.ee_site_id] - goal_point[:3])
        self.last_control = np.zeros(self.nu)

        self.solver.solver.set_f_ext_batch(self.solver.f_ext_batch)

        self.realfext = np.array([0.0, 0.0, -5.0]) * (self.usefext)

    def update_xs_batch(self):
        self.xs_batch = np.tile(self.xs, self.batch_size)

    def update_XU_batch(self):
        self.XU_batch = np.tile(self.XU, (self.batch_size, 1))
    
    def update_goal_trace_batch(self, goal_point=np.array([0.0, 0.0, 0.0])):
        self.goal_trace = np.tile(np.concatenate([goal_point, np.zeros(3)]), (self.N, 1))
        self.goal_trace_batch = np.tile(self.goal_trace, (self.batch_size, 1))

    def update_batches(self):
        self.update_xs_batch()
        self.update_XU_batch()
    
    def getfext(self):
        if self.usefext:
            self.realfext = np.clip(self.realfext_generator.normal(self.realfext, 2.0), -50.0, 50.0)
        return self.realfext
    
    def reset_solver(self):
        # reset primals and duals to zeros
        self.solver.reset_dual()
        self.solver.reset_rho()

    def runMPC(self, viewer, goal_point):
        sim_steps = 0
        solves = 0
        total_cost = 0
        total_dist = 0
        total_ctrl = 0.0
        total_vel = 0.0
        best_cost = np.inf
        avg_solve_time = 0
        max_solve_time = 0

        self.XU_batch = np.zeros((self.batch_size, self.solver.N*(self.nx+self.nu)-self.nu))
        self.update_XU_batch()
        self.update_goal_trace_batch(self.eepos_zero)
        goal_set = False
        
        while sim_steps < 1000:
            if (self.dist_to_goal(goal_point) < 5e-2 and np.linalg.norm(self.data.qvel, ord=1) < 1.0):
                print(f'Got to goal in {sim_steps} steps')
                break

            # set goal
            if not goal_set and self.dist_to_goal(self.eepos_zero) < 0.4:
                print(f'goal set')
                self.update_goal_trace_batch(goal_point)
                goal_set = True

            # get current state
            self.xs = np.hstack((self.data.qpos, self.data.qvel))
            self.update_xs_batch()

            # solve
            solvestart = time.monotonic()
            XU_batch_new, gpu_solve_time = self.solver.solve(self.xs_batch, self.goal_trace_batch, self.XU_batch)
            solve_time = time.monotonic() - solvestart
            # print(f'Solve time: {1000 * (solve_time):.2f} ms')
            print(f'{XU_batch_new[:,:18]}')

            # if any XU_batch_new is nan or inf, reset solver
            if np.any(np.isnan(XU_batch_new)) or np.any(np.isinf(XU_batch_new)):
                print("solve returned nan or inf")
                self.reset_solver()
                self.update_XU_batch()
                continue

            # simulate forward with last control
            sim_time = solve_time
            while sim_time > 0:
                total_ctrl += np.linalg.norm(self.data.ctrl)
                total_vel += np.linalg.norm(self.data.qvel, ord=1)
                # total_cost += self.solver.eepos_cost(np.hstack((self.data.qpos, self.data.qvel)), self.goal_trace, 1)
                total_dist += self.dist_to_goal(self.goal_trace[0,:3])
                # print(self.data.site_xpos[self.ee_site_id])
                
                self.data.xfrc_applied[7,:3] = self.getfext()
                # print(self.data.xfrc_applied[6,:3])
                mujoco.mj_step(self.model, self.data)
                sim_time -= self.model.opt.timestep
                
                if viewer is not None:
                    viewer.sync()
                sim_steps += 1
                
                # stats to print
                if sim_steps % 100 == 0:
                    linear_dist = self.dist_to_goal(goal_point)
                    # orientation_dist = np.linalg.norm(self.solver.compute_rotation_error(self.solver.eepos(self.xs[:self.nq])[1], self.solver.goal_orientation))
                    orientation_dist = 0
                    abs_vel = np.linalg.norm(self.data.qvel, ord=1)
                    print(np.round(linear_dist, 3), np.round(orientation_dist, 3), np.round(abs_vel, 3), np.round(1000 * (solve_time), 0), sep='\t')
                if not self.realtime:
                    break
            xnext = np.hstack((self.data.qpos, self.data.qvel))
            
            # get best control
            predictions = self.solver.sim_forward(self.xs, self.data.ctrl, self.model.opt.timestep * math.ceil(solve_time / self.model.opt.timestep))

            best_tracker = None
            best_error = np.inf
            for i, result in enumerate(predictions):
                error = np.linalg.norm(result - xnext)
                if error < best_error:
                    best_error = error
                    best_tracker = i
            # print(f'Best fext: {self.fext_batch[best_tracker]}')
            if best_tracker is not None:
                bestctrl = XU_batch_new[best_tracker][self.nx:self.nx+self.nu]
            else:
                print("solve failed")
                bestctrl = np.zeros(self.nu)
                self.XU = np.zeros(self.solver.N*(self.nx+self.nu)-self.nu)
                self.reset_solver()
                self.update_XU_batch()

            if self.resample_fext:
                self.fext_batch[:] = self.fext_batch[best_tracker]
                self.fext_batch = self.fext_generator.normal(self.fext_batch, 2.0)
                self.solver.batch_set_fext(self.fext_batch)


            # set control for next step (maybe make this a moving avg so you don't give up gravity comp?)
            self.data.ctrl = bestctrl * 0.8 # + self.last_control * 0.2
            self.last_control = self.data.ctrl
            self.XU_batch[:] = XU_batch_new[best_tracker]

            # update stats
            avg_solve_time += solve_time
            max_solve_time = max(max_solve_time, solve_time)
            # best_cost = min(best_cost, self.solver.last_state_cost)
            solves += 1

        stats = {
            'failed': sim_steps>=1000,
            'cumulative_dist': total_dist,
            'cumulative_cost': total_cost,
            'best_cost': best_cost,
            'avg_solve_time': avg_solve_time / solves,
            'max_solve_time': max_solve_time,
            'steps': sim_steps,
            'solves': solves,
            'total_ctrl': total_ctrl,
            'total_vel': total_vel
        }
        print(f'average vel: {total_vel / sim_steps}')
        print(f'average ctrl: {total_ctrl / sim_steps}')
        return stats

    def runBench(self, headless=False):

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
        
        if headless:
            viewer = None
        else:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        
        

        # warmup
        for _ in range(10):
            self.solver.solve(self.xs_batch, self.goal_trace_batch, self.XU_batch)
        
        for i in tqdm(range(len(self.points)-1)):
            print(f'Point{i}: {self.points[i]}, {self.points[i+1]}')
            # reset to zero
            self.xs_batch = np.zeros(self.batch_size*self.nx)
            self.data.qpos = np.zeros(self.nq)
            self.data.qvel = np.zeros(self.nv)
            self.data.ctrl = np.zeros(self.nu)

            # go to point 1
            p1 = self.points[i]
            self.reset_solver()
            leg1 = self.runMPC(viewer, p1)

            # go to point 2
            p2 = self.points[i+1]
            self.reset_solver()
            leg2 = self.runMPC(viewer, p2)

            # return to zero
            self.reset_solver()
            leg3 = self.runMPC(viewer, self.eepos_zero)

            failed = leg1['failed'] or leg2['failed'] or leg3['failed']
            print(f'Failed: {failed}')
            allstats['failed'].append(failed)
            allstats['cumulative_cost'].append(leg1['cumulative_cost'] + leg2['cumulative_cost'] + leg3['cumulative_cost'])
            allstats['cumulative_dist'].append(leg1['cumulative_dist'] + leg2['cumulative_dist'] + leg3['cumulative_dist'])
            allstats['best_cost'].append(min(leg1['best_cost'], leg2['best_cost'], leg3['best_cost']))
            allstats['avg_solve_time'].append((leg1['avg_solve_time'] * leg1['steps'] + leg2['avg_solve_time'] * leg2['steps'] + leg3['avg_solve_time'] * leg3['steps']) / (leg1['steps'] + leg2['steps'] + leg3['steps']))
            allstats['max_solve_time'].append(max(leg1['max_solve_time'], leg2['max_solve_time'], leg3['max_solve_time']))
            allstats['steps'].append(leg1['steps'] + leg2['steps'] + leg3['steps'])
            
            # save stats
            if i % 20 == 0:
                pickle.dump(allstats, open(f'benchmark_stats{self.file_prefix}_stats_{i}.pkl', 'wb'))
            if i==20:
                break
        pickle.dump(allstats, open(f'benchmark_stats{self.file_prefix}_stats_final.pkl', 'wb'))

        if not headless:
            viewer.close()

        return allstats


        

if __name__ == '__main__':
    b = Benchmark(file_prefix='stockcpu_batch2', batch_size=2, usefext=False)
    b.runBench()
    # b = Benchmark(file_prefix='stockcpu_batch1_fext', batch_size=1, usefext=True)
    # b.runBench()
    # b = Benchmark(file_prefix='stockcpu_batch2_fext', batch_size=2, usefext=True)
    # b.runBench()
    # b = Benchmark(file_prefix='stockcpu_batch4_fext', batch_size=4, usefext=True)
    # b.runBench()
    # b = Benchmark(file_prefix='stockcpu_batch8_fext', batch_size=8, usefext=True)
    # b.runBench()
    # b = Benchmark(file_prefix='stockcpu_batch16_fext', batch_size=16, usefext=True)
    # b.runBench()
