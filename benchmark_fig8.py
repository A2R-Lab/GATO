#!/usr/bin/env python3
"""
Benchmark script for GATO solver tracking a figure 8 trajectory.
Sweeps over different batch sizes and knot points configurations.
"""

import sys
import os
import time
import numpy as np
import pickle
from datetime import datetime
import pinocchio as pin

# Add the bsqp interface to path
sys.path.append('./python/bsqp')
sys.path.append('./python')
from bsqp.interface import BSQP
from bsqp.common import figure8, rk4, get_ee_position, initialize_warm_start

def run_benchmark_iteration(batch_size, N, dt=0.01, sim_time=5.0, sim_dt=0.001):
    """Run a single benchmark iteration with given batch size and horizon."""
    
    # Load Pinocchio model for forward simulation
    urdf_path = "examples/indy7_description/indy7.urdf"
    model_dir = "examples/indy7_description/"
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, model_dir)
    model.gravity.linear = np.array([0, 0, -9.81])
    data = model.createData()
    
    # Initialize solver
    solver = BSQP(
        model_path="examples/indy7_description/indy7.urdf",
        batch_size=batch_size,
        N=N,
        dt=dt,
        max_sqp_iters=1,
        kkt_tol=0.001,
        max_pcg_iters=100,
        pcg_tol=1e-6,
        solve_ratio=1.0,
        mu=10.0,
        q_cost=2.0,
        qd_cost=1e-3,
        u_cost=1e-8 * N,
        N_cost=20.0,
        q_lim_cost=0.0,
        rho=0.1
    )
    
    # Setup dimensions
    nq = model.nq
    nv = model.nv
    nx = nq + nv
    nu = model.nv
    
    # Generate figure 8 trajectory
    fig8_traj = figure8(dt)
    
    # Initial state
    x_start = np.hstack((np.array([-1.096711, -0.09903229, 0.83125766, -0.10907673, 0.49704404, 0.01499449]), 
                        np.zeros(nv)))
    x_curr = x_start.copy()
    q = x_start[:nq]
    dq = x_start[nq:nx]
    
    # Prepare batch inputs
    x_curr_batch = np.tile(x_curr, (batch_size, 1))
    ee_g = fig8_traj[:6*N]
    ee_g_batch = np.tile(ee_g, (batch_size, 1))
    
    # Initialize warm start using common function
    XU = initialize_warm_start(x_start, N, nx, nu)
    XU_batch = np.tile(XU, (batch_size, 1))
    XU_best = XU.copy()
    
    # Reset dual variables
    solver.reset_dual()
    
    # Warm up solve
    XU_batch, _ = solver.solve(x_curr_batch, ee_g_batch, XU_batch)
    XU_best = XU_batch[0, :]  # Use first trajectory for single-threaded sim
    
    # Benchmark loop
    solve_times = []
    goal_distances = []
    sqp_iters_list = []
    iterations = 0
    total_sim_time = 0.0
    accumulated_time = 0.0
    
    while total_sim_time < sim_time:
        # Get solve time for this step
        solve_time = solve_times[-1]['gpu_time_ms'] / 1000.0 if solve_times else dt
        timestep = min(solve_time, dt)  # Cap at dt
        
        # Forward simulate with RK4
        nsteps = int(timestep/sim_dt)
        for i in range(nsteps):
            offset = int(i/(dt/sim_dt))  # get correct control input
            u_idx = nx + (nx+nu)*min(offset, N-1)
            u = XU_best[u_idx:u_idx+nu]
            q, dq = rk4(model, data, q, dq, u, sim_dt)
            total_sim_time += sim_dt
        
        # Handle residual time
        if timestep % sim_dt > 1e-5:
            accumulated_time += timestep % sim_dt
            
        if accumulated_time >= sim_dt:
            accumulated_time -= sim_dt
            offset = int(nsteps/(dt/sim_dt))
            u_idx = nx + (nx+nu)*min(offset, N-1)
            u = XU_best[u_idx:u_idx+nu]
            q, dq = rk4(model, data, q, dq, u, sim_dt)
            total_sim_time += sim_dt
        
        x_curr = np.concatenate([q, dq])
        
        # Shift goal trajectory
        eepos_offset = int(total_sim_time / dt)
        if eepos_offset >= len(fig8_traj)/6 - 6*N:
            break
            
        x_curr_batch = np.tile(x_curr, (batch_size, 1))
        ee_g = fig8_traj[6*eepos_offset:6*(eepos_offset+N)]
        ee_g_batch[:, :] = ee_g
        XU_batch[:, :nx] = x_curr
        
        # Solve
        start = time.perf_counter()
        XU_batch_new, solve_time_us = solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        end = time.perf_counter()
        
        # For benchmarking, just use first trajectory
        XU_best = XU_batch_new[0, :]
        XU_batch[:, :] = XU_best
        
        # Record solve time (both GPU reported and wall clock)
        solve_times.append({
            'gpu_time_ms': solve_time_us / 1000.0,
            'wall_time_ms': (end - start) * 1000.0
        })
        
        # Get solver stats including SQP iterations
        stats = solver.get_stats()
        sqp_iters_list.append(stats['sqp_iters'])
        
        # Calculate goal distance for tracking quality
        ee_pos = get_ee_position(model, data, q)
        goal_pos = ee_g[6:9]  # First goal position
        goal_dist = np.linalg.norm(ee_pos[:3] - goal_pos)
        goal_distances.append(goal_dist)
        
        iterations += 1
        
    return {
        'batch_size': batch_size,
        'N': N,
        'iterations': iterations,
        'solve_times': solve_times,
        'goal_distances': goal_distances,
        'sqp_iters': sqp_iters_list,
        'avg_gpu_time_ms': np.mean([t['gpu_time_ms'] for t in solve_times]) if solve_times else 0,
        'avg_wall_time_ms': np.mean([t['wall_time_ms'] for t in solve_times]) if solve_times else 0,
        'std_gpu_time_ms': np.std([t['gpu_time_ms'] for t in solve_times]) if solve_times else 0,
        'std_wall_time_ms': np.std([t['wall_time_ms'] for t in solve_times]) if solve_times else 0,
        'avg_goal_distance': np.mean(goal_distances) if goal_distances else 0,
        'std_goal_distance': np.std(goal_distances) if goal_distances else 0,
        'avg_sqp_iters': np.mean(sqp_iters_list) if sqp_iters_list else 0,
        'std_sqp_iters': np.std(sqp_iters_list) if sqp_iters_list else 0,
    }

def main():
    # Benchmark configurations
    batch_sizes = [1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    knot_points = [64]  # Available compiled horizon lengths
    
    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("GATO Figure 8 Tracking Benchmark")
    print("=" * 60)
    
    for N in knot_points:
        for batch_size in batch_sizes:
            try:
                print(f"\nRunning: Batch={batch_size}, N={N}")
                print("-" * 40)
                
                result = run_benchmark_iteration(batch_size, N, sim_time=10.0)
                results.append(result)
                
                print(f"Completed: {result['iterations']} iterations")
                print(f"Avg GPU time: {result['avg_gpu_time_ms']:.3f} ± {result['std_gpu_time_ms']:.3f} ms")
                print(f"Avg Wall time: {result['avg_wall_time_ms']:.3f} ± {result['std_wall_time_ms']:.3f} ms")
                print(f"Avg Goal distance: {result['avg_goal_distance']:.4f} ± {result['std_goal_distance']:.4f} m")
                print(f"Avg SQP iters: {result['avg_sqp_iters']:.2f} ± {result['std_sqp_iters']:.2f}")
                
            except Exception as e:
                print(f"Failed: {e}")
                results.append({
                    'batch_size': batch_size,
                    'N': N,
                    'error': str(e)
                })
    
    # Save results
    output_file = f"benchmark_fig8_{timestamp}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Batch':<8} {'N':<6} {'Avg GPU (ms)':<15} {'Avg Wall (ms)':<15} {'Avg Goal Dist':<15} {'Avg SQP Iters':<15}")
    print("-" * 100)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['batch_size']:<8} {r['N']:<6} "
                  f"{r['avg_gpu_time_ms']:<15.3f} {r['avg_wall_time_ms']:<15.3f} "
                  f"{r['avg_goal_distance']:<15.4f} {r['avg_sqp_iters']:<15.2f}")
        else:
            print(f"{r['batch_size']:<8} {r['N']:<6} ERROR: {r['error'][:30]}")

if __name__ == "__main__":
    main()
