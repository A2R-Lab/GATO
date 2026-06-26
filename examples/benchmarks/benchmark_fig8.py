import os
import sys
import time
import argparse
import numpy as np
import pickle
from datetime import datetime
import pinocchio as pin

# benchmark pkls land in examples/benchmarks/data/ (next to this script), cwd-independent
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Add paths
sys.path.append('./python/bsqp')
sys.path.append('./python')

from bsqp.mpc_controller import MPC_GATO
from bsqp.common import figure8
from bsqp.config import (
    STANDARD_BATCH_SIZES,
    FIG8_DEFAULT_PARAMS,
    INDY7_START_CONFIGS,
    BATCH_COLORS
)


def run_single_benchmark(model, batch_size, N, dt, sim_time, sim_dt, fig8_traj, x_start, model_path=None):
    """Run a single benchmark configuration."""
    
    print(f"\nBatch={batch_size}, N={N}")
    print("-" * 40)
    
    try:
        # Create controller
        mpc = MPC_GATO(
            model=model,
            N=N,
            dt=dt,
            batch_size=batch_size,
            model_path=model_path,
            plant_type='indy7',
            constant_f_ext=None,  # No external force
            track_full_stats=True  # Track SQP iterations
        )
        
        # Run simulation
        start_time = time.perf_counter()
        _, stats = mpc.run_mpc_fig8(x_start, fig8_traj, sim_dt=sim_dt, sim_time=sim_time)
        total_time = time.perf_counter() - start_time
        
        # Compute metrics
        result = {
            'batch_size': batch_size,
            'N': N,
            'success': True,
            'total_time': total_time,
            'iterations': len(stats['timestamps']),
            'avg_gpu_time_ms': np.mean(stats['solve_times']),
            'std_gpu_time_ms': np.std(stats['solve_times']),
            'avg_goal_distance': np.mean(stats['goal_distances']),
            'std_goal_distance': np.std(stats['goal_distances']),
            'max_goal_distance': np.max(stats['goal_distances']),
            'avg_sqp_iters': np.mean(stats['sqp_iters']) if 'sqp_iters' in stats else 0,
        }
        
        print(f"✓ Completed: {result['iterations']} iterations")
        print(f"  Avg GPU time: {result['avg_gpu_time_ms']:.3f} ± {result['std_gpu_time_ms']:.3f} ms")
        print(f"  Avg tracking error: {result['avg_goal_distance']:.4f} ± {result['std_goal_distance']:.4f} m")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        result = {
            'batch_size': batch_size,
            'N': N,
            'success': False,
            'error': str(e)
        }
    
    return result


def _parse_int_list(s):
    return [int(x) for x in str(s).split(',') if x != '']


def run_sweep(model, model_path, batch_sizes, N, dt, sim_time, sim_dt, x_start,
              fig8_traj, save=True):
    """Run a batch-size sweep at a single horizon N. Saves a heatmap-compatible
    per-N pickle (a bare list of per-batch result dicts, named
    ``benchmark_fig8_{N}N.pkl`` — the format plots/fig8_benchmark_heatmap.ipynb
    globs/loads) and returns the results list."""
    results = []
    print("=" * 60)
    print(f"GATO Figure-8 Tracking Benchmark — N={N}, dt={dt}, sim_time={sim_time}s")
    print("=" * 60)
    for batch_size in batch_sizes:
        results.append(run_single_benchmark(
            model=model, batch_size=batch_size, N=N, dt=dt, sim_time=sim_time,
            sim_dt=sim_dt, fig8_traj=fig8_traj, x_start=x_start, model_path=model_path,
        ))
    if save:
        os.makedirs(_DATA_DIR, exist_ok=True)
        output_file = os.path.join(_DATA_DIR, f"benchmark_fig8_{N}N.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)  # bare list -> heatmap notebook format
        print(f"\nResults saved to: {output_file}")
    return results


def main():
    """Main benchmark runner (Fig 3 scalability: batch-size sweep over horizons N)."""
    p = argparse.ArgumentParser(description="GATO Fig-3 scalability benchmark.")
    p.add_argument('--plant', default='indy7', help="plant_type (indy7/iiwa14)")
    p.add_argument('--urdf', default=None, help="override URDF path")
    p.add_argument('--N', default='64',
                   help="comma-separated horizons, e.g. '8,16,32,64,128' for the heatmap")
    p.add_argument('--batch-sizes', default=','.join(map(str, STANDARD_BATCH_SIZES)),
                   help="comma-separated batch sizes")
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--sim-time', type=float, default=10.0)
    p.add_argument('--sim-dt', type=float, default=0.001)
    p.add_argument('--start-config', default='ready')
    p.add_argument('--quick', action='store_true',
                   help="small subset (batch 1,32,128 @ N=64) for a wiring smoke")
    args = p.parse_args()

    urdf_path = args.urdf or f"examples/{args.plant}_description/{args.plant}.urdf"
    model_dir = urdf_path.rsplit('/', 1)[0] + '/'
    N_list = _parse_int_list(args.N)
    batch_sizes = _parse_int_list(args.batch_sizes)
    if args.quick:
        N_list, batch_sizes = [64], [1, 32, 128]

    model, _, _ = pin.buildModelsFromUrdf(urdf_path, model_dir)
    fig8_traj = figure8(args.dt, **FIG8_DEFAULT_PARAMS)
    start_cfg = INDY7_START_CONFIGS[args.start_config] if args.plant == 'indy7' \
        else np.zeros(model.nq)
    x_start = np.hstack((start_cfg, np.zeros(model.nv)))

    results = []
    for N in N_list:
        results.extend(run_sweep(
            model, urdf_path, batch_sizes, N, args.dt, args.sim_time, args.sim_dt,
            x_start, fig8_traj, save=True,
        ))

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Batch':<8} {'N':<6} {'Status':<10} {'Avg GPU (ms)':<15} {'Tracking (m)':<15} {'SQP Iters':<12}")
    print("-" * 100)
    
    for r in results:
        if r['success']:
            print(f"{r['batch_size']:<8} {r['N']:<6} {'✓ OK':<10} "
                  f"{r['avg_gpu_time_ms']:<15.3f} "
                  f"{r['avg_goal_distance']:.4f} ± {r['std_goal_distance']:.4f}  "
                  f"{r['avg_sqp_iters']:<12.2f}")
        else:
            print(f"{r['batch_size']:<8} {r['N']:<6} {'✗ FAIL':<10} "
                  f"Error: {r['error'][:40]}")
    
    # Performance summary
    successful = [r for r in results if r['success']]
    if successful:
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Find best configurations
        best_error = min(successful, key=lambda x: x['avg_goal_distance'])
        best_speed = min(successful, key=lambda x: x['avg_gpu_time_ms'])
        
        print(f"Best tracking: Batch={best_error['batch_size']} "
              f"({best_error['avg_goal_distance']:.4f}m)")
        print(f"Fastest solve: Batch={best_speed['batch_size']} "
              f"({best_speed['avg_gpu_time_ms']:.3f}ms)")
        
        # Speedup analysis
        if any(r['batch_size'] == 1 for r in successful):
            single = next(r for r in successful if r['batch_size'] == 1)
            print("\nSpeedup vs single solver:")
            for r in successful:
                if r['batch_size'] > 1:
                    speedup = r['batch_size'] / (r['avg_gpu_time_ms'] / single['avg_gpu_time_ms'])
                    print(f"  Batch={r['batch_size']:4d}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
