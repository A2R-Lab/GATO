import sys
import time
import numpy as np
import pickle
from datetime import datetime
import pinocchio as pin

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


def run_single_benchmark(model, batch_size, N, dt, sim_time, sim_dt, fig8_traj, x_start):
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


def main():
    """Main benchmark runner."""
    
    # Configuration
    config = {
        'urdf_path': "examples/indy7-mpc/description/indy7.urdf",
        'model_dir': "examples/indy7-mpc/description/",
        'batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'N': 64,  # Horizon length
        'dt': 0.01,
        'sim_time': 10.0,
        'sim_dt': 0.001,
        'start_config': 'ready',  # Use named config from INDY7_START_CONFIGS
    }
    
    # Override batch sizes if needed for testing
    # config['batch_sizes'] = [1, 32, 128]  # Quick test
    
    # Load robot model
    model, _, _ = pin.buildModelsFromUrdf(config['urdf_path'], config['model_dir'])
    
    # Generate figure-8 trajectory
    fig8_traj = figure8(config['dt'], **FIG8_DEFAULT_PARAMS)
    
    # Get starting configuration
    x_start = np.hstack((
        INDY7_START_CONFIGS[config['start_config']], 
        np.zeros(6)
    ))
    
    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("GATO Figure-8 Tracking Benchmark")
    print("=" * 60)
    print(f"Config: N={config['N']}, dt={config['dt']}, sim_time={config['sim_time']}s")
    print(f"Starting from: {config['start_config']} configuration")
    
    # Run benchmarks
    for batch_size in config['batch_sizes']:
        result = run_single_benchmark(
            model=model,
            batch_size=batch_size,
            N=config['N'],
            dt=config['dt'],
            sim_time=config['sim_time'],
            sim_dt=config['sim_dt'],
            fig8_traj=fig8_traj,
            x_start=x_start
        )
        results.append(result)
    
    # Save results
    output_file = f"benchmark_fig8_{timestamp}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': results,
            'timestamp': timestamp
        }, f)
    print(f"\nResults saved to: {output_file}")
    
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
