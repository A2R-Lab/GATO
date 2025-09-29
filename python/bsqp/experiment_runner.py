"""
Experiment runner for GATO benchmarks and analysis.
Provides utilities for running batch experiments with different configurations.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import pinocchio as pin
from .mpc_controller import MPC_GATO
from .common import figure8
from .config import (
    EXPERIMENT_BATCH_SIZES, 
    FIG8_DEFAULT_PARAMS,
    INDY7_START_CONFIGS,
    DEFAULT_SIM_PARAMS
)


class ExperimentRunner:
    """Manages and runs GATO experiments with multiple batch sizes."""
    
    def __init__(self, urdf_path: str, model_dir: str = None):
        """
        Initialize experiment runner with robot model.
        
        Args:
            urdf_path: Path to robot URDF file
            model_dir: Directory containing URDF (for mesh loading)
        """
        if model_dir is None:
            model_dir = urdf_path.rsplit('/', 1)[0] + '/'
            
        self.urdf_path = urdf_path
        self.model_dir = model_dir
        self.model, self.visual_model, self.collision_model = pin.buildModelsFromUrdf(
            urdf_path, model_dir
        )
        
        self.results = {}
        
    def run_batch_experiments(
        self,
        batch_sizes: List[int] = None,
        N: int = 32,
        dt: float = 0.01,
        sim_time: float = 16.0,
        sim_dt: float = 0.001,
        f_ext: Optional[np.ndarray] = None,
        start_config: str = 'zero',
        fig8_params: Dict = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run experiments with multiple batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            N: MPC horizon length
            dt: MPC timestep
            sim_time: Total simulation time
            sim_dt: Simulation timestep
            f_ext: External force (6D)
            start_config: Starting configuration name or array
            fig8_params: Figure-8 trajectory parameters
            verbose: Print progress
            
        Returns:
            Dictionary with results for each batch size
        """
        if batch_sizes is None:
            batch_sizes = EXPERIMENT_BATCH_SIZES
            
        if fig8_params is None:
            fig8_params = FIG8_DEFAULT_PARAMS.copy()
            
        # Get starting configuration
        if isinstance(start_config, str):
            x_start = np.hstack((INDY7_START_CONFIGS[start_config], np.zeros(6)))
        else:
            x_start = np.hstack((start_config, np.zeros(6)))
            
        # Generate trajectory
        fig8_traj = figure8(dt, **fig8_params)
        
        results = {}
        
        for batch_size in batch_sizes:
            if verbose:
                print(f"\\nRunning experiment with batch_size={batch_size}...")
                
            try:
                # Create MPC controller
                mpc = MPC_GATO(
                    self.model, 
                    N=N, 
                    dt=dt, 
                    batch_size=batch_size,
                    constant_f_ext=f_ext
                )
                
                # Run simulation
                q_traj, stats = mpc.run_mpc_fig8(
                    x_start, 
                    fig8_traj,
                    sim_dt=sim_dt,
                    sim_time=sim_time
                )
                
                results[batch_size] = {
                    'q_trajectory': q_traj,
                    'stats': stats,
                    'success': True
                }
                
                if verbose:
                    avg_error = np.mean(stats['goal_distances'])
                    avg_time = np.mean(stats['solve_times'])
                    print(f"  Batch {batch_size}: avg_error={avg_error:.4f}m, "
                          f"avg_time={avg_time:.3f}ms")
                    
            except Exception as e:
                results[batch_size] = {
                    'error': str(e),
                    'success': False
                }
                if verbose:
                    print(f"  Batch {batch_size}: FAILED - {e}")
                    
        self.results = results
        return results
        
    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics across all batch sizes.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for batch_size, result in self.results.items():
            if result['success']:
                stats = result['stats']
                summary[batch_size] = {
                    'avg_tracking_error': np.mean(stats['goal_distances']),
                    'std_tracking_error': np.std(stats['goal_distances']),
                    'max_tracking_error': np.max(stats['goal_distances']),
                    'avg_solve_time_ms': np.mean(stats['solve_times']),
                    'std_solve_time_ms': np.std(stats['solve_times']),
                    'avg_sqp_iters': np.mean(stats['sqp_iters']),
                    'total_steps': len(stats['timestamps'])
                }
            else:
                summary[batch_size] = {'error': result['error']}
                
        return summary
        
    def save_results(self, filename: str):
        """Save experiment results to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'summary': self.get_summary_statistics()
            }, f)
            
    @staticmethod
    def load_results(filename: str) -> Dict:
        """Load experiment results from file."""
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)


def run_standard_benchmark(
    urdf_path: str = "examples/indy7-mpc/description/indy7.urdf",
    batch_sizes: List[int] = None,
    **kwargs
) -> Dict:
    """
    Run standard benchmark with default settings.
    
    Args:
        urdf_path: Path to robot URDF
        batch_sizes: List of batch sizes to test
        **kwargs: Additional parameters for run_batch_experiments
        
    Returns:
        Dictionary with results
    """
    runner = ExperimentRunner(urdf_path)
    results = runner.run_batch_experiments(batch_sizes=batch_sizes, **kwargs)
    summary = runner.get_summary_statistics()
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for batch_size in sorted(summary.keys()):
        if 'error' in summary[batch_size]:
            print(f"Batch {batch_size:4d}: FAILED")
        else:
            s = summary[batch_size]
            print(f"Batch {batch_size:4d}: "
                  f"err={s['avg_tracking_error']:.4f}±{s['std_tracking_error']:.4f}m, "
                  f"time={s['avg_solve_time_ms']:.3f}±{s['std_solve_time_ms']:.3f}ms")
    
    return results
