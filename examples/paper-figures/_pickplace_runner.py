"""
Experiment runner for GATO benchmarks and analysis.
Provides utilities for running batch experiments with different configurations.
"""

import numpy as np
from typing import Dict, List
import pinocchio as pin
from gato.config import IIWA14_START_CONFIGS
from _common import (
    PICKPLACE_SOLVER_PARAMS,
    PICKPLACE_MPC_DEFAULTS,
    PICKPLACE_DEFAULT_GOALS,
    PENDULUM_DEFAULT_PARAMS,
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
        
    def run_pickplace_sweep(
        self,
        batch_sizes: List[int] = None,
        N: int = 16,
        dt: float = 0.05,
        sim_dt: float = 0.001,
        plant_type: str = 'iiwa14',
        goal_sequences: List[List[np.ndarray]] = None,
        pendulum_config: Dict = None,
        solver_params: Dict = None,
        mpc_defaults: Dict = None,
        start_config: str = 'home',
        verbose: bool = True,
    ) -> Dict:
        """
        CS3a / Table-I: pick-and-place success rate vs. batch size.

        For each batch size, runs the goal-reaching MPC (with an end-effector
        pendulum/payload) over one or more goal sequences and records the success
        rate. A goal counts as 'reached' iff the modern `run_mpc_goals` returns
        'reached' for it (``||ee - goal|| < goal_threshold`` AND ``L1(dq) <
        velocity_threshold`` before the per-goal timeout). The paper's finding is
        that this success rate climbs with batch size.

        Args:
            batch_sizes: batch sizes to sweep (default config.STANDARD_BATCH_SIZES).
            N, dt, sim_dt: horizon / MPC step / sim step.
            plant_type: dynamics plant ('iiwa14' for the paper's CS3a).
            goal_sequences: list of goal-position lists. Default = a single
                sequence (PICKPLACE_DEFAULT_GOALS). Pass several (e.g. randomized
                sequences) to get a true multi-trial success *rate* per batch.
            pendulum_config: EE payload (default PENDULUM_DEFAULT_PARAMS).
            solver_params: BSQP params (default PICKPLACE_SOLVER_PARAMS).
            mpc_defaults: goal_timeout / goal_threshold / velocity_threshold
                (default PICKPLACE_MPC_DEFAULTS).
            start_config: IIWA14_START_CONFIGS key for the initial robot state.

        Returns:
            {batch_size: {success_rate, n_reached, n_total, per_sequence:[...],
                          avg_solve_time_ms, success}} (also stored on self.results).
        """
        from gato.mpc_gato import MPC_GATO

        if batch_sizes is None:
            from .config import STANDARD_BATCH_SIZES
            batch_sizes = STANDARD_BATCH_SIZES
        if goal_sequences is None:
            goal_sequences = [PICKPLACE_DEFAULT_GOALS]
        if pendulum_config is None:
            pendulum_config = PENDULUM_DEFAULT_PARAMS
        if solver_params is None:
            solver_params = PICKPLACE_SOLVER_PARAMS
        if mpc_defaults is None:
            mpc_defaults = PICKPLACE_MPC_DEFAULTS

        nv = self.model.nv
        x_start = np.hstack((IIWA14_START_CONFIGS[start_config], np.zeros(nv)))

        results = {}
        for batch_size in batch_sizes:
            if verbose:
                print(f"\nPick&place sweep: batch_size={batch_size} "
                      f"({len(goal_sequences)} sequence(s))...")
            try:
                per_sequence = []
                solve_times = []
                n_reached = n_total = 0
                for seq in goal_sequences:
                    mpc = MPC_GATO(
                        self.model,
                        model_path=self.urdf_path,
                        N=N,
                        dt=dt,
                        batch_size=batch_size,
                        plant_type=plant_type,
                        pendulum_config=pendulum_config,
                        solver_params=solver_params,
                        track_full_stats=True,
                    )
                    _, stats = mpc.run_mpc_goals(
                        x_start, seq, sim_dt=sim_dt,
                        goal_timeout=mpc_defaults['goal_timeout'],
                        goal_threshold=mpc_defaults['goal_threshold'],
                        velocity_threshold=mpc_defaults['velocity_threshold'],
                    )
                    outcomes = stats['goal_outcomes']
                    reached = sum(1 for o in outcomes if o == 'reached')
                    n_reached += reached
                    n_total += len(seq)
                    solve_times.extend(stats['solve_times'])
                    per_sequence.append({
                        'goal_outcomes': outcomes,
                        'n_reached': reached,
                        'n_goals': len(seq),
                        'time_to_all_reached': stats.get('time_to_all_reached'),
                    })

                results[batch_size] = {
                    'success_rate': n_reached / n_total if n_total else 0.0,
                    'n_reached': n_reached,
                    'n_total': n_total,
                    'per_sequence': per_sequence,
                    'avg_solve_time_ms': float(np.mean(solve_times)) if solve_times else None,
                    'success': True,
                }
                if verbose:
                    r = results[batch_size]
                    print(f"  Batch {batch_size}: success_rate="
                          f"{r['success_rate']*100:.1f}% ({n_reached}/{n_total}), "
                          f"avg_solve={r['avg_solve_time_ms']:.3f}ms")
            except Exception as e:
                results[batch_size] = {'error': str(e), 'success': False}
                if verbose:
                    print(f"  Batch {batch_size}: FAILED - {e}")

        self.results = results
        return results

