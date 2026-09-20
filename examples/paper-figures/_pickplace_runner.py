"""CS3 pick-and-place experiment runner + its constants (Fig-7 / Table-I).

The paper-experiment settings below (solver params, success gates, pendulum
payload, goal sequence) are experiment config, not solver API — they moved here
from gato.config/gato.common (2026-07) and from _common.py (2026-09-20). Only
reproduce_fig7_pickplace.py and the archived Phase-0 diagnostic use them.
"""
import numpy as np
from typing import Dict, List

import _common as C
from gato.config import IIWA14_START_CONFIGS

PICKPLACE_SOLVER_PARAMS = {
    'max_sqp_iters': 5,
    'max_pcg_iters': 100,
    'pcg_tol': 1e-6,
    'solve_ratio': 1.0,
    'mu': 10.0,
    'q_cost': 5.0,
    'qd_cost': 1e-2,
    'u_cost': 5e-7,
    'N_cost': 50.0,
    'q_lim_cost': 0.0,
    'vel_lim_cost': 0.0,
    'ctrl_lim_cost': 0.0,
    'rho': 0.001
}

PICKPLACE_MPC_DEFAULTS = {
    'goal_timeout': 5.0,
    'goal_threshold': 0.05,
    'velocity_threshold': 1.0,
    # Paper-comparable metrics (2026-07-08 gap investigation): the paper's
    # "total joint velocity < 1.0 rad/s" is read as the Euclidean norm (the L1
    # sum over 7 joints is materially stricter and capped success), and fixed
    # pacing makes the completion clock physical task time instead of
    # cumulative wall time (also makes runs deterministic).
    'velocity_norm': 2,
    'pace_by_solve_time': False,
}

# Pendulum parameter defaults
PENDULUM_DEFAULT_PARAMS = {
    'mass': 15.0,           # kg
    'length': 0.3,          # m
    'damping': 0.4,         # Nms/rad
    'initial_angle': np.array([0.3, 0.0, 0.0])  # axis-angle (radians)
}

# Default pick&place goal sequence (IIWA14 workspace)
PICKPLACE_DEFAULT_GOALS = [
    np.array([0.5, -0.1865, 0.5]),
    np.array([0.5, 0.5, 0.2]),
    np.array([0.3, 0.3, 0.8]),
    np.array([0.6, -0.5, 0.2]),
    np.array([0.0, -0.5, 0.8])
]


def sample_axis_angle(mag_range=(0.0, 0.6)):
    """Random axis-angle vector (uniform magnitude in mag_range, uniform direction)
    for the pendulum initial condition."""
    mag = np.random.uniform(*mag_range)
    v = np.random.normal(size=3)
    axis = v / (np.linalg.norm(v) + 1e-12)
    return axis * mag


def sample_pendulum_params(length_range=(0.3, 0.7), damping_range=(0.1, 0.6),
                           angle_range=(0.0, 0.6), mass=15.0):
    """Random pendulum configuration for the scenario sweeps: {mass, length,
    damping, initial_angle} with length/damping/|angle| uniform in their ranges."""
    return {
        'mass': mass,
        'length': np.random.uniform(*length_range),
        'damping': np.random.uniform(*damping_range),
        'initial_angle': sample_axis_angle(angle_range)
    }


class ExperimentRunner:
    """Manages and runs GATO pick-place experiments with multiple batch sizes."""

    def __init__(self, plant: str = 'iiwa14'):
        """Build the robot model for a registered plant (the paper's CS3a is iiwa14)."""
        self.plant = plant
        self.urdf_path, self.model_dir, self.model = C.resolve_model(plant)
        self.results = {}

    def run_pickplace_sweep(
        self,
        batch_sizes: List[int] = None,
        N: int = 16,
        dt: float = 0.05,
        sim_dt: float = 0.001,
        plant_type: str = None,
        goal_sequences: List[List[np.ndarray]] = None,
        pendulum_config: Dict = None,
        solver_params: Dict = None,
        mpc_defaults: Dict = None,
        start_config: str = 'home',
        fc_config: Dict = None,
        wrench_id: Dict = None,
        verbose: bool = True,
    ) -> Dict:
        """
        CS3a / Table-I: pick-and-place success rate vs. batch size.

        For each batch size, runs the goal-reaching MPC (with an end-effector
        pendulum/payload) over one or more goal sequences and records the success
        rate. A goal counts as 'reached' iff the modern `run_mpc_goals` returns
        'reached' for it (``||ee - goal|| < goal_threshold`` AND
        ``norm(dq, ord=velocity_norm) < velocity_threshold`` before the per-goal
        timeout; PICKPLACE_MPC_DEFAULTS uses the Euclidean norm + fixed pacing —
        paper-comparable physical completion clock). The paper's finding is
        that this success rate climbs with batch size.

        Args:
            batch_sizes: batch sizes to sweep (default _common.STANDARD_BATCH_SIZES).
            N, dt, sim_dt: horizon / MPC step / sim step.
            plant_type: dynamics plant (default: the runner's plant).
            goal_sequences: list of goal-position lists. Default = a single
                sequence (PICKPLACE_DEFAULT_GOALS). Pass several (e.g. randomized
                sequences) to get a true multi-trial success *rate* per batch.
            pendulum_config: EE payload (default PENDULUM_DEFAULT_PARAMS).
            solver_params: BSQP params (default PICKPLACE_SOLVER_PARAMS).
            mpc_defaults: goal_timeout / goal_threshold / velocity_threshold /
                velocity_norm / pace_by_solve_time (default PICKPLACE_MPC_DEFAULTS).
            start_config: IIWA14_START_CONFIGS key for the initial robot state.
            wrench_id: least-squares wrench-identification arm (B=1): dict of
                OneStepWrenchIdentifier options ({} for defaults). Mutually
                exclusive with the ForceEstimator batch.
            fc_config: contact-force arm — {'cost', 'pin_torque_rows'} passed to
                MPC_GATO so the solver's own contact-wrench slots (the "fc"
                module variant) explain the payload (the B=1 alternative to the
                hypothesis batch). None = the ForceEstimator arm of record.

        Returns:
            {batch_size: {success_rate, n_reached, n_total, per_sequence:[...],
                          avg_solve_time_ms, success}} (also stored on self.results).
        """
        from gato.mpc_gato import MPC_GATO

        if batch_sizes is None:
            batch_sizes = C.STANDARD_BATCH_SIZES
        if goal_sequences is None:
            goal_sequences = [PICKPLACE_DEFAULT_GOALS]
        if pendulum_config is None:
            pendulum_config = PENDULUM_DEFAULT_PARAMS
        if solver_params is None:
            solver_params = PICKPLACE_SOLVER_PARAMS
        if mpc_defaults is None:
            mpc_defaults = PICKPLACE_MPC_DEFAULTS
        plant_type = plant_type or self.plant

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
                        # paper numbers were measured under the pcg path
                        # (controller default is "auto" since 08-12); an
                        # explicit caller linsys still wins
                        params=solver_params, linsys="pcg",
                        track_full_stats=True,
                        variant="fc" if fc_config else None,
                        fc_config=fc_config,
                        wrench_id=wrench_id,
                    )
                    _, stats = mpc.run_mpc_goals(
                        x_start, seq, sim_dt=sim_dt,
                        goal_timeout=mpc_defaults['goal_timeout'],
                        goal_threshold=mpc_defaults['goal_threshold'],
                        velocity_threshold=mpc_defaults['velocity_threshold'],
                        velocity_norm=mpc_defaults.get('velocity_norm', 1),
                        pace_by_solve_time=mpc_defaults.get('pace_by_solve_time', True),
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
