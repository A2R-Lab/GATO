"""
Configuration settings for GATO experiments and benchmarks.
"""

import numpy as np

# Standard batch sizes for experiments
STANDARD_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
EXPERIMENT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]  # For interactive experiments

# Figure-8 trajectory parameters
FIG8_DEFAULT_PARAMS = {
    'A_x': 0.4,           # X amplitude
    'A_z': 0.4,           # Z amplitude  
    'offset': [0.0, 0.5, 0.6],  # Center offset
    'period': 6,          # Period for one cycle
    'cycles': 5,          # Number of cycles
    'theta': np.pi/4      # Rotation angle
}

# Standard starting configurations for Indy7
INDY7_START_CONFIGS = {
    'zero': np.zeros(6),
    'home': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'ready': np.array([-1.096711, -0.09903229, 0.83125766, -0.10907673, 0.49704404, 0.01499449])
}

# Standard starting configurations for IIWA14
IIWA14_START_CONFIGS = {
    'zero': np.zeros(7),
    'home': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

# MPC solver parameters
DEFAULT_SOLVER_PARAMS = {
    'max_sqp_iters': 1,
    'kkt_tol': 0.001,
    'max_pcg_iters': 200,
    'pcg_tol': 1e-4,
    'solve_ratio': 1.0,
    'mu': 10.0,
    'q_cost': 2.0,
    'qd_cost': 1e-2,
    'u_cost': 2e-6,
    'N_cost': 50.0,
    'q_lim_cost': 0.01,
    'vel_lim_cost': 0.0,
    'ctrl_lim_cost': 0.0,
    'rho': 0.01
}

PICKPLACE_SOLVER_PARAMS = {
    'max_sqp_iters': 5,
    'kkt_tol': 0.0,
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
    'velocity_threshold': 1.0
}

# Visualization colors for different batch sizes
BATCH_COLORS = {
    1: '#003192',    # Barnard Blue
    4: '#747474',    # Gray
    8: '#7030A0',    # Purple
    16: '#F19759',   # Orange
    32: '#00693E',   # Dartmouth Green
    64: '#56B4E9',   # Sky Blue
    128: '#C90016',  # Harvard Crimson
    256: '#FF69B4',  # Pink
    512: '#8B4513',  # Brown
    1024: '#000000'  # Black
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
