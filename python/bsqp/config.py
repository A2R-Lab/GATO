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
    'rho': 0.01
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
