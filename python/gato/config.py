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

# Standard starting configurations for IIWA14.
#
# ⚠ 'zero' and 'home' are BOTH all-zeros: the arm extended straight up. That is a
# kinematic singularity, and a consequential one — a VERTICAL end-effector force
# produces zero joint torque there (|J^T w| = 1.4e-6 vs > 1 once bent), so a
# hanging payload is completely UNOBSERVABLE to any wrench estimator until the arm
# moves away. Manipulability is 0.0 and cond(J) ~ 2e10.
#
# They are kept as-is because they are the inputs of record for the bitwise parity
# baseline (tools/parity_baseline.py) and the test suite. Start closed-loop
# EXPERIMENTS from 'ready' instead.
IIWA14_START_CONFIGS = {
    'zero': np.zeros(7),
    'home': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # Mid-workspace elbow pose. The signs ALTERNATE down the arm (shoulder +,
    # elbow -, wrist +): curling joints 2/4/6 the same way stacks the bends and
    # leaves the arm folded near the base, which is both less manipulable and
    # further from the workspace. Measured against the all-zeros start:
    #   |J^T w| (vertical-force observability)  0.00  -> 77.8
    #   sigma_min(J)                            0.000 -> 0.159
    #   cond(J)                                 2e10  -> 11.6
    #   manipulability sqrt(det(J J^T))         0.000 -> 0.103
    # EE lands at [0.64, 0.17, 0.64], inside joint limits with margin.
    'ready': np.array([0.0, np.pi/6, np.pi/8, -np.pi/3, 0.0, np.pi/3, 0.0]),
}

# Standard starting configurations for the go2 quadruped (STORED layout:
# [p(3); quat xyzw(4); 12 actuated joints hip/thigh/calf x FL/FR/RL/RR]).
# 'standing' is the nominal stance (base 0.35 m up, legs [0, 0.9, -1.8]);
# it is the fingerprint rest posture and every go2 gate's start.
GO2_START_CONFIGS = {
    'standing': np.concatenate([
        [0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0],
        np.tile([0.0, 0.9, -1.8], 4),
    ]),
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


