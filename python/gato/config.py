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

# ---------------------------------------------------------------------------
# SolverParams — THE solver configuration (plan 2.A, 2026-09-20).
#
# One default set. These are the values every paper experiment and closed-loop
# harness ran with (the former config.DEFAULT_SOLVER_PARAMS dict); the old
# BSQP constructor carried a second, never-exercised default set (mu=1,
# qd_cost=1e-4, rho=1e-3, max_sqp_iters=10, ...) that disagreed on 8 knobs —
# it is gone. max_sqp_iters=1 is the real-time-iteration MPC setting: a
# one-shot solve to convergence should raise it (e.g. params.replace(
# max_sqp_iters=10)). The public kkt_tol knob was removed: convergence is
# merit/step based; the device-side KKT check it promised was never wired.
# ---------------------------------------------------------------------------
from dataclasses import dataclass, asdict, replace as _dc_replace


@dataclass(frozen=True)
class SolverParams:
    """BSQP solver configuration. Immutable; derive variants with ``.replace()``.

    SQP / linear system
        max_sqp_iters: SQP iterations per solve (1 = RTI/MPC; raise for one-shot).
        max_pcg_iters, pcg_tol: PCG budget / tolerance (pcg and bdsv_first paths).
        solve_ratio: fraction of the batch that must converge before early exit.
        mu: merit-function constraint weight.
        rho: trust-region / regularization floor (> 0: f32 bdsv returns garbage
            steps at rho=0); adapt_rho lets the line search adapt it per solve.
        linsys: "pcg" | "bdsv" | "bdsv_first" | None (None = wired default:
            pcg fixed-base, bdsv floating-base — the static arm of the
            controller's per-step policy; see gato.linsys_autotune.resolve_linsys).
    Cost weights (grid_plant tracking cost)
        q_cost (running EE position), N_cost (terminal EE position), qd_cost,
        u_cost, and the clamped-log-barrier limit weights q_lim_cost /
        vel_lim_cost / ctrl_lim_cost. Per-knot and per-joint refinements are
        runtime setters on BSQP (set_cost_weights_per_knot, set_q_pos_cost,
        set_u_cost_vec, set_fc_cost/ref) and take precedence over these.
    exact_hessian: SO-SQP stage-Hessian projection (needs an "eh" variant module).
    """
    max_sqp_iters: int = 1
    max_pcg_iters: int = 200
    pcg_tol: float = 1e-4
    solve_ratio: float = 1.0
    mu: float = 10.0
    rho: float = 0.01
    adapt_rho: bool = True
    linsys: str | None = None
    q_cost: float = 2.0
    qd_cost: float = 1e-2
    u_cost: float = 2e-6
    N_cost: float = 50.0
    q_lim_cost: float = 0.01
    vel_lim_cost: float = 0.0
    ctrl_lim_cost: float = 0.0
    exact_hessian: bool = False

    def replace(self, **changes):
        return _dc_replace(self, **changes)

    def asdict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, m):
        """Build from a dict of field overrides (unknown keys are an error)."""
        if isinstance(m, SolverParams):
            return m
        m = dict(m or {})
        bad = set(m) - set(cls.__dataclass_fields__)
        if bad:
            raise TypeError(f"unknown SolverParams field(s): {sorted(bad)}")
        return cls(**m)


COST_FIELDS = ("q_cost", "qd_cost", "u_cost", "N_cost", "q_lim_cost", "vel_lim_cost", "ctrl_lim_cost")



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


