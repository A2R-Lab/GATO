"""
Common utilities for GATO trajectory optimization.
Shared functions used by both benchmark scripts and notebooks.
"""

import numpy as np
import pinocchio as pin


def figure8(dt, A_x=0.4, A_z=0.4, offset=[0.0, 0.5, 0.6], period=6, cycles=5, theta=np.pi/4):
    """
    Generate figure 8 trajectory for end-effector tracking.
    
    Args:
        dt: Time step
        A_x: Amplitude in X direction  
        A_z: Amplitude in Z direction
        offset: 3D offset for trajectory center
        period: Period of one figure-8 cycle
        cycles: Number of cycles to generate
        theta: Rotation angle around Z-axis
    
    Returns:
        Flattened array of trajectory points [x, y, z, 0, 0, 0] for each timestep
    """
    x_unrot = lambda t: offset[0] + A_x * np.sin(t)
    y_unrot = lambda t: offset[1]
    z_unrot = lambda t: offset[2] + A_z * np.sin(2*t)/2 + A_z/2
    
    # Rotation matrix around Z-axis
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    
    def get_rotated_coords(t):
        unrot = np.array([x_unrot(t), y_unrot(t), z_unrot(t)])
        rot = R @ unrot
        return rot[0], rot[1], rot[2]
    
    x = lambda t: get_rotated_coords(t)[0]
    y = lambda t: get_rotated_coords(t)[1]
    z = lambda t: get_rotated_coords(t)[2]
    
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    fig_8 = np.array([[x(t), y(t), z(t), 0.0, 0.0, 0.0] for t in timesteps]).reshape(-1)
    return np.tile(fig_8, int(cycles))


def rk4(model, data, q, dq, u, dt, fext=None):
    """
    RK4 integration for forward dynamics.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint positions
        dq: Joint velocities
        u: Control torques
        dt: Time step
        fext: External forces (optional)
    
    Returns:
        q_next: Joint positions at next timestep
        dq_next: Joint velocities at next timestep
    """
    if fext is None:
        fext = pin.StdVec_Force()
        for _ in range(model.njoints):
            fext.append(pin.Force.Zero())
    
    # RK4 integration steps
    k1q = dq
    k1v = pin.aba(model, data, q, dq, u, fext)
    
    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = dq + k1v * dt/2
    k2v = pin.aba(model, data, q2, k2q, u, fext)
    
    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = dq + k2v * dt/2
    k3v = pin.aba(model, data, q3, k3q, u, fext)
    
    q4 = pin.integrate(model, q, k3q * dt)
    k4q = dq + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u, fext)
    
    dq_next = dq + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    avg_dq = (k1q + 2*k2q + 2*k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_dq * dt)
    
    return q_next, dq_next

def initialize_warm_start(x_start, N, nx, nu):
    """Initialize warm start trajectory."""
    XU = np.zeros(N*(nx+nu)-nu)
    for i in range(N):
        start_idx = i * (nx + nu)
        XU[start_idx:start_idx+nx] = x_start
    return XU

def collect_tracking_stats(q, dq, ee_goal, model, data, gpu_time_us, solver_stats):
    """
    Collect tracking statistics for a single MPC step.
    
    Returns:
        Dictionary with tracking metrics
    """
    ee_pos = get_ee_position(model, data, q)
    goal_dist = np.linalg.norm(ee_pos[:3] - ee_goal[6:9])
    
    return {
        'goal_distance': goal_dist,
        'ee_actual': ee_pos.copy(),
        'ee_goal': ee_goal[6:9].copy(),
        'gpu_time_ms': gpu_time_us / 1000.0,
        'sqp_iters': solver_stats.get('sqp_iters', 0),
        'pcg_iters': solver_stats.get('pcg_iters', [0])[0] if 'pcg_iters' in solver_stats else 0
    }


def sample_axis_angle(mag_range=(0.0, 0.6)):
    """
    Sample random axis-angle vector for pendulum initial condition.
    
    Args:
        mag_range: Tuple of (min_magnitude, max_magnitude) in radians
        
    Returns:
        3D axis-angle vector
    """
    mag = np.random.uniform(*mag_range)
    # Random direction on unit sphere
    v = np.random.normal(size=3)
    n = np.linalg.norm(v) + 1e-12
    axis = v / n
    return axis * mag


def sample_pendulum_params(length_range=(0.3, 0.7), damping_range=(0.1, 0.6), 
                          angle_range=(0.0, 0.6), mass=15.0):
    """
    Sample random pendulum configuration for parameter sweeps.
    
    Args:
        length_range: Tuple of (min, max) pendulum length in meters
        damping_range: Tuple of (min, max) damping coefficient in Nms/rad
        angle_range: Tuple of (min, max) initial angle magnitude in radians
        mass: Fixed mass in kg
        
    Returns:
        Dictionary with pendulum configuration
    """
    return {
        'mass': mass,
        'length': np.random.uniform(*length_range),
        'damping': np.random.uniform(*damping_range),
        'initial_angle': sample_axis_angle(angle_range)
    }
