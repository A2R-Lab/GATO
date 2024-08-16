import numpy as np
import gato

def solve_sqp_pcg_n(solve_count, eePos_goal_traj, xu_traj, pcg_exit_tol=1e-5, pcg_max_iter=1000, rho_init=1e-3, rho_reset=1e-3):
    """
    Solve the SQP problem using PCG on CUDA for multiple trajectories.

    Args:
        solve_count (int): Number of trajectories to solve.
        eePos_goal_traj (np.ndarray): End effector goal trajectory.
        xu_traj (np.ndarray): Initial state and control trajectory.
        pcg_exit_tol (float): PCG exit tolerance.
        pcg_max_iter (int): Maximum number of PCG iterations.
        rho_init (float): Initial rho value.
        rho_reset (float): Rho reset value.

    Returns:
        SQPResult: Object containing the results of the SQP solve.
    """
    # Ensure input arrays are contiguous and in single precision
    eePos_goal_traj = np.ascontiguousarray(eePos_goal_traj, dtype=np.float32)
    xu_traj = np.ascontiguousarray(xu_traj, dtype=np.float32)

    result = gato.solve_sqp_pcg_n(
        solve_count,
        eePos_goal_traj,
        xu_traj,
        pcg_exit_tol,
        pcg_max_iter,
        rho_init,
        rho_reset
    )

    return result


# Example usage
if __name__ == "__main__":
    # Load your data here (example shapes, adjust as needed)
    solve_count = 1
    rho_init = 1e-3
    rho_reset = 1e-3
    pcg_max_iter = 173
    pcg_exit_tol = 1e-5

    eePos_goal_traj = np.random.rand(solve_count, 32, 6).astype(np.float32)  # Example shape
    xu_traj = np.random.rand(solve_count, 32, 18).astype(np.float32)

    result = solve_sqp_pcg_n(
        solve_count,
        eePos_goal_traj,
        xu_traj,
        pcg_exit_tol,
        pcg_max_iter,
        rho_init,
        rho_reset
    )

    print("SQP solve time:", result.sqp_solve_time)
    print("SQP iterations:", result.sqp_iterations_vec)
    print("PCG iterations matrix shape:", np.array(result.pcg_iters_matrix).shape)
    print("PCG times vector length:", len(result.pcg_times_vec))