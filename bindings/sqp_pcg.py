import numpy as np
import gato

# Prepare input data
eePos_goal_traj = np.random.rand(32 * 6).astype(np.float32)
xu = np.random.rand((14 + 7) * 32 - 7).astype(np.float32)
lambda_ = np.zeros(14 * 32, dtype=np.float32)

# Set parameters
rho = 1e-3
rho_reset = 1e-3
pcg_max_iter = 1000
pcg_exit_tol = 1e-6

print("Running SQP-PCG solver using Python bindings...\n")
result = gato.solve_sqp_pcg(
    eePos_goal_traj, xu, lambda_, rho, rho_reset, pcg_max_iter, pcg_exit_tol
)

print("Results:")
print("PCG iterations:", result.pcg_iter_vec)
print("Linear system solve times (us):", result.linsys_time_vec)
print("SQP solve time (us):", result.sqp_solve_time)
print("SQP iterations:", result.sqp_iter)
print("SQP time exit:", result.sqp_time_exit)
print("PCG exit flags:", result.pcg_exit_vec)

# xu and lambda arrays are modified in-place
print("Updated xu:", xu)