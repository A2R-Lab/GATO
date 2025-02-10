import numpy as np
import torch
import batch_sqp
import time

def read_csv_to_array(filename):
    return np.loadtxt(filename, delimiter=',', dtype=np.float32)

def main():
    BATCH_SIZE = 64
    
    # Read trajectory files
    ee_pos_trajs = read_csv_to_array("../examples/trajfiles/32_ee_pos_trajs.csv")
    xu_trajs = read_csv_to_array("../examples/trajfiles/32_xu_trajs.csv")
    
    xu_traj = xu_trajs[0]
    ee_pos_traj = ee_pos_trajs[0]
    
    xu_traj_batch = np.tile(xu_traj, (BATCH_SIZE, 1))
    x_s_batch = np.tile(xu_traj[:14], (BATCH_SIZE, 1))  # STATE_SIZE = 14
    reference_traj_batch = np.tile(ee_pos_traj, (BATCH_SIZE, 1))
    
    # Create solver instance (using float32)
    solver = batch_sqp.SQPSolverfloat_64()
    
    timestep = 0.015625
    result = solver.solve(
        xu_traj_batch,
        timestep,
        x_s_batch,
        reference_traj_batch
    )
    
    solver.reset()
    
    result = solver.solve(
        xu_traj_batch,
        timestep,
        x_s_batch,
        reference_traj_batch
    )
    
    # Print statistics
    print("***** Stats *****")
    
    trajectories = result["xu_trajectory"]
    trajectories_equal = np.allclose(
        trajectories[0], 
        trajectories[1:],
        rtol=1e-5,
        atol=1e-5
    )
    print(f"All trajectories equal: {trajectories_equal}")
    
    # Print SQP iterations
    sqp_iterations = result["sqp_iterations"]
    print("SQP num iterations:", end=" ")
    for i in range(min(BATCH_SIZE, 10)):
        print(f"{sqp_iterations[i]}", end=" ")
    if BATCH_SIZE > 10:
        print("...")
    else:
        print()
    
    # Print solve time
    print(f"SQP solve time (us): {result['solve_time_us']}")
    
    # Print PCG iterations
    print("PCG num iterations:")
    for i, pcg_stat in enumerate(result["pcg_stats"]):
        print(f"  SQP iteration {i}:", end=" ")
        iterations = pcg_stat["pcg_iterations"]
        for j in range(min(BATCH_SIZE, 10)):
            print(f"{iterations[j]}", end=" ")
        if BATCH_SIZE > 10:
            print("...")
        print()
    
    # Print PCG solve times
    print("PCG solve times (us):", end=" ")
    for pcg_stat in result["pcg_stats"]:
        print(f"{pcg_stat['solve_time_us']}", end=" ")
    print()

if __name__ == "__main__":
    main()