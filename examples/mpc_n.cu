#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>

#include "sim/mpcsim_n.cuh"
#include "solvers/sqp/sqp_pcg_n.cuh"
#include "gato.cuh"
#include "utils/utils.cuh"

int main() {
    // Set up parameters
    const uint32_t solve_count = 20; 
    const uint32_t state_size = gato::STATE_SIZE;
    const uint32_t control_size = gato::CONTROL_SIZE;
    const uint32_t knot_points = gato::KNOT_POINTS;
    
    const float timestep = 0.015625f;
    const linsys_t linsys_exit_tol = 1e-5f;
    //const uint32_t traj_len = (state_size + control_size) * knot_points - control_size;


    //read in input goal end effector position trajectory
    auto eePos_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_eepos.traj"); 
    auto xu_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_traj.csv"); 
    if(eePos_traj2d.size() < knot_points){ std::cout << "precomputed traj length < knotpoints, not implemented\n"; return 1; }
    const uint32_t traj_steps = eePos_traj2d.size();
    std::vector<linsys_t> h_eePos_traj, h_xu_traj, h_xs;
    for (uint32_t i = 0; i < solve_count; ++i) {
        // Flatten and copy data
        for (const auto& row : eePos_traj2d) { h_eePos_traj.insert(h_eePos_traj.end(), row.begin(), row.end()); }
        for (const auto& row : xu_traj2d) { h_xu_traj.insert(h_xu_traj.end(), row.begin(), row.end()); }
        // Copy initial state
        h_xs.insert(h_xs.end(), xu_traj2d[0].begin(), xu_traj2d[0].begin() + state_size);
    }


    linsys_t *d_eePos_trajs, *d_xu_trajs, *d_xs;
    gpuErrchk(cudaMalloc(&d_eePos_trajs, h_eePos_traj.size() * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xu_trajs, h_xu_traj.size() * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xs, h_xs.size() * sizeof(linsys_t)));

    // Copy data to device
    gpuErrchk(cudaMemcpy(d_eePos_trajs, h_eePos_traj.data(), h_eePos_traj.size() * sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xu_trajs, h_xu_traj.data(), h_xu_traj.size() * sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xs, h_xs.data(), h_xs.size() * sizeof(linsys_t), cudaMemcpyHostToDevice));

    // Run batched MPC simulation
    auto result = simulateMPC_n<linsys_t>(
        solve_count, state_size, control_size, knot_points, traj_steps,
        timestep, d_eePos_trajs, d_xu_trajs, d_xs, linsys_exit_tol
    );

    // Process and print results
    auto& sqp_times_vec = std::get<0>(result);
    auto& linsys_times_vec = std::get<1>(result);
    auto& tracking_errors_vec = std::get<2>(result);
    auto& final_tracking_errors = std::get<3>(result);

    for (uint32_t i = 0; i < solve_count; ++i) {
        std::cout << "Solve " << i << " results:" << std::endl;
        std::cout << "  Average SQP time: " << std::accumulate(sqp_times_vec[i].begin(), sqp_times_vec[i].end(), 0.0) / sqp_times_vec[i].size() << " us" << std::endl;
        std::cout << "  Average PCG solve time: " << std::accumulate(linsys_times_vec[i].begin(), linsys_times_vec[i].end(), 0.0) / linsys_times_vec[i].size() << " us" << std::endl;
        std::cout << "  Average tracking error: " << std::accumulate(tracking_errors_vec[i].begin(), tracking_errors_vec[i].end(), 0.0) / tracking_errors_vec[i].size() << std::endl;
        std::cout << "  Final tracking error: " << final_tracking_errors[i] << std::endl;
    }

    // Clean up
    gpuErrchk(cudaFree(d_eePos_trajs));
    gpuErrchk(cudaFree(d_xu_trajs));
    gpuErrchk(cudaFree(d_xs));


    return 0;
}