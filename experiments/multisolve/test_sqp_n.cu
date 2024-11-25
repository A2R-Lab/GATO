#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>
#include <chrono>
#include <algorithm>

#include "solvers/sqp/sqp_pcg_n_DEV.cuh"
#include "gato.cuh"
#include "utils/utils.cuh"

// Function to run the experiment for a given batch size
std::tuple<double, double> runExperiment(uint32_t solve_count) {
    const uint32_t state_size = gato::STATE_SIZE;
    const uint32_t control_size = gato::CONTROL_SIZE;
    const uint32_t knot_points = gato::KNOT_POINTS;
    const linsys_t timestep = gato::TIMESTEP;
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;

    pcg_config<linsys_t> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = 1e-5;
    config.pcg_max_iter = PCG_MAX_ITER;

    // Read input data (assuming this part remains the same)
    auto eePos_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_eepos.traj"); 
    auto xu_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_traj.csv"); 
    if(eePos_traj2d.size() < knot_points){ 
        std::cout << "precomputed traj length < knotpoints, not implemented\n"; 
        return std::make_tuple(0.0, 0.0); 
    }

    // Prepare host data
    std::vector<std::vector<linsys_t>> h_eePos_trajs(solve_count);
    std::vector<std::vector<linsys_t>> h_xu_trajs(solve_count); 
    for (uint32_t i = 0; i < solve_count; ++i) {
        for (uint32_t j = 0; j < knot_points; ++j) {
            h_eePos_trajs[i].insert(h_eePos_trajs[i].end(), eePos_traj2d[j].begin(), eePos_traj2d[j].end());
            h_xu_trajs[i].insert(h_xu_trajs[i].end(), xu_traj2d[j].begin(), xu_traj2d[j].end());
        }   
    }

    // Allocate and copy device memory
    void *d_dynmem = gato::plant::initializeDynamicsConstMem<linsys_t>();
    linsys_t *d_eePos_trajs, *d_xu_trajs, *d_lambdas, *d_rhos;
    gpuErrchk(cudaMalloc(&d_eePos_trajs, 6 * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xu_trajs, traj_size * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_lambdas, state_size * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_rhos, solve_count * sizeof(linsys_t)));

    gpuErrchk(cudaMemset(d_lambdas, 0, state_size * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMemset(d_rhos, 1e-3, solve_count * sizeof(linsys_t)));

    for (uint32_t i = 0; i < solve_count; ++i) {
        gpuErrchk(cudaMemcpy(d_eePos_trajs + i * 6 * knot_points, h_eePos_trajs[i].data(), 6 * knot_points * sizeof(linsys_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_xu_trajs + i * traj_size, h_xu_trajs[i].data(), traj_size * sizeof(linsys_t), cudaMemcpyHostToDevice));
    }

    // Warm-up run
    sqpSolvePcgN<linsys_t>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_trajs, d_lambdas, d_xu_trajs, d_dynmem, config, d_rhos, 1e-3);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    auto result = sqpSolvePcgN<linsys_t>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_trajs, d_lambdas, d_xu_trajs, d_dynmem, config, d_rhos, 1e-3);
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_per_solve = total_time / solve_count;

    // Clean up
    gato::plant::freeDynamicsConstMem<linsys_t>(d_dynmem);
    gpuErrchk(cudaFree(d_eePos_trajs));
    gpuErrchk(cudaFree(d_xu_trajs));
    gpuErrchk(cudaFree(d_lambdas));
    gpuErrchk(cudaFree(d_rhos));

    return std::make_tuple(total_time, avg_time_per_solve);
}

int main() {
    printCudaDeviceProperties();
    printf("\n\n-----");

    std::vector<uint32_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<std::tuple<uint32_t, double, double>> results;

    for (uint32_t batch_size : batch_sizes) {
        std::cout << "Running experiment with batch size: " << batch_size << std::endl;
        auto [total_time, avg_time] = runExperiment(batch_size);
        results.emplace_back(batch_size, total_time, avg_time);

        // Optional: Print intermediate results
        std::cout << "Batch Size: " << batch_size 
                  << ", Total Time: " << total_time << " ms"
                  << ", Avg Time per Solve: " << avg_time << " ms" << std::endl;
    }

    // Print final results
    std::cout << "\n\nFinal Results:\n";
    std::cout << "Batch Size | Total Time (ms) | Avg Time per Solve (ms)\n";
    std::cout << "-------------------------------------------------------\n";
    for (const auto& [batch_size, total_time, avg_time] : results) {
        std::cout << std::setw(10) << batch_size << " | " 
                  << std::setw(15) << std::fixed << std::setprecision(3) << total_time << " | "
                  << std::setw(22) << std::fixed << std::setprecision(3) << avg_time << "\n";
    }

    // Save results to a CSV file
    std::ofstream csv_file("batch_experiment_results.csv");
    csv_file << "Batch Size,Total Time (ms),Avg Time per Solve (ms)\n";
    for (const auto& [batch_size, total_time, avg_time] : results) {
        csv_file << batch_size << "," << total_time << "," << avg_time << "\n";
    }
    csv_file.close();

    return 0;
}



