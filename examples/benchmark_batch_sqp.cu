#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <limits>
#include <iomanip>

#include "multisolve/batch_sqp_solver.cuh"
#include "utils/types.cuh"
#include "utils/utils.h"

// Define a macro to run benchmark for a specific batch size
#define RUN_BENCHMARK(SIZE) \
    { \
        T *d_xu_traj_batch; \
        gpuErrchk(cudaMalloc(&d_xu_traj_batch, TRAJ_SIZE * SIZE * sizeof(T))); \
        \
        ProblemInputs<T, SIZE> inputs; \
        inputs.timestep = static_cast<T>(TIMESTEP); \
        gpuErrchk(cudaMalloc(&inputs.d_x_s_batch, STATE_SIZE * SIZE * sizeof(T))); \
        gpuErrchk(cudaMalloc(&inputs.d_reference_traj_batch, REFERENCE_TRAJ_SIZE * SIZE * sizeof(T))); \
        inputs.d_GRiD_mem = gato::plant::initializeDynamicsConstMem<T>(); \
        \
        SQPSolver<T, SIZE> solver; \
        SQPStats<T, SIZE> stats; \
        \
        for (int k = 0; k < SIZE; k++) { \
            gpuErrchk(cudaMemcpy(d_xu_traj_batch + k * TRAJ_SIZE, xu_trajs[0].data(), \
                TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice)); \
            gpuErrchk(cudaMemcpy(inputs.d_x_s_batch + k * STATE_SIZE, xu_trajs[0].data(), \
                STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice)); \
            gpuErrchk(cudaMemcpy(inputs.d_reference_traj_batch + k * REFERENCE_TRAJ_SIZE, \
                ee_pos_trajs[0].data(), REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice)); \
        } \
        \
        /* warm up run */ \
        stats = solver.solve(d_xu_traj_batch, inputs); \
        \
        for (unsigned j = 0; j < ee_pos_trajs.size(); j++) { \
            solver.reset(); \
            \
            for (int k = 0; k < SIZE; k++) { \
                gpuErrchk(cudaMemcpy(d_xu_traj_batch + k * TRAJ_SIZE, xu_trajs[j].data(), \
                    TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice)); \
                gpuErrchk(cudaMemcpy(inputs.d_x_s_batch + k * STATE_SIZE, xu_trajs[j].data(), \
                    STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice)); \
                gpuErrchk(cudaMemcpy(inputs.d_reference_traj_batch + k * REFERENCE_TRAJ_SIZE, \
                    ee_pos_trajs[j].data(), REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice)); \
                solver.setLambdas(lambda_trajs[j].data(), k); \
            } \
            \
            stats = solver.solve(d_xu_traj_batch, inputs); \
            \
            SQP_solve_times.push_back(stats.solve_time_us); \
            PCG_solve_times.push_back(stats.pcg_stats[0].solve_time_us); \
            PCG_num_iterations.push_back(stats.pcg_stats[0].num_iterations[0]); \
        } \
        \
        gpuErrchk(cudaFree(d_xu_traj_batch)); \
        gpuErrchk(cudaFree(inputs.d_x_s_batch)); \
        gpuErrchk(cudaFree(inputs.d_reference_traj_batch)); \
        gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem); \
    }

int main() {
    //setL2PersistingAccess(1); //TODO play with this param
    //resetL2PersistingAccess();

    constexpr int BATCH_SIZES[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    constexpr int NUM_BATCH_SIZES = sizeof(BATCH_SIZES) / sizeof(BATCH_SIZES[0]);
    
    std::vector<std::vector<T>> ee_pos_trajs = readCSVToVecVec<T>("examples/trajfiles/8_ee_pos_trajs.csv");
    std::vector<std::vector<T>> xu_trajs = readCSVToVecVec<T>("examples/trajfiles/8_xu_trajs.csv");
    std::vector<std::vector<T>> lambda_trajs = readCSVToVecVec<T>("examples/trajfiles/8_lambda_trajs.csv");

    std::vector<float> SQP_solve_times;
    std::vector<float> PCG_solve_times;
    std::vector<int> PCG_num_iterations;
    
    // Add vectors to store statistics for each batch size
    std::vector<float> sqp_means, sqp_stds, sqp_medians, sqp_mins, sqp_maxs;
    std::vector<float> pcg_means, pcg_stds, pcg_medians, pcg_mins, pcg_maxs;
    std::vector<float> iter_means, iter_stds, iter_medians, iter_mins, iter_maxs;

    // Run benchmarks for each batch size
    RUN_BENCHMARK(1);
    RUN_BENCHMARK(2);
    RUN_BENCHMARK(4);
    RUN_BENCHMARK(8);
    RUN_BENCHMARK(16);
    RUN_BENCHMARK(32);
    RUN_BENCHMARK(64);
    RUN_BENCHMARK(128);
    RUN_BENCHMARK(256);
    RUN_BENCHMARK(512);
    
    // Calculate statistics for each batch size
    for (int i = 0; i < NUM_BATCH_SIZES; i++) {
        double sqp_mean = 0.0f, sqp_std = 0.0f;
        double pcg_mean = 0.0f, pcg_std = 0.0f;
        double iter_mean = 0.0f, iter_std = 0.0f;
        
        int offset = i * ee_pos_trajs.size();
        
        // Create temporary vectors for this batch's data to calculate median
        std::vector<float> sqp_batch(ee_pos_trajs.size());
        std::vector<float> pcg_batch(ee_pos_trajs.size());
        std::vector<float> iter_batch(ee_pos_trajs.size());
        
        // Calculate means and collect batch data
        float sqp_min = std::numeric_limits<float>::max();
        float sqp_max = std::numeric_limits<float>::lowest();
        float pcg_min = std::numeric_limits<float>::max();
        float pcg_max = std::numeric_limits<float>::lowest();
        float iter_min = std::numeric_limits<float>::max();
        float iter_max = std::numeric_limits<float>::lowest();
        
        for (size_t j = 0; j < ee_pos_trajs.size(); j++) {
            float sqp_val = SQP_solve_times[offset + j];
            float pcg_val = PCG_solve_times[offset + j];
            float iter_val = PCG_num_iterations[offset + j];
            
            sqp_batch[j] = sqp_val;
            pcg_batch[j] = pcg_val;
            iter_batch[j] = iter_val;
            
            sqp_mean += sqp_val;
            pcg_mean += pcg_val;
            iter_mean += iter_val;
            
            sqp_min = std::min(sqp_min, sqp_val);
            sqp_max = std::max(sqp_max, sqp_val);
            pcg_min = std::min(pcg_min, pcg_val);
            pcg_max = std::max(pcg_max, pcg_val);
            iter_min = std::min(iter_min, iter_val);
            iter_max = std::max(iter_max, iter_val);
        }
        
        sqp_mean /= ee_pos_trajs.size();
        pcg_mean /= ee_pos_trajs.size();
        iter_mean /= ee_pos_trajs.size();
        
        // Calculate standard deviations
        for (size_t j = 0; j < ee_pos_trajs.size(); j++) {
            sqp_std += pow(SQP_solve_times[offset + j] - sqp_mean, 2);
            pcg_std += pow(PCG_solve_times[offset + j] - pcg_mean, 2);
            iter_std += pow(PCG_num_iterations[offset + j] - iter_mean, 2);
        }
        sqp_std = sqrt(sqp_std / ee_pos_trajs.size());
        pcg_std = sqrt(pcg_std / ee_pos_trajs.size());
        iter_std = sqrt(iter_std / ee_pos_trajs.size());
        
        // Calculate medians
        std::sort(sqp_batch.begin(), sqp_batch.end());
        std::sort(pcg_batch.begin(), pcg_batch.end());
        std::sort(iter_batch.begin(), iter_batch.end());
        
        float sqp_median = sqp_batch[ee_pos_trajs.size() / 2];
        float pcg_median = pcg_batch[ee_pos_trajs.size() / 2];
        float iter_median = iter_batch[ee_pos_trajs.size() / 2];
        
        // Store statistics
        sqp_means.push_back(sqp_mean);
        sqp_stds.push_back(sqp_std);
        sqp_medians.push_back(sqp_median);
        sqp_mins.push_back(sqp_min);
        sqp_maxs.push_back(sqp_max);
        
        pcg_means.push_back(pcg_mean);
        pcg_stds.push_back(pcg_std);
        pcg_medians.push_back(pcg_median);
        pcg_mins.push_back(pcg_min);
        pcg_maxs.push_back(pcg_max);
        
        iter_means.push_back(iter_mean);
        iter_stds.push_back(iter_std);
        iter_medians.push_back(iter_median);
        iter_mins.push_back(iter_min);
        iter_maxs.push_back(iter_max);
    }

    // Create directory if it doesn't exist
    std::filesystem::create_directories("benchmark_results");

    // Save statistics to CSV file
    std::ofstream outfile("benchmark_results/benchmark_stats.csv");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open benchmark_results/benchmark_stats.csv for writing" << std::endl;
        return 1;
    }

    // Set precision for floating-point numbers
    outfile << std::fixed << std::setprecision(2);

    // Write SQP solve time statistics
    outfile << "SQP Solve Time Statistics (microseconds)\n";
    outfile << "batch_size,     mean,   std_dev,    median,      min,      max\n";
    for (int i = 0; i < NUM_BATCH_SIZES; i++) {
        outfile << std::setw(10) << BATCH_SIZES[i] << ","
               << std::setw(10) << sqp_means[i] << ","
               << std::setw(10) << sqp_stds[i] << ","
               << std::setw(10) << sqp_medians[i] << ","
               << std::setw(10) << sqp_mins[i] << ","
               << std::setw(10) << sqp_maxs[i] << "\n";
    }
    outfile << "\n";

    // Write PCG solve time statistics
    outfile << "PCG Solve Time Statistics (microseconds)\n";
    outfile << "batch_size,     mean,   std_dev,    median,      min,      max\n";
    for (int i = 0; i < NUM_BATCH_SIZES; i++) {
        outfile << std::setw(10) << BATCH_SIZES[i] << ","
               << std::setw(10) << pcg_means[i] << ","
               << std::setw(10) << pcg_stds[i] << ","
               << std::setw(10) << pcg_medians[i] << ","
               << std::setw(10) << pcg_mins[i] << ","
               << std::setw(10) << pcg_maxs[i] << "\n";
    }
    outfile << "\n";

    // Write PCG iteration statistics
    outfile << "PCG Iteration Statistics\n";
    outfile << "batch_size,     mean,   std_dev,    median,      min,      max\n";
    for (int i = 0; i < NUM_BATCH_SIZES; i++) {
        outfile << std::setw(10) << BATCH_SIZES[i] << ","
               << std::setw(10) << iter_means[i] << ","
               << std::setw(10) << iter_stds[i] << ","
               << std::setw(10) << iter_medians[i] << ","
               << std::setw(10) << iter_mins[i] << ","
               << std::setw(10) << iter_maxs[i] << "\n";
    }

    outfile.close();

    return 0;
}
