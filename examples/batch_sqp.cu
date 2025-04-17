#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "multisolve/batch_sqp_solver.cuh"
#include "types.cuh"
#include "utils/utils.h"

int main() {
    constexpr int BATCH_SIZE = 16;
    
    //setL2PersistingAccess(1); //TODO play with this param
    //resetL2PersistingAccess();
    
    std::vector<T> h_ee_pos_traj = readCSVToVec<T>("examples/trajfiles/ee_pos_traj.csv");
    std::vector<T> h_xu_traj = readCSVToVec<T>("examples/trajfiles/xu_traj.csv");

    // Allocate batch memory and copy the same initial trajectory BATCH_SIZE times
    T *d_xu_traj_batch;
    gpuErrchk(cudaMalloc(&d_xu_traj_batch, TRAJ_SIZE * BATCH_SIZE * sizeof(T)));
    for (int i = 0; i < BATCH_SIZE; i++) {
        gpuErrchk(cudaMemcpy(d_xu_traj_batch + i * TRAJ_SIZE, h_xu_traj.data(), 
            TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Setup identical problem inputs for each batch
    ProblemInputs<T, BATCH_SIZE> inputs;
    inputs.timestep = static_cast<T>(0.01);
    gpuErrchk(cudaMalloc(&inputs.d_x_s_batch, STATE_SIZE * BATCH_SIZE * sizeof(T)));
    gpuErrchk(cudaMalloc(&inputs.d_reference_traj_batch, REFERENCE_TRAJ_SIZE * BATCH_SIZE * sizeof(T)));
    for (int i = 0; i < BATCH_SIZE; i++) { // Copy the same initial state and reference trajectory for each batch
        gpuErrchk(cudaMemcpy(inputs.d_x_s_batch + i * STATE_SIZE, 
            h_xu_traj.data(), STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(inputs.d_reference_traj_batch + i * REFERENCE_TRAJ_SIZE,
            h_ee_pos_traj.data(), REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));
    }
    inputs.d_GRiD_mem = gato::plant::initializeDynamicsConstMem<T>();

    SQPSolver<T, BATCH_SIZE> solver;

    // warm up run
    SQPStats<T, BATCH_SIZE> stats = solver.solve(
        d_xu_traj_batch,
        inputs
    );

    solver.reset();

    stats = solver.solve(
        d_xu_traj_batch,
        inputs
    );

    std::cout << "***** Stats *****" << std::endl;
    bool trajectories_equal = checkIfBatchTrajsMatch<T, BATCH_SIZE>(d_xu_traj_batch);
    std::cout << "All trajectories equal: " << (trajectories_equal ? "true" : "false") << std::endl;
    std::cout << "SQP num iterations: ";
    for (int i = 0; i < std::min(BATCH_SIZE, 10); i++) {
        std::cout << stats.sqp_iterations[i] << " ";
    }
    if (BATCH_SIZE > 10) std::cout << "...";
    std::cout << std::endl;
    std::cout << "SQP solve time (us): " << stats.solve_time_us << std::endl;
    std::cout << "PCG num iterations: " << std::endl;
    for (unsigned i = 0; i < stats.pcg_stats.size(); i++) {
        std::cout << "  SQP iteration " << i << ": ";
        for (int j = 0; j < std::min(BATCH_SIZE, 10); j++) {
            std::cout << stats.pcg_stats[i].num_iterations[j] << " ";
        }
        if (BATCH_SIZE > 10) std::cout << "...";
        std::cout << std::endl;
    }
    std::cout << "PCG solve times (us): ";
    for (unsigned i = 0; i < stats.pcg_stats.size(); i++) {
        std::cout << stats.pcg_stats[i].solve_time_us << " ";
    }
    std::cout << std::endl;

    gpuErrchk(cudaFree(d_xu_traj_batch));
    gpuErrchk(cudaFree(inputs.d_x_s_batch));
    gpuErrchk(cudaFree(inputs.d_reference_traj_batch));
    gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem);

    return 0;
}
