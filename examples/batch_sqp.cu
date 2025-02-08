#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "multisolve/batch_sqp_solver.cuh"
#include "utils/types.cuh"
#include "utils/utils.h"


template<typename T, uint32_t BatchSize>
bool checkIfBatchTrajsMatch(T* d_xu_traj_batch) {
    std::vector<T> h_xu_traj_batch(TRAJ_SIZE * BatchSize);
    gpuErrchk(cudaMemcpy(h_xu_traj_batch.data(), d_xu_traj_batch, 
        TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

    // Compare each trajectory to the first one
    for (uint32_t i = 1; i < BatchSize; i++) {
        for (uint32_t j = 0; j < TRAJ_SIZE; j++) {
            if (std::abs(h_xu_traj_batch[j] - h_xu_traj_batch[i * TRAJ_SIZE + j]) > 1e-10) {
                std::cout << "Mismatch found at trajectory " << i << ", index " << j << std::endl;
                std::cout << "Expected: " << h_xu_traj_batch[j] 
                         << ", Got: " << h_xu_traj_batch[i * TRAJ_SIZE + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    constexpr int BATCH_SIZE = 16;
    
    setL2PersistingAccess(1); //TODO play with this param
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
    inputs.timestep = static_cast<T>(TIMESTEP);
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

    SQPStats<T, BATCH_SIZE> stats = solver.solve(
        d_xu_traj_batch,
        inputs
    );

    bool trajectories_equal = checkIfBatchTrajsMatch<T, BATCH_SIZE>(d_xu_traj_batch);
    std::cout << "All trajectories equal: " << (trajectories_equal ? "true" : "false") << std::endl;

    std::cout << "\nSQP solve time: " << stats.solve_time_us << " us" << std::endl;
    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "\nBatch " << i << " statistics:" << std::endl;
        std::cout << "SQP iterations: " << stats.sqp_iterations[i] << std::endl;
        std::cout << "Rho max reached: " << (stats.rho_max_reached[i] ? "true" : "false") << std::endl;
    }

    // Cleanup
    gpuErrchk(cudaFree(d_xu_traj_batch));
    gpuErrchk(cudaFree(inputs.d_x_s_batch));
    gpuErrchk(cudaFree(inputs.d_reference_traj_batch));
    gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem);

    return 0;
}
