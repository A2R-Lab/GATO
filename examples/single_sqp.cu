#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "multisolve/batch_sqp_solver.cuh"
#include "types.cuh"
#include "utils/utils.h"

int main() {
    setL2PersistingAccess(1);
    std::vector<T> h_ee_pos_traj = readCSVToVec<T>("examples/trajfiles/ee_pos_traj.csv"); // trajectory needs to be >= KNOT_POINTS
    std::vector<T> h_xu_traj = readCSVToVec<T>("examples/trajfiles/xu_traj.csv");

    T *d_xu_traj;
    gpuErrchk(cudaMalloc(&d_xu_traj, TRAJ_SIZE * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemset(d_xu_traj, 0, TRAJ_SIZE * sizeof(T)));

    ProblemInputs<T, 1> inputs;
    
    inputs.timestep = static_cast<T>(TIMESTEP);

    gpuErrchk(cudaMalloc(&inputs.d_x_s_batch, STATE_SIZE * sizeof(T)));
    gpuErrchk(cudaMemcpy(inputs.d_x_s_batch, h_xu_traj.data(), STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&inputs.d_reference_traj_batch, REFERENCE_TRAJ_SIZE * sizeof(T)));
    gpuErrchk(cudaMemcpy(inputs.d_reference_traj_batch, h_ee_pos_traj.data(), REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));

    inputs.d_GRiD_mem = gato::plant::initializeDynamicsConstMem<T>();

    SQPSolver<T, 1> solver;

    SQPStats<T, 1> stats = solver.solve(
        d_xu_traj,
        inputs
    );

    std::cout << "***** Stats *****" << std::endl;
    std::cout << "SQP num iterations: " << stats.sqp_iterations[0] << std::endl;
    std::cout << "SQP solve time: " << stats.solve_time_us << " us" << std::endl;
    std::cout << "PCG num iterations: " << stats.pcg_stats[0].num_iterations[0] << std::endl;
    std::cout << "PCG solve time: " << stats.pcg_stats[0].solve_time_us << " us" << std::endl;

    gpuErrchk(cudaFree(d_xu_traj));
    gpuErrchk(cudaFree(inputs.d_x_s_batch));
    gpuErrchk(cudaFree(inputs.d_reference_traj_batch));
    gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem);

    return 0;
}