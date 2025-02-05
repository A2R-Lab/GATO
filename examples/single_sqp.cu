#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "multisolve/batch_sqp.cuh"
#include "utils/types.cuh"
#include "utils/utils.h"

int main() {
    setL2PersistingAccess(1);
    std::vector<T> h_ee_pos_traj = readCSVToVec<T>("examples/trajfiles/ee_pos_traj.csv"); // trajectory needs to be >= KNOT_POINTS
    std::vector<T> h_xu_traj = readCSVToVec<T>("examples/trajfiles/xu_traj.csv");

    T *d_xu_traj;
    gpuErrchk(cudaMalloc(&d_xu_traj, TRAJ_SIZE * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemset(d_xu_traj, 0, TRAJ_SIZE * sizeof(T)));

    T *d_lambda;
    gpuErrchk(cudaMalloc(&d_lambda, VEC_SIZE_PADDED * sizeof(T)));
    gpuErrchk(cudaMemset(d_lambda, 0, VEC_SIZE_PADDED * sizeof(T)));

    ProblemInputs<T, 1> inputs;
    inputs.timestep = static_cast<T>(TIMESTEP);
    gpuErrchk(cudaMalloc(&inputs.d_x_s_batch, STATE_SIZE * sizeof(T)));
    gpuErrchk(cudaMemcpy(inputs.d_x_s_batch, h_xu_traj.data(), STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&inputs.d_reference_traj_batch, REFERENCE_TRAJ_SIZE * sizeof(T)));
    gpuErrchk(cudaMemcpy(inputs.d_reference_traj_batch, h_ee_pos_traj.data(), REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));
    inputs.d_GRiD_mem = gato::plant::initializeDynamicsConstMem<T>();
    
    T rho_penalty = static_cast<T>(1e-3);

    SQPStats<T, 1> stats = solveSQPBatched<T, 1>(
        d_xu_traj,
        d_lambda,
        inputs,
        rho_penalty
    );

    std::cout << "SQP solve time: " << stats.solve_time_us << " us" << std::endl;
    std::cout << "SQP iterations: " << stats.sqp_iterations[0] << std::endl;
    std::cout << "PCG solve time: " << stats.pcg_solve_times[0] << " us" << std::endl;
    std::cout << "Rho max reached: " << (stats.rho_max_reached[0] ? "true" : "false") << std::endl;

    std::cout << "\nPCG stats per iteration:" << std::endl;
    for (size_t i = 0; i < stats.pcg_stats.size(); i++) {
        std::cout << "Iteration " << i << ":" << std::endl;
        std::cout << "  Solve time: " << stats.pcg_stats[i].solve_time_us << " us" << std::endl;
        std::cout << "  Num iterations: " << stats.pcg_stats[i].num_iterations[0] << std::endl;
        std::cout << "  Converged: " << (stats.pcg_stats[i].converged[0] ? "true" : "false") << std::endl;
    }

    std::cout << "\nLine search stats per iteration:" << std::endl; 
    for (size_t i = 0; i < stats.line_search_stats.size(); i++) {
        std::cout << "Iteration " << i << ":" << std::endl;
        std::cout << "  Step size: " << stats.line_search_stats[i].step_size[0] << std::endl;
        std::cout << "  Min merit: " << stats.line_search_stats[i].min_merit[0] << std::endl;
    }

    gpuErrchk(cudaFree(d_xu_traj));
    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(inputs.d_x_s_batch));
    gpuErrchk(cudaFree(inputs.d_reference_traj_batch));
    gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem);

    return 0;
}