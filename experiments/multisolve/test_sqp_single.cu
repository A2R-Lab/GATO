#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include "solvers/sqp/sqp_pcg.cuh"
#include "gato.cuh"
#include "utils/utils.cuh"

int main() {
    printCudaDeviceProperties();
    printf("\n\n-----\n");

    const uint32_t state_size = gato::STATE_SIZE;
    const uint32_t control_size = gato::CONTROL_SIZE;
    const uint32_t knot_points = gato::KNOT_POINTS;
    //const linsys_t timestep = gato::TIMESTEP;
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;

    pcg_config<linsys_t> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = 1e-5;
    config.pcg_max_iter = PCG_MAX_ITER;

    // Read input data
    auto eePos_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_eepos.traj");
    auto xu_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_traj.csv");
    if(eePos_traj2d.size() < knot_points) {
        std::cout << "precomputed traj length < knotpoints, not implemented\n";
        return 1;
    }

    // Prepare host data
    std::vector<linsys_t> h_eePos_traj;
    std::vector<linsys_t> h_xu_traj;
    for (uint32_t j = 0; j < knot_points; ++j) {
        h_eePos_traj.insert(h_eePos_traj.end(), eePos_traj2d[j].begin(), eePos_traj2d[j].end());
        h_xu_traj.insert(h_xu_traj.end(), xu_traj2d[j].begin(), xu_traj2d[j].end());
    }

    // Allocate and copy device memory
    void *d_dynmem = gato::plant::initializeDynamicsConstMem<linsys_t>();
    linsys_t *d_eePos_traj, *d_xu_traj, *d_lambdas;
    linsys_t rho = 1e-4;
    gpuErrchk(cudaMalloc(&d_eePos_traj, 6 * knot_points * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xu_traj, traj_size * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_lambdas, state_size * knot_points * sizeof(linsys_t)));

    gpuErrchk(cudaMemset(d_lambdas, 0, state_size * knot_points * sizeof(linsys_t)));

    gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), 6 * knot_points * sizeof(linsys_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), traj_size * sizeof(linsys_t), cudaMemcpyHostToDevice));


    // Warm-up run
    sqpSolvePcg<linsys_t>(d_eePos_traj, d_xu_traj, d_lambdas, d_dynmem, config, rho, 1e-3);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    auto result = sqpSolvePcg<linsys_t>(d_eePos_traj, d_xu_traj, d_lambdas, d_dynmem, config, rho, 1e-3);
    auto end = std::chrono::high_resolution_clock::now();
    
    double solve_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    std::cout << "Single solve time: " << solve_time << " ms" << std::endl;
    std::cout << "Solve result: " << std::get<2>(result) << " us" << std::endl;

    //Print sqp iterations
    printf("sqp iterations: %d\n", std::get<3>(result));


    // Clean up
    gato::plant::freeDynamicsConstMem<linsys_t>(d_dynmem);
    gpuErrchk(cudaFree(d_eePos_traj));
    gpuErrchk(cudaFree(d_xu_traj));
    gpuErrchk(cudaFree(d_lambdas));

    return 0;
}