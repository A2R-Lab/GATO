
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>

#include "solvers/sqp/sqp_pcg_n.cuh"
#include "gato.cuh"
#include "utils/utils.cuh"

int main(){
    printCudaDeviceProperties();
    printf("\n\n-----");
    // ----------------- Constants -----------------
    const uint32_t solve_count = 512;
    constexpr uint32_t state_size = gato::STATE_SIZE;
    constexpr uint32_t control_size = gato::CONTROL_SIZE;
    constexpr uint32_t knot_points = gato::KNOT_POINTS;
    const linsys_t timestep = gato::TIMESTEP; // 1/64 s
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;

    pcg_config<linsys_t> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = 1e-5;      //1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6
    config.pcg_max_iter = PCG_MAX_ITER;
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);   // TODO: change for batched PCG solver
    
    printf("Solve count: %d\n", solve_count);
    print_test_config();

    // ----------------- Host Memory -----------------

    // sqp return value
    std::tuple<std::vector<std::vector<int>>, std::vector<double>, float, std::vector<uint32_t>, std::vector<char>, std::vector<std::vector<bool>>> sqp_return;

    // data storage for sqp stats
    std::vector<std::vector<int>> pcg_iters_matrix(solve_count);
    std::vector<double> pcg_times_vec;
    float sqp_solve_time = 0.0;
    std::vector<uint32_t> sqp_iterations_vec(solve_count);
    std::vector<char> sqp_time_exit_vec(solve_count);
    std::vector<std::vector<bool>> pcg_exits_matrix(solve_count);

    //read in input goal end effector position trajectory
    auto eePos_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_eepos.traj"); 
    auto xu_traj2d = readCSVToVecVec<linsys_t>("data/trajfiles/0_0_traj.csv"); 
    if(eePos_traj2d.size() < knot_points){ std::cout << "precomputed traj length < knotpoints, not implemented\n"; return 1; }
    std::vector<std::vector<linsys_t>> h_eePos_trajs(solve_count);
    std::vector<std::vector<linsys_t>> h_xu_trajs(solve_count); 
    // Duplicate the trajectory data for each solve (for now)
    for (uint32_t i = 0; i < solve_count; ++i) {
        for (uint32_t j = 0; j < knot_points; ++j) {
            h_eePos_trajs[i].insert(h_eePos_trajs[i].end(), eePos_traj2d[j].begin(), eePos_traj2d[j].end());
            h_xu_trajs[i].insert(h_xu_trajs[i].end(), xu_traj2d[j].begin(), xu_traj2d[j].end());
        }   
    }

    // ----------------- Device Memory -----------------

    void *d_dynmem = gato::plant::initializeDynamicsConstMem<linsys_t>();

    linsys_t *d_eePos_trajs; 
    linsys_t *d_xu_trajs; 
    gpuErrchk(cudaMalloc(&d_eePos_trajs, 6 * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xu_trajs, traj_size * solve_count * sizeof(linsys_t)));

    linsys_t *d_lambdas;
    linsys_t *d_rhos;
    gpuErrchk(cudaMalloc(&d_lambdas, state_size * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMemset(d_lambdas, 0, state_size * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_rhos, solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMemset(d_rhos, 1e-3, solve_count * sizeof(linsys_t)));

    for (uint32_t i = 0; i < solve_count; ++i) {
        gpuErrchk(cudaMemcpy(d_eePos_trajs + i * 6 * knot_points, h_eePos_trajs[i].data(), 6 * knot_points * sizeof(linsys_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_xu_trajs + i * traj_size, h_xu_trajs[i].data(), traj_size * sizeof(linsys_t), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // ----------------- SOLVE -----------------

    sqp_return = sqpSolvePcgN<linsys_t>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_trajs, d_lambdas, d_xu_trajs, d_dynmem, config, d_rhos, 1e-3);

    pcg_iters_matrix = std::get<0>(sqp_return);
    pcg_times_vec = std::get<1>(sqp_return);
    sqp_solve_time = std::get<2>(sqp_return);
    sqp_iterations_vec = std::get<3>(sqp_return);
    sqp_time_exit_vec = std::get<4>(sqp_return);
    pcg_exits_matrix = std::get<5>(sqp_return);

    // ----------------- Print Results -----------------

    std::cout << "\n\nResults:\n";
    std::cout << "SQP iterations: " << sqp_iterations_vec[0] << "\n";
    std::cout << "PCG iters: ";
    for (unsigned long i = 0; i < pcg_iters_matrix[0].size(); ++i) {
        std::cout << pcg_iters_matrix[0][i] << " ";
    }
    std::cout << "\nSQP exits: ";
    for (unsigned long i = 0; i < sqp_time_exit_vec.size(); ++i) {
        std::cout << sqp_time_exit_vec[i] << " ";    
    }
    std::cout << "\n";

        std::cout << "\n-----\n";
  
    std::cout << "\nPCG times (ms): ";
    for (unsigned long i = 0; i < pcg_times_vec.size(); ++i) {
        std::cout <<  i << ": " << pcg_times_vec[i]/1000.0 << " ";
    }
    std::cout << "\n";
    std::cout << "SQP solve time: " << sqp_solve_time/1000.0 << " ms\n";

    // ----------------- Free Memory -----------------

    gato::plant::freeDynamicsConstMem<linsys_t>(d_dynmem);

    gpuErrchk(cudaFree(d_xu_trajs));
    gpuErrchk(cudaFree(d_eePos_trajs));
    gpuErrchk(cudaFree(d_lambdas));
    gpuErrchk(cudaFree(d_rhos));
    
    gpuErrchk(cudaPeekAtLastError());

    return 0;
}



