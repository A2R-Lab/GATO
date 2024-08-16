#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>

#include "gato.cuh"
#include "sim/mpcsim.cuh"
#include "solvers/sqp/sqp_pcg.cuh"

int main(){
    
    // ----------------- CONFIG -----------------
    constexpr uint32_t state_size = gato::STATE_SIZE; // for kuka iiwa: joints and velocities
    constexpr uint32_t knot_points = gato::KNOT_POINTS; // number of knot points in trajectory
    std::vector<float> pcg_exit_vals = {1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6}; // hardcoded exit tolerances for PCG
    float pcg_exit_tol = pcg_exit_vals[2];

    // check GPU space for pcg
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);   // TODO: change for initialization of batched PCG solver
    print_test_config();


    // ----------------- input trajectory stuff -----------------
    //read in precomputed end effector position

    auto eePos_traj2d = readCSVToVecVec<linsys_t>("../data/trajfiles/0_0_eepos.traj"); 
    auto xu_traj2d = readCSVToVecVec<linsys_t>("../data/trajfiles/0_0_traj.csv");
    if(eePos_traj2d.size() < knot_points){ std::cout << "precomputed traj length < knotpoints, not implemented\n"; exit(1); }
    std::vector<linsys_t> h_eePos_traj, h_xu_traj;
    for (const auto& vec : eePos_traj2d) { h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end()); } //flatten 2D vector into 1D vector
    for (const auto& xu_vec : xu_traj2d) { h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end()); }

    // ----------------- variables and memory allocation -----------------
    MPCLogParams mpc_log_params = {0, 0, 0, "output"}; // start_state_index, goal_state_index, current iteration, test_output_prefix
    std::vector<double> linsys_times; // for storing linsys solve times (if TIME_LINSYS == 1)
    std::vector<uint32_t> sqp_iters; // for storing sqp iterations (if TIME_LINSYS == 0)
    std::vector<toplevel_return_type> current_results; // current results for each test iteration (linsys_times or sqp_iters) (toplevel_return_type is a double in this case)
    std::vector<float> tracking_errs, cur_tracking_errs; // tracking_errs is a vector of tracking errors for all test iterations, cur_tracking_errs is for each test iteration
    double tot_final_tracking_err = 0; // total final tracking error for all test iterations
    
    linsys_t *d_eePos_traj; //device pointer to end effector position trajectory, same size as h_eePos_traj
    linsys_t *d_xu_traj; //device pointer to control trajectory, same size as h_xu_traj
    linsys_t *d_xs; //initial state, size = state_size

    gpuErrchk(cudaMalloc(&d_eePos_traj, h_eePos_traj.size()*sizeof(linsys_t))); // device vector for end effector trajectory
    gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), h_eePos_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_xu_traj, h_xu_traj.size()*sizeof(linsys_t))); // device vector for state and control trajectory
    gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), h_xu_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(linsys_t))); // device vector for initial state
    gpuErrchk(cudaMemcpy(d_xs, h_xu_traj.data(), state_size*sizeof(linsys_t), cudaMemcpyHostToDevice));


    // ----------------- run MPC simulation -----------------
    std::cout << "************************************************\n";
    std::cout << "Running MPC simulation...\n";

    std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> mpc_return = simulateMPC<linsys_t, toplevel_return_type>(
        static_cast<uint32_t>(eePos_traj2d.size()), // total knot points in input trajectory
        d_eePos_traj, 
        d_xu_traj,
        d_xs,
        pcg_exit_tol,
        mpc_log_params // struct [start_state_index, goal_state_index, current iteration, test_output_prefix]
    ); 

#if TIME_LINSYS
        linsys_times = std::get<0>(mpc_return); 
#else
        sqp_iters = std::get<0>(mpc_return); 
#endif
    tracking_errs = std::get<1>(mpc_return); 
    tot_final_tracking_err += std::get<2>(mpc_return);


    // ----------------- free memory and print results -----------------
    gpuErrchk(cudaFree(d_xu_traj));
    gpuErrchk(cudaFree(d_eePos_traj));
    gpuErrchk(cudaFree(d_xs));
    gpuErrchk(cudaPeekAtLastError());

    std::cout << "\n----------------- RESULTS -----------------\n";
    std::cout << "\nTracking err:";
    std::string trackingStats = printStats<float>(&tracking_errs, "trackingerr");
    std::cout << "\nAverage final tracking err: " << tot_final_tracking_err << std::endl;
    std::string linsysOrSqpStats;
    if (linsys_times.size() > 0 || sqp_iters.size() > 0) {
        if (TIME_LINSYS == 1)
        {
            std::cout << "\nLinsys times";
            linsysOrSqpStats = printStats<double>(&linsys_times, "linsystimes");
        }
        else
        {
            std::cout << "\nSqp iters";
            linsysOrSqpStats = printStats<uint32_t>(&sqp_iters, "sqpiters");
        }
        std::cout << "************************************************\n\n";
    }
    return 0;
}


