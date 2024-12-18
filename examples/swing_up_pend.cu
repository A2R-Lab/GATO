#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>

#include "sim/mpcsim.cuh"
#include "gato.cuh"
#include "utils/utils.cuh"
#include "GBD-PCG/include/pcg.cuh"

int main(){

    constexpr uint32_t state_size = 2;
    constexpr uint32_t knot_points = gato::KNOT_POINTS;
    const uint32_t total_trajsteps = 1000;
    const uint32_t traj_test_iters = TEST_ITERS;

    // checks GPU space for pcg
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);    

    print_test_config();
    // where to store test results — manually create this directory
    std::string output_directory_path = "build/results/";

    const uint32_t recorded_states = 5;
    const uint32_t start_goal_combinations = recorded_states*recorded_states;

    int start_state, goal_state;
    linsys_t *d_eePos_traj, *d_xu_traj, *d_xs;

    for(uint32_t ind = 0; ind < start_goal_combinations; ind++){

        start_state = ind % recorded_states;
        goal_state = ind / recorded_states;
        if(start_state == goal_state && start_state != 0){ continue; }
        std::cout << "start: " << start_state << " goal: " << goal_state << std::endl;

        uint32_t num_exit_vals = 5;
        float pcg_exit_vals[num_exit_vals];
        if(knot_points==32){
            pcg_exit_vals[0] = 5e-6;
            pcg_exit_vals[1] = 7.5e-6;
            pcg_exit_vals[2] = 5e-6;
            pcg_exit_vals[3] = 2.5e-6;
            pcg_exit_vals[4] = 1e-6;
        }
        else if(knot_points==64){
            pcg_exit_vals[0] = 5e-5;
            pcg_exit_vals[1] = 7.5e-5;
            pcg_exit_vals[2] = 5e-5;
            pcg_exit_vals[3] = 2.5e-5;
            pcg_exit_vals[4] = 1e-5;
        }
        else{
            pcg_exit_vals[0] = 1e-5;
            pcg_exit_vals[1] = 5e-5;
            pcg_exit_vals[2] = 1e-4;
            pcg_exit_vals[3] = 5e-4;
            pcg_exit_vals[4] = 1e-3;
        }


        for (uint32_t pcg_exit_ind = 0; pcg_exit_ind < num_exit_vals; pcg_exit_ind++){

            float pcg_exit_tol = pcg_exit_vals[pcg_exit_ind];
			std::vector<double> linsys_times;
			std::vector<uint32_t> sqp_iters;
			std::vector<toplevel_return_type> current_results;
			std::vector<float> tracking_errs;
			std::vector<float> cur_tracking_errs;
			double tot_final_tracking_err = 0;

			std::string test_output_prefix = output_directory_path + std::to_string(knot_points) + "_" + ( (LINSYS_SOLVE == 1) ? "PCG" : "QDLDL") + "_" + std::to_string(pcg_exit_tol);
			printf("Logging test results to files with prefix %s \n", test_output_prefix.c_str()); 

			for (uint32_t single_traj_test_iter = 0; single_traj_test_iter < traj_test_iters; single_traj_test_iter++){

			std::vector<linsys_t> h_eePos_traj;
			std::vector<linsys_t> h_xu_traj;
			for (unsigned i = 0; i < total_trajsteps; i ++) {
				// adding the goal
				h_eePos_traj.push_back(3.14159);
				h_eePos_traj.push_back(0);
				// init to zeros
				h_xu_traj.push_back(0);
				h_xu_traj.push_back(0);
				h_xu_traj.push_back(0);
			}


			gpuErrchk(cudaMalloc(&d_eePos_traj, h_eePos_traj.size()*sizeof(linsys_t)));
			gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), h_eePos_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
			
			gpuErrchk(cudaMalloc(&d_xu_traj, h_xu_traj.size()*sizeof(linsys_t)));
			gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), h_xu_traj.size()*sizeof(linsys_t), cudaMemcpyHostToDevice));
			
			gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(linsys_t)));
			gpuErrchk(cudaMemcpy(d_xs, h_xu_traj.data(), state_size*sizeof(linsys_t), cudaMemcpyHostToDevice));

			MPCLogParams mpc_log_params = {start_state, goal_state, single_traj_test_iter, test_output_prefix};

			std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> trackingstats = simulateMPC<linsys_t, toplevel_return_type>(static_cast<uint32_t>(total_trajsteps),
				d_eePos_traj, d_xu_traj, d_xs, pcg_exit_tol, mpc_log_params);
			
			current_results = std::get<0>(trackingstats);
			if (TIME_LINSYS == 1) {
				linsys_times.insert(linsys_times.end(), current_results.begin(), current_results.end());
			} else {
				sqp_iters.insert(sqp_iters.end(), current_results.begin(), current_results.end());
			}

			cur_tracking_errs = std::get<1>(trackingstats);
			tracking_errs.insert(tracking_errs.end(), cur_tracking_errs.begin(), cur_tracking_errs.end());

			tot_final_tracking_err += std::get<2>(trackingstats);

			gpuErrchk(cudaFree(d_xu_traj));
			gpuErrchk(cudaFree(d_eePos_traj));
			gpuErrchk(cudaFree(d_xs));
			gpuErrchk(cudaPeekAtLastError());
		
		}

		std::cout << "Completed at " << getCurrentTimestamp() << std::endl;
		std::cout << "\nRESULTS*************************************\n";
		std::cout << "Exit tol: " << pcg_exit_tol << std::endl;
		std::cout << "\nTracking err";
		std::string trackingStats = printStats<float>(&tracking_errs, "trackingerr");
		std::cout << "Average final tracking err: " << tot_final_tracking_err / traj_test_iters << std::endl;
		std::string linsysOrSqpStats;
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

		// Specify the CSV file path
		const std::string csvFilePath = test_output_prefix + "_" + "overall_stats.csv";

		// Open the CSV file for writing
		std::ofstream csvFile(csvFilePath);
		if (!csvFile.is_open()) {
		std::cerr << "Error opening CSV file for writing." << std::endl;
		return 1;
		}

		// Write the header row
		csvFile << "Average,Std Dev, Min, Max, Median, Q1, Q3\n";

		// Write the data rows
		csvFile << getStatsString(trackingStats) << "\n";
		csvFile << getStatsString(linsysOrSqpStats) << "\n";

		// Close the CSV file
		csvFile.close();
	}
        break;
    }




    return 0;
}