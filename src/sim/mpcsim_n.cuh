
#pragma once
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cublas_v2.h>
#include <math.h>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <tuple>
#include <time.h>

#include "gato.cuh"
#include "solvers/sqp/sqp_pcg_n.cuh"
#include "utils/utils.cuh"
#include "mpcsim.cuh"


template <typename T>
auto simulateMPC_n(uint32_t solve_count,
                    const uint32_t state_size,
                    const uint32_t control_size,
                    const uint32_t knot_points,
                    const uint32_t traj_steps,
                    float timestep,
                    T *d_eePos_traj,
                    T *d_xu_traj,
                    T *d_xs,
                    T linsys_exit_tol
){

    // constants
    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;
    const T shift_threshold = SHIFT_THRESHOLD; // shift xu when this far through timestep
    const int max_control_updates = 100000;
    
    std::vector<double> sqp_solve_time_us_vec(solve_count, 0);               // current sqp solve time
    std::vector<double> simulation_time_vec(solve_count, 0);                 // current simulation time
    std::vector<double> prev_simulation_time_vec(solve_count, 0);            // last simulation time
    std::vector<double> time_since_timestep_vec(solve_count, 0);             // time since last timestep of original trajectory
    std::vector<bool> shifted_vec(solve_count, false);                       // has xu been shifted
    std::vector<uint32_t> traj_offset_vec(solve_count, 0);                        // current goal states of original trajectory

    // vars for recording data
    std::vector<std::vector<std::vector<T>>> tracking_path_vec(solve_count);      // list of traversed traj
    std::vector<std::vector<int>> linsys_iters_vec(solve_count);
    std::vector<std::vector<double>> linsys_times_vec(solve_count);
    std::vector<std::vector<double>> sqp_times_vec(solve_count);
    std::vector<std::vector<uint32_t>> sqp_iters_vec(solve_count);
    std::vector<std::vector<bool>> sqp_exits_vec(solve_count);
    std::vector<std::vector<bool>> linsys_exits_vec(solve_count);
    std::vector<std::vector<T>> tracking_errors_vec(solve_count);

    // std::vector<std::vector<int>> cur_linsys_iters_vec(solve_count);
    // std::vector<std::vector<bool>> cur_linsys_exits_vec(solve_count);
    // std::vector<std::vector<double>> cur_linsys_times_vec(solve_count);
    // std::vector<std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>>> sqp_stats_vec(solve_count);
    // std::vector<uint32_t> cur_sqp_iters_vec(solve_count);
    // std::vector<T> cur_tracking_error_vec(solve_count);
    // std::vector<int> control_update_step_vec(solve_count);

    // mpc iterates
    // TODO probably need to be instanced: one per simultaneous solve
    T *d_lambda, *d_eePos_goal, *d_xu, *d_xu_old, *d_rhos;
    gpuErrchk(cudaMalloc(&d_lambda, state_size * knot_points * solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len * solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len * solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eePos_goal, 6 * knot_points * solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_rhos, solve_count * sizeof(T)));

    gpuErrchk(cudaMemset(d_lambda, 0, state_size * knot_points * solve_count * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_eePos_goal, d_eePos_traj, 6 * knot_points * solve_count * sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_xu_traj, traj_len * solve_count * sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len * solve_count * sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemset(d_rhos, 1e-3, solve_count * sizeof(T)));

    void *d_dynmem = gato::plant::initializeDynamicsConstMem<T>();

    // Temporary host and device memory
    T *h_xs = new T[state_size * solve_count];
    T *h_eePos = new T[6 * solve_count];
    T *h_eePos_goal = new T[6 * solve_count];
    T *d_eePos;
    gpuErrchk(cudaMalloc(&d_eePos, 6 * solve_count * sizeof(T)));

    gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size * solve_count * sizeof(T), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < solve_count; i++) {
        tracking_path_vec[i].push_back(std::vector<T>(h_xs + i * state_size, h_xs + (i + 1) * state_size));
    }


     // PCG configuration
    pcg_config<T> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;

    T rho_reset = 1e-3;


    // Pre-processing step to remove jitters
    // pcg_config<T> jitter_config = config;
    // jitter_config.pcg_exit_tol = 1e-11;
    // jitter_config.pcg_max_iter = 10000;
    
    // for(int j = 0; j < 100; j++){
    //     sqpSolvePcgN<T>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, jitter_config, d_rhos, rho_reset);
    //     gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, solve_count * traj_len * sizeof(T), cudaMemcpyDeviceToDevice));
    // }

    // Reset rho after jitter removal
    gpuErrchk(cudaMemset(d_rhos, 1e-3, solve_count * sizeof(T)));

    // Main simulation loop
    for (int control_update_step = 0; control_update_step < max_control_updates; control_update_step++) {
        

        if (std::all_of(traj_offset_vec.begin(), traj_offset_vec.end(), [traj_steps](uint32_t offset) { return offset == traj_steps; })) {
            break;
        }

        std::tuple<std::vector<std::vector<int>>, std::vector<double>, float, std::vector<uint32_t>, std::vector<char>, std::vector<std::vector<bool>>> sqp_stats;
        sqp_stats = sqpSolvePcgN<T>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, d_rhos, rho_reset);

        // Process SQP results
        auto& pcg_iters_matrix = std::get<0>(sqp_stats);
        auto& pcg_times_vec = std::get<1>(sqp_stats);
        float sqp_solve_time = std::get<2>(sqp_stats);
        auto& sqp_iterations_vec = std::get<3>(sqp_stats);
        auto& sqp_time_exit_vec = std::get<4>(sqp_stats);
        auto& pcg_exits_matrix = std::get<5>(sqp_stats);

        for (uint32_t i = 0; i < solve_count; i++) {
            linsys_iters_vec[i].insert(linsys_iters_vec[i].end(), pcg_iters_matrix[i].begin(), pcg_iters_matrix[i].end());
            linsys_times_vec[i].insert(linsys_times_vec[i].end(), pcg_times_vec.begin(), pcg_times_vec.end());
            sqp_solve_time_us_vec[i] = sqp_solve_time;
            sqp_iters_vec[i].push_back(sqp_iterations_vec[i]);
            sqp_exits_vec[i].push_back(sqp_time_exit_vec[i]);
            linsys_exits_vec[i].insert(linsys_exits_vec[i].end(), pcg_exits_matrix[i].begin(), pcg_exits_matrix[i].end());
            // Simulate trajectory
#if CONST_UPDATE_FREQ
            simulation_time_vec[i] = SIMULATION_PERIOD;
#else
            simulation_time_vec[i] = sqp_solve_time_us_vec[i];
#endif
            //TODO: batch this
            simple_simulate<T>(state_size, control_size, knot_points, 
                               d_xs + i * state_size, 
                               d_xu + i * traj_len, 
                               d_dynmem, timestep, 
                               prev_simulation_time_vec[i], 
                               simulation_time_vec[i]);

            //TODO: xu_old??? in mpcsim
                    // old xu = new xu
            //gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));

            time_since_timestep_vec[i] += simulation_time_vec[i] * 1e-6;

            // Handle trajectory shifting and goal updates
            if (!shifted_vec[i] && time_since_timestep_vec[i] > shift_threshold) {
                // Record tracking error
                grid::end_effector_positions_kernel<T><<<1, 128>>>(d_eePos + i * 6, d_xs + i * state_size, grid::NUM_JOINTS, (grid::robotModel<T> *)d_dynmem, 1);
                gpuErrchk(cudaMemcpy(h_eePos + i * 6, d_eePos + i * 6, 6 * sizeof(T), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(h_eePos_goal + i * 6, d_eePos_goal + i * 6 * knot_points, 6 * sizeof(T), cudaMemcpyDeviceToHost));

                T cur_tracking_error = 0.0;
                for (uint32_t j = 0; j < 3; j++) {
                    cur_tracking_error += abs(h_eePos[j + i * 6] - h_eePos_goal[j + i * 6]);
                }
                tracking_errors_vec[i].push_back(cur_tracking_error);

                traj_offset_vec[i]++;

                // Shift xu, eePos_goal, and lambda
                just_shift<T>(state_size, control_size, d_xu + i * traj_len);
                just_shift<T>(6, 0, d_eePos_goal + i * 6 * knot_points);
                just_shift<T>(state_size, 0, d_lambda + i * state_size * knot_points);

                // Update last elements of xu and eePos_goal
                if (traj_offset_vec[i] + knot_points < traj_steps) {
                    gpuErrchk(cudaMemcpy(d_xu + i * traj_len + traj_len - (state_size + control_size), 
                                         d_xu_traj + i * traj_len + (state_size + control_size) * traj_offset_vec[i] - control_size, 
                                         (state_size + control_size) * sizeof(T), cudaMemcpyDeviceToDevice));
                    gpuErrchk(cudaMemcpy(d_eePos_goal + i * 6 * knot_points + (knot_points - 1) * 6, 
                                         d_eePos_traj + i * 6 * traj_steps + (traj_offset_vec[i] + knot_points - 1) * 6, 
                                         6 * sizeof(T), cudaMemcpyDeviceToDevice));
                } else {
                    // Fill with final goal state
                    gpuErrchk(cudaMemcpy(d_xu + i * traj_len + traj_len - state_size, 
                                         d_xu_traj + i * traj_len + (traj_steps - 1) * (state_size + control_size), 
                                         state_size * sizeof(T), cudaMemcpyDeviceToDevice));
                    gpuErrchk(cudaMemset(d_xu + i * traj_len + traj_len - (state_size + control_size), 0, control_size * sizeof(T)));
                    gpuErrchk(cudaMemcpy(d_eePos_goal + i * 6 * knot_points + (knot_points - 1) * 6, 
                                         d_eePos_traj + i * 6 * traj_steps + (traj_steps - 1) * 6, 
                                         6 * sizeof(T), cudaMemcpyDeviceToDevice));
                }

                shifted_vec[i] = true;
            }

            if (time_since_timestep_vec[i] > timestep) {
                shifted_vec[i] = false;
                time_since_timestep_vec[i] = std::fmod(time_since_timestep_vec[i], timestep);
            }

            prev_simulation_time_vec[i] = simulation_time_vec[i];

            // Update d_xu with new states
            gpuErrchk(cudaMemcpy(d_xu + i * traj_len, d_xs + i * state_size, state_size * sizeof(T), cudaMemcpyDeviceToDevice));
               
            // Record trajectory data
            //gpuErrchk(cudaMemcpy(h_xs + i * state_size, d_xs + i * state_size, state_size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        for (uint32_t i = 0; i < solve_count; i++) {
            //tracking_path_vec[i].emplace_back(h_xs + i * state_size, h_xs + (i + 1) * state_size);
            sqp_times_vec[i].push_back(sqp_solve_time_us_vec[i]);
        }

        gpuErrchk(cudaPeekAtLastError());
    }

    // --- DONE WITH SIMULATION LOOP ---

    // Compute final tracking errors
    std::vector<T> final_tracking_errors(solve_count);
    for (uint32_t i = 0; i < solve_count; i++) {
        grid::end_effector_positions_kernel<T><<<1, 128>>>(d_eePos + i * 6, d_xs + i * state_size, grid::NUM_JOINTS, (grid::robotModel<T> *)d_dynmem, 1);
        gpuErrchk(cudaMemcpy(h_eePos + i * 6, d_eePos + i * 6, 6 * sizeof(T), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_eePos_goal + i * 6, d_eePos_goal + i * 6 * knot_points, 6 * sizeof(T), cudaMemcpyDeviceToHost));
        
        final_tracking_errors[i] = 0.0;
        for (uint32_t j = 0; j < 3; j++) {
            final_tracking_errors[i] += abs(h_eePos[i * 6 + j] - h_eePos_goal[i * 6 + j]);
        }
    }

    // Clean up
    gato::plant::freeDynamicsConstMem<T>(d_dynmem);
    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_eePos_goal));
    //gpuErrchk(cudaFree(d_xu_old));
    gpuErrchk(cudaFree(d_eePos));
    gpuErrchk(cudaFree(d_rhos));
    delete[] h_xs;
    delete[] h_eePos;
    delete[] h_eePos_goal;

    return std::make_tuple(sqp_times_vec, linsys_times_vec, tracking_errors_vec, final_tracking_errors);

}

