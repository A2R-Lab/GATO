#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <time.h>

#include "gato.cuh"
#include "kernels/setup_kkt_n.cuh"
#include "kernels/setup_schur_pcg_n.cuh" 
#include "kernels/pcg_n.cuh" // batched pcg
#include "kernels/compute_dz.cuh"
#include "kernels/compute_merit_n.cuh" // merit function
#include "kernels/line_search_n.cuh" // line search

/**
 * @brief Solve trajectory optimization problem using sequential quadratic programming (SQP) with preconditioned conjugate gradient (PCG) method.
 * 
 * @tparam T float or double
 * @param solve_count number of solves to run in parallel
 * @param state_size state size (joint angles and velocities)
 * @param control_size control size (torques)
 * @param knot_points number of knot points in trajectory
 * @param timestep timestep between knot points
 * @param d_eePos_goal_tensor end effector goal trajectory (6 * knot_points * solve_count)
 * @param d_lambdas initial guess for lambdas (state_size * knot_points * solve_count)
 * @param d_xu_tensor initial guess for state and control trajectory ((state_size + control_size) * knot_points - control_size) * solve_count
 * @param d_dynMem_const pointer to dynamics memory
 * @param config 
 * @param d_rhos 
 * @param rho_reset 
 * @return auto 
 */
template <typename T>
auto sqpSolvePcgN(const uint32_t solve_count, const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, float timestep, 
                    T* d_eePos_goal_tensor, 
                    T* d_lambdas, 
                    T* d_xu_tensor, 
                    void* d_dynMem_const, 
                    pcg_config<T>& config, 
                    T* d_rhos, 
                    T rho_reset) {
    // ------------------ Constants --------------
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;
    const uint32_t G_size = (state_size * state_size + control_size * control_size) * knot_points - control_size * control_size;
    const uint32_t C_size = (state_size * state_size + state_size * control_size) * (knot_points - 1);
    
    // ------------------ Data storage --------------
    std::vector<std::vector<int>> pcg_iters_matrix(solve_count);
    std::vector<double> pcg_times_vec;
    std::vector<uint32_t> sqp_iterations_vec(solve_count, 0);
    std::vector<char> sqp_time_exit_vec(solve_count, true);
    std::vector<std::vector<bool>> pcg_exits_matrix(solve_count);

    std::vector<uint32_t> h_pcg_iters(solve_count);
    std::vector<char> h_pcg_exit(solve_count); //char replaces bool

    uint32_t *d_sqp_iterations_vec;
    bool *d_sqp_time_exit_vec;
    gpuErrchk(cudaMalloc(&d_sqp_iterations_vec, solve_count * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_sqp_time_exit_vec, solve_count * sizeof(bool)));
    
    T *d_xs; // current state
    gpuErrchk(cudaMalloc(&d_xs, solve_count * state_size * sizeof(T)));
    for (uint32_t i = 0; i < solve_count; ++i) {
        gpuErrchk(cudaMemcpy(d_xs + i * state_size, d_xu_tensor + i * traj_size, state_size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    // ------------------ KKT Matrices --------------
    T *d_G_dense, *d_C_dense, *d_g, *d_c, *d_dz;
    gpuErrchk(cudaMalloc(&d_G_dense, solve_count * G_size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_C_dense, solve_count * C_size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_g, solve_count * traj_size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_c, solve_count * state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_dz, solve_count * traj_size * sizeof(T)));
    
    // ------------------ Schur Matrices --------------
    T *d_S, *d_Pinv, *d_gamma;
    gpuErrchk(cudaMalloc(&d_S, solve_count * state_size * state_size * knot_points * sizeof(T) * 3));
    gpuErrchk(cudaMalloc(&d_Pinv, solve_count * state_size * state_size * knot_points * sizeof(T) * 3));
    gpuErrchk(cudaMalloc(&d_gamma, solve_count * state_size * knot_points * sizeof(T)));

    // ------------------ Rho --------------
    std::vector<T> rhos(solve_count);
    std::vector<T> drho_vec(solve_count, 1.0);
    const T rho_factor = RHO_FACTOR;
    const T rho_max = RHO_MAX;
    const T rho_min = RHO_MIN;

    T *d_drho_vec;
    gpuErrchk(cudaMalloc(&d_drho_vec, solve_count * sizeof(T)));
    gpuErrchk(cudaMemset(d_drho_vec, 1.0, solve_count * sizeof(T)));

    // ------------------ PCG --------------
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, solve_count * state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, solve_count * state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, solve_count * (knot_points + 1) * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, solve_count * (knot_points + 1) * sizeof(T)));

    uint32_t *d_pcg_iters; // number of PCG iterations for each solve (return value)
    bool *d_pcg_exit; // whether PCG converged for each solve (return value)
    gpuErrchk(cudaMalloc(&d_pcg_iters, solve_count * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_pcg_exit, solve_count * sizeof(bool)));
    gpuErrchk(cudaMemset(d_pcg_exit, false, solve_count * sizeof(bool)));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // ------------------ Merit function and line search --------------
    const size_t merit_smem_size = get_merit_kernel_smem_size<T>();
    const float mu = 10.0f;
    const uint32_t num_alphas = 8;
    T h_merit_news[solve_count * num_alphas];
    T h_merit_initial[solve_count];
    std::vector<T> h_min_merit(solve_count, 0);
    std::vector<uint32_t> h_line_search_step(solve_count, 0);

    T *d_min_merit;
    T *d_merit_initial;
    T *d_merit_news;
    T *d_merit_temp;
    uint32_t *d_line_search_step;
    gpuErrchk(cudaMalloc(&d_min_merit, solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_initial, solve_count * sizeof(T)));
    gpuErrchk(cudaMemset(d_merit_initial, 0, solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_news, solve_count * num_alphas * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_temp, num_alphas * knot_points * solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_line_search_step, solve_count * sizeof(uint32_t)));

    
    // ------------------ CUDA stuff --------------
    cudaStream_t streams[num_alphas];
    for(uint32_t str = 0; str < num_alphas; str++){
        cudaStreamCreate(&streams[str]);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // ------------------ Timing --------------
    cudaEvent_t kkt_start, kkt_stop, schur_start, schur_stop, pcg_start, pcg_stop, dz_start, dz_stop, line_search_start, line_search_stop, sqp_start, sqp_stop;
    cudaEventCreate(&kkt_start);
    cudaEventCreate(&kkt_stop);
    cudaEventCreate(&schur_start);
    cudaEventCreate(&schur_stop);
    cudaEventCreate(&pcg_start);
    cudaEventCreate(&pcg_stop);
    cudaEventCreate(&dz_start);
    cudaEventCreate(&dz_stop);
    cudaEventCreate(&line_search_start);
    cudaEventCreate(&line_search_stop);
    cudaEventCreate(&sqp_start);
    cudaEventCreate(&sqp_stop);

    cudaEventRecord(sqp_start);
    float milliseconds = 0;

    struct timespec sqp_solve_start, sqp_solve_end;
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);
    

#if CONST_UPDATE_FREQ
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif

    // ------------------ Compute Initial Merit --------------

    initial_merit_n<T>(solve_count, state_size, control_size, knot_points, d_xu_tensor, 
        d_eePos_goal_tensor, static_cast<T>(10), timestep, d_dynMem_const, d_merit_initial);
    
    gpuErrchk(cudaMemcpy(&h_merit_initial, d_merit_initial, solve_count*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // ------------------ SQP loop --------------

    for (uint32_t iter = 0; iter < SQP_MAX_ITER; ++iter) {
        
        // ------------------ KKT Matrices --------------
        cudaEventRecord(kkt_start);

        setup_kkt_n<T>(solve_count, knot_points, state_size, control_size, d_G_dense, d_C_dense, 
            d_g, d_c, d_dynMem_const, timestep, d_eePos_goal_tensor, d_xs, d_xu_tensor);
        
        gpuErrchk(cudaPeekAtLastError());        
        cudaEventRecord(kkt_stop);
        cudaEventSynchronize(kkt_stop);
        cudaEventElapsedTime(&milliseconds, kkt_start, kkt_stop);
        std::cout << "\nTime elapsed for forming KKT system: " << milliseconds << std::endl;
        if (sqpTimecheck()){ break; }
        // ------------------ Schur Matrices --------------
        cudaEventRecord(schur_start);

        form_schur_system_n<T>(solve_count, state_size, control_size, knot_points, 
            d_G_dense, d_C_dense, d_g, d_c, d_S, d_Pinv, d_gamma, d_rhos);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaEventRecord(schur_stop);
        cudaEventSynchronize(schur_stop);
        cudaEventElapsedTime(&milliseconds, schur_start, schur_stop);
        std::cout << "Time elapsed for forming Schur system: " << milliseconds << std::endl;
        if (sqpTimecheck()){ break; }
        // ------------------ PCG --------------
        cudaEventRecord(pcg_start);

        pcg_n(solve_count, state_size, knot_points, d_S, d_Pinv, d_gamma, d_lambdas, 
            d_r, d_p, d_v_temp, d_eta_new_temp, d_pcg_iters, d_pcg_exit, &config);

        gpuErrchk(cudaMemcpy(h_pcg_iters.data(), d_pcg_iters, solve_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_pcg_exit.data(), d_pcg_exit, solve_count * sizeof(char), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < solve_count; ++i) {
            pcg_iters_matrix[i].push_back(h_pcg_iters[i]);
            pcg_exits_matrix[i].push_back(h_pcg_exit[i]);
        }
        gpuErrchk(cudaPeekAtLastError());
        cudaEventRecord(pcg_stop);
        cudaEventSynchronize(pcg_stop);
        cudaEventElapsedTime(&milliseconds, pcg_start, pcg_stop);
        pcg_times_vec.push_back(milliseconds);
        std::cout << "Time elapsed for PCG: " << milliseconds << std::endl;
        if (sqpTimecheck()){ break; }
        // ------------------ Recover dz --------------
        cudaEventRecord(dz_start);

        //TODO: batch this
        for (uint32_t i = 0; i < solve_count; ++i) {
            compute_dz<T>(d_G_dense + i * G_size, d_C_dense + i * C_size, 
                        d_g + i * traj_size, d_lambdas + i * state_size * knot_points, 
                        d_dz + i * traj_size);
        }

        gpuErrchk(cudaPeekAtLastError());
        cudaEventRecord(dz_stop);
        cudaEventSynchronize(dz_stop);
        cudaEventElapsedTime(&milliseconds, dz_start, dz_stop);
        std::cout << "Time elapsed for computing dz: " << milliseconds << std::endl;
        if (sqpTimecheck()){ break; }
        // ------------------ Line Search --------------
        cudaEventRecord(line_search_start);

        compute_ls_merit_n<T>(solve_count, state_size, control_size, knot_points, 
                                    d_xs, d_xu_tensor, d_eePos_goal_tensor, 
                                    mu, timestep, d_dynMem_const, d_dz, 
                                    num_alphas, d_merit_news, d_merit_temp);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        find_alpha_n(solve_count, num_alphas,
            d_merit_news, d_merit_initial, d_min_merit, d_line_search_step);

        gpuErrchk(cudaMemcpyAsync(h_min_merit.data(), d_min_merit, solve_count * sizeof(T), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpyAsync(h_line_search_step.data(), d_line_search_step, solve_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpyAsync(rhos.data(), d_rhos, solve_count * sizeof(T), cudaMemcpyDeviceToHost));
        if (sqpTimecheck()){ break; }

        // update xu and rho
        for (uint32_t i = 0; i < solve_count; ++i) {
            uint32_t line_search_step = h_line_search_step[i];
            T min_merit = h_min_merit[i];
            if (min_merit == h_merit_initial[i]) {
                // No improvement
                drho_vec[i] = std::max(drho_vec[i] * rho_factor, rho_factor);
                rhos[i] = std::max(rhos[i] * drho_vec[i], rho_min);
                sqp_iterations_vec[i]++;
                if (rhos[i] > rho_max) {
                    sqp_time_exit_vec[i] = false;
                    rhos[i] = rho_reset;
                }   
                continue;
            }

            T alphafinal = -1.0 / (1 << line_search_step);

            // Update drho and rho
            drho_vec[i] = std::min(drho_vec[i] / rho_factor, 1 / rho_factor);
            rhos[i] = std::max(rhos[i] * drho_vec[i], rho_min);
            
#if USE_DOUBLES // Update xu with dz and alpha      
            cublasDaxpy(handle, 
                traj_size,
                &alphafinal,
                d_dz + i * traj_size, 1,
                d_xu_tensor + i * traj_size, 1
            );
#else
            cublasSaxpy(handle, 
                traj_size,
                &alphafinal,
                d_dz + i * traj_size, 1,
                d_xu_tensor + i * traj_size, 1
            );
#endif
            // Increment SQP iterations
            sqp_iterations_vec[i]++;

            // Update merit for next iteration
            h_merit_initial[i] = min_merit;
        }
        gpuErrchk(cudaMemcpyAsync(d_rhos, rhos.data(), solve_count * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpyAsync(d_merit_initial, h_merit_initial, solve_count * sizeof(T), cudaMemcpyHostToDevice));

        gpuErrchk(cudaPeekAtLastError());
        cudaEventRecord(line_search_stop);
        cudaEventSynchronize(line_search_stop);
        cudaEventElapsedTime(&milliseconds, line_search_start, line_search_stop);
        std::cout << "Time elapsed for merit-function and line-search: " << milliseconds << std::endl;
        if (sqpTimecheck()){ break; }
        // ------------------ Check convergence --------------

        bool all_converged = true;
        for (uint32_t i = 0; i < solve_count; ++i) {
            if (sqp_time_exit_vec[i]) {
                all_converged = false;
                break;
            }
        }
        if (all_converged) { break; }
    }

    // ------------------ End of SQP loop --------------

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);
    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);
    cudaEventRecord(sqp_stop);
    cudaEventSynchronize(sqp_stop);
    cudaEventElapsedTime(&milliseconds, sqp_start, sqp_stop);
    std::cout << "Time elapsed for SQP: " << milliseconds << std::endl;
    //std::cout << "Average time per iteration: " << sqp_solve_time / SQP_MAX_ITER << std::endl;

    // ------------------ Clean up --------------

    cudaFree(d_xs);
    cudaFree(d_G_dense);
    cudaFree(d_C_dense);
    cudaFree(d_g);
    cudaFree(d_c);
    cudaFree(d_dz);
    cudaFree(d_S);
    cudaFree(d_Pinv);
    cudaFree(d_gamma);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_v_temp);
    cudaFree(d_eta_new_temp);
    cudaFree(d_pcg_iters);
    cudaFree(d_pcg_exit);
    cudaFree(d_merit_news);
    cudaFree(d_merit_temp);

    cudaEventDestroy(kkt_start);
    cudaEventDestroy(kkt_stop);
    cudaEventDestroy(schur_start);
    cudaEventDestroy(schur_stop);
    cudaEventDestroy(pcg_start);
    cudaEventDestroy(pcg_stop);
    cudaEventDestroy(line_search_start);
    cudaEventDestroy(line_search_stop);
    cudaEventDestroy(sqp_start);
    cudaEventDestroy(sqp_stop);

    for (uint32_t str = 0; str < num_alphas; str++) {
        cudaStreamDestroy(streams[str]);
    }

    cublasDestroy(handle);

    // ------------------ Return values --------------

    return std::make_tuple(pcg_iters_matrix, pcg_times_vec, sqp_solve_time, sqp_iterations_vec, sqp_time_exit_vec, pcg_exits_matrix);
}



