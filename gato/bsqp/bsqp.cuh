#pragma once

#include <iostream>
#include <cstdint>
#include <chrono>
#include "settings.h"
#include "constants.h"
#include "types.cuh"
#include "kernels/setup_kkt.cuh"
#include "kernels/schur_linsys.cuh"
#include "kernels/pcg.cuh"
#include "kernels/merit.cuh"
#include "kernels/line_search.cuh"
#include "kernels/sim.cuh"

using namespace sqp;

template<typename T, uint32_t BatchSize>
class BSQP {
      public:
        BSQP()
            : dt_(0.01), max_sqp_iters_(5), kkt_tol_(0.0001), max_pcg_iters_(100), pcg_tol_(1e-5), solve_ratio_(1.0), mu_(10.0), q_cost_(settings_q_COST), qd_cost_(settings_dq_COST),
              u_cost_(settings_u_COST), N_cost_(settings_N_COST), q_lim_cost_(settings_q_lim_COST), vel_lim_cost_(settings_vel_lim_COST), ctrl_lim_cost_(settings_ctrl_lim_COST), rho_(1e-4)
        {
                allocateMemory();
                for (uint32_t i = 0; i < BatchSize; i++) {
                        h_drho_batch_init_[i] = static_cast<T>(1.0);
                        h_rho_penalty_batch_init_[i] = static_cast<T>(rho_);
                }
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaDeviceSynchronize());
        }

        BSQP(T dt, uint32_t max_sqp_iters, T kkt_tol, uint32_t max_pcg_iters, T pcg_tol, T solve_ratio, T mu, T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T rho)
            : dt_(dt), max_sqp_iters_(max_sqp_iters), kkt_tol_(kkt_tol), max_pcg_iters_(max_pcg_iters), pcg_tol_(pcg_tol), solve_ratio_(solve_ratio), mu_(mu), q_cost_(q_cost), qd_cost_(qd_cost),
              u_cost_(u_cost), N_cost_(N_cost), q_lim_cost_(q_lim_cost), vel_lim_cost_(vel_lim_cost), ctrl_lim_cost_(ctrl_lim_cost), rho_(rho)
        {
                allocateMemory();
                for (uint32_t i = 0; i < BatchSize; i++) {
                        h_drho_batch_init_[i] = static_cast<T>(1.0);
                        h_rho_penalty_batch_init_[i] = static_cast<T>(rho_);
                }
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaDeviceSynchronize());
        }

        ~BSQP() { freeMemory(); }

        void set_f_ext_batch(T* h_f_ext_batch) { gpuErrchk(cudaMemcpy(d_f_ext_batch_, h_f_ext_batch, 6 * BatchSize * sizeof(T), cudaMemcpyHostToDevice)); }

        void reset_dual() { gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T))); }

        void reset_rho()
        {
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        }
        // void warmstart()
        // {
        //         // TODO: run a bunch of times with low sqp & pcg tolerances so lambda is warm started
        //         return;
        // }

        void sim_forward(T* d_xkp1_batch, T* d_xk, T* d_uk, T dt) { simForwardBatched<T, BatchSize>(d_xkp1_batch, d_xk, d_uk, d_GRiD_mem_, d_f_ext_batch_, dt); }

        SQPStats<T, BatchSize> solve(T* d_xu_traj_batch, ProblemInputs<T, BatchSize> inputs)
        {
                SQPStats<T, BatchSize>        sqp_stats;
                PCGStats<BatchSize>           pcg_stats;
                LineSearchStats<T, BatchSize> ls_stats;

                auto sqp_start_time = std::chrono::high_resolution_clock::now();

                // set d_dz_batch_ to zero
                gpuErrchk(cudaMemset(d_dz_batch_, 0, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMemset(d_pcg_iterations_, 0, sizeof(uint32_t) * BatchSize));
                gpuErrchk(cudaMemset(d_kkt_converged_batch_, 0, sizeof(int32_t) * BatchSize));

                computeMeritBatched<T, BatchSize, 1>(
                    d_merit_initial_batch_, d_merit_batch_temp_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, mu_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_);

                // SQP Loop
                for (uint32_t i = 0; i < max_sqp_iters_; i++) {
                        setupKKTSystemBatched<T, BatchSize>(kkt_system_batch_, inputs, d_xu_traj_batch, d_f_ext_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_);
                        formSchurSystemBatched<T, BatchSize>(schur_system_batch_, kkt_system_batch_, d_rho_penalty_batch_);

                        // gpuErrchk(cudaEventRecord(pcg_start_event_));
                        solvePCGBatched<T, BatchSize>(d_lambda_batch_, schur_system_batch_, pcg_tol_, max_pcg_iters_, d_kkt_converged_batch_, d_pcg_iterations_);
                        // gpuErrchk(cudaEventRecord(pcg_stop_event_));
                        // gpuErrchk(cudaEventSynchronize(pcg_stop_event_));

                        computeDzBatched<T, BatchSize>(d_dz_batch_, d_lambda_batch_, kkt_system_batch_);

                        // d_q_batch, d_r_batch contain the KKT residuals after computeDzBatched
                        gpuErrchk(cudaMemcpyAsync(h_q_batch_, kkt_system_batch_.d_q_batch, STATE_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
                        gpuErrchk(cudaMemcpyAsync(h_c_batch_, kkt_system_batch_.d_c_batch, STATE_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
                        // gpuErrchk(cudaMemcpy(h_r_batch_, kkt_system_batch_.d_r_batch, CONTROL_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

                        gpuErrchk(cudaMemcpyAsync(pcg_stats.num_iterations.data(), d_pcg_iterations_, sizeof(uint32_t) * BatchSize, cudaMemcpyDeviceToHost)); // throwing an error
                        // gpuErrchk(cudaEventElapsedTime(&pcg_time_us_, pcg_start_event_, pcg_stop_event_)); // this was throwing an error
                        pcg_stats.solve_time_us = 0; //  pcg_time_us_ * 1000;

                        // KKT condition check on cpu is async with gpu
                        uint32_t num_solved = 0;
                        for (uint32_t b = 0; b < BatchSize; ++b) {
                                const T* q_ptr = h_q_batch_ + b * STATE_P_KNOTS;
                                const T* c_ptr = h_c_batch_ + b * STATE_P_KNOTS;

                                auto abs_cmp = [](T a, T b) { return std::abs(a) < std::abs(b); };

                                T q_max = std::abs(*std::max_element(q_ptr, q_ptr + STATE_P_KNOTS, abs_cmp));
                                T c_max = std::abs(*std::max_element(c_ptr, c_ptr + STATE_P_KNOTS, abs_cmp));

                                // within kkt exit tol or pcg exit tol (no steps taken)
                                if (pcg_stats.num_iterations[b] == 0 || (q_max < kkt_tol_ && c_max < kkt_tol_)) {
                                        h_kkt_converged_batch_[b] = 1;
                                        h_sqp_iters_B_[b] += 1;
                                }

                                if (h_kkt_converged_batch_[b]) {
                                        num_solved++;
                                } else {
                                        h_sqp_iters_B_[b] += 1;
                                }
                        }

                        if (num_solved >= BatchSize * solve_ratio_) break;

                        gpuErrchk(cudaMemcpyAsync(d_kkt_converged_batch_, h_kkt_converged_batch_, BatchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

                        computeMeritBatched<T, BatchSize, NUM_ALPHAS>(
                            d_merit_batch_, d_merit_batch_temp_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, mu_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_);
                        lineSearchAndUpdateBatched<T, BatchSize, NUM_ALPHAS>(
                            d_xu_traj_batch, d_dz_batch_, d_merit_batch_, d_merit_initial_batch_, d_step_size_batch_, d_rho_penalty_batch_, d_drho_batch_);

                        gpuErrchk(cudaMemcpyAsync(ls_stats.min_merit.data(), d_merit_initial_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
                        gpuErrchk(cudaMemcpyAsync(ls_stats.step_size.data(), d_step_size_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
                        sqp_stats.line_search_stats.push_back(ls_stats);
                        sqp_stats.pcg_stats.push_back(pcg_stats);
                }

                gpuErrchk(cudaDeviceSynchronize());
                auto sqp_end_time = std::chrono::high_resolution_clock::now();
                gpuErrchk(cudaMemset(d_sqp_iters_B_, 0, BatchSize * sizeof(uint32_t)));
                gpuErrchk(cudaMemset(d_all_kkt_converged_, 0, sizeof(int32_t)));
                gpuErrchk(cudaMemset(d_kkt_converged_batch_, 0, BatchSize * sizeof(int32_t)));
                gpuErrchk(cudaMemcpyAsync(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sqp_end_time - sqp_start_time).count();
                memcpy(sqp_stats.kkt_converged.data(), h_kkt_converged_batch_, BatchSize * sizeof(int32_t));
                memcpy(sqp_stats.sqp_iterations.data(), h_sqp_iters_B_, BatchSize * sizeof(uint32_t));
                memset(h_kkt_converged_batch_, 0, BatchSize * sizeof(int32_t));
                memset(h_sqp_iters_B_, 0, BatchSize * sizeof(uint32_t));

                return sqp_stats;
        }

      private:
        void allocateMemory()
        {
                size_t BT = BatchSize * sizeof(T);
                size_t BI = BatchSize * sizeof(uint32_t);

                d_GRiD_mem_ = gato::plant::initializeDynamicsConstMem<T>();

                gpuErrchk(cudaEventCreate(&pcg_start_event_));
                gpuErrchk(cudaEventCreate(&pcg_stop_event_));

                // Allocate KKT system memory
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_Q_batch, STATE_SQ_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_R_batch, CONTROL_SQ_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_q_batch, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_r_batch, CONTROL_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_A_batch, STATE_SQ_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_B_batch, STATE_P_CONTROL_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&kkt_system_batch_.d_c_batch, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMalloc(&d_dz_batch_, TRAJ_SIZE * BT));
                gpuErrchk(cudaMalloc(&d_lambda_batch_, VEC_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BT));

                // Allocate Schur system memory
                gpuErrchk(cudaMallocManaged(&schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemAdvise(schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BT, cudaMemAdviseSetPreferredLocation, 0));
                gpuErrchk(cudaMallocManaged(&schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemAdvise(schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BT, cudaMemAdviseSetPreferredLocation, 0));
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_gamma_batch, VEC_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_S_batch, 0, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_P_inv_batch, 0, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_gamma_batch, 0, VEC_SIZE_PADDED * BT));

                gpuErrchk(cudaMalloc(&d_merit_initial_batch_, BT));
                gpuErrchk(cudaMalloc(&d_merit_batch_, NUM_ALPHAS * BT));
                gpuErrchk(cudaMalloc(&d_merit_batch_temp_, NUM_ALPHAS * BT * KNOT_POINTS));

                gpuErrchk(cudaMalloc(&d_sqp_iters_B_, BI));
                gpuErrchk(cudaMalloc(&d_pcg_iterations_, BI));
                gpuErrchk(cudaMalloc(&d_step_size_batch_, BT));
                gpuErrchk(cudaMalloc(&d_all_kkt_converged_, sizeof(int32_t)));
                gpuErrchk(cudaMalloc(&d_kkt_converged_batch_, BI));
                gpuErrchk(cudaMalloc(&d_rho_penalty_batch_, BT));
                gpuErrchk(cudaMalloc(&d_drho_batch_, BT));

                gpuErrchk(cudaMalloc(&d_f_ext_batch_, 6 * BT));
                gpuErrchk(cudaMemset(d_f_ext_batch_, 0, 6 * BT));

                gpuErrchk(cudaMallocHost(&h_q_batch_, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMallocHost(&h_r_batch_, CONTROL_P_KNOTS * BT));
                gpuErrchk(cudaMallocHost(&h_c_batch_, STATE_P_KNOTS * BT));
                gpuErrchk(cudaMallocHost(&h_kkt_converged_batch_, BI));
                gpuErrchk(cudaMallocHost(&h_sqp_iters_B_, BI));
                memset(h_kkt_converged_batch_, 0, BI);
                memset(h_sqp_iters_B_, 0, BI);
        }

        void freeMemory()
        {
                gato::plant::freeDynamicsConstMem<T>(d_GRiD_mem_);

                gpuErrchk(cudaEventDestroy(pcg_start_event_));
                gpuErrchk(cudaEventDestroy(pcg_stop_event_));

                gpuErrchk(cudaFree(kkt_system_batch_.d_Q_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_R_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_q_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_r_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_A_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_B_batch));
                gpuErrchk(cudaFree(kkt_system_batch_.d_c_batch));

                gpuErrchk(cudaFree(schur_system_batch_.d_S_batch));
                gpuErrchk(cudaFree(schur_system_batch_.d_P_inv_batch));
                gpuErrchk(cudaFree(schur_system_batch_.d_gamma_batch));

                gpuErrchk(cudaFree(d_lambda_batch_));
                gpuErrchk(cudaFree(d_dz_batch_));
                gpuErrchk(cudaFree(d_kkt_converged_batch_));
                gpuErrchk(cudaFree(d_merit_initial_batch_));
                gpuErrchk(cudaFree(d_merit_batch_));
                gpuErrchk(cudaFree(d_merit_batch_temp_));
                gpuErrchk(cudaFree(d_sqp_iters_B_));
                gpuErrchk(cudaFree(d_pcg_iterations_));
                gpuErrchk(cudaFree(d_step_size_batch_));
                gpuErrchk(cudaFree(d_all_kkt_converged_));
                gpuErrchk(cudaFree(d_f_ext_batch_));
                gpuErrchk(cudaFree(d_rho_penalty_batch_));
                gpuErrchk(cudaFree(d_drho_batch_));

                gpuErrchk(cudaFreeHost(h_q_batch_));
                gpuErrchk(cudaFreeHost(h_r_batch_));
                gpuErrchk(cudaFreeHost(h_c_batch_));
                gpuErrchk(cudaFreeHost(h_kkt_converged_batch_));
        }

        // Device memory
        void*                     d_GRiD_mem_;
        KKTSystem<T, BatchSize>   kkt_system_batch_;
        SchurSystem<T, BatchSize> schur_system_batch_;
        T*                        d_lambda_batch_;
        T*                        d_dz_batch_;
        // PCG
        uint32_t* d_pcg_iterations_;
        // Merit
        T* d_merit_initial_batch_;
        T* d_merit_batch_;
        T* d_merit_batch_temp_;
        // Line search
        T*        d_step_size_batch_;
        int32_t*  d_all_kkt_converged_;
        int32_t*  d_kkt_converged_batch_;
        uint32_t* d_sqp_iters_B_;
        T*        d_f_ext_batch_;

        T* d_rho_penalty_batch_;
        T  h_rho_penalty_batch_init_[BatchSize];
        T  h_drho_batch_init_[BatchSize];
        T* d_drho_batch_;

        // Host-side buffers for KKT check
        T*          h_q_batch_;
        T*          h_r_batch_;
        T*          h_c_batch_;
        int32_t*    h_kkt_converged_batch_;
        cudaEvent_t pcg_start_event_, pcg_stop_event_;
        float       pcg_time_us_;
        uint32_t*   h_sqp_iters_B_;
        T           dt_;
        uint32_t    max_sqp_iters_;
        T           kkt_tol_;
        uint32_t    max_pcg_iters_;
        T           pcg_tol_;
        T           solve_ratio_;
        T           mu_;
        T           q_cost_;
        T           qd_cost_;
        T           u_cost_;
        T           N_cost_;
        T           q_lim_cost_;
        T           vel_lim_cost_;
        T           ctrl_lim_cost_;
        T           rho_;
};
