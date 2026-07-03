#pragma once

#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cstring>
#include <algorithm>
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

// batch_size is a RUNTIME parameter: the batch index is only ever the outermost
// grid dimension (one block per solve) and a buffer-extent multiplier, so nothing
// in device code needs it at compile time.
template<typename T>
class BSQP {
      public:
        explicit BSQP(uint32_t batch_size)
            : batch_size_(batch_size), dt_(0.01), max_sqp_iters_(5), kkt_tol_(0.0001), max_pcg_iters_(100), pcg_tol_(1e-5), solve_ratio_(1.0), mu_(10.0),
              q_cost_(1.0), qd_cost_(1e-3), u_cost_(1e-6), N_cost_(50.0), q_lim_cost_(1e-3), vel_lim_cost_(0.0), ctrl_lim_cost_(0.0),
              rho_(1e-3), adapt_rho_(true)
        {
                allocateMemory();
                initBatchedHyperparams();
        }

        BSQP(uint32_t batch_size, T dt, uint32_t max_sqp_iters, T kkt_tol, uint32_t max_pcg_iters, T pcg_tol, T solve_ratio, T mu, T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, T rho)
            : batch_size_(batch_size), dt_(dt), max_sqp_iters_(max_sqp_iters), kkt_tol_(kkt_tol), max_pcg_iters_(max_pcg_iters), pcg_tol_(pcg_tol), solve_ratio_(solve_ratio), mu_(mu), q_cost_(q_cost), qd_cost_(qd_cost),
              u_cost_(u_cost), N_cost_(N_cost), q_lim_cost_(q_lim_cost), vel_lim_cost_(vel_lim_cost), ctrl_lim_cost_(ctrl_lim_cost), rho_(rho), adapt_rho_(true)
        {
                allocateMemory();
                initBatchedHyperparams();
        }

        ~BSQP() { freeMemory(); }

        uint32_t batch_size() const { return batch_size_; }

        void set_f_ext_batch(T* h_f_ext_batch) { gpuErrchk(cudaMemcpy(d_f_ext_batch_, h_f_ext_batch, 6 * grid::NUM_BODIES * batch_size_ * sizeof(T), cudaMemcpyHostToDevice)); }

        // Hyperparameter setters (batched)
        void set_rho_penalty_batch(const T* h_rho_penalty_batch, bool set_as_reset_default = true)
        {
                if (set_as_reset_default) { memcpy(h_rho_penalty_batch_init_.data(), h_rho_penalty_batch, batch_size_ * sizeof(T)); }
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch, batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
        }

        void set_drho_batch(const T* h_drho_batch, bool set_as_reset_default = true)
        {
                if (set_as_reset_default) { memcpy(h_drho_batch_init_.data(), h_drho_batch, batch_size_ * sizeof(T)); }
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch, batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
        }

        void set_mu_batch(const T* h_mu_batch) { gpuErrchk(cudaMemcpy(d_mu_batch_, h_mu_batch, batch_size_ * sizeof(T), cudaMemcpyHostToDevice)); }
        void set_pcg_tol_batch(const T* h_pcg_tol_batch) { gpuErrchk(cudaMemcpy(d_pcg_tol_batch_, h_pcg_tol_batch, batch_size_ * sizeof(T), cudaMemcpyHostToDevice)); }

        void reset_dual() { gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * batch_size_ * sizeof(T))); }

        void reset_rho()
        {
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
        }

        void set_rho_adaptation(bool enabled) { adapt_rho_ = enabled; }
        void set_collect_stats(bool enabled) { collect_stats_ = enabled; }

        // runtime scalar cost weights (plain members threaded into every launch)
        void set_cost_weights(T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
        {
                q_cost_ = q_cost;
                qd_cost_ = qd_cost;
                u_cost_ = u_cost;
                N_cost_ = N_cost;
                q_lim_cost_ = q_lim_cost;
                vel_lim_cost_ = vel_lim_cost;
                ctrl_lim_cost_ = ctrl_lim_cost;
        }

        // per-knot [ee, qd, u] weight triples (KNOT_POINTS x 3, knot-major); overrides the
        // scalar q/qd/u/N weights (terminal EE weight = last knot's ee entry). Enables
        // via-points / terminal-only goals / horizon masking at runtime.
        void set_cost_weights_per_knot(const T* h_knot_weights)
        {
                gpuErrchk(cudaMemcpy(d_knot_cost_weights_, h_knot_weights, 3 * KNOT_POINTS * sizeof(T), cudaMemcpyHostToDevice));
                use_knot_cost_weights_ = true;
        }
        void clear_cost_weights_per_knot() { use_knot_cost_weights_ = false; }

        void sim_forward(T* d_xkp1_batch, T* d_xk, T* d_uk, T dt) { simForwardBatched<T>(batch_size_, d_xkp1_batch, d_xk, d_uk, d_GRiD_mem_, d_f_ext_batch_, dt); }

        void copy_final_merit_to_host(T* h_out)
        {
                gpuErrchk(cudaMemcpy(h_out, d_merit_initial_batch_, batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void copy_initial_merit0_to_host(T* h_out)
        {
                gpuErrchk(cudaMemcpy(h_out, d_merit_initial0_batch_, batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }

        SQPStats<T> solve(T* d_xu_traj_batch, ProblemInputs<T> inputs)
        {
                SQPStats<T> sqp_stats(batch_size_);
                PCGStats    pcg_stats(batch_size_);

                const T* d_knot_w = use_knot_cost_weights_ ? d_knot_cost_weights_ : nullptr;
                uint32_t ls_iters_run = 0;  // iterations that reached merit+line-search

                auto sqp_start_time = std::chrono::high_resolution_clock::now();

                // set d_dz_batch_ to zero
                gpuErrchk(cudaMemset(d_dz_batch_, 0, TRAJ_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMemset(d_pcg_iterations_, 0, sizeof(uint32_t) * batch_size_));
                gpuErrchk(cudaMemset(d_kkt_converged_batch_, 0, sizeof(int32_t) * batch_size_));

                computeMeritBatched<T, 1>(
                    batch_size_, /*d_kkt_converged=*/nullptr, d_knot_w, d_merit_initial_batch_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, d_mu_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_);
                gpuErrchk(cudaMemcpy(d_merit_initial0_batch_, d_merit_initial_batch_, batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice));

                // SQP Loop
                for (uint32_t i = 0; i < max_sqp_iters_; i++) {
                        setupKKTSystemBatched<T>(batch_size_, kkt_system_batch_, inputs, d_xu_traj_batch, d_f_ext_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_, d_kkt_converged_batch_, d_knot_w);
                        formSchurSystemBatched<T>(batch_size_, schur_system_batch_, kkt_system_batch_, d_rho_penalty_batch_, d_kkt_converged_batch_);

                        solvePCGBatched<T>(batch_size_, d_lambda_batch_, schur_system_batch_, d_pcg_tol_batch_, max_pcg_iters_, d_kkt_converged_batch_, d_pcg_iterations_);

                        computeDzBatched<T>(batch_size_, d_dz_batch_, d_lambda_batch_, kkt_system_batch_, d_kkt_converged_batch_);

                        // convergence signal: PCG iteration counts (pinned staging + event —
                        // a pageable-destination copy here would silently serialize the host).
                        // NOTE: the old max|q|/max|c| KKT check was dead (commented out); its
                        // ~1MB/iter q/c device->host traffic is gone. kkt_tol_ is currently
                        // unused — a device-side KKT check is backlogged.
                        gpuErrchk(cudaMemcpyAsync(h_pcg_iters_, d_pcg_iterations_, sizeof(uint32_t) * batch_size_, cudaMemcpyDeviceToHost));
                        gpuErrchk(cudaEventRecord(sync_event_));
                        gpuErrchk(cudaEventSynchronize(sync_event_));

                        for (uint32_t b = 0; b < batch_size_; ++b) { pcg_stats.num_iterations[b] = static_cast<int>(h_pcg_iters_[b]); }
                        pcg_stats.solve_time_us = 0;
                        if (collect_stats_) { sqp_stats.pcg_stats.push_back(pcg_stats); }

                        uint32_t num_solved = 0;
                        for (uint32_t b = 0; b < batch_size_; ++b) {
                                // converged when PCG took no steps (already at the linsys solution)
                                if (h_pcg_iters_[b] == 0) { h_kkt_converged_batch_[b] = 1; }

                                if (h_kkt_converged_batch_[b]) {
                                        num_solved++;
                                }
                                h_sqp_iters_B_[b] += 1;
                        }

                        if (num_solved >= batch_size_ * solve_ratio_) break;

                        gpuErrchk(cudaMemcpyAsync(d_kkt_converged_batch_, h_kkt_converged_batch_, batch_size_ * sizeof(int32_t), cudaMemcpyHostToDevice));

                        computeMeritBatched<T, NUM_ALPHAS>(
                            batch_size_, d_kkt_converged_batch_, d_knot_w, d_merit_batch_, d_dz_batch_, d_xu_traj_batch, d_f_ext_batch_, inputs, d_mu_batch_, d_GRiD_mem_, q_cost_, qd_cost_, u_cost_, N_cost_, q_lim_cost_, vel_lim_cost_, ctrl_lim_cost_);
                        lineSearchAndUpdateBatched<T, NUM_ALPHAS>(
                            batch_size_, d_xu_traj_batch, d_dz_batch_, d_merit_batch_, d_merit_initial_batch_, d_step_size_batch_, d_rho_penalty_batch_, d_drho_batch_, adapt_rho_ ? 1 : 0, d_kkt_converged_batch_);

                        // stage line-search stats into per-iteration pinned slots; read once
                        // after the final device sync (no per-iteration host stall)
                        if (collect_stats_) {
                                gpuErrchk(cudaMemcpyAsync(h_ls_min_merit_ + ls_iters_run * batch_size_, d_merit_initial_batch_, batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
                                gpuErrchk(cudaMemcpyAsync(h_ls_step_size_ + ls_iters_run * batch_size_, d_step_size_batch_, batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
                        }
                        ls_iters_run++;
                }

                // NOTE: no final merit recompute — d_merit_initial_batch_ invariantly holds the
                // merit of the CURRENT trajectory (line search writes the accepted merit on
                // success and leaves both trajectory and merit untouched on failure).

                gpuErrchk(cudaDeviceSynchronize());

                if (collect_stats_) {
                        for (uint32_t it = 0; it < ls_iters_run; ++it) {
                                LineSearchStats<T> ls_stats(batch_size_);
                                memcpy(ls_stats.min_merit.data(), h_ls_min_merit_ + it * batch_size_, batch_size_ * sizeof(T));
                                memcpy(ls_stats.step_size.data(), h_ls_step_size_ + it * batch_size_, batch_size_ * sizeof(T));
                                sqp_stats.line_search_stats.push_back(ls_stats);
                        }
                }
                auto sqp_end_time = std::chrono::high_resolution_clock::now();
                gpuErrchk(cudaMemset(d_sqp_iters_B_, 0, batch_size_ * sizeof(uint32_t)));
                gpuErrchk(cudaMemset(d_all_kkt_converged_, 0, sizeof(int32_t)));
                gpuErrchk(cudaMemset(d_kkt_converged_batch_, 0, batch_size_ * sizeof(int32_t)));
                gpuErrchk(cudaMemcpyAsync(d_drho_batch_, h_drho_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sqp_end_time - sqp_start_time).count();
                memcpy(sqp_stats.kkt_converged.data(), h_kkt_converged_batch_, batch_size_ * sizeof(int32_t));
                memcpy(sqp_stats.sqp_iterations.data(), h_sqp_iters_B_, batch_size_ * sizeof(uint32_t));
                memset(h_kkt_converged_batch_, 0, batch_size_ * sizeof(int32_t));
                memset(h_sqp_iters_B_, 0, batch_size_ * sizeof(uint32_t));

                return sqp_stats;
        }

      private:
        void initBatchedHyperparams()
        {
                h_rho_penalty_batch_init_.assign(batch_size_, static_cast<T>(rho_));
                h_drho_batch_init_.assign(batch_size_, static_cast<T>(1.0));
                h_mu_batch_init_.assign(batch_size_, static_cast<T>(mu_));
                h_pcg_tol_batch_init_.assign(batch_size_, static_cast<T>(pcg_tol_));
                gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_mu_batch_, h_mu_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_pcg_tol_batch_, h_pcg_tol_batch_init_.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaDeviceSynchronize());
        }

        void allocateMemory()
        {
                size_t BT = batch_size_ * sizeof(T);
                size_t BI = batch_size_ * sizeof(uint32_t);

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
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMalloc(&schur_system_batch_.d_gamma_batch, VEC_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_S_batch, 0, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_P_inv_batch, 0, B3D_MATRIX_SIZE_PADDED * BT));
                gpuErrchk(cudaMemset(schur_system_batch_.d_gamma_batch, 0, VEC_SIZE_PADDED * BT));

                gpuErrchk(cudaMalloc(&d_merit_initial_batch_, BT));
                gpuErrchk(cudaMalloc(&d_merit_initial0_batch_, BT));
                gpuErrchk(cudaMalloc(&d_merit_batch_, NUM_ALPHAS * BT));
                // the multi-alpha merit buffer accumulates via atomicAdd and is re-zeroed by
                // the line-search kernel after each read — zero it ONCE here, not per call
                gpuErrchk(cudaMemset(d_merit_batch_, 0, NUM_ALPHAS * BT));

                // per-knot [ee,qd,u] cost-weight triples (optional override)
                gpuErrchk(cudaMalloc(&d_knot_cost_weights_, 3 * KNOT_POINTS * sizeof(T)));

                gpuErrchk(cudaMalloc(&d_sqp_iters_B_, BI));
                gpuErrchk(cudaMalloc(&d_pcg_iterations_, BI));
                gpuErrchk(cudaMalloc(&d_step_size_batch_, BT));
                gpuErrchk(cudaMalloc(&d_all_kkt_converged_, sizeof(int32_t)));
                gpuErrchk(cudaMalloc(&d_kkt_converged_batch_, BI));
                gpuErrchk(cudaMalloc(&d_rho_penalty_batch_, BT));
                gpuErrchk(cudaMalloc(&d_drho_batch_, BT));

                gpuErrchk(cudaMalloc(&d_f_ext_batch_, 6 * grid::NUM_BODIES * BT));
                gpuErrchk(cudaMemset(d_f_ext_batch_, 0, 6 * grid::NUM_BODIES * BT));

                // Batched hyperparameters
                gpuErrchk(cudaMalloc(&d_mu_batch_, BT));
                gpuErrchk(cudaMalloc(&d_pcg_tol_batch_, BT));

                gpuErrchk(cudaMallocHost(&h_kkt_converged_batch_, BI));
                gpuErrchk(cudaMallocHost(&h_sqp_iters_B_, BI));
                gpuErrchk(cudaMallocHost(&h_pcg_iters_, BI));
                // per-iteration line-search stat staging (read once after the final sync)
                gpuErrchk(cudaMallocHost(&h_ls_min_merit_, max_sqp_iters_ * BT));
                gpuErrchk(cudaMallocHost(&h_ls_step_size_, max_sqp_iters_ * BT));
                gpuErrchk(cudaEventCreateWithFlags(&sync_event_, cudaEventDisableTiming));
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
                gpuErrchk(cudaFree(d_merit_initial0_batch_));
                gpuErrchk(cudaFree(d_merit_batch_));
                gpuErrchk(cudaFree(d_sqp_iters_B_));
                gpuErrchk(cudaFree(d_pcg_iterations_));
                gpuErrchk(cudaFree(d_step_size_batch_));
                gpuErrchk(cudaFree(d_all_kkt_converged_));
                gpuErrchk(cudaFree(d_f_ext_batch_));
                gpuErrchk(cudaFree(d_rho_penalty_batch_));
                gpuErrchk(cudaFree(d_drho_batch_));
                gpuErrchk(cudaFree(d_mu_batch_));
                gpuErrchk(cudaFree(d_pcg_tol_batch_));
                gpuErrchk(cudaFree(d_knot_cost_weights_));

                gpuErrchk(cudaFreeHost(h_kkt_converged_batch_));
                gpuErrchk(cudaFreeHost(h_sqp_iters_B_));
                gpuErrchk(cudaFreeHost(h_pcg_iters_));
                gpuErrchk(cudaFreeHost(h_ls_min_merit_));
                gpuErrchk(cudaFreeHost(h_ls_step_size_));
                gpuErrchk(cudaEventDestroy(sync_event_));
        }

        // Device memory
        uint32_t            batch_size_;
        void*               d_GRiD_mem_;
        KKTSystem<T>        kkt_system_batch_;
        SchurSystem<T>      schur_system_batch_;
        T*                  d_lambda_batch_;
        T*                  d_dz_batch_;
        // PCG
        uint32_t* d_pcg_iterations_;
        // Merit
        T* d_merit_initial_batch_;
        T* d_merit_initial0_batch_;
        T* d_merit_batch_;
        // Line search
        T*        d_step_size_batch_;
        int32_t*  d_all_kkt_converged_;
        int32_t*  d_kkt_converged_batch_;
        uint32_t* d_sqp_iters_B_;
        T*        d_f_ext_batch_;

        T*             d_rho_penalty_batch_;
        std::vector<T> h_rho_penalty_batch_init_;
        std::vector<T> h_drho_batch_init_;
        T*             d_drho_batch_;

        // Batched hyperparameters
        T*             d_mu_batch_;
        std::vector<T> h_mu_batch_init_;
        T*             d_pcg_tol_batch_;
        std::vector<T> h_pcg_tol_batch_init_;

        // per-knot cost weights (optional)
        T*   d_knot_cost_weights_;
        bool use_knot_cost_weights_ = false;
        bool collect_stats_ = true;

        // Host-side pinned staging (convergence + stats)
        int32_t*    h_kkt_converged_batch_;
        uint32_t*   h_pcg_iters_;
        T*          h_ls_min_merit_;
        T*          h_ls_step_size_;
        cudaEvent_t sync_event_;
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
        bool        adapt_rho_;
};
