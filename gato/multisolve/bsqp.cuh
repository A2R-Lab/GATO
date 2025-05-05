#pragma once

#include <iostream>
#include <cstdint>
#include <chrono>
#include <Eigen/Core>
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

template <typename T, uint32_t BatchSize>
class BSQP {
public:
    BSQP() : 
        dt_(0.01),
        max_sqp_iters_(5), 
        kkt_tol_(0.0001), // 1e-4
        max_pcg_iters_(100), 
        pcg_tol_(1e-5),
        solve_ratio_(1.0), 
        mu_(10.0), 
        q_cost_(q_COST), 
        qd_cost_(dq_COST), 
        u_cost_(u_COST), 
        N_cost_(N_COST), 
        q_lim_cost_(q_lim_COST) {
        allocateMemory();
    }

    BSQP(T dt, 
         uint32_t max_sqp_iters, 
         T kkt_tol, 
         uint32_t max_pcg_iters, 
         T pcg_tol, 
         T solve_ratio, 
         T mu, T q_cost, 
         T qd_cost, 
         T u_cost, 
         T N_cost, 
         T q_lim_cost
        ) : 
        dt_(dt),
        max_sqp_iters_(max_sqp_iters), 
        kkt_tol_(kkt_tol),
        max_pcg_iters_(max_pcg_iters), 
        pcg_tol_(pcg_tol),
        solve_ratio_(solve_ratio),
        mu_(mu),
        q_cost_(q_cost),
        qd_cost_(qd_cost),
        u_cost_(u_cost),
        N_cost_(N_cost),
        q_lim_cost_(q_lim_cost) {
        allocateMemory();
    }

    ~BSQP() {
        freeMemory();
    }

    void set_f_ext_batch(T *h_f_ext_batch) {
        gpuErrchk(cudaMemcpy(d_f_ext_batch_, h_f_ext_batch, 6 * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void reset_dual(){
        gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
    }
    
    void reset_primal(){
        // TODO: reset primal variables
        return;
    }

    void warmstart() {
        // TODO: run a bunch of times so lambda is warm started
        return;
    }

    void sim_forward(T *d_xkp1_batch, T *d_xk, T *d_uk, T dt) { // simulates forward for each batch element

        simForwardBatched<T, BatchSize>(
            d_xkp1_batch, // batch of next states
            d_xk, // current state
            d_uk, // control input
            d_GRiD_mem_, 
            d_f_ext_batch_, // external wrenches
            dt // sim timestep
        );
    }

    SQPStats<T, BatchSize> solve(
        T *d_xu_traj_batch,
        ProblemInputs<T, BatchSize> inputs
    ) {
        SQPStats<T, BatchSize> sqp_stats;
        PCGStats<BatchSize> pcg_stats;
        LineSearchStats<T, BatchSize> ls_stats;
        
        auto sqp_start_time = std::chrono::high_resolution_clock::now();

        computeMeritBatched<T, BatchSize, 1>(
            d_merit_initial_batch_,
            d_merit_batch_temp_,
            d_dz_batch_,
            d_xu_traj_batch,
            d_f_ext_batch_,
            inputs,
            mu_,
            d_GRiD_mem_
        );

        // SQP Loop
        for (uint32_t i = 0; i < max_sqp_iters_; i++) {
            
            setupKKTSystemBatched<T, BatchSize>(
                kkt_system_batch_,
                inputs,
                d_xu_traj_batch,
                d_f_ext_batch_,
                d_GRiD_mem_
            );
            
            formSchurSystemBatched<T, BatchSize>(
                schur_system_batch_,
                kkt_system_batch_
            );
            
            gpuErrchk(cudaEventRecord(pcg_start_event_));
            solvePCGBatched<T, BatchSize>(
                d_lambda_batch_,
                schur_system_batch_,
                pcg_tol_,
                max_pcg_iters_,
                d_kkt_converged_batch_,
                d_pcg_iterations_
            );
            gpuErrchk(cudaEventRecord(pcg_stop_event_));
            gpuErrchk(cudaEventSynchronize(pcg_stop_event_));
            
            gpuErrchk(cudaMemcpyAsync(pcg_stats.num_iterations.data(), d_pcg_iterations_, sizeof(uint32_t) * BatchSize, cudaMemcpyDeviceToHost));
            gpuErrchk(cudaEventElapsedTime(&pcg_time_us_, pcg_start_event_, pcg_stop_event_));
            pcg_stats.solve_time_us = pcg_time_us_ * 1000;
            
            computeDzBatched<T, BatchSize>(
                d_dz_batch_,
                d_lambda_batch_,
                kkt_system_batch_
            );

            // Note: d_q_batch, d_r_batch contain the KKT residuals after computeDzBatched
            // Use pinned host memory (allocated with cudaMallocHost) for async copy
            gpuErrchk(cudaMemcpy(h_q_batch_, kkt_system_batch_.d_q_batch, STATE_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            //gpuErrchk(cudaMemcpy(h_r_batch_, kkt_system_batch_.d_r_batch, CONTROL_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_c_batch_, kkt_system_batch_.d_c_batch, STATE_P_KNOTS * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            
            // Perform KKT residual check on cpu
            uint32_t num_solved = 0;
            for (uint32_t b = 0; b < BatchSize; ++b) {
                Eigen::Map<Eigen::Matrix<T, STATE_P_KNOTS, 1>> q_vec(h_q_batch_ + b * STATE_P_KNOTS);
                //Eigen::Map<Eigen::Matrix<T, CONTROL_P_KNOTS, 1>> r_vec(h_r_batch_ + b * CONTROL_P_KNOTS);
                Eigen::Map<Eigen::Matrix<T, STATE_P_KNOTS, 1>> c_vec(h_c_batch_ + b * STATE_P_KNOTS);

                T q_max = q_vec.cwiseAbs().maxCoeff();
                T c_max = c_vec.cwiseAbs().maxCoeff();
                //T r_max = r_vec.cwiseAbs().maxCoeff();

                // within kkt exit tol or pcg exit tol (no steps taken)
                if  (pcg_stats.num_iterations[b] == 0 || (q_max < kkt_tol_ && c_max < kkt_tol_)) {
                    h_kkt_converged_batch_[b] = 1;
                    h_sqp_iters_B_[b] += 1;
                }

                if (h_kkt_converged_batch_[b]) { 
                    num_solved++;
                } else {
                    h_sqp_iters_B_[b] += 1;
                }
            }
           
            gpuErrchk(cudaMemcpyAsync(d_kkt_converged_batch_, h_kkt_converged_batch_, BatchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

            computeMeritBatched<T, BatchSize, NUM_ALPHAS>(
                d_merit_batch_,
                d_merit_batch_temp_,
                d_dz_batch_,
                d_xu_traj_batch,
                d_f_ext_batch_,
                inputs,
                mu_,
                d_GRiD_mem_
            );


            lineSearchAndUpdateBatched<T, BatchSize, NUM_ALPHAS>(
                d_xu_traj_batch,
                d_dz_batch_,
                d_merit_batch_,
                d_merit_initial_batch_,
                d_step_size_batch_
            );

            gpuErrchk(cudaMemcpyAsync(ls_stats.min_merit.data(), d_merit_initial_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpyAsync(ls_stats.step_size.data(), d_step_size_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            sqp_stats.line_search_stats.push_back(ls_stats);
            sqp_stats.pcg_stats.push_back(pcg_stats);

            if (num_solved >= BatchSize * solve_ratio_) break;
        }

        gpuErrchk(cudaDeviceSynchronize());
        auto sqp_end_time = std::chrono::high_resolution_clock::now();
        sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sqp_end_time - sqp_start_time).count();
        memcpy(sqp_stats.kkt_converged.data(), h_kkt_converged_batch_, BatchSize * sizeof(int32_t));
        memcpy(sqp_stats.sqp_iterations.data(), h_sqp_iters_B_, BatchSize * sizeof(uint32_t));

        gpuErrchk(cudaMemset(d_sqp_iters_B_, 0, BatchSize * sizeof(uint32_t)));
        gpuErrchk(cudaMemset(d_all_kkt_converged_, 0, sizeof(int32_t)));
        gpuErrchk(cudaMemset(d_kkt_converged_batch_, 0, BatchSize * sizeof(int32_t)));
        memset(h_kkt_converged_batch_, 0, BatchSize * sizeof(int32_t));
        memset(h_sqp_iters_B_, 0, BatchSize * sizeof(uint32_t));
        return sqp_stats;
    }

private:
    void allocateMemory() {
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

    void freeMemory() {
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
        gpuErrchk(cudaFree(d_f_ext_batch_));

        gpuErrchk(cudaFreeHost(h_q_batch_));
        gpuErrchk(cudaFreeHost(h_r_batch_));
        gpuErrchk(cudaFreeHost(h_c_batch_));
        gpuErrchk(cudaFreeHost(h_kkt_converged_batch_));
    }

    // Device memory
    void *d_GRiD_mem_;
    KKTSystem<T, BatchSize> kkt_system_batch_;
    SchurSystem<T, BatchSize> schur_system_batch_;
    T *d_lambda_batch_;
    T *d_dz_batch_;
    T *d_drho_batch_;
    // PCG
    uint32_t *d_pcg_iterations_;
    // Merit
    T *d_merit_initial_batch_;
    T *d_merit_batch_;
    T *d_merit_batch_temp_;
    // Line search
    T *d_step_size_batch_;
    int32_t *d_all_kkt_converged_;
    int32_t *d_kkt_converged_batch_;
    uint32_t *d_sqp_iters_B_;
    T *d_f_ext_batch_;

    // Host-side buffers for KKT check
    T *h_q_batch_;
    T *h_r_batch_;
    T *h_c_batch_;
    int32_t *h_kkt_converged_batch_;
    cudaEvent_t pcg_start_event_, pcg_stop_event_;
    float pcg_time_us_;
    uint32_t *h_sqp_iters_B_;
    T dt_;
    uint32_t max_sqp_iters_;
    T kkt_tol_;
    uint32_t max_pcg_iters_;
    T pcg_tol_;
    T solve_ratio_;
    T mu_;
    T q_cost_;
    T qd_cost_;
    T u_cost_;
    T N_cost_;
    T q_lim_cost_;
};





            
                // // Compute average L2 norm of each residual (q, r, c) over all knot points
                // T q_norm_sum = 0.0;
                // T c_norm_sum = 0.0;

                // for (uint32_t k = 0; k < KNOT_POINTS; ++k) {
                //     T* q_k = h_q_batch_ + b * STATE_P_KNOTS + k * STATE_SIZE;
                //     T* c_k = h_c_batch_ + b * STATE_P_KNOTS + k * STATE_SIZE;

                //     T q_norm = 0.0;
                //     T c_norm = 0.0;

                //     for (uint32_t j = 0; j < STATE_SIZE; ++j) {
                //         q_norm += q_k[j] * q_k[j];
                //         c_norm += c_k[j] * c_k[j];
                //     }

                //     q_norm_sum += std::sqrt(q_norm);
                //     c_norm_sum += std::sqrt(c_norm);
                // }

                // T avg_q_norm = q_norm_sum / KNOT_POINTS;
                // T avg_c_norm = c_norm_sum / KNOT_POINTS;

                // if (avg_q_norm < kkt_tol_ && avg_c_norm < kkt_tol_) { // ignore r_norm for now
                //     h_kkt_converged_batch_[b] = 1;
                //     num_solved++;
                // }