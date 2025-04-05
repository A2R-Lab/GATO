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

using namespace sqp;

template <typename T, uint32_t BatchSize>
class SQPSolver {
public:
    SQPSolver() {
        allocateMemory();
        for (uint32_t i = 0; i < BatchSize; i++) {
            h_drho_batch_init_[i] = static_cast<T>(1.0);
            h_rho_penalty_batch_init_[i] = static_cast<T>(RHO_INIT);
        }
        gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
    }

    ~SQPSolver() {
        freeMemory();
    }

    void set_external_wrench(T *h_f_ext_batch, uint32_t solve_idx) {
        gpuErrchk(cudaMemcpy(d_f_ext_batch_ + solve_idx * 6, h_f_ext_batch, 6 * sizeof(T), cudaMemcpyHostToDevice));
    }

    void set_external_wrench_batch(T *h_f_ext_batch) {
        gpuErrchk(cudaMemcpy(d_f_ext_batch_, h_f_ext_batch, 6 * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    void reset() {
        // Reset penalty parameters
        // gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        // gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        
        // Reset Lagrange multipliers
        // gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
        
        // Reset step direction
        gpuErrchk(cudaMemset(d_dz_batch_, 0, TRAJ_SIZE * BatchSize * sizeof(T)));
        
        // Reset merit function values
        gpuErrchk(cudaMemset(d_merit_initial_batch_, 0, BatchSize * sizeof(T)));
        
        // Reset line search variables
        gpuErrchk(cudaMemset(d_step_size_batch_, 0, BatchSize * sizeof(T)));
        
        // Reset rho max reached flags
        gpuErrchk(cudaMemset(d_all_rho_max_reached_, 0, sizeof(int32_t)));
        gpuErrchk(cudaMemset(d_rho_max_reached_batch_, 0, BatchSize * sizeof(int32_t)));
        
        // Reset iteration counters
        gpuErrchk(cudaMemset(d_iterations_batch_, 0, BatchSize * sizeof(uint32_t)));
        
        // Reset PCG state if needed
        gpuErrchk(cudaMemset(d_pcg_converged_, 0, sizeof(int32_t) * BatchSize));
        gpuErrchk(cudaMemset(d_pcg_iterations_, 0, sizeof(uint32_t) * BatchSize));
        gpuErrchk(cudaDeviceSynchronize());
    }

    void resetRho(){
        gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    void resetLambda(){
        gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
    }

    void setLambdas(T *h_lambda_batch, uint32_t solve_idx) {
        gpuErrchk(cudaMemcpy(d_lambda_batch_ + solve_idx * VEC_SIZE_PADDED + STATE_SIZE, h_lambda_batch, STATE_P_KNOTS * sizeof(T), cudaMemcpyHostToDevice));
    }

    void warmStart() {
        // TODO: run a bunch of times so lambda is warm started
        return;
    }

    SQPStats<T, BatchSize> solve(
        T *d_xu_traj_batch,
        ProblemInputs<T, BatchSize> inputs
    ) {
        SQPStats<T, BatchSize> sqp_stats;
        PCGStats<BatchSize> pcg_stats;
        LineSearchStats<T, BatchSize> line_search_stats;

        cudaEvent_t pcg_start, pcg_stop;
        gpuErrchk(cudaEventCreate(&pcg_start));
        gpuErrchk(cudaEventCreate(&pcg_stop));

        auto solve_start_time = std::chrono::high_resolution_clock::now();

        computeMeritBatched<T, BatchSize, 1>(
            d_merit_initial_batch_,
            d_merit_batch_temp_,
            d_dz_batch_,
            d_xu_traj_batch,
            d_f_ext_batch_,
            inputs
        );

        // SQP Loop
        for (uint32_t i = 0; i < SQP_MAX_ITER; i++) {
            
            setupKKTSystemBatched<T, BatchSize>(
                kkt_system_batch_,
                inputs,
                d_xu_traj_batch,
                d_f_ext_batch_
            );
            
            formSchurSystemBatched<T, BatchSize>(
                schur_system_batch_,
                kkt_system_batch_,
                d_rho_penalty_batch_
            );
            
            gpuErrchk(cudaEventRecord(pcg_start));
            solvePCGBatched<T, BatchSize>(
                d_lambda_batch_,
                schur_system_batch_,
                PCG_TOLERANCE,
                d_rho_max_reached_batch_,
                d_pcg_converged_,
                d_pcg_iterations_
            );
            gpuErrchk(cudaEventRecord(pcg_stop));
            gpuErrchk(cudaEventSynchronize(pcg_stop));
            
            float pcg_time_ms;
            gpuErrchk(cudaEventElapsedTime(&pcg_time_ms, pcg_start, pcg_stop));
        
            gpuErrchk(cudaMemcpy(pcg_stats.converged.data(), d_pcg_converged_, sizeof(int32_t) * BatchSize, cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(pcg_stats.num_iterations.data(), d_pcg_iterations_, sizeof(uint32_t) * BatchSize, cudaMemcpyDeviceToHost));
            pcg_stats.solve_time_us = pcg_time_ms * 1000;

            computeDzBatched<T, BatchSize>(
                d_dz_batch_,
                d_lambda_batch_,
                kkt_system_batch_
            );
            
            computeMeritBatched<T, BatchSize, NUM_ALPHAS>(
                d_merit_batch_,
                d_merit_batch_temp_,
                d_dz_batch_,
                d_xu_traj_batch,
                d_f_ext_batch_,
                inputs
            );
            
            lineSearchAndUpdateBatched<T, BatchSize, NUM_ALPHAS>(
                d_xu_traj_batch,
                d_dz_batch_,
                d_merit_batch_,
                d_merit_initial_batch_,
                d_rho_penalty_batch_,
                d_drho_batch_,
                d_step_size_batch_,
                d_all_rho_max_reached_,
                d_rho_max_reached_batch_,
                d_iterations_batch_
            );

            sqp_stats.pcg_stats.push_back(pcg_stats);
            gpuErrchk(cudaMemcpy(line_search_stats.min_merit.data(), d_merit_initial_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(line_search_stats.step_size.data(), d_step_size_batch_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(&line_search_stats.all_rho_max_reached, d_all_rho_max_reached_, sizeof(int32_t), cudaMemcpyDeviceToHost));
            sqp_stats.line_search_stats.push_back(line_search_stats);
            if (line_search_stats.all_rho_max_reached) break;
        }

        gpuErrchk(cudaDeviceSynchronize());
        auto solve_end_time = std::chrono::high_resolution_clock::now();

        sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time).count();
        gpuErrchk(cudaMemcpy(sqp_stats.sqp_iterations.data(), d_iterations_batch_, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(sqp_stats.rho_max_reached.data(), d_rho_max_reached_batch_, BatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost));

        // Reset drho batch 
        gpuErrchk(cudaMemcpyAsync(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));

        // Reset other arrays
        gpuErrchk(cudaMemset(d_dz_batch_, 0, TRAJ_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_merit_initial_batch_, 0, BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_step_size_batch_, 0, BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_all_rho_max_reached_, 0, sizeof(int32_t)));
        gpuErrchk(cudaMemset(d_rho_max_reached_batch_, 0, BatchSize * sizeof(int32_t)));
        gpuErrchk(cudaMemset(d_iterations_batch_, 0, BatchSize * sizeof(uint32_t)));

        gpuErrchk(cudaEventDestroy(pcg_start));
        gpuErrchk(cudaEventDestroy(pcg_stop));

        return sqp_stats;
    }

private:
    void allocateMemory() {
        // Allocate KKT system memory
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_Q_batch, STATE_SQ_P_KNOTS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_R_batch, CONTROL_SQ_P_KNOTS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_q_batch, STATE_P_KNOTS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_r_batch, CONTROL_P_KNOTS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_A_batch, STATE_SQ_P_KNOTS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_B_batch, STATE_P_CONTROL_P_KNOTS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&kkt_system_batch_.d_c_batch, STATE_P_KNOTS * BatchSize * sizeof(T)));

        // Allocate Schur system memory
        gpuErrchk(cudaMallocManaged(&schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemAdvise(schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T), cudaMemAdviseSetPreferredLocation, 0));
        gpuErrchk(cudaMallocManaged(&schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemAdvise(schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T), cudaMemAdviseSetPreferredLocation, 0));
        gpuErrchk(cudaMalloc(&schur_system_batch_.d_gamma_batch, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(schur_system_batch_.d_S_batch, 0, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(schur_system_batch_.d_P_inv_batch, 0, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(schur_system_batch_.d_gamma_batch, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T)));

        // Allocate other memory
        gpuErrchk(cudaMalloc(&d_lambda_batch_, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_lambda_batch_, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_dz_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_rho_penalty_batch_, BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_drho_batch_, BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_pcg_converged_, sizeof(int32_t) * BatchSize));
        gpuErrchk(cudaMalloc(&d_pcg_iterations_, sizeof(uint32_t) * BatchSize));
        gpuErrchk(cudaMalloc(&d_merit_initial_batch_, BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_merit_batch_, NUM_ALPHAS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_merit_batch_temp_, NUM_ALPHAS * BatchSize * KNOT_POINTS * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_step_size_batch_, BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_all_rho_max_reached_, sizeof(int32_t)));
        gpuErrchk(cudaMalloc(&d_rho_max_reached_batch_, BatchSize * sizeof(int32_t)));
        gpuErrchk(cudaMalloc(&d_iterations_batch_, BatchSize * sizeof(uint32_t)));
        gpuErrchk(cudaMalloc(&d_f_ext_batch_, 6 * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_f_ext_batch_, 0, 6 * BatchSize * sizeof(T)));
    }

    void freeMemory() {
        // Free KKT system memory
        gpuErrchk(cudaFree(kkt_system_batch_.d_Q_batch));
        gpuErrchk(cudaFree(kkt_system_batch_.d_R_batch));
        gpuErrchk(cudaFree(kkt_system_batch_.d_q_batch));
        gpuErrchk(cudaFree(kkt_system_batch_.d_r_batch));
        gpuErrchk(cudaFree(kkt_system_batch_.d_A_batch));
        gpuErrchk(cudaFree(kkt_system_batch_.d_B_batch));
        gpuErrchk(cudaFree(kkt_system_batch_.d_c_batch));

        // Free Schur system memory
        gpuErrchk(cudaFree(schur_system_batch_.d_S_batch));
        gpuErrchk(cudaFree(schur_system_batch_.d_P_inv_batch));
        gpuErrchk(cudaFree(schur_system_batch_.d_gamma_batch));

        // Free other memory
        gpuErrchk(cudaFree(d_lambda_batch_));
        gpuErrchk(cudaFree(d_dz_batch_));
        gpuErrchk(cudaFree(d_rho_penalty_batch_));
        gpuErrchk(cudaFree(d_drho_batch_));
        gpuErrchk(cudaFree(d_rho_max_reached_batch_));
        gpuErrchk(cudaFree(d_merit_initial_batch_));
        gpuErrchk(cudaFree(d_merit_batch_));
        gpuErrchk(cudaFree(d_merit_batch_temp_));
        gpuErrchk(cudaFree(d_iterations_batch_));
        gpuErrchk(cudaFree(d_f_ext_batch_));
    }

    // Member variables
    KKTSystem<T, BatchSize> kkt_system_batch_;
    SchurSystem<T, BatchSize> schur_system_batch_;
    T *d_lambda_batch_;
    T *d_dz_batch_;
    T *d_rho_penalty_batch_;
    T h_rho_penalty_batch_init_[BatchSize];
    T h_drho_batch_init_[BatchSize];
    T *d_drho_batch_;
    // PCG
    int32_t *d_pcg_converged_;
    uint32_t *d_pcg_iterations_;
    // Merit
    T *d_merit_initial_batch_;
    T *d_merit_batch_;
    T *d_merit_batch_temp_;
    // Line search
    T *d_step_size_batch_;
    int32_t *d_all_rho_max_reached_;
    int32_t *d_rho_max_reached_batch_;
    uint32_t *d_iterations_batch_;
    T *d_f_ext_batch_;
};
