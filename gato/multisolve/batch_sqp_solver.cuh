#pragma once

#include <iostream>
#include <cstdint>
#include <chrono>

#include "config/settings.h"
#include "config/constants.h"
#include "utils/types.cuh"
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
        }
        T h_rho_penalty_batch_init[BatchSize] = {static_cast<T>(RHO_INIT)};
        gpuErrchk(cudaMemcpy(d_rho_penalty_batch_, h_rho_penalty_batch_init, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
    }

    ~SQPSolver() {
        freeMemory();
    }

    void reset() {
        // TODO: reset rhos (what else?)
        return;
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
        PCGStats pcg_stats;
        LineSearchStats<T> line_search_stats;

        auto solve_start_time = std::chrono::high_resolution_clock::now();

        // Reset drho batch 
        gpuErrchk(cudaMemcpy(d_drho_batch_, h_drho_batch_init_, BatchSize * sizeof(T), cudaMemcpyHostToDevice));

        // Reset other arrays
        gpuErrchk(cudaMemset(d_dz_batch_, 0, TRAJ_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_rho_max_reached_batch_, 0, BatchSize * sizeof(int32_t)));
        gpuErrchk(cudaMemset(d_merit_initial_batch_, 0, BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_merit_batch_, 0, NUM_ALPHAS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemset(d_iterations_batch_, 0, BatchSize * sizeof(uint32_t)));

        // Initial Merit
        computeMeritBatched<T, BatchSize, 1>(
            d_merit_initial_batch_,
            d_dz_batch_,
            d_xu_traj_batch,
            inputs
        );

        // SQP Loop
        for (uint32_t i = 0; i < SQP_MAX_ITER; i++) {
            setupKKTSystemBatched<T, BatchSize>(
                kkt_system_batch_,
                inputs,
                d_xu_traj_batch
            );

            formSchurSystemBatched<T, BatchSize>(
                schur_system_batch_,
                kkt_system_batch_,
                d_rho_penalty_batch_
            );

            pcg_stats = solvePCGBatched<T, BatchSize>(
                d_lambda_batch_,
                schur_system_batch_,
                PCG_TOLERANCE,
                d_rho_max_reached_batch_
            );

            computeDzBatched<T, BatchSize>(
                d_dz_batch_,
                d_lambda_batch_,
                kkt_system_batch_
            );

            computeMeritBatched<T, BatchSize, NUM_ALPHAS>(
                d_merit_batch_,
                d_dz_batch_,
                d_xu_traj_batch,
                inputs
            );

            line_search_stats = lineSearchAndUpdateBatched<T, BatchSize, NUM_ALPHAS>(
                d_xu_traj_batch,
                d_dz_batch_,
                d_merit_batch_,
                d_merit_initial_batch_,
                d_rho_penalty_batch_,
                d_drho_batch_,
                d_rho_max_reached_batch_,
                d_iterations_batch_
            );

            sqp_stats.pcg_stats.push_back(pcg_stats);
            sqp_stats.line_search_stats.push_back(line_search_stats);
            if (line_search_stats.all_rho_max_reached) break;
        }

        gpuErrchk(cudaDeviceSynchronize());
        auto solve_end_time = std::chrono::high_resolution_clock::now();

        // Populate stats
        sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time).count();
        gpuErrchk(cudaMemcpy(sqp_stats.sqp_iterations.data(), d_iterations_batch_, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(sqp_stats.rho_max_reached.data(), d_rho_max_reached_batch_, BatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost));

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
        gpuErrchk(cudaMalloc(&schur_system_batch_.d_S_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&schur_system_batch_.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
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
        gpuErrchk(cudaMalloc(&d_rho_max_reached_batch_, BatchSize * sizeof(int32_t)));
        gpuErrchk(cudaMalloc(&d_merit_initial_batch_, BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_merit_batch_, NUM_ALPHAS * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_iterations_batch_, BatchSize * sizeof(uint32_t)));
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
        gpuErrchk(cudaFree(d_dz_batch_));
        gpuErrchk(cudaFree(d_rho_penalty_batch_));
        gpuErrchk(cudaFree(d_drho_batch_));
        gpuErrchk(cudaFree(d_rho_max_reached_batch_));
        gpuErrchk(cudaFree(d_merit_initial_batch_));
        gpuErrchk(cudaFree(d_merit_batch_));
        gpuErrchk(cudaFree(d_iterations_batch_));
    }

    // Member variables
    KKTSystem<T, BatchSize> kkt_system_batch_;
    SchurSystem<T, BatchSize> schur_system_batch_;
    T *d_lambda_batch_;
    T *d_dz_batch_;
    T *d_rho_penalty_batch_;
    T h_drho_batch_init_[BatchSize];
    T *d_drho_batch_;
    int32_t *d_rho_max_reached_batch_;
    T *d_merit_initial_batch_;
    T *d_merit_batch_;
    uint32_t *d_iterations_batch_;
};
