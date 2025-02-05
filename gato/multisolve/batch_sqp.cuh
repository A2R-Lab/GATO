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
__host__
SQPStats<T, BatchSize> solveSQPBatched(
    T *d_xu_traj_batch, // modified in-place
    T *d_lambda_batch,
    ProblemInputs<T, BatchSize> inputs,
    T rho_penalty
) {
    // ----- Setup -----
    SQPStats<T, BatchSize> sqp_stats;
    PCGStats pcg_stats;
    LineSearchStats<T> line_search_stats;

    auto solve_start_time = std::chrono::high_resolution_clock::now();

    // ----- GPU memory -----
    KKTSystem<T, BatchSize> kkt_system_batch;
    SchurSystem<T, BatchSize> schur_system_batch;

    gpuErrchk(cudaMalloc(&kkt_system_batch.d_Q_batch, STATE_SQ_P_KNOTS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&kkt_system_batch.d_R_batch, CONTROL_SQ_P_KNOTS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&kkt_system_batch.d_q_batch, STATE_P_KNOTS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&kkt_system_batch.d_r_batch, CONTROL_P_KNOTS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&kkt_system_batch.d_A_batch, STATE_SQ_P_KNOTS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&kkt_system_batch.d_B_batch, STATE_P_CONTROL_P_KNOTS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&kkt_system_batch.d_c_batch, STATE_P_KNOTS * BatchSize * sizeof(T)));

    gpuErrchk(cudaMalloc(&schur_system_batch.d_S_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&schur_system_batch.d_P_inv_batch, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&schur_system_batch.d_gamma_batch, VEC_SIZE_PADDED * BatchSize * sizeof(T)));
    gpuErrchk(cudaMemset(schur_system_batch.d_S_batch, 0, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
    gpuErrchk(cudaMemset(schur_system_batch.d_P_inv_batch, 0, B3D_MATRIX_SIZE_PADDED * BatchSize * sizeof(T)));
    gpuErrchk(cudaMemset(schur_system_batch.d_gamma_batch, 0, VEC_SIZE_PADDED * BatchSize * sizeof(T)));

    // -----

    T *d_dz_batch;
    gpuErrchk(cudaMalloc(&d_dz_batch, TRAJ_SIZE * BatchSize * sizeof(T)));
    gpuErrchk(cudaMemset(d_dz_batch, 0, TRAJ_SIZE * BatchSize * sizeof(T)));
    
    T h_rho_penalty_batch[BatchSize] = {rho_penalty};
    T *d_rho_penalty_batch; // penalty for constraint violations (added to Q and R in formSchurSystemBatched)
    gpuErrchk(cudaMalloc(&d_rho_penalty_batch, BatchSize * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_rho_penalty_batch, h_rho_penalty_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice));

    T h_drho_batch[BatchSize] = {static_cast<T>(1.0)};
    T *d_drho_batch; // rate of change for rho (used to update rho in lineSearchAndUpdateBatched)
    gpuErrchk(cudaMalloc(&d_drho_batch, BatchSize * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_drho_batch, h_drho_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
    
    int32_t *d_rho_max_reached_batch;
    gpuErrchk(cudaMalloc(&d_rho_max_reached_batch, BatchSize * sizeof(int32_t)));
    gpuErrchk(cudaMemset(d_rho_max_reached_batch, 0, BatchSize * sizeof(int32_t)));
    
    T h_merit_initial_batch[BatchSize] = {0};
    T *d_merit_initial_batch;
    gpuErrchk(cudaMalloc(&d_merit_initial_batch, BatchSize * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_merit_initial_batch, h_merit_initial_batch, BatchSize * sizeof(T), cudaMemcpyHostToDevice));
    
    T *d_merit_batch;
    gpuErrchk(cudaMalloc(&d_merit_batch, NUM_ALPHAS * BatchSize * sizeof(T)));
    gpuErrchk(cudaMemset(d_merit_batch, 0, NUM_ALPHAS * BatchSize * sizeof(T)));

    uint32_t *d_iterations_batch;
    gpuErrchk(cudaMalloc(&d_iterations_batch, BatchSize * sizeof(uint32_t)));
    gpuErrchk(cudaMemset(d_iterations_batch, 0, BatchSize * sizeof(uint32_t)));

    // ----- Initial Merit -----
    computeMeritBatched<T, BatchSize, 1>(
        d_merit_initial_batch,
        d_dz_batch,
        d_xu_traj_batch,
        inputs
    ); // outputs: initial merit -> (d_merit_initial_batch)

    // ----- SQP Loop -----
    for (uint32_t i = 0; i < SQP_MAX_ITER; i++) {

        // ----- Setup KKT system -----
        // [ G C^T ] [ -dZ ] = [ g ]
        // [ C  0  ] [  λ  ] = [ c ]
        setupKKTSystemBatched<T, BatchSize>(
            kkt_system_batch,
            inputs,
            d_xu_traj_batch
        ); // outputs: KKT system (G (Q, R), C (A, B), g (q, r), c) -> (kkt_system_batch)

        T *h_rho_penalty_batch = new T[BatchSize];
        
        // ----- Form Schur system -----
        // P_inv * S * λ = P_inv * gamma
        formSchurSystemBatched<T, BatchSize>(
            schur_system_batch,
            kkt_system_batch,
            d_rho_penalty_batch
        ); // outputs: Schur system (S, gamma, P_inv) -> (schur_system_batch), Q_inv (in place), R_inv (in place)
        
        // ----- Solve system with PCG -----
        pcg_stats = solvePCGBatched<T, BatchSize>(
            d_lambda_batch,
            schur_system_batch,
            PCG_TOLERANCE,
            d_rho_max_reached_batch // so we can skip solves where rho_max_reached
        ); // outputs: pcg_stats, lambda -> (d_lambda_batch)

        computeDzBatched<T, BatchSize>(
            d_dz_batch,
            d_lambda_batch,
            kkt_system_batch
        ); // outputs: dz -> (d_dz_batch)

        // ----- Line search and update -----
        computeMeritBatched<T, BatchSize, NUM_ALPHAS>(
            d_merit_batch,
            d_dz_batch,
            d_xu_traj_batch,
            inputs
        ); // outputs: merit -> (d_merit_batch)

        line_search_stats = lineSearchAndUpdateBatched<T, BatchSize, NUM_ALPHAS>(
            d_xu_traj_batch,
            d_dz_batch,
            d_merit_batch,
            d_merit_initial_batch,
            d_rho_penalty_batch,
            d_drho_batch,
            d_rho_max_reached_batch,
            d_iterations_batch
        ); // outputs: line_search_stats, xu_traj_new (in place), rho, drho, rho_max_reached, iterations
        
        sqp_stats.pcg_stats.push_back(pcg_stats);
        sqp_stats.line_search_stats.push_back(line_search_stats);
        if (line_search_stats.all_rho_max_reached) break;
    }

    gpuErrchk(cudaDeviceSynchronize());
    auto solve_end_time = std::chrono::high_resolution_clock::now();
    
    // finish populating sqp_stats
    sqp_stats.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time).count();
    gpuErrchk(cudaMemcpy(sqp_stats.sqp_iterations.data(), d_iterations_batch, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(sqp_stats.rho_max_reached.data(), d_rho_max_reached_batch, BatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost));


    gpuErrchk(cudaFree(kkt_system_batch.d_Q_batch));
    gpuErrchk(cudaFree(kkt_system_batch.d_R_batch));
    gpuErrchk(cudaFree(kkt_system_batch.d_q_batch));
    gpuErrchk(cudaFree(kkt_system_batch.d_r_batch));
    gpuErrchk(cudaFree(kkt_system_batch.d_A_batch));
    gpuErrchk(cudaFree(kkt_system_batch.d_B_batch));
    gpuErrchk(cudaFree(kkt_system_batch.d_c_batch));

    gpuErrchk(cudaFree(schur_system_batch.d_S_batch));
    gpuErrchk(cudaFree(schur_system_batch.d_P_inv_batch));
    gpuErrchk(cudaFree(schur_system_batch.d_gamma_batch));

    gpuErrchk(cudaFree(d_dz_batch));
    gpuErrchk(cudaFree(d_rho_penalty_batch));
    gpuErrchk(cudaFree(d_drho_batch));
    gpuErrchk(cudaFree(d_rho_max_reached_batch));
    gpuErrchk(cudaFree(d_merit_initial_batch));
    gpuErrchk(cudaFree(d_merit_batch));
    gpuErrchk(cudaFree(d_iterations_batch));
    
    return sqp_stats;
}
