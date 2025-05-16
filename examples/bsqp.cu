#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/utils.h"
#include "utils/cuda.cuh"

int main()
{

        // Define constants
        const uint32_t BatchSize = 16;
        const uint32_t TRAJ_SIZE = ((12 + 6) * (BatchSize - 1) + 12);

        T        dt = 0.03;
        uint32_t N = 16;
        uint32_t batch_size = 16;
        uint32_t max_sqp_iters = 10;
        T        kkt_tol = 1e-3;
        uint32_t max_pcg_iters = 100;
        T        pcg_tol = 1e-3;
        T        solve_ratio = 1.0;
        T        mu = 1.0;

        BSQP<T, 16> bsqp(dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, solve_ratio, mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        std::vector<T> reference_traj = readCSVToVec<T>("/home/alex/a2r/gato/GATO/examples/fig8_0.03.csv");
        T*             d_reference_traj_batch;
        gpuErrchk(cudaMalloc(&d_reference_traj_batch, 6 * N * batch_size * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_reference_traj_batch, reference_traj.data(), 6 * N * batch_size * sizeof(T), cudaMemcpyHostToDevice));

        std::vector<T> x_0 = {-1.096711, -0.09903229, 0.83125766, -0.10907673, 0.49704404, 0.01499449, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<T> x_0_batch(12 * batch_size);
        for (uint32_t i = 0; i < batch_size; i++) { std::copy(x_0.begin(), x_0.end(), x_0_batch.begin() + i * 12); }
        T* d_x_0_batch;
        gpuErrchk(cudaMalloc(&d_x_0_batch, 12 * batch_size * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_x_0_batch, x_0_batch.data(), 12 * batch_size * sizeof(T), cudaMemcpyHostToDevice));

        std::vector<T> zeros(6, 0);
        std::vector<T> xu_traj_batch((12 + 6) * (N - 1) * batch_size + 12 * batch_size);
        for (uint32_t b = 0; b < batch_size; b++) {
                for (uint32_t i = 0; i < N - 1; i++) {
                        std::copy(x_0.begin(), x_0.end(), xu_traj_batch.begin() + b * ((12 + 6) * (N - 1) + 12) + i * (12 + 6));
                        std::copy(zeros.begin(), zeros.end(), xu_traj_batch.begin() + b * ((12 + 6) * (N - 1) + 12) + i * (12 + 6) + 12);
                }
                std::copy(x_0.begin(), x_0.end(), xu_traj_batch.begin() + b * ((12 + 6) * (N - 1) + 12) + (N - 1) * (12 + 6));
        }

        T* d_xu_traj_batch;
        gpuErrchk(cudaMalloc(&d_xu_traj_batch, ((12 + 6) * (N - 1) + 12) * batch_size * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_xu_traj_batch, xu_traj_batch.data(), ((12 + 6) * (N - 1) + 12) * batch_size * sizeof(T), cudaMemcpyHostToDevice));

        ProblemInputs<T, BatchSize> inputs;
        inputs.timestep = dt;
        inputs.d_x_s_batch = d_x_0_batch;
        inputs.d_reference_traj_batch = d_reference_traj_batch;

        SQPStats<T, BatchSize> stats = bsqp.solve(d_xu_traj_batch, inputs);

        std::vector<T> h_xu_traj(TRAJ_SIZE * BatchSize);
        gpuErrchk(cudaMemcpy(h_xu_traj.data(), d_xu_traj_batch, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        printf("XU Traj: %.6f, %.6f, %.6f, %.6f\n", h_xu_traj[0], h_xu_traj[1], h_xu_traj[2], h_xu_traj[3]);

        // Free allocated memory
        cudaFree(d_reference_traj_batch);
        cudaFree(d_x_0_batch);
        cudaFree(d_xu_traj_batch);

        return 0;
}
