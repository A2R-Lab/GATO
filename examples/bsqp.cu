#include <iostream>
#include <vector>
#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/cuda.cuh"

int main()
{
    // Define constants
    const uint32_t BatchSize = 16;

    T        dt = 0.03;
    uint32_t N = 16;
    uint32_t batch_size = 16;
    const uint32_t STATE = gato::constants::STATE_SIZE;
    const uint32_t CONTROL = gato::constants::CONTROL_SIZE;
    const uint32_t TRAJ_SIZE = ((STATE + CONTROL) * (N - 1) + STATE);
    uint32_t max_sqp_iters = 10;
    T        kkt_tol = 1e-3;
    uint32_t max_pcg_iters = 100;
    T        pcg_tol = 1e-3;
    T        solve_ratio = 1.0;
    T        mu = 1.0;

    BSQP<T, 16> bsqp(dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, solve_ratio, mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Generate synthetic reference trajectory data instead of loading from file
    std::vector<T> reference_traj(grid::EE_POS_SIZE * N * batch_size, 0.0);
    // Fill with some simple pattern
    for (uint32_t i = 0; i < reference_traj.size(); i++) {
        reference_traj[i] = 0.1 * (i % grid::EE_POS_SIZE);
    }
    
    T* d_reference_traj_batch;
    gpuErrchk(cudaMalloc(&d_reference_traj_batch, grid::EE_POS_SIZE * N * batch_size * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_reference_traj_batch, reference_traj.data(), grid::EE_POS_SIZE * N * batch_size * sizeof(T), cudaMemcpyHostToDevice));

    // Use sample initial state (zeros with correct DOF)
    std::vector<T> x_0(STATE, static_cast<T>(0));
    std::vector<T> x_0_batch(STATE * batch_size);
    for (uint32_t i = 0; i < batch_size; i++) { std::copy(x_0.begin(), x_0.end(), x_0_batch.begin() + i * STATE); }
    T* d_x_0_batch;
    gpuErrchk(cudaMalloc(&d_x_0_batch, STATE * batch_size * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_x_0_batch, x_0_batch.data(), STATE * batch_size * sizeof(T), cudaMemcpyHostToDevice));

    std::vector<T> zeros(CONTROL, static_cast<T>(0));
    std::vector<T> xu_traj_batch((STATE + CONTROL) * (N - 1) * batch_size + STATE * batch_size);
    for (uint32_t b = 0; b < batch_size; b++) {
            for (uint32_t i = 0; i < N - 1; i++) {
                    std::copy(x_0.begin(), x_0.end(), xu_traj_batch.begin() + b * ((STATE + CONTROL) * (N - 1) + STATE) + i * (STATE + CONTROL));
                    std::copy(zeros.begin(), zeros.end(), xu_traj_batch.begin() + b * ((STATE + CONTROL) * (N - 1) + STATE) + i * (STATE + CONTROL) + STATE);
            }
            std::copy(x_0.begin(), x_0.end(), xu_traj_batch.begin() + b * ((STATE + CONTROL) * (N - 1) + STATE) + (N - 1) * (STATE + CONTROL));
    }

    T* d_xu_traj_batch;
    gpuErrchk(cudaMalloc(&d_xu_traj_batch, ((STATE + CONTROL) * (N - 1) + STATE) * batch_size * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xu_traj_batch, xu_traj_batch.data(), ((STATE + CONTROL) * (N - 1) + STATE) * batch_size * sizeof(T), cudaMemcpyHostToDevice));

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
