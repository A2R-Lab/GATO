// A batched trajectory-optimization solve straight on the C++/CUDA solver —
// the same BSQP<T> class the Python package wraps, with no wrapper in the loop.
//
// The plant (robot), horizon and plant adapter are compile-time defines
// (KNOT_POINTS, GATO_PLANT_NAME, GATO_PLANT_HEADER); `python -m gato.paths
// --robot indy7 --N 32` prints them with the include directories, and the
// CMakeLists.txt next to this file consumes that output. Everything the solver
// reads or writes lives in device memory the caller owns:
//
//   d_x_s_batch            (B, nx)                  measured states
//   d_reference_traj_batch (B, 6 * N)               EE reference per knot: xyz + 3 zeros
//   d_xu_traj_batch        (B, N*(nx+nu) - nu)      warm start in, solution out
//
// Built against the vendored Indy7 by default: B solves of the paper's figure-8
// reference from the "ready" posture, then one warm-started solve. Printed
// durations are diagnostics on whatever GPU this runs on, not a benchmark.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/cuda.cuh"

using gato::constants::CONTROL_SIZE;
using gato::constants::XU_STATE_SIZE;
using gato::constants::XU_TRAJ_SIZE;
using gato::constants::REFERENCE_TRAJ_SIZE;

// The figure-8 of gato.common.figure8 with its defaults (A 0.4 m, centre
// (0, 0.5, 0.6), 6 s period, rotated 45 degrees about z), sampled every dt.
static std::vector<T> figure8_reference(T dt, uint32_t knots, T t0)
{
        const T A = 0.4, ox = 0.0, oy = 0.5, oz = 0.6, period = 6.0, theta = M_PI / 4;
        std::vector<T> ref(6 * knots, T(0));
        for (uint32_t k = 0; k < knots; ++k) {
                const T t = (t0 + k * dt) * (2 * M_PI / period);
                const T x = ox + A * std::sin(t), y = oy, z = oz + A * std::sin(2 * t) / 2 + A / 2;
                ref[6 * k + 0] = std::cos(theta) * x - std::sin(theta) * y;
                ref[6 * k + 1] = std::sin(theta) * x + std::cos(theta) * y;
                ref[6 * k + 2] = z;
        }
        return ref;
}

int main()
{
        const uint32_t B = 8, N = KNOT_POINTS;
        const T dt = 0.01;
        // Indy7 "ready" posture (gato.config.INDY7_START_CONFIGS["ready"]); other plants: fill nq values
        std::vector<T> q0 = {-1.096711, -0.09903229, 0.83125766, -0.10907673, 0.49704404, 0.01499449};
        if (q0.size() * 2 != XU_STATE_SIZE) { q0.assign(XU_STATE_SIZE / 2, T(0)); }

        // batch inputs on the host: each row starts from the posture, nudged on joint 0
        std::vector<T> h_x(B * XU_STATE_SIZE, T(0)), h_ref(B * REFERENCE_TRAJ_SIZE), h_xu(B * XU_TRAJ_SIZE, T(0));
        for (uint32_t b = 0; b < B; ++b) {
                for (uint32_t i = 0; i < q0.size(); ++i) h_x[b * XU_STATE_SIZE + i] = q0[i];
                h_x[b * XU_STATE_SIZE] += T(0.02) * b;
                const auto ref = figure8_reference(dt, N, T(0.1) * b);
                std::copy(ref.begin(), ref.end(), h_ref.begin() + b * REFERENCE_TRAJ_SIZE);
                // cold start = hold at x: every knot's state is x, controls zero
                for (uint32_t k = 0; k < N; ++k)
                        for (uint32_t i = 0; i < XU_STATE_SIZE; ++i)
                                h_xu[b * XU_TRAJ_SIZE + k * (XU_STATE_SIZE + CONTROL_SIZE) + i] = h_x[b * XU_STATE_SIZE + i];
        }

        T *d_x, *d_ref, *d_xu;
        gpuErrchk(cudaMalloc(&d_x, h_x.size() * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_ref, h_ref.size() * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_xu, h_xu.size() * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_ref, h_ref.data(), h_ref.size() * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_xu, h_xu.data(), h_xu.size() * sizeof(T), cudaMemcpyHostToDevice));

        // (batch, max_sqp_iters, max_pcg_iters, pcg_tol, solve_ratio, mu,
        //  q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost, rho)
        // — the same knobs as gato.SolverParams; these are its defaults with 5 SQP iterations.
        BSQP<T> solver(B, 5, 200, 1e-4, 1.0, 10.0, 1.0, 1e-2, 2e-6, 50.0, 1e-2, 0.0, 0.0, 1e-2);

        ProblemInputs<T> inputs;
        inputs.timestep = dt;
        inputs.d_x_s_batch = d_x;
        inputs.d_reference_traj_batch = d_ref;

        auto t0 = std::chrono::high_resolution_clock::now();
        SQPStats<T> stats = solver.solve(d_xu, inputs);              // d_xu: warm start in, solution out
        gpuErrchk(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<T> merit(B);
        solver.copy_final_merit_to_host(merit.data());
        gpuErrchk(cudaMemcpy(h_xu.data(), d_xu, h_xu.size() * sizeof(T), cudaMemcpyDeviceToHost));
        std::printf("GATO C++/CUDA batched solve: plant %s, N=%u, B=%u, nx=%u, nu=%u\n",
                    GATO_PLANT_NAME_STR, N, B, XU_STATE_SIZE, CONTROL_SIZE);
        std::printf("  first solve %.1f ms (includes CUDA context and kernel initialization), SQP iterations per row:",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
        for (uint32_t b = 0; b < B; ++b) std::printf(" %d", stats.sqp_iterations[b]);
        std::printf("\n  final merit per row:");
        for (uint32_t b = 0; b < B; ++b) std::printf(" %.3f", merit[b]);
        std::printf("\n  first control of row 0:");
        for (uint32_t i = 0; i < CONTROL_SIZE; ++i) std::printf(" %.2f", h_xu[XU_STATE_SIZE + i]);
        std::printf("\n");

        // warm start: the solution is already in d_xu; only the measured state is refreshed
        gpuErrchk(cudaMemcpy2D(d_xu, XU_TRAJ_SIZE * sizeof(T), d_x, XU_STATE_SIZE * sizeof(T),
                               XU_STATE_SIZE * sizeof(T), B, cudaMemcpyDeviceToDevice));
        SQPStats<T> again = solver.solve(d_xu, inputs);
        gpuErrchk(cudaDeviceSynchronize());
        solver.copy_final_merit_to_host(merit.data());
        std::printf("  warm-started solve: %.3f ms (internal solver time), merit row 0 %.3f\n", again.solve_time_us / 1000.0, merit[0]);

        bool finite = true;
        gpuErrchk(cudaMemcpy(h_xu.data(), d_xu, h_xu.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (T v : h_xu) finite = finite && std::isfinite(v);
        std::printf("  trajectories finite: %s\n", finite ? "yes" : "NO");
        gpuErrchk(cudaFree(d_x));
        gpuErrchk(cudaFree(d_ref));
        gpuErrchk(cudaFree(d_xu));
        return finite ? 0 : 1;
}
