#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include "solvers/sqp/sqp_pcg_n.cuh"
#include "gato.cuh"

namespace py = pybind11;

struct SQPStatsN {
    std::vector<std::vector<int>> pcg_iters_matrix;
    std::vector<double> pcg_times_vec;
    double sqp_solve_time;
    std::vector<uint32_t> sqp_iterations_vec;
    std::vector<char> sqp_time_exit_vec;
    std::vector<std::vector<bool>> pcg_exits_matrix;
};

SQPStatsN sqp_pcg_n_wrapper(
    uint32_t solve_count,
    py::array_t<float> eePos_goal_traj,
    py::array_t<float> xu_traj,
    float pcg_exit_tol,
    uint32_t pcg_max_iter,
    float rho_init,
    float rho_reset
) {
    // Constants
    constexpr uint32_t state_size = gato::STATE_SIZE;
    constexpr uint32_t control_size = gato::CONTROL_SIZE;
    constexpr uint32_t knot_points = gato::KNOT_POINTS;
    const float timestep = gato::TIMESTEP;
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;

    // Get input array info
    py::buffer_info eePos_buf = eePos_goal_traj.request();
    py::buffer_info xu_buf = xu_traj.request();

    // Allocate device memory
    float *d_eePos_trajs, *d_xu_trajs, *d_lambdas, *d_rhos;
    gpuErrchk(cudaMalloc(&d_eePos_trajs, 6 * knot_points * solve_count * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_xu_trajs, traj_size * solve_count * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_lambdas, state_size * knot_points * solve_count * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_rhos, solve_count * sizeof(float)));

    // Copy data to device
    gpuErrchk(cudaMemcpy(d_eePos_trajs, eePos_buf.ptr, 6 * knot_points * solve_count * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xu_trajs, xu_buf.ptr, traj_size * solve_count * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_lambdas, 0, state_size * knot_points * solve_count * sizeof(float)));
    gpuErrchk(cudaMemset(d_rhos, rho_init, solve_count * sizeof(float)));

    // Initialize dynamics memory
    void *d_dynMem_const = gato::plant::initializeDynamicsConstMem<float>();

    // Set up PCG configuration
    pcg_config<float> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = pcg_exit_tol;
    config.pcg_max_iter = pcg_max_iter;

    gpuErrchk(cudaPeekAtLastError());

    // Call the CUDA function
    auto result = sqpSolvePcgN<float>(solve_count, state_size, control_size, knot_points, timestep,
                                      d_eePos_trajs, d_lambdas, d_xu_trajs, d_dynMem_const,
                                      config, d_rhos, rho_reset);

    // Copy results back to host
    SQPStatsN sqp_result;
    sqp_result.pcg_iters_matrix = std::get<0>(result);
    sqp_result.pcg_times_vec = std::get<1>(result);
    sqp_result.sqp_solve_time = std::get<2>(result);
    sqp_result.sqp_iterations_vec = std::get<3>(result);
    sqp_result.sqp_time_exit_vec = std::get<4>(result);
    sqp_result.pcg_exits_matrix = std::get<5>(result);

    // Update input arrays with new values
    gpuErrchk(cudaMemcpy(xu_buf.ptr, d_xu_trajs, traj_size * solve_count * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    gpuErrchk(cudaFree(d_eePos_trajs));
    gpuErrchk(cudaFree(d_xu_trajs));
    gpuErrchk(cudaFree(d_lambdas));
    gpuErrchk(cudaFree(d_rhos));
    gato::plant::freeDynamicsConstMem<float>(d_dynMem_const);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return sqp_result;
}

PYBIND11_MODULE(gato, m) {
    m.doc() = "Python bindings for GATO SQP-PCG solver";

    py::class_<SQPStatsN>(m, "SQPStatsN")
        .def_readwrite("pcg_iters_matrix", &SQPStatsN::pcg_iters_matrix)
        .def_readwrite("pcg_times_vec", &SQPStatsN::pcg_times_vec)
        .def_readwrite("sqp_solve_time", &SQPStatsN::sqp_solve_time)
        .def_readwrite("sqp_iterations_vec", &SQPStatsN::sqp_iterations_vec)
        .def_readwrite("sqp_time_exit_vec", &SQPStatsN::sqp_time_exit_vec)
        .def_readwrite("pcg_exits_matrix", &SQPStatsN::pcg_exits_matrix);

    m.def("solve_sqp_pcg_n", &sqp_pcg_n_wrapper,
        py::arg("solve_count"),
        py::arg("eePos_goal_traj"),
        py::arg("xu_traj"),
        py::arg("pcg_exit_tol"),
        py::arg("pcg_max_iter"),
        py::arg("rho_init"),
        py::arg("rho_reset"),
        "Solve SQP problem using PCG on CUDA for multiple trajectories");
}