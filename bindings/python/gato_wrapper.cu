#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include "solvers/sqp/sqp_pcg.cuh"
#include "solvers/sqp/sqp_pcg_n.cuh"
#include "gato.cuh"

namespace py = pybind11;

// We need a struct for pybind11 to store our return values
struct SQPStats {
    std::vector<int> pcg_iter_vec;
    std::vector<double> linsys_time_vec;
    double sqp_solve_time;
    int sqp_iter;
    bool sqp_time_exit;
    std::vector<bool> pcg_exit_vec;
};

//same for multisolve
struct SQPStatsN {
    std::vector<std::vector<int>> pcg_iters_matrix;
    std::vector<double> pcg_times_vec;
    double sqp_solve_time;
    std::vector<uint32_t> sqp_iterations_vec;
    std::vector<char> sqp_time_exit_vec;
    std::vector<std::vector<bool>> pcg_exits_matrix;
};


SQPStats sqp_pcg_wrapper(py::array_t<float> eePos_goal_traj,
    py::array_t<float> xu,
    py::array_t<float> lambda,
    float rho,
    float rho_reset,
    int pcg_max_iter,
    float pcg_exit_tol
) {

    py::buffer_info eePos_buf = eePos_goal_traj.request();
    py::buffer_info xu_buf = xu.request();
    py::buffer_info lambda_buf = lambda.request();

    // device memory
    float *d_eePos_goal_traj, *d_xu, *d_lambda;
    gpuErrchk(cudaMalloc(&d_eePos_goal_traj, eePos_buf.size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_xu, xu_buf.size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_lambda, lambda_buf.size * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_eePos_goal_traj, eePos_buf.ptr, eePos_buf.size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xu, xu_buf.ptr, xu_buf.size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lambda, lambda_buf.ptr, lambda_buf.size * sizeof(float), cudaMemcpyHostToDevice));

    void *d_dynMem_const = gato::plant::initializeDynamicsConstMem<float>();


    pcg_config<float> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = pcg_exit_tol;
    config.pcg_max_iter = pcg_max_iter;

    gpuErrchk(cudaPeekAtLastError());

    auto result = sqpSolvePcg<float>(d_eePos_goal_traj, d_xu, d_lambda, d_dynMem_const, config, rho, rho_reset);

    std::vector<int> pcg_iter_vec = std::get<0>(result);
    std::vector<double> linsys_time_vec = std::get<1>(result);
    double sqp_solve_time = std::get<2>(result);
    int sqp_iter = std::get<3>(result);
    bool sqp_time_exit = std::get<4>(result);
    std::vector<bool> pcg_exit_vec = std::get<5>(result);

    // Update input arrays inplace
    gpuErrchk(cudaMemcpy(xu_buf.ptr, d_xu, xu_buf.size * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(lambda_buf.ptr, d_lambda, lambda_buf.size * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_eePos_goal_traj));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_lambda));
    gato::plant::freeDynamicsConstMem<float>(d_dynMem_const);

    gpuErrchk(cudaPeekAtLastError());

    return SQPStats{
    pcg_iter_vec,
    linsys_time_vec,
    sqp_solve_time,
    sqp_iter,
    sqp_time_exit,
    pcg_exit_vec
    };
}



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
    m.doc() = "Python bindings for GATO";

    py::class_<SQPStats>(m, "SQPStats")
        .def_readwrite("pcg_iter_vec", &SQPStats::pcg_iter_vec)
        .def_readwrite("linsys_time_vec", &SQPStats::linsys_time_vec)
        .def_readwrite("sqp_solve_time", &SQPStats::sqp_solve_time)
        .def_readwrite("sqp_iter", &SQPStats::sqp_iter)
        .def_readwrite("sqp_time_exit", &SQPStats::sqp_time_exit)
        .def_readwrite("pcg_exit_vec", &SQPStats::pcg_exit_vec);

    py::class_<SQPStatsN>(m, "SQPStatsN")
        .def_readwrite("pcg_iters_matrix", &SQPStatsN::pcg_iters_matrix)
        .def_readwrite("pcg_times_vec", &SQPStatsN::pcg_times_vec)
        .def_readwrite("sqp_solve_time", &SQPStatsN::sqp_solve_time)
        .def_readwrite("sqp_iterations_vec", &SQPStatsN::sqp_iterations_vec)
        .def_readwrite("sqp_time_exit_vec", &SQPStatsN::sqp_time_exit_vec)
        .def_readwrite("pcg_exits_matrix", &SQPStatsN::pcg_exits_matrix);

    m.def("solve_sqp_pcg", &sqp_pcg_wrapper, 
        py::arg("eePos_goal_traj"),
        py::arg("xu"),
        py::arg("lambda"),
        py::arg("rho"),
        py::arg("rho_reset"),
        py::arg("pcg_max_iter"),
        py::arg("pcg_exit_tol"),
        "Solve SQP problem using PCG");

    m.def("solve_sqp_pcg_n", &sqp_pcg_n_wrapper,
        py::arg("solve_count"),
        py::arg("eePos_goal_traj"),
        py::arg("xu_traj"),
        py::arg("pcg_exit_tol"),
        py::arg("pcg_max_iter"),
        py::arg("rho_init"),
        py::arg("rho_reset"),
        "Solve SQP problem using PCG for multiple trajectories");
}