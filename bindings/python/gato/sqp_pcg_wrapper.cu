#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <vector>
#include <tuple>
#include "solvers/sqp/sqp_pcg.cuh"

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

SQPStats sqp_pcg_wrapper(py::array_t<float> eePos_goal_traj,
                        py::array_t<float> xu,
                        py::array_t<float> lambda,
                        float rho,
                        float rho_reset,
                        int pcg_max_iter,
                        float pcg_exit_tol) {

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


PYBIND11_MODULE(gato, m) {
    m.doc() = "Python bindings for GATO";

    py::class_<SQPStats>(m, "SQPStats")
        .def_readwrite("pcg_iter_vec", &SQPStats::pcg_iter_vec)
        .def_readwrite("linsys_time_vec", &SQPStats::linsys_time_vec)
        .def_readwrite("sqp_solve_time", &SQPStats::sqp_solve_time)
        .def_readwrite("sqp_iter", &SQPStats::sqp_iter)
        .def_readwrite("sqp_time_exit", &SQPStats::sqp_time_exit)
        .def_readwrite("pcg_exit_vec", &SQPStats::pcg_exit_vec);

    m.def("solve_sqp_pcg", &sqp_pcg_wrapper, 
        py::arg("eePos_goal_traj"),
        py::arg("xu"),
        py::arg("lambda"),
        py::arg("rho"),
        py::arg("rho_reset"),
        py::arg("pcg_max_iter"),
        py::arg("pcg_exit_tol"),
        "Solve SQP problem using PCG");
}