#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "multisolve/batch_sqp_solver.cuh"
#include "types.cuh"
#include "utils/utils.h"

namespace py = pybind11;

template <typename T, uint32_t BatchSize>
class PySQPSolver {
public:
    PySQPSolver() : solver_() {}

    py::dict solve(
        py::array_t<T> xu_traj_batch,
        T timestep,
        py::array_t<T> x_s_batch,
        py::array_t<T> reference_traj_batch
    ) {
        // Get buffer info for numpy arrays
        py::buffer_info xu_buf = xu_traj_batch.request();
        py::buffer_info xs_buf = x_s_batch.request();
        py::buffer_info ref_buf = reference_traj_batch.request();
            
        // Allocate and copy device memory
        T *d_xu_traj_batch;
        gpuErrchk(cudaMalloc(&d_xu_traj_batch, TRAJ_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_xu_traj_batch, xu_buf.ptr, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));

        // Setup problem inputs
        ProblemInputs<T, BatchSize> inputs;
        inputs.timestep = timestep;
        gpuErrchk(cudaMalloc(&inputs.d_x_s_batch, STATE_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&inputs.d_reference_traj_batch, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMemcpy(inputs.d_x_s_batch, xs_buf.ptr, STATE_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(inputs.d_reference_traj_batch, ref_buf.ptr, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
        inputs.d_GRiD_mem = gato::plant::initializeDynamicsConstMem<T>();

        // Solve
        SQPStats<T, BatchSize> stats = solver_.solve(d_xu_traj_batch, inputs);

        // Copy trajectory back to host
        std::vector<T> h_xu_traj(TRAJ_SIZE * BatchSize);
        gpuErrchk(cudaMemcpy(h_xu_traj.data(), d_xu_traj_batch, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

        // Create return dictionary
        py::dict result;
        result["xu_trajectory"] = py::array_t<T>({BatchSize, TRAJ_SIZE}, h_xu_traj.data());
        result["solve_time_us"] = stats.solve_time_us;
        result["sqp_iterations"] = py::array_t<int32_t>(
            {BatchSize},                    // shape
            {sizeof(int32_t)},            // stride
            stats.sqp_iterations.data()     // data
        );
        result["rho_max_reached"] = py::array_t<int32_t>(
            {BatchSize},                    // shape
            {sizeof(int32_t)},             // stride
            stats.rho_max_reached.data()    // data
        );

        // Add PCG stats
        py::list pcg_stats_list;
        for (const auto& pcg_stat : stats.pcg_stats) {
            py::dict pcg_dict;
            pcg_dict["solve_time_us"] = pcg_stat.solve_time_us;
            pcg_dict["pcg_iterations"] = py::array_t<int>(
                {BatchSize},                    // shape
                {sizeof(int)},                 // stride
                pcg_stat.num_iterations.data()  // data
            );
            pcg_dict["converged"] = py::array_t<int>(
                {BatchSize},                    // shape
                {sizeof(int)},                 // stride
                pcg_stat.converged.data()      // data
            );
            pcg_stats_list.append(pcg_dict);
        }
        result["pcg_stats"] = pcg_stats_list;

        // Cleanup
        gpuErrchk(cudaFree(d_xu_traj_batch));
        gpuErrchk(cudaFree(inputs.d_x_s_batch));
        gpuErrchk(cudaFree(inputs.d_reference_traj_batch));
        gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem);

        return result;
    }

    void reset() {
        solver_.reset();
    }

private:
    SQPSolver<T, BatchSize> solver_;
};

PYBIND11_MODULE(batch_sqp, m) {
    py::class_<PySQPSolver<float, 1>>(m, "SQPSolverfloat_1")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 1>::solve)
        .def("reset", &PySQPSolver<float, 1>::reset);

    py::class_<PySQPSolver<float, 2>>(m, "SQPSolverfloat_2")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 2>::solve)
        .def("reset", &PySQPSolver<float, 2>::reset);

    py::class_<PySQPSolver<float, 4>>(m, "SQPSolverfloat_4")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 4>::solve)
        .def("reset", &PySQPSolver<float, 4>::reset);

    py::class_<PySQPSolver<float, 8>>(m, "SQPSolverfloat_8")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 8>::solve)
        .def("reset", &PySQPSolver<float, 8>::reset);

    py::class_<PySQPSolver<float, 16>>(m, "SQPSolverfloat_16")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 16>::solve)
        .def("reset", &PySQPSolver<float, 16>::reset);

    py::class_<PySQPSolver<float, 32>>(m, "SQPSolverfloat_32")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 32>::solve)
        .def("reset", &PySQPSolver<float, 32>::reset);

    py::class_<PySQPSolver<float, 64>>(m, "SQPSolverfloat_64")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 64>::solve)
        .def("reset", &PySQPSolver<float, 64>::reset);

    py::class_<PySQPSolver<float, 128>>(m, "SQPSolverfloat_128")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 128>::solve)
        .def("reset", &PySQPSolver<float, 128>::reset);

    py::class_<PySQPSolver<float, 256>>(m, "SQPSolverfloat_256")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 256>::solve)
        .def("reset", &PySQPSolver<float, 256>::reset);
} 