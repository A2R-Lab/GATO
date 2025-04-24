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

        py::list line_search_stats_list;
        for (const auto& line_search_stat : stats.line_search_stats) {
            py::dict line_search_dict;
            line_search_dict["min_merit"] = line_search_stat.min_merit;
            line_search_dict["step_size"] = line_search_stat.step_size;
            line_search_dict["all_rho_max_reached"] = line_search_stat.all_rho_max_reached;
            line_search_stats_list.append(line_search_dict);
        }
        result["line_search_stats"] = line_search_stats_list;

        // Cleanup
        gpuErrchk(cudaFree(d_xu_traj_batch));
        gpuErrchk(cudaFree(inputs.d_x_s_batch));
        gpuErrchk(cudaFree(inputs.d_reference_traj_batch));
        gato::plant::freeDynamicsConstMem<T>(inputs.d_GRiD_mem);

        return result;
    }
    
    void set_external_wrench(py::array_t<T> f_ext, uint32_t solve_idx) {
        py::buffer_info f_ext_buf = f_ext.request();
        solver_.set_external_wrench(static_cast<float*>(f_ext_buf.ptr), solve_idx);
    }

    void set_external_wrench_batch(py::array_t<T> f_ext_batch) {
        py::buffer_info f_ext_buf = f_ext_batch.request();
        solver_.set_external_wrench_batch(static_cast<float*>(f_ext_buf.ptr));
    }

    py::array_t<T> sim_forward(py::array_t<T> xk, py::array_t<T> uk, T dt) {
        py::buffer_info xk_buf = xk.request();
        py::buffer_info uk_buf = uk.request();

        T *d_xkp1_batch, *d_xk, *d_uk;
        gpuErrchk(cudaMalloc(&d_xkp1_batch, STATE_SIZE * BatchSize * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_xk, STATE_SIZE * sizeof(T)));
        gpuErrchk(cudaMalloc(&d_uk, CONTROL_SIZE * sizeof(T)));
        
        gpuErrchk(cudaMemcpy(d_xk, xk_buf.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_uk, uk_buf.ptr, CONTROL_SIZE * sizeof(T), cudaMemcpyHostToDevice));

        solver_.sim_forward(d_xkp1_batch, d_xk, d_uk, dt);
        gpuErrchk(cudaDeviceSynchronize());

        std::vector<T> h_xkp1_batch(STATE_SIZE * BatchSize);
        gpuErrchk(cudaMemcpy(h_xkp1_batch.data(), d_xkp1_batch, STATE_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_xkp1_batch));
        gpuErrchk(cudaFree(d_xk));
        gpuErrchk(cudaFree(d_uk));

        return py::array_t<T>({BatchSize, STATE_SIZE}, h_xkp1_batch.data());
    }

    void reset() {
        solver_.reset();
    }

    void resetRho(){
        solver_.resetRho();
    }

    void resetLambda(){
        solver_.resetLambda();
    }
private:
    SQPSolver<T, BatchSize> solver_;
};

PYBIND11_MODULE(batch_sqp, m) {
    py::class_<PySQPSolver<float, 1>>(m, "SQPSolverfloat_1")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 1>::solve)
        .def("reset", &PySQPSolver<float, 1>::reset)
        .def("resetRho", &PySQPSolver<float, 1>::resetRho)
        .def("resetLambda", &PySQPSolver<float, 1>::resetLambda)
        .def("set_external_wrench", &PySQPSolver<float, 1>::set_external_wrench)
        .def("set_external_wrench_batch", &PySQPSolver<float, 1>::set_external_wrench_batch)
        .def("sim_forward", &PySQPSolver<float, 1>::sim_forward);
    py::class_<PySQPSolver<float, 16>>(m, "SQPSolverfloat_16")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 16>::solve)
        .def("reset", &PySQPSolver<float, 16>::reset)
        .def("resetRho", &PySQPSolver<float, 16>::resetRho)
        .def("resetLambda", &PySQPSolver<float, 16>::resetLambda)
        .def("set_external_wrench", &PySQPSolver<float, 16>::set_external_wrench)
        .def("set_external_wrench_batch", &PySQPSolver<float, 16>::set_external_wrench_batch)
        .def("sim_forward", &PySQPSolver<float, 16>::sim_forward);
    py::class_<PySQPSolver<float, 32>>(m, "SQPSolverfloat_32")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 32>::solve)
        .def("reset", &PySQPSolver<float, 32>::reset)
        .def("resetRho", &PySQPSolver<float, 32>::resetRho)
        .def("resetLambda", &PySQPSolver<float, 32>::resetLambda)
        .def("set_external_wrench", &PySQPSolver<float, 32>::set_external_wrench)
        .def("set_external_wrench_batch", &PySQPSolver<float, 32>::set_external_wrench_batch)
        .def("sim_forward", &PySQPSolver<float, 32>::sim_forward);
    py::class_<PySQPSolver<float, 64>>(m, "SQPSolverfloat_64")
        .def(py::init<>())
        .def("solve", &PySQPSolver<float, 64>::solve)
        .def("reset", &PySQPSolver<float, 64>::reset)
        .def("resetRho", &PySQPSolver<float, 64>::resetRho)
        .def("resetLambda", &PySQPSolver<float, 64>::resetLambda)
        .def("set_external_wrench", &PySQPSolver<float, 64>::set_external_wrench)
        .def("set_external_wrench_batch", &PySQPSolver<float, 64>::set_external_wrench_batch)
        .def("sim_forward", &PySQPSolver<float, 64>::sim_forward);
} 