#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/utils.h"
#include "utils/cuda.cuh"

namespace py = pybind11;

template<typename T, uint32_t BatchSize>
class PyBSQP {
      public:
        PyBSQP() : solver_()
        {
                printDeviceInfo();
                setL2PersistingAccess(1.0);
                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * BatchSize);
        }

        PyBSQP(const T        dt,
               const uint32_t max_sqp_iters,
               const T        kkt_tol,
               const uint32_t max_pcg_iters,
               const T        pcg_tol,
               const T        solve_ratio,
               const T        mu,
               const T        q_cost,
               const T        qd_cost,
               const T        u_cost,
               const T        N_cost,
               const T        q_lim_cost)
            : solver_(dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, solve_ratio, mu, q_cost, qd_cost, u_cost, N_cost, q_lim_cost)
        {
                printDeviceInfo();
                setL2PersistingAccess(1.0);
                std::cout << "T : " << typeid(T).name() << std::endl;

                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * BatchSize);
        }

        ~PyBSQP()
        {
                gpuErrchk(cudaFree(d_xu_traj_batch_));
                gpuErrchk(cudaFree(d_x_s_batch_));
                gpuErrchk(cudaFree(d_reference_traj_batch_));
                gpuErrchk(cudaFree(d_xkp1_batch_));
                gpuErrchk(cudaFree(d_xk_));
                gpuErrchk(cudaFree(d_uk_));
        }

        py::dict solve(py::array_t<T> xu_traj_batch, T timestep, py::array_t<T> x_s_batch, py::array_t<T> reference_traj_batch)
        {
                py::buffer_info xu_buf = xu_traj_batch.request();
                py::buffer_info xs_buf = x_s_batch.request();
                py::buffer_info ref_buf = reference_traj_batch.request();

                gpuErrchk(cudaMemcpy(d_xu_traj_batch_, xu_buf.ptr, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_x_s_batch_, xs_buf.ptr, STATE_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_reference_traj_batch_, ref_buf.ptr, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyHostToDevice));

                ProblemInputs<T, BatchSize> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                // Solve
                SQPStats<T, BatchSize> stats = solver_.solve(d_xu_traj_batch_, inputs);

                // Copy trajectory back to host
                std::vector<T> h_xu_traj(TRAJ_SIZE * BatchSize);
                gpuErrchk(cudaMemcpy(h_xu_traj.data(), d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

                py::dict result;
                result["XU"] = py::array_t<T>({BatchSize, TRAJ_SIZE}, h_xu_traj.data());
                result["sqp_time_us"] = stats.solve_time_us;
                result["sqp_iters"] = py::array_t<int32_t>({BatchSize},                 // shape
                                                           {sizeof(int32_t)},           // stride
                                                           stats.sqp_iterations.data()  // data
                );
                result["kkt_converged"] = py::array_t<int32_t>({BatchSize},                // shape
                                                               {sizeof(int32_t)},          // stride
                                                               stats.kkt_converged.data()  // data
                );

                std::vector<float> pcg_times_us;
                std::vector<int>   pcg_iters;
                pcg_times_us.reserve(BatchSize);
                pcg_iters.reserve(BatchSize);
                for (const auto& pcg_stat : stats.pcg_stats) {
                        pcg_times_us.push_back(pcg_stat.solve_time_us);
                        for (size_t i = 0; i < BatchSize; ++i) { pcg_iters.push_back(pcg_stat.num_iterations[i]); }
                }
                result["pcg_iters"] = py::array_t<int>(BatchSize, pcg_iters.data());
                result["pcg_times_us"] = py::array_t<float>(BatchSize, pcg_times_us.data());

                std::vector<float> ls_min_merit;
                std::vector<float> ls_step_size;
                ls_min_merit.reserve(BatchSize);
                ls_step_size.reserve(BatchSize);
                for (const auto& line_search_stat : stats.line_search_stats) {
                        for (size_t i = 0; i < BatchSize; ++i) {
                                ls_min_merit.push_back(line_search_stat.min_merit[i]);
                                ls_step_size.push_back(line_search_stat.step_size[i]);
                        }
                }
                result["ls_min_merit"] = py::array_t<float>(BatchSize, ls_min_merit.data());
                result["ls_step_size"] = py::array_t<float>(BatchSize, ls_step_size.data());

                return py::dict(result);
        }

        void set_f_ext_batch(py::array_t<T> f_ext_batch)
        {
                py::buffer_info f_ext_buf = f_ext_batch.request();
                solver_.set_f_ext_batch(static_cast<T*>(f_ext_buf.ptr));
        }

        py::array_t<T> sim_forward(py::array_t<T> xk, py::array_t<T> uk, T dt)
        {
                py::buffer_info xk_buf = xk.request();
                py::buffer_info uk_buf = uk.request();

                gpuErrchk(cudaMemcpy(d_xk_, xk_buf.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_uk_, uk_buf.ptr, CONTROL_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                solver_.sim_forward(d_xkp1_batch_, d_xk_, d_uk_, dt);
                gpuErrchk(cudaDeviceSynchronize());

                gpuErrchk(cudaMemcpy(h_xkp1_batch_.data(), d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T), cudaMemcpyDeviceToHost));

                return py::array_t<T>({BatchSize, STATE_SIZE}, h_xkp1_batch_.data());
        }

        void reset_dual() { solver_.reset_dual(); }

      private:
        BSQP<T, BatchSize> solver_;
        T*                 d_xu_traj_batch_;
        T*                 d_x_s_batch_;
        T*                 d_reference_traj_batch_;

        // for sim_forward
        T *            d_xkp1_batch_, *d_xk_, *d_uk_;
        std::vector<T> h_xkp1_batch_;
};


#define MODULE_NAME_HELPER(knot) bsqpN##knot
#define MODULE_NAME(knot) MODULE_NAME_HELPER(knot)

// Macro to register a PyBSQP class with the given precision type and batch size
#define REGISTER_BSQP_CLASS(Type, BatchSize)                                                                                                                                         \
        py::class_<PyBSQP<Type, BatchSize>>(m, "BSQP_" #BatchSize "_" #Type)                                                                                                         \
            .def(py::init<const Type, const uint32_t, const Type, const uint32_t, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type>()) \
            .def("solve", &PyBSQP<Type, BatchSize>::solve)                                                                                                                           \
            .def("reset_dual", &PyBSQP<Type, BatchSize>::reset_dual)                                                                                                                 \
            .def("set_f_ext_batch", &PyBSQP<Type, BatchSize>::set_f_ext_batch)                                                                                                       \
            .def("sim_forward", &PyBSQP<Type, BatchSize>::sim_forward)

PYBIND11_MODULE(MODULE_NAME(KNOT_POINTS), m)
{
        m.attr("KNOT_POINTS") = KNOT_POINTS;  // to check num knots for current module


#ifdef USE_DOUBLES
        REGISTER_BSQP_CLASS(double, 1);
        REGISTER_BSQP_CLASS(double, 2);
        REGISTER_BSQP_CLASS(double, 4);
        REGISTER_BSQP_CLASS(double, 8);
        REGISTER_BSQP_CLASS(double, 16);
        REGISTER_BSQP_CLASS(double, 32);
        REGISTER_BSQP_CLASS(double, 64);
        REGISTER_BSQP_CLASS(double, 128);
#else
        REGISTER_BSQP_CLASS(float, 1);
        REGISTER_BSQP_CLASS(float, 2);
        REGISTER_BSQP_CLASS(float, 4);
        REGISTER_BSQP_CLASS(float, 8);
        REGISTER_BSQP_CLASS(float, 16);
        REGISTER_BSQP_CLASS(float, 32);
        REGISTER_BSQP_CLASS(float, 64);
        REGISTER_BSQP_CLASS(float, 128);
#endif
}
