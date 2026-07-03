#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/cuda.cuh"

namespace py = pybind11;

// batch_size is a runtime constructor argument (the solver kernels only use it
// as the outermost grid dimension), so ONE binding class covers every batch size.
template<typename T>
class PyBSQP {
      public:
        PyBSQP(const uint32_t batch_size,
               const T        dt,
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
               const T        q_lim_cost,
               const T        vel_lim_cost,
               const T        ctrl_lim_cost,
               const T        rho)
            : batch_size_(batch_size),
              solver_(batch_size, dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, solve_ratio, mu, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost, rho)
        {
                setL2PersistingAccess(1.0);

                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));

                // pinned staging: numpy -> pinned -> device beats numpy(pageable) -> device
                gpuErrchk(cudaMallocHost(&h_xu_staging_, TRAJ_SIZE * batch_size_ * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * batch_size_);
        }

        ~PyBSQP()
        {
                gpuErrchk(cudaFreeHost(h_xu_staging_));
                gpuErrchk(cudaFree(d_xu_traj_batch_));
                gpuErrchk(cudaFree(d_x_s_batch_));
                gpuErrchk(cudaFree(d_reference_traj_batch_));
                gpuErrchk(cudaFree(d_xkp1_batch_));
                gpuErrchk(cudaFree(d_xk_));
                gpuErrchk(cudaFree(d_uk_));
        }

        // shape/size validation: a wrong-width numpy array would otherwise silently
        // over-read host memory in the cudaMemcpy below
        void check_size(const py::buffer_info& buf, size_t expected, const char* name)
        {
                if (static_cast<size_t>(buf.size) != expected) {
                        throw py::value_error(std::string(name) + ": expected " + std::to_string(expected) + " elements (batch_size=" + std::to_string(batch_size_) + "), got "
                                              + std::to_string(buf.size));
                }
        }

        py::dict solve(py::array_t<T> xu_traj_batch, T timestep, py::array_t<T> x_s_batch, py::array_t<T> reference_traj_batch)
        {
                py::buffer_info xu_buf = xu_traj_batch.request();
                py::buffer_info xs_buf = x_s_batch.request();
                py::buffer_info ref_buf = reference_traj_batch.request();
                check_size(xu_buf, (size_t)TRAJ_SIZE * batch_size_, "xu_traj_batch");
                check_size(xs_buf, (size_t)STATE_SIZE * batch_size_, "x_s_batch");
                check_size(ref_buf, (size_t)REFERENCE_TRAJ_SIZE * batch_size_, "reference_traj_batch");

                memcpy(h_xu_staging_, xu_buf.ptr, TRAJ_SIZE * batch_size_ * sizeof(T));
                gpuErrchk(cudaMemcpy(d_xu_traj_batch_, h_xu_staging_, TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_x_s_batch_, xs_buf.ptr, STATE_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_reference_traj_batch_, ref_buf.ptr, REFERENCE_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));

                ProblemInputs<T> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                // Solve
                SQPStats<T> stats = solver_.solve(d_xu_traj_batch_, inputs);

                // Copy trajectory back: device -> pinned -> straight into the output
                // py::array (no intermediate std::vector + second copy)
                const py::ssize_t Bs = static_cast<py::ssize_t>(batch_size_);
                py::array_t<T>    xu_out({Bs, (py::ssize_t)TRAJ_SIZE});
                gpuErrchk(cudaMemcpy(h_xu_staging_, d_xu_traj_batch_, TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
                memcpy(xu_out.request().ptr, h_xu_staging_, TRAJ_SIZE * batch_size_ * sizeof(T));

                // Copy final merits (computed at end of solve) and initial (pre-iteration) merits back to host
                std::vector<T> h_final_merit(batch_size_);
                std::vector<T> h_initial_merit(batch_size_);
                solver_.copy_final_merit_to_host(h_final_merit.data());
                solver_.copy_initial_merit0_to_host(h_initial_merit.data());

                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);

                py::dict result;
                result["XU"] = xu_out;
                result["sqp_time_us"] = stats.solve_time_us;
                result["sqp_iters"] = py::array_t<int32_t>({B}, {sizeof(int32_t)}, stats.sqp_iterations.data());
                result["kkt_converged"] = py::array_t<int32_t>({B}, {sizeof(int32_t)}, stats.kkt_converged.data());
                result["final_merit"] = py::array_t<T>({B}, {sizeof(T)}, h_final_merit.data());
                result["initial_merit"] = py::array_t<T>({B}, {sizeof(T)}, h_initial_merit.data());

                // Per-iteration stats: shape them as (iters, batch_size)
                const size_t num_iters = stats.line_search_stats.size();
                result["ls_num_iters"] = static_cast<int>(num_iters);

                std::vector<float> pcg_times_us;
                std::vector<int>   pcg_iters;
                pcg_times_us.reserve(num_iters);
                pcg_iters.reserve(num_iters * batch_size_);
                for (const auto& pcg_stat : stats.pcg_stats) {
                        pcg_times_us.push_back(pcg_stat.solve_time_us);
                        for (size_t i = 0; i < batch_size_; ++i) { pcg_iters.push_back(pcg_stat.num_iterations[i]); }
                }
                {
                        std::vector<py::ssize_t> sh_times = {static_cast<py::ssize_t>(stats.pcg_stats.size())};
                        result["pcg_times_us"] = py::array_t<float>(sh_times, pcg_times_us.data());
                }
                {
                        std::vector<py::ssize_t> sh_iters = {static_cast<py::ssize_t>(stats.pcg_stats.size()), B};
                        result["pcg_iters"] = py::array_t<int>(sh_iters, pcg_iters.data());
                }

                std::vector<float> ls_min_merit;
                std::vector<float> ls_step_size;
                ls_min_merit.reserve(num_iters * batch_size_);
                ls_step_size.reserve(num_iters * batch_size_);
                for (const auto& line_search_stat : stats.line_search_stats) {
                        for (size_t i = 0; i < batch_size_; ++i) {
                                ls_min_merit.push_back(line_search_stat.min_merit[i]);
                                ls_step_size.push_back(line_search_stat.step_size[i]);
                        }
                }
                {
                        std::vector<py::ssize_t> sh = {static_cast<py::ssize_t>(num_iters), B};
                        result["ls_min_merit"] = py::array_t<float>(sh, ls_min_merit.data());
                        result["ls_step_size"] = py::array_t<float>(sh, ls_step_size.data());
                }

                return py::dict(result);
        }

        void set_f_ext_batch(py::array_t<T> f_ext_batch)
        {
                py::buffer_info f_ext_buf = f_ext_batch.request();
                check_size(f_ext_buf, (size_t)6 * grid::NUM_BODIES * batch_size_, "f_ext_batch");
                solver_.set_f_ext_batch(static_cast<T*>(f_ext_buf.ptr));
        }

        void set_rho_penalty_batch(py::array_t<T> rho_batch, bool set_as_reset_default = true)
        {
                py::buffer_info buf = rho_batch.request();
                check_size(buf, batch_size_, "rho_batch");
                solver_.set_rho_penalty_batch(static_cast<T*>(buf.ptr), set_as_reset_default);
        }

        void set_drho_batch(py::array_t<T> drho_batch, bool set_as_reset_default = true)
        {
                py::buffer_info buf = drho_batch.request();
                check_size(buf, batch_size_, "drho_batch");
                solver_.set_drho_batch(static_cast<T*>(buf.ptr), set_as_reset_default);
        }

        void set_mu_batch(py::array_t<T> mu_batch)
        {
                py::buffer_info buf = mu_batch.request();
                check_size(buf, batch_size_, "mu_batch");
                solver_.set_mu_batch(static_cast<T*>(buf.ptr));
        }

        void set_pcg_tol_batch(py::array_t<T> eps_batch)
        {
                py::buffer_info buf = eps_batch.request();
                check_size(buf, batch_size_, "pcg_tol_batch");
                solver_.set_pcg_tol_batch(static_cast<T*>(buf.ptr));
        }

        py::array_t<T> sim_forward(py::array_t<T> xk, py::array_t<T> uk, T dt)
        {
                py::buffer_info xk_buf = xk.request();
                py::buffer_info uk_buf = uk.request();
                if (static_cast<size_t>(xk_buf.size) != STATE_SIZE) { throw py::value_error("xk: expected " + std::to_string(STATE_SIZE) + " elements, got " + std::to_string(xk_buf.size)); }
                if (static_cast<size_t>(uk_buf.size) != CONTROL_SIZE) { throw py::value_error("uk: expected " + std::to_string(CONTROL_SIZE) + " elements, got " + std::to_string(uk_buf.size)); }

                gpuErrchk(cudaMemcpy(d_xk_, xk_buf.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_uk_, uk_buf.ptr, CONTROL_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                solver_.sim_forward(d_xkp1_batch_, d_xk_, d_uk_, dt);
                gpuErrchk(cudaDeviceSynchronize());

                gpuErrchk(cudaMemcpy(h_xkp1_batch_.data(), d_xkp1_batch_, STATE_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));

                return py::array_t<T>({static_cast<py::ssize_t>(batch_size_), (py::ssize_t)STATE_SIZE}, h_xkp1_batch_.data());
        }

        void reset_dual() { solver_.reset_dual(); }
        void reset_rho() { solver_.reset_rho(); }
        void set_rho_adaptation(bool enabled) { solver_.set_rho_adaptation(enabled); }
        void set_collect_stats(bool enabled) { solver_.set_collect_stats(enabled); }

        void set_cost_weights(T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
        {
                solver_.set_cost_weights(q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
        }

        void set_cost_weights_per_knot(py::array_t<T> knot_weights)
        {
                py::buffer_info buf = knot_weights.request();
                if (static_cast<size_t>(buf.size) != 3 * KNOT_POINTS) {
                        throw py::value_error("knot_weights: expected " + std::to_string(3 * KNOT_POINTS) + " elements ((KNOT_POINTS, 3) [ee, qd, u] triples), got " + std::to_string(buf.size));
                }
                solver_.set_cost_weights_per_knot(static_cast<T*>(buf.ptr));
        }
        void clear_cost_weights_per_knot() { solver_.clear_cost_weights_per_knot(); }

      private:
        uint32_t       batch_size_;
        BSQP<T>        solver_;
        T*             h_xu_staging_;
        T*             d_xu_traj_batch_;
        T*             d_x_s_batch_;
        T*             d_reference_traj_batch_;

        // for sim_forward
        T *            d_xkp1_batch_, *d_xk_, *d_uk_;
        std::vector<T> h_xkp1_batch_;
};


// the plant name token is injected at compile time (-DGATO_PLANT_NAME=<name>)
#ifndef GATO_PLANT_NAME
#error "GATO_PLANT_NAME must be defined (plant name token for the module name)"
#endif

#define MODULE_NAME_HELPER(knot, plant) bsqpN##knot##_##plant
#define MODULE_NAME(knot, plant) MODULE_NAME_HELPER(knot, plant)

// Register the runtime-batch-size PyBSQP class for the given precision type
#define REGISTER_BSQP_CLASS(Type)                                                                                                                                                                  \
        py::class_<PyBSQP<Type>>(m, "BSQP_" #Type)                                                                                                                                                 \
            .def(py::init<const uint32_t, const Type, const uint32_t, const Type, const uint32_t, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, \
                          const Type, const Type, const Type>())                                                                                                                                    \
            .def("solve", &PyBSQP<Type>::solve)                                                                                                                                                    \
            .def("reset_dual", &PyBSQP<Type>::reset_dual)                                                                                                                                          \
            .def("set_f_ext_batch", &PyBSQP<Type>::set_f_ext_batch)                                                                                                                                \
            .def("set_rho_penalty_batch", &PyBSQP<Type>::set_rho_penalty_batch, py::arg("rho_batch"), py::arg("set_as_reset_default") = true)                                                      \
            .def("set_drho_batch", &PyBSQP<Type>::set_drho_batch, py::arg("drho_batch"), py::arg("set_as_reset_default") = true)                                                                   \
            .def("set_mu_batch", &PyBSQP<Type>::set_mu_batch)                                                                                                                                      \
            .def("set_pcg_tol_batch", &PyBSQP<Type>::set_pcg_tol_batch)                                                                                                                            \
            .def("sim_forward", &PyBSQP<Type>::sim_forward)                                                                                                                                        \
            .def("reset_rho", &PyBSQP<Type>::reset_rho)                                                                                                                                            \
            .def("set_rho_adaptation", &PyBSQP<Type>::set_rho_adaptation)                                                                                                                              \
            .def("set_collect_stats", &PyBSQP<Type>::set_collect_stats)                                                                                                                                \
            .def("set_cost_weights", &PyBSQP<Type>::set_cost_weights)                                                                                                                                  \
            .def("set_cost_weights_per_knot", &PyBSQP<Type>::set_cost_weights_per_knot)                                                                                                                \
            .def("clear_cost_weights_per_knot", &PyBSQP<Type>::clear_cost_weights_per_knot)

PYBIND11_MODULE(MODULE_NAME(KNOT_POINTS, GATO_PLANT_NAME), m)
{
        m.attr("KNOT_POINTS") = KNOT_POINTS;      // to check num knots for current module
        m.attr("NUM_BODIES") = grid::NUM_BODIES;  // body-major f_ext is 6*NUM_BODIES per solve

#ifdef USE_DOUBLES
        REGISTER_BSQP_CLASS(double);
#else
        REGISTER_BSQP_CLASS(float);
#endif
}
