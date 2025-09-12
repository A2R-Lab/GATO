#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <array>
#include <memory>
#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/utils.h"
#include "utils/cuda.cuh"
#include "bsqp/kernels/batch_utils.cuh"
#include "bsqp/force_estimator.hpp"

namespace py = pybind11;

template<typename T, uint32_t BatchSize>
class PyBSQP {
      public:
        PyBSQP() : solver_()
        {
                // printDeviceInfo();
                setL2PersistingAccess(1.0);
                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xu_single_, TRAJ_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xs_single_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_ref_single_, REFERENCE_TRAJ_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_curr_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_errors_, BatchSize * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * BatchSize);
                h_xu_best_.resize(TRAJ_SIZE);
                h_errors_.resize(BatchSize);
                // identity wrench transform (row-major)
                for (int i = 0; i < 36; ++i) wrench_transform_[i] = (i % 7 == 0) ? T(1) : T(0);

                // Force estimator is only meaningful when BatchSize > 3
                use_force_estimator_ = (BatchSize > 3);
                if (use_force_estimator_) {
                        estimator_ = std::make_unique<gato::ImprovedForceEstimator<T>>(BatchSize);
                }
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
               const T        q_lim_cost,
               const T        vel_lim_cost,
               const T        ctrl_lim_cost,
               const T        rho)
            : solver_(dt, max_sqp_iters, kkt_tol, max_pcg_iters, pcg_tol, solve_ratio, mu, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost, rho)
        {
                // printDeviceInfo();
                setL2PersistingAccess(1.0);
                std::cout << "T : " << typeid(T).name() << std::endl;

                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, STATE_SIZE * BatchSize * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xu_single_, TRAJ_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xs_single_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_ref_single_, REFERENCE_TRAJ_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_curr_, STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_errors_, BatchSize * sizeof(T)));

                h_xkp1_batch_.resize(STATE_SIZE * BatchSize);
                h_xu_best_.resize(TRAJ_SIZE);
                h_errors_.resize(BatchSize);
                // identity wrench transform (row-major)
                for (int i = 0; i < 36; ++i) wrench_transform_[i] = (i % 7 == 0) ? T(1) : T(0);

                // Force estimator is only meaningful when BatchSize > 3
                use_force_estimator_ = (BatchSize > 3);
                if (use_force_estimator_) {
                        estimator_ = std::make_unique<gato::ImprovedForceEstimator<T>>(BatchSize);
                }
        }

        ~PyBSQP()
        {
                gpuErrchk(cudaFree(d_xu_traj_batch_));
                gpuErrchk(cudaFree(d_x_s_batch_));
                gpuErrchk(cudaFree(d_reference_traj_batch_));
                gpuErrchk(cudaFree(d_xkp1_batch_));
                gpuErrchk(cudaFree(d_xk_));
                gpuErrchk(cudaFree(d_uk_));
                gpuErrchk(cudaFree(d_xu_single_));
                gpuErrchk(cudaFree(d_xs_single_));
                gpuErrchk(cudaFree(d_ref_single_));
                gpuErrchk(cudaFree(d_x_curr_));
                gpuErrchk(cudaFree(d_errors_));
        }

        // // Set 6x6 wrench transform (world -> joint local)
        // void set_wrench_transform(py::array_t<T> A6)
        // {
        //         py::buffer_info Abuf = A6.request();
        //         if (Abuf.ndim != 2 || Abuf.shape[0] != 6 || Abuf.shape[1] != 6)
        //                 throw std::runtime_error("wrench transform must be 6x6");
        //         std::memcpy(wrench_transform_.data(), Abuf.ptr, sizeof(T) * 36);
        // }

        py::dict solve(py::array_t<T> xu_traj_batch, T timestep, py::array_t<T> x_s_batch, py::array_t<T> reference_traj_batch)
        {
                py::buffer_info xu_buf = xu_traj_batch.request();
                py::buffer_info xs_buf = x_s_batch.request();
                py::buffer_info ref_buf = reference_traj_batch.request();

                // Update external force batch from estimator if enabled
                if (use_force_estimator_ && estimator_) {
                        auto batch = estimator_->generate_batch();
                        solver_.set_f_ext_batch(batch.data());
                }

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
                {
                        const ssize_t B = static_cast<ssize_t>(BatchSize);
                        const ssize_t L = static_cast<ssize_t>(TRAJ_SIZE);
                        py::array xu_arr = py::array(py::dtype::of<T>(),
                                                     {B, L},
                                                     {L * (ssize_t)sizeof(T), (ssize_t)sizeof(T)},
                                                     h_xu_traj.data(),
                                                     py::none());
                        result["XU"] = xu_arr;
                }
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

        // // Expose force estimator: generate batch in world frame (B x 6)
        // py::array fe_generate()
        // {
        //         auto batch = estimator_.generate_batch();
        //         const ssize_t B = static_cast<ssize_t>(BatchSize);
        //         const ssize_t cols = 6;
        //         // own a heap copy so lifetime is independent of the estimator buffer
        //         T* data = new T[B * cols];
        //         std::memcpy(data, batch.data(), sizeof(T) * B * cols);
        //         py::capsule owner(data, [](void* p){ delete[] reinterpret_cast<T*>(p); });
        //         return py::array(py::dtype::of<T>(),
        //                          {B, cols},
        //                          {cols * (ssize_t)sizeof(T), (ssize_t)sizeof(T)},
        //                          data,
        //                          owner);
        // }

        py::dict solve_single(py::array_t<T> xu_traj,
                              T timestep,
                              py::array_t<T> x_s,
                              py::array_t<T> reference_traj)
        {
                py::buffer_info xu_buf = xu_traj.request();
                py::buffer_info xs_buf = x_s.request();
                py::buffer_info ref_buf = reference_traj.request();

                if (xu_buf.size != TRAJ_SIZE) throw std::runtime_error("XU size mismatch");
                if (xs_buf.size != STATE_SIZE) throw std::runtime_error("x_s size mismatch");
                if (ref_buf.size != REFERENCE_TRAJ_SIZE) throw std::runtime_error("reference traj size mismatch");

                // Update external force batch from estimator if enabled
                if (use_force_estimator_ && estimator_) {
                        auto batch = estimator_->generate_batch();
                        solver_.set_f_ext_batch(batch.data());
                }

                // Copy singles to device
                gpuErrchk(cudaMemcpy(d_xu_single_, xu_buf.ptr, TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_xs_single_, xs_buf.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_ref_single_, ref_buf.ptr, REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                // Replicate on device
                gato::batch::replicateVectorToBatch<T, BatchSize>(d_xu_traj_batch_, d_xu_single_, TRAJ_SIZE);
                gato::batch::replicateVectorToBatch<T, BatchSize>(d_x_s_batch_, d_xs_single_, STATE_SIZE);
                gato::batch::replicateVectorToBatch<T, BatchSize>(d_reference_traj_batch_, d_ref_single_, REFERENCE_TRAJ_SIZE);
                cudaDeviceSynchronize();

                ProblemInputs<T, BatchSize> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                SQPStats<T, BatchSize> stats = solver_.solve(d_xu_traj_batch_, inputs);

                py::dict result;
                result["sqp_time_us"] = stats.solve_time_us;
                result["sqp_iters"] = py::array_t<int32_t>({BatchSize}, {sizeof(int32_t)}, stats.sqp_iterations.data());
                // result["kkt_converged"] = py::array_t<int32_t>({BatchSize}, {sizeof(int32_t)}, stats.kkt_converged.data());

                // std::vector<float> pcg_times_us;
                // std::vector<int>   pcg_iters;
                // pcg_times_us.reserve(BatchSize);
                // pcg_iters.reserve(BatchSize);
                // for (const auto& pcg_stat : stats.pcg_stats) {
                //         pcg_times_us.push_back(pcg_stat.solve_time_us);
                //         for (size_t i = 0; i < BatchSize; ++i) pcg_iters.push_back(pcg_stat.num_iterations[i]);
                // }
                // result["pcg_iters"] = py::array_t<int>(BatchSize, pcg_iters.data());
                // result["pcg_times_us"] = py::array_t<float>(BatchSize, pcg_times_us.data());

                // std::vector<float> ls_min_merit;
                // std::vector<float> ls_step_size;
                // ls_min_merit.reserve(BatchSize);
                // ls_step_size.reserve(BatchSize);
                // for (const auto& ls : stats.line_search_stats) {
                //         for (size_t i = 0; i < BatchSize; ++i) {
                //                 ls_min_merit.push_back(ls.min_merit[i]);
                //                 ls_step_size.push_back(ls.step_size[i]);
                //         }
                // }
                // result["ls_min_merit"] = py::array_t<float>(BatchSize, ls_min_merit.data());
                // result["ls_step_size"] = py::array_t<float>(BatchSize, ls_step_size.data());
                return result;
        }

        // Reuse existing GPU batch: x_s and reference updated
        py::dict solve_existing_with_ref(py::array_t<T> x_s, py::array_t<T> reference_traj, T timestep)
        {
                py::buffer_info xs_buf = x_s.request();
                py::buffer_info ref_buf = reference_traj.request();
                if (xs_buf.size != STATE_SIZE) throw std::runtime_error("x_s size mismatch");
                if (ref_buf.size != REFERENCE_TRAJ_SIZE) throw std::runtime_error("reference traj size mismatch");
                gpuErrchk(cudaMemcpy(d_xs_single_, xs_buf.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_ref_single_, ref_buf.ptr, REFERENCE_TRAJ_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                // Update external force batch from estimator if enabled
                if (use_force_estimator_ && estimator_) {
                        auto batch = estimator_->generate_batch();
                        solver_.set_f_ext_batch(batch.data());
                }

                gato::batch::replicateVectorToBatch<T, BatchSize>(d_x_s_batch_, d_xs_single_, STATE_SIZE);
                gato::batch::replicateVectorToBatch<T, BatchSize>(d_reference_traj_batch_, d_ref_single_, REFERENCE_TRAJ_SIZE);
                cudaDeviceSynchronize();

                ProblemInputs<T, BatchSize> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                SQPStats<T, BatchSize> stats = solver_.solve(d_xu_traj_batch_, inputs);
                py::dict result;
                result["sqp_time_us"] = stats.solve_time_us;
                result["sqp_iters"] = py::array_t<int32_t>({BatchSize}, {sizeof(int32_t)}, stats.sqp_iterations.data());
                // result["kkt_converged"] = py::array_t<int32_t>({BatchSize}, {sizeof(int32_t)}, stats.kkt_converged.data());
                return result;
        }

        py::array_t<T> get_XU() //just copy first trajectory to host
        {
                gpuErrchk(cudaMemcpy(h_xu_best_.data(), d_xu_traj_batch_, TRAJ_SIZE * sizeof(T), cudaMemcpyDeviceToHost));
                py::array_t<T> out_arr(TRAJ_SIZE);
                std::memcpy(out_arr.request().ptr, h_xu_best_.data(), sizeof(T) * TRAJ_SIZE);
                return out_arr;
        }

        // After solve: simulate one step for all hypotheses, pick best trajectory, copy it to all batches, return it
        py::array_t<T> post_solve_select_and_copy(py::array_t<T> x_last,
                                                  py::array_t<T> u_last,
                                                  py::array_t<T> x_curr,
                                                  T dt)
        {
                py::buffer_info xL = x_last.request();
                py::buffer_info uL = u_last.request();
                py::buffer_info xC = x_curr.request();
                if (xL.size != STATE_SIZE || xC.size != STATE_SIZE || uL.size != CONTROL_SIZE)
                        throw std::runtime_error("State/control size mismatch in post_solve_select_and_copy");

                gpuErrchk(cudaMemcpy(d_xk_, xL.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_uk_, uL.ptr, CONTROL_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_x_curr_, xC.ptr, STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                // x_{k+1} predictions for each batch hypothesis
                solver_.sim_forward(d_xkp1_batch_, d_xk_, d_uk_, dt);
                cudaDeviceSynchronize();
                // compute errors vs measured x_curr
                gato::batch::computeSquaredErrorsBatched<T, BatchSize>(d_xkp1_batch_, d_x_curr_, d_errors_);
                gpuErrchk(cudaMemcpy(h_errors_.data(), d_errors_, BatchSize * sizeof(T), cudaMemcpyDeviceToHost));
                cudaDeviceSynchronize();
                uint32_t best_id = 0; 
                T best_val = h_errors_[0];
                for (uint32_t i = 1; i < BatchSize; ++i) {
                        if (h_errors_[i] < best_val) { best_val = h_errors_[i]; best_id = i; }
                }

                if (use_force_estimator_ && estimator_) {
                        estimator_->update(best_id, h_errors_);
                }

                // copy best trajectory back to host
                gpuErrchk(cudaMemcpy(h_xu_best_.data(), d_xu_traj_batch_ + static_cast<size_t>(best_id) * TRAJ_SIZE,
                                     TRAJ_SIZE * sizeof(T), cudaMemcpyDeviceToHost));
                // replicate best over entire batch on device for warm-starting next iteration
                gato::batch::copyBestTrajectoryToAll<T, BatchSize>(d_xu_traj_batch_, d_xu_traj_batch_ + static_cast<size_t>(best_id) * TRAJ_SIZE, TRAJ_SIZE);
                cudaDeviceSynchronize();

                py::array_t<T> out_arr(TRAJ_SIZE);
                std::memcpy(out_arr.request().ptr, h_xu_best_.data(), sizeof(T) * TRAJ_SIZE);
                return out_arr;
        }

        void set_f_ext_batch(py::array_t<T> f_ext_batch)
        {
                py::buffer_info f_ext_buf = f_ext_batch.request();
                solver_.set_f_ext_batch(static_cast<T*>(f_ext_buf.ptr));
        }

        void set_use_force_estimator(bool enabled)
        {
                // Only allow enabling when BatchSize > 3
                if (enabled && BatchSize <= 3) {
                        use_force_estimator_ = false;
                        estimator_.reset();
                        return;
                }
                use_force_estimator_ = enabled;
                if (use_force_estimator_ && !estimator_) {
                        estimator_ = std::make_unique<gato::ImprovedForceEstimator<T>>(BatchSize);
                } else if (!use_force_estimator_ && estimator_) {
                        estimator_.reset();
                }
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

                {
                        const ssize_t B = static_cast<ssize_t>(BatchSize);
                        const ssize_t L = static_cast<ssize_t>(STATE_SIZE);
                        return py::array(py::dtype::of<T>(),
                                         {B, L},
                                         {L * (ssize_t)sizeof(T), (ssize_t)sizeof(T)},
                                         h_xkp1_batch_.data(),
                                         py::none());
                }
        }

        void reset_dual() { solver_.reset_dual(); }
        void reset_rho() { solver_.reset_rho(); }

      private:
        BSQP<T, BatchSize> solver_;
        std::unique_ptr<gato::ImprovedForceEstimator<T>> estimator_;
        bool use_force_estimator_ = false;
        std::array<T,36> wrench_transform_;
        T*                 d_xu_traj_batch_;
        T*                 d_x_s_batch_;
        T*                 d_reference_traj_batch_;

        // for sim_forward
        T *            d_xkp1_batch_, *d_xk_, *d_uk_;
        // single inputs and helpers
        T *            d_xu_single_, *d_xs_single_, *d_ref_single_;
        T *            d_x_curr_, *d_errors_;
        std::vector<T> h_xkp1_batch_;
        std::vector<T> h_xu_best_;
        std::vector<T> h_errors_;
        std::vector<T> h_fe_batch_;
};


#define MODULE_NAME_HELPER(knot) bsqpN##knot
#define MODULE_NAME(knot) MODULE_NAME_HELPER(knot)

// Macro to register a PyBSQP class with the given precision type and batch size
#define REGISTER_BSQP_CLASS(Type, BatchSize)                                                                                                                                                     \
        py::class_<PyBSQP<Type, BatchSize>>(m, "BSQP_" #BatchSize "_" #Type)                                                                                                                     \
            .def(py::init<const Type, const uint32_t, const Type, const uint32_t, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type>())\
            .def("solve_single", &PyBSQP<Type, BatchSize>::solve_single)                                                                                                                         \
            .def("solve", &PyBSQP<Type, BatchSize>::solve)                                                                                                                         \
            .def("solve_existing_with_ref", &PyBSQP<Type, BatchSize>::solve_existing_with_ref)                                                                                                   \
            .def("reset_dual", &PyBSQP<Type, BatchSize>::reset_dual)                                                                                                                             \
            .def("set_f_ext_batch", &PyBSQP<Type, BatchSize>::set_f_ext_batch)                                                                                                                   \
            .def("set_use_force_estimator", &PyBSQP<Type, BatchSize>::set_use_force_estimator)                                                                                                   \
            .def("sim_forward", &PyBSQP<Type, BatchSize>::sim_forward)                                                                                                                           \
            .def("post_solve_select_and_copy", &PyBSQP<Type, BatchSize>::post_solve_select_and_copy)                                                                                             \
            .def("reset_rho", &PyBSQP<Type, BatchSize>::reset_rho)       \
            .def("get_XU", &PyBSQP<Type, BatchSize>::get_XU)

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
