#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bsqp/bsqp.cuh"
#include "dynamics/manifold.cuh"
#include "types.cuh"
#include "utils/cuda.cuh"
#include "utils/dlpack.h"

#include <memory>

namespace py = pybind11;

// A device-resident float array handed to the caller through the DLPack
// protocol (torch.from_dlpack / jax.dlpack.from_dlpack): zero-copy outputs of
// the device-pointer solve path. The memory lives as long as any consumer holds it.
struct DeviceArray {
        std::shared_ptr<void>  mem;
        std::vector<int64_t>   shape;
        int                    device = 0;

        static DeviceArray alloc(std::vector<int64_t> shape)
        {
                size_t n = 1;
                for (auto d : shape) n *= static_cast<size_t>(d);
                void* p = nullptr;
                gpuErrchk(cudaMalloc(&p, n * sizeof(float)));
                DeviceArray a;
                a.mem = std::shared_ptr<void>(p, [](void* q) { cudaFree(q); });
                a.shape = std::move(shape);
                gpuErrchk(cudaGetDevice(&a.device));
                return a;
        }
        uintptr_t data_ptr() const { return reinterpret_cast<uintptr_t>(mem.get()); }
        size_t    numel() const { size_t n = 1; for (auto d : shape) n *= static_cast<size_t>(d); return n; }

        py::capsule dlpack() const
        {
                struct Holder {
                        std::shared_ptr<void> mem;
                        std::vector<int64_t>  shape, strides;
                        DLManagedTensor       tensor;
                };
                auto* h = new Holder{mem, shape, {}, {}};
                h->strides.resize(shape.size());
                int64_t s = 1;
                for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) { h->strides[i] = s; s *= shape[i]; }
                h->tensor.dl_tensor.data = mem.get();
                h->tensor.dl_tensor.device = DLDevice{kDLCUDA, device};
                h->tensor.dl_tensor.ndim = static_cast<int32_t>(shape.size());
                h->tensor.dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
                h->tensor.dl_tensor.shape = h->shape.data();
                h->tensor.dl_tensor.strides = h->strides.data();
                h->tensor.dl_tensor.byte_offset = 0;
                h->tensor.manager_ctx = h;
                h->tensor.deleter = [](DLManagedTensor* t) { delete static_cast<Holder*>(t->manager_ctx); };
                return py::capsule(&h->tensor, "dltensor", [](PyObject* cap) {
                        if (PyCapsule_IsValid(cap, "dltensor")) {   // never consumed: release here
                                auto* t = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
                                t->deleter(t);
                        }
                });
        }
        py::tuple dlpack_device() const { return py::make_tuple(static_cast<int>(kDLCUDA), device); }
        py::array_t<float> to_numpy() const
        {
                std::vector<py::ssize_t> sh(shape.begin(), shape.end());
                py::array_t<float> out(sh);
                gpuErrchk(cudaMemcpy(out.request().ptr, mem.get(), numel() * sizeof(float), cudaMemcpyDeviceToHost));
                return out;
        }
};

// hold-at-x cold start written directly on the device: every knot's state = x_b, controls 0
template<typename T>
__global__ void fill_hold_warm_start_kernel(T* xu, const T* x_s, uint32_t knots, uint32_t nx, uint32_t nu)
{
        const uint32_t stride = nx + nu;
        T* row = xu + blockIdx.x * (knots * stride - nu);
        const T* x = x_s + blockIdx.x * nx;
        for (uint32_t i = threadIdx.x; i < knots * stride - nu; i += blockDim.x) {
                const uint32_t k = i / stride, j = i - k * stride;
                row[i] = (j < nx) ? x[j] : T(0);
        }
}

// ---- device-resident warm start (MPCController's device path) ----------------
// one source row broadcast to every batch row
template<typename T>
__global__ void tile_rows_kernel(T* dst, const T* src, uint32_t row_len)
{
        T* row = dst + blockIdx.x * row_len;
        for (uint32_t i = threadIdx.x; i < row_len; i += blockDim.x) row[i] = src[i];
}

// the controller's warm start for the next tick, written to every batch row from
// the winning row: shift = numpy's concatenate([xu[stride:], xu[-stride:]]) (the
// plan advanced one knot, the last knot repeated), hold = the plan as solved
template<typename T>
__global__ void shift_rows_kernel(T* xu, const T* src_row, uint32_t row_len, uint32_t stride, bool hold)
{
        T* row = xu + blockIdx.x * row_len;
        for (uint32_t i = threadIdx.x; i < row_len; i += blockDim.x) {
                row[i] = (hold || i + stride >= row_len) ? src_row[i] : src_row[i + stride];
        }
}

// ||x - xu0|| for the controller's warm-startedness signal: knot 0 of the resident
// plan (the model's one-step-ahead prediction after last tick's shift) against the
// measured state. Floating base with a unit base quaternion: the tangent norm
// ||x ⊟ x_pred|| (the host path's rule); otherwise the Euclidean norm on the stored
// layout. ONE block; the result is a single T.
template<typename T>
__global__ void pred_err_kernel(T* out, const T* xu0, const T* x)
{
        using namespace gato::constants;
        constexpr uint32_t NMAX = (STATE_SIZE > XU_STATE_SIZE) ? STATE_SIZE : XU_STATE_SIZE;
        __shared__ T s_diff[NMAX];
        __shared__ T s_red[256];
        uint32_t n = XU_STATE_SIZE;
        bool tangent = false;
        if constexpr (FLOATING_BASE) {
                const T qn = sqrt(xu0[3] * xu0[3] + xu0[4] * xu0[4] + xu0[5] * xu0[5] + xu0[6] * xu0[6]);
                tangent = fabs(qn - static_cast<T>(1)) < static_cast<T>(1e-3);
        }
        if (tangent) {
                gato::plant::state_difference<T>(s_diff, xu0, x);   // x ⊟ x_pred
                n = STATE_SIZE;
        } else {
                for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) s_diff[i] = x[i] - xu0[i];
        }
        __syncthreads();
        T acc = static_cast<T>(0);
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) acc += s_diff[i] * s_diff[i];
        s_red[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
                __syncthreads();
        }
        if (threadIdx.x == 0) *out = sqrt(s_red[0]);
}

// batch_size is a runtime constructor argument (the solver kernels only use it
// as the outermost grid dimension), so ONE binding class covers every batch size.
template<typename T>
class PyBSQP {
        // c_style|forcecast: the host consumes .ptr as a dense buffer, so pybind MUST
        // deliver a contiguous copy for strided inputs (a 0-stride np.broadcast_to
        // view passed as lo/hi read 1 valid element + heap garbage as bounds —
        // per-process-constant corruption, found 2026-08-02 via the fc-pin flake)
        using DenseArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

      public:
        PyBSQP(const uint32_t batch_size,
               const uint32_t max_sqp_iters,
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
              solver_(batch_size, max_sqp_iters, max_pcg_iters, pcg_tol, solve_ratio, mu, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost, rho)
        {
                gpuErrchk(cudaMalloc(&d_xu_traj_batch_, XU_TRAJ_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_x_s_batch_, XU_STATE_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_reference_traj_batch_, REFERENCE_TRAJ_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xkp1_batch_, XU_STATE_SIZE * batch_size_ * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xk_, XU_STATE_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_uk_, CONTROL_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_xu_row_, XU_TRAJ_SIZE * sizeof(T)));
                gpuErrchk(cudaMalloc(&d_scalar_, sizeof(T)));

                // pinned staging: numpy -> pinned -> device beats numpy(pageable) -> device
                gpuErrchk(cudaMallocHost(&h_xu_staging_, XU_TRAJ_SIZE * batch_size_ * sizeof(T)));

                h_xkp1_batch_.resize(XU_STATE_SIZE * batch_size_);
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
                gpuErrchk(cudaFree(d_xu_row_));
                gpuErrchk(cudaFree(d_scalar_));
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
        // the host pointer of a dense array holding exactly `expected` elements
        T* checked_ptr(const DenseArray& a, size_t expected, const char* name)
        {
                py::buffer_info buf = a.request();
                check_size(buf, expected, name);
                return static_cast<T*>(buf.ptr);
        }
        // optional vector: empty -> nullptr (reset to the solver default), else exactly `expected` entries
        T* optional_ptr(const DenseArray& a, size_t expected, const char* name)
        {
                py::buffer_info buf = a.request();
                if (buf.size == 0) return nullptr;
                if (static_cast<size_t>(buf.size) != expected) { throw std::runtime_error(std::string(name) + " must have " + std::to_string(expected) + " entries (or be empty to reset)"); }
                return static_cast<T*>(buf.ptr);
        }

        py::dict solve(DenseArray xu_traj_batch, T timestep, DenseArray x_s_batch, DenseArray reference_traj_batch)
        {
                auto inputs = upload_problem(xu_traj_batch, timestep, x_s_batch, reference_traj_batch);

                // Solve
                SQPStats<T> stats = solver_.solve(d_xu_traj_batch_, inputs);

                // Copy trajectory back: device -> pinned -> straight into the output
                // py::array (no intermediate std::vector + second copy)
                const py::ssize_t Bs = static_cast<py::ssize_t>(batch_size_);
                py::array_t<T>    xu_out({Bs, (py::ssize_t)XU_TRAJ_SIZE});
                gpuErrchk(cudaMemcpy(h_xu_staging_, d_xu_traj_batch_, XU_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
                memcpy(xu_out.request().ptr, h_xu_staging_, XU_TRAJ_SIZE * batch_size_ * sizeof(T));

                py::dict result = collect_stats(stats);
                result["XU"] = xu_out;
                return result;
        }

        // The device-side solve both device paths end in: the staged x_s / reference
        // and the trajectory buffer are complete, solve, hand the solution out as a
        // DeviceArray copy (the buffer itself keeps the solution for solve_resident).
        py::dict solve_staged_to_device(T timestep)
        {
                gpuErrchk(cudaDeviceSynchronize());
                ProblemInputs<T> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                SQPStats<T> stats = solver_.solve(d_xu_traj_batch_, inputs);

                DeviceArray xu_out = DeviceArray::alloc({static_cast<int64_t>(batch_size_), static_cast<int64_t>(XU_TRAJ_SIZE)});
                gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(xu_out.data_ptr()), d_xu_traj_batch_, XU_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaDeviceSynchronize());
                py::dict result = collect_stats(stats);
                result["XU"] = xu_out;
                return result;
        }

        // Device-pointer solve (the torch / JAX path): x_s and the reference are read
        // from CUDA memory the caller owns (float32, contiguous, exactly batch_size
        // rows); xu_ptr == 0 seeds the hold-at-x cold start on the device, otherwise
        // the warm start is copied device-to-device and its knot-0 states are
        // overwritten by x_s. The returned "XU" is a DeviceArray (DLPack-exportable);
        // the caller's buffers are never written. Inputs must be complete on the
        // device: work queued on non-blocking streams needs a sync first.
        py::dict solve_device(uintptr_t xu_ptr, T timestep, uintptr_t x_ptr, uintptr_t ref_ptr)
        {
                if (!x_ptr || !ref_ptr) { throw py::value_error("solve_device: x_s and reference device pointers are required"); }
                gpuErrchk(cudaMemcpy(d_x_s_batch_, reinterpret_cast<const void*>(x_ptr), XU_STATE_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemcpy(d_reference_traj_batch_, reinterpret_cast<const void*>(ref_ptr), REFERENCE_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
                if (xu_ptr) {
                        gpuErrchk(cudaMemcpy(d_xu_traj_batch_, reinterpret_cast<const void*>(xu_ptr), XU_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
                        // knot-0 state := measured state (one strided copy over the batch)
                        gpuErrchk(cudaMemcpy2D(d_xu_traj_batch_, XU_TRAJ_SIZE * sizeof(T), d_x_s_batch_, XU_STATE_SIZE * sizeof(T),
                                               XU_STATE_SIZE * sizeof(T), batch_size_, cudaMemcpyDeviceToDevice));
                } else {
                        fill_hold_warm_start_kernel<T><<<batch_size_, 256>>>(d_xu_traj_batch_, d_x_s_batch_, KNOT_POINTS, XU_STATE_SIZE, CONTROL_SIZE);
                        gpuErrchk(cudaPeekAtLastError());
                }
                return solve_staged_to_device(timestep);
        }

        // ---- device-resident warm start (MPCController's device path) ----------
        // d_xu_traj_batch_ doubles as the resident warm-start buffer: solve_resident
        // reads it as the warm start and leaves the solution in it, shift_resident
        // advances it one knot for the next tick, resident_set_* / resident_hold
        // seed it. x and the reference are CUDA memory the caller owns (float32,
        // contiguous; one row broadcast to the batch, or exactly batch_size rows);
        // the caller's buffers are only ever read. Inputs must be complete on the
        // device: work queued on non-blocking streams needs a sync first.
        void tile_rows(T* dst, uintptr_t src, uint32_t row_len, bool batched, const char* name)
        {
                if (!src) { throw py::value_error(std::string(name) + ": a device pointer is required"); }
                if (batched) {
                        gpuErrchk(cudaMemcpy(dst, reinterpret_cast<const void*>(src), row_len * batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
                } else {
                        tile_rows_kernel<T><<<batch_size_, 256>>>(dst, reinterpret_cast<const T*>(src), row_len);
                        gpuErrchk(cudaPeekAtLastError());
                }
        }
        void resident_set_device(uintptr_t xu_ptr, bool batched)
        {
                tile_rows(d_xu_traj_batch_, xu_ptr, XU_TRAJ_SIZE, batched, "resident_set_device");
                gpuErrchk(cudaDeviceSynchronize());
        }
        void resident_set_host(DenseArray xu)
        {
                py::buffer_info buf = xu.request();
                const size_t n = static_cast<size_t>(buf.size);
                if (n == (size_t)XU_TRAJ_SIZE * batch_size_) {
                        memcpy(h_xu_staging_, buf.ptr, n * sizeof(T));
                } else if (n == XU_TRAJ_SIZE) {
                        for (uint32_t b = 0; b < batch_size_; ++b) memcpy(h_xu_staging_ + (size_t)b * XU_TRAJ_SIZE, buf.ptr, XU_TRAJ_SIZE * sizeof(T));
                } else {
                        throw py::value_error("resident_set_host: expected " + std::to_string(XU_TRAJ_SIZE) + " or " + std::to_string((size_t)XU_TRAJ_SIZE * batch_size_) + " elements, got " + std::to_string(n));
                }
                gpuErrchk(cudaMemcpy(d_xu_traj_batch_, h_xu_staging_, XU_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
        py::array_t<T> resident_get_host()
        {
                const py::ssize_t Bs = static_cast<py::ssize_t>(batch_size_);
                py::array_t<T>    out({Bs, (py::ssize_t)XU_TRAJ_SIZE});
                gpuErrchk(cudaMemcpy(out.request().ptr, d_xu_traj_batch_, XU_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));
                return out;
        }
        // hold-at-x cold seed (the host path's initialize_warm_start), on the device
        void resident_hold(uintptr_t x_ptr, bool batched)
        {
                tile_rows(d_x_s_batch_, x_ptr, XU_STATE_SIZE, batched, "resident_hold");
                fill_hold_warm_start_kernel<T><<<batch_size_, 256>>>(d_xu_traj_batch_, d_x_s_batch_, KNOT_POINTS, XU_STATE_SIZE, CONTROL_SIZE);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
        }
        // ||x - knot 0 of resident row 0|| (x: one state row on the device)
        T resident_pred_err(uintptr_t x_ptr)
        {
                if (!x_ptr) { throw py::value_error("resident_pred_err: a device pointer is required"); }
                pred_err_kernel<T><<<1, 256>>>(d_scalar_, d_xu_traj_batch_, reinterpret_cast<const T*>(x_ptr));
                gpuErrchk(cudaPeekAtLastError());
                T out;
                gpuErrchk(cudaMemcpy(&out, d_scalar_, sizeof(T), cudaMemcpyDeviceToHost));
                return out;
        }
        // solve from the resident warm start (knot-0 states := x); the solution stays
        // resident and a copy is returned as a DeviceArray
        py::dict solve_resident(uintptr_t x_ptr, bool x_batched, uintptr_t ref_ptr, bool ref_batched, T timestep)
        {
                tile_rows(d_x_s_batch_, x_ptr, XU_STATE_SIZE, x_batched, "solve_resident: x");
                tile_rows(d_reference_traj_batch_, ref_ptr, REFERENCE_TRAJ_SIZE, ref_batched, "solve_resident: reference");
                gpuErrchk(cudaMemcpy2D(d_xu_traj_batch_, XU_TRAJ_SIZE * sizeof(T), d_x_s_batch_, XU_STATE_SIZE * sizeof(T),
                                       XU_STATE_SIZE * sizeof(T), batch_size_, cudaMemcpyDeviceToDevice));
                return solve_staged_to_device(timestep);
        }
        // next tick's warm start from resident row `best`, into every row
        void resident_shift(uint32_t best, bool hold)
        {
                if (best >= batch_size_) { throw py::value_error("resident_shift: best_id " + std::to_string(best) + " >= batch_size " + std::to_string(batch_size_)); }
                gpuErrchk(cudaMemcpy(d_xu_row_, d_xu_traj_batch_ + (size_t)best * XU_TRAJ_SIZE, XU_TRAJ_SIZE * sizeof(T), cudaMemcpyDeviceToDevice));
                shift_rows_kernel<T><<<batch_size_, 256>>>(d_xu_traj_batch_, d_xu_row_, XU_TRAJ_SIZE, XU_STATE_SIZE + CONTROL_SIZE, hold);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
        }

        // everything solve() returns besides the trajectory itself
        py::dict collect_stats(const SQPStats<T>& stats)
        {
                // Copy final merits (computed at end of solve) and initial (pre-iteration) merits back to host
                std::vector<T> h_final_merit(batch_size_);
                std::vector<T> h_initial_merit(batch_size_);
                solver_.copy_final_merit_to_host(h_final_merit.data());
                solver_.copy_initial_merit0_to_host(h_initial_merit.data());

                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);

                py::dict result;
                result["sqp_time_us"] = stats.solve_time_us;
                result["sqp_iters"] = py::array_t<int32_t>({B}, {sizeof(int32_t)}, stats.sqp_iterations.data());
                result["kkt_converged"] = py::array_t<int32_t>({B}, {sizeof(int32_t)}, stats.kkt_converged.data());
                result["final_merit"] = py::array_t<T>({B}, {sizeof(T)}, h_final_merit.data());
                result["initial_merit"] = py::array_t<T>({B}, {sizeof(T)}, h_initial_merit.data());
                result["linsys_mode"] = solver_.linsys_mode();
                result["budget_hit"] = stats.budget_hit;
                result["defect_exit"] = stats.defect_exit;
                result["initial_merit_time_us"] = stats.initial_merit_time_us;
                result["merit_fused"] = stats.merit_fused;
                {
                        const py::ssize_t n_it = static_cast<py::ssize_t>(stats.stage_times_us.size() / 5);
                        py::array_t<float> st({n_it, (py::ssize_t)5});
                        if (n_it > 0) { std::memcpy(st.request().ptr, stats.stage_times_us.data(), stats.stage_times_us.size() * sizeof(float)); }
                        result["stage_times_us"] = st;
                }
                if (!stats.defect.empty()) { result["defect"] = py::array_t<T>({B}, {sizeof(T)}, stats.defect.data()); }

                // Per-iteration stats: shape them as (iters, batch_size)
                result["ls_num_iters"] = static_cast<int>(stats.line_search_stats.size());
                std::vector<float> pcg_times_us;
                for (const auto& pcg_stat : stats.pcg_stats) { pcg_times_us.push_back(pcg_stat.solve_time_us); }
                result["pcg_times_us"] = py::array_t<float>({(py::ssize_t)pcg_times_us.size()}, pcg_times_us.data());
                result["pcg_iters"] = per_iter_array<int>(stats.pcg_stats, [](const auto& st, size_t b) { return st.num_iterations[b]; });
                result["ls_min_merit"] = per_iter_array<float>(stats.line_search_stats, [](const auto& st, size_t b) { return st.min_merit[b]; });
                result["ls_step_size"] = per_iter_array<float>(stats.line_search_stats, [](const auto& st, size_t b) { return st.step_size[b]; });

                // row-group telemetry: (n_groups, batch) {max, sum} true violation of
                // the returned trajectory (present only when row groups are enabled)
                const int32_t n_grp = solver_.num_row_groups();
                if (n_grp > 0) {
                        std::vector<T> h_telem(2 * gato::rows::MAX_ROW_GROUPS * batch_size_);
                        solver_.copy_row_telemetry_to_host(h_telem.data());
                        py::array_t<T> vmax({(py::ssize_t)n_grp, B});
                        py::array_t<T> vsum({(py::ssize_t)n_grp, B});
                        T* pmax = static_cast<T*>(vmax.request().ptr);
                        T* psum = static_cast<T*>(vsum.request().ptr);
                        for (int32_t g = 0; g < n_grp; ++g) {
                                for (size_t b = 0; b < batch_size_; ++b) {
                                        pmax[g * batch_size_ + b] = h_telem[b * 2 * gato::rows::MAX_ROW_GROUPS + 2 * g + 0];
                                        psum[g * batch_size_ + b] = h_telem[b * 2 * gato::rows::MAX_ROW_GROUPS + 2 * g + 1];
                                }
                        }
                        result["row_max_violation"] = vmax;
                        result["row_sum_violation"] = vsum;
                        if (solver_.admm_active()) {
                                std::vector<T> h_res(2 * batch_size_);
                                solver_.copy_admm_residuals_to_host(h_res.data());
                                py::array_t<T> rp({B}), rd({B});
                                T* prp = static_cast<T*>(rp.request().ptr);
                                T* prd = static_cast<T*>(rd.request().ptr);
                                for (size_t b = 0; b < batch_size_; ++b) {
                                        prp[b] = h_res[2 * b + 0];
                                        prd[b] = h_res[2 * b + 1];
                                }
                                result["admm_r_prim"] = rp;
                                result["admm_r_dual"] = rd;
                                result["admm_iters"] = py::array_t<int32_t>({B}, {sizeof(int32_t)}, stats.admm_iters.data());
                                result["admm_linsys_iters"] = py::array_t<int32_t>({B}, {sizeof(int32_t)}, stats.admm_linsys_iters.data());
                        }
                }

                return py::dict(result);
        }

        // per-solve wrench (B, 6*NUM_BODIES) / per-knot wrench band (B, KNOT_POINTS, 6*NUM_BODIES), flattened
        void set_f_ext_batch(DenseArray f_ext_batch) { solver_.set_f_ext_batch(checked_ptr(f_ext_batch, (size_t)6 * grid::NUM_BODIES * batch_size_, "f_ext_batch")); }
        void set_f_ext_knot_batch(DenseArray f_ext_knot_batch) { solver_.set_f_ext_knot_batch(checked_ptr(f_ext_knot_batch, (size_t)6 * grid::NUM_BODIES * KNOT_POINTS * batch_size_, "f_ext_knot_batch")); }
        // per-solve scalars (B,)
        void set_rho_penalty_batch(DenseArray rho_batch, bool set_as_reset_default = true) { solver_.set_rho_penalty_batch(checked_ptr(rho_batch, batch_size_, "rho_batch"), set_as_reset_default); }
        void set_drho_batch(DenseArray drho_batch, bool set_as_reset_default = true) { solver_.set_drho_batch(checked_ptr(drho_batch, batch_size_, "drho_batch"), set_as_reset_default); }
        void set_mu_batch(DenseArray mu_batch) { solver_.set_mu_batch(checked_ptr(mu_batch, batch_size_, "mu_batch")); }
        void set_pcg_tol_batch(DenseArray eps_batch) { solver_.set_pcg_tol_batch(checked_ptr(eps_batch, batch_size_, "pcg_tol_batch")); }

        py::array_t<T> sim_forward(DenseArray xk, DenseArray uk, T dt)
        {
                py::buffer_info xk_buf = xk.request();
                py::buffer_info uk_buf = uk.request();
                if (static_cast<size_t>(xk_buf.size) != XU_STATE_SIZE) { throw py::value_error("xk: expected " + std::to_string(XU_STATE_SIZE) + " elements, got " + std::to_string(xk_buf.size)); }
                if (static_cast<size_t>(uk_buf.size) != CONTROL_SIZE) { throw py::value_error("uk: expected " + std::to_string(CONTROL_SIZE) + " elements, got " + std::to_string(uk_buf.size)); }

                gpuErrchk(cudaMemcpy(d_xk_, xk_buf.ptr, XU_STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_uk_, uk_buf.ptr, CONTROL_SIZE * sizeof(T), cudaMemcpyHostToDevice));

                solver_.sim_forward(d_xkp1_batch_, d_xk_, d_uk_, dt);
                gpuErrchk(cudaDeviceSynchronize());

                gpuErrchk(cudaMemcpy(h_xkp1_batch_.data(), d_xkp1_batch_, XU_STATE_SIZE * batch_size_ * sizeof(T), cudaMemcpyDeviceToHost));

                return py::array_t<T>({static_cast<py::ssize_t>(batch_size_), (py::ssize_t)XU_STATE_SIZE}, h_xkp1_batch_.data());
        }

        void enable_limit_telemetry() { solver_.enable_limit_telemetry(); }
        void enable_limit_barrier(T mu, T delta) { solver_.enable_limit_barrier(mu, delta); }
        void enable_limit_admm(T rho, uint32_t iters) { solver_.enable_limit_admm(rho, iters); }
        void enable_limit_al(T rho) { solver_.enable_limit_al(rho); }
        void enable_ee_terminal_equality(DenseArray target, T rho)
        {
                py::buffer_info buf = target.request();
                if (buf.size != 3) { throw py::value_error("enable_ee_terminal_equality: target must be xyz (3 elements)"); }
                solver_.enable_ee_terminal_equality(static_cast<T*>(buf.ptr), rho);
        }
        void disable_row_groups() { solver_.disable_row_groups(); }

        // LIN_U row-group append (CL-2): C is (m, NU); d/lo/hi length m (d may be
        // empty -> zero offset; lo/hi ignored for cone rows). mech is the
        // rows::Mechanism enum value (0 telemetry, 1 barrier, 2 admm, 3 al).
        void add_lin_u_group(int32_t mech, DenseArray C, DenseArray d, DenseArray lo, DenseArray hi,
                             bool cone, T rho, T delta, T sigma, int32_t knot_lo, int32_t knot_hi, uint32_t admm_iters, bool equilibrate)
        {
                py::buffer_info bC = C.request();
                if (bC.ndim != 2 || bC.shape[1] != (py::ssize_t)CONTROL_SIZE) { throw py::value_error("add_lin_u_group: C must be (m, " + std::to_string(CONTROL_SIZE) + ")"); }
                const int32_t   m = (int32_t)bC.shape[0];
                py::buffer_info bd = d.request(), blo = lo.request(), bhi = hi.request();
                if (bd.size != 0 && bd.size != m) { throw py::value_error("add_lin_u_group: d must be empty or length m"); }
                if (!cone && (blo.size != m || bhi.size != m)) { throw py::value_error("add_lin_u_group: interval rows need lo/hi of length m"); }
                solver_.add_lin_u_group(mech, m, static_cast<T*>(bC.ptr), bd.size ? static_cast<T*>(bd.ptr) : nullptr, blo.size ? static_cast<T*>(blo.ptr) : nullptr,
                                        bhi.size ? static_cast<T*>(bhi.ptr) : nullptr, cone, rho, delta, sigma, knot_lo, knot_hi, admm_iters, equilibrate);
        }

        // CONTACT_POS rows (CL-4 §1.3): 3*NUM_CONTACT_FRAMES residual rows
        // p_f(q_k) - tgt_k on the state block; lo/hi (n_rows, empty = 0 = equality),
        // tgt (KNOT_POINTS x n_rows, empty = zeros). Returns the group index.
        int32_t add_contact_pos_group(int32_t mech, DenseArray lo, DenseArray hi, DenseArray tgt,
                                      T rho, T delta, T sigma, int32_t knot_lo, int32_t knot_hi, uint32_t admm_iters)
        {
                constexpr py::ssize_t R = (py::ssize_t)gato::rows::CONTACT_ROWS;
                py::buffer_info blo = lo.request(), bhi = hi.request(), bt = tgt.request();
                if (blo.size != 0 && blo.size != R) { throw py::value_error("add_contact_pos_group: lo must be empty or length " + std::to_string(R)); }
                if (bhi.size != 0 && bhi.size != R) { throw py::value_error("add_contact_pos_group: hi must be empty or length " + std::to_string(R)); }
                if (bt.size != 0 && bt.size != (py::ssize_t)KNOT_POINTS * R) { throw py::value_error("add_contact_pos_group: targets must be empty or (KNOT_POINTS, " + std::to_string(R) + ")"); }
                return solver_.add_contact_pos_group(mech, blo.size ? static_cast<T*>(blo.ptr) : nullptr, bhi.size ? static_cast<T*>(bhi.ptr) : nullptr,
                                                     bt.size ? static_cast<T*>(bt.ptr) : nullptr, rho, delta, sigma, knot_lo, knot_hi, admm_iters);
        }
        // per-knot target table of one group ((KNOT_POINTS, n_rows), row-major)
        void set_row_group_targets(int32_t g, DenseArray tgt)
        {
                py::buffer_info bt = tgt.request();
                solver_.set_row_group_targets(g, static_cast<T*>(bt.ptr), (size_t)bt.size);
        }

        // per-solve row state: AL duals {lam_hi, lam_lo} / ADMM state {z, y}, either
        // the dense row_state_index layout (B, MAX_ROW_GROUPS, KNOT_POINTS,
        // MAX_ROWS_PER_GROUP) or the collision band (B, KNOT_POINTS, NCC)
        py::dict get_row_duals() { return state_dict(/*admm=*/false, /*collision=*/false); }
        py::dict get_admm_state() { return state_dict(/*admm=*/true, /*collision=*/false); }
        py::dict get_collision_row_duals() { return state_dict(/*admm=*/false, /*collision=*/true); }
        py::dict get_collision_admm_state() { return state_dict(/*admm=*/true, /*collision=*/true); }

        // flat (n, k) float arrays per primitive: spheres (n,4) {x,y,z,r};
        // capsules (n,7) {a(3),b(3),r}; cuboids (n,15) {c(3),u(3),hu,v(3),hv,
        // w(3),hw}; planes (n,4) {n_unit(3),d} — geometry-header layouts
        void set_collision_environment(DenseArray spheres, DenseArray capsules, DenseArray cuboids, DenseArray planes)
        {
                auto check = [](py::buffer_info& b, py::ssize_t w, const char* nm) {
                        if (b.size == 0) return (py::ssize_t)0;
                        if (b.ndim != 2 || b.shape[1] != w) { throw py::value_error(std::string("set_collision_environment: ") + nm + " must be (n, " + std::to_string(w) + ")"); }
                        return b.shape[0];
                };
                auto bs = spheres.request(), bc = capsules.request(), bb = cuboids.request(), bp = planes.request();
                const int32_t ns = (int32_t)check(bs, 4, "spheres"), nc = (int32_t)check(bc, 7, "capsules");
                const int32_t nb = (int32_t)check(bb, 15, "cuboids"), np_ = (int32_t)check(bp, 4, "planes");
                solver_.set_collision_environment(ns ? static_cast<T*>(bs.ptr) : nullptr, ns, nc ? static_cast<T*>(bc.ptr) : nullptr, nc,
                                                  nb ? static_cast<T*>(bb.ptr) : nullptr, nb, np_ ? static_cast<T*>(bp.ptr) : nullptr, np_);
        }
        void enable_collision(int32_t mech, T margin, T rho, T delta, T sigma, int32_t knot_lo, uint32_t admm_iters)
        {
                solver_.enable_collision(mech, margin, rho, delta, sigma, knot_lo, admm_iters);
        }

        void set_row_group_soft(int32_t g, T sigma) { solver_.set_row_group_soft(g, sigma); }
        void set_row_group_rho(int32_t g, T rho) { solver_.set_row_group_rho(g, rho); }
        void set_admm_merit(bool on) { solver_.set_admm_merit(on); }
        void set_admm_rho_adaptation(bool on) { solver_.set_admm_rho_adaptation(on); }
        // per-solve adapted rho scale (B,): rho_effective = group rho * scale
        py::array_t<T> get_admm_rho_scale()
        {
                py::array_t<T> out((py::ssize_t)batch_size_);
                solver_.copy_admm_rho_scale_to_host(static_cast<T*>(out.request().ptr));
                return out;
        }

        // per-knot row-activity mask: (KNOT_POINTS,) uint64, bit i = row i live at that knot
        void set_row_group_mask(int32_t g, py::array_t<uint64_t, py::array::c_style | py::array::forcecast> active)
        {
                py::buffer_info b = active.request();
                if (b.size != (py::ssize_t)KNOT_POINTS) { throw py::value_error("set_row_group_mask: expected " + std::to_string(KNOT_POINTS) + " knot words"); }
                solver_.set_row_group_mask(g, static_cast<uint64_t*>(b.ptr));
        }

        void set_row_group_bounds(int32_t g, DenseArray lo, DenseArray hi)
        {
                py::buffer_info blo = lo.request(), bhi = hi.request();
                std::vector<gato::rows::RowGroupDesc<T>> h_groups(gato::rows::MAX_ROW_GROUPS);
                solver_.copy_row_groups_to_host(h_groups.data());
                if (g < 0 || g >= solver_.num_row_groups()) { throw py::value_error("set_row_group_bounds: group index out of range"); }
                const py::ssize_t n = h_groups[g].n_rows;
                if (blo.size != n || bhi.size != n) { throw py::value_error("set_row_group_bounds: expected " + std::to_string(n) + " bounds per side"); }
                solver_.set_row_group_bounds(g, static_cast<T*>(blo.ptr), static_cast<T*>(bhi.ptr));
        }

        // descriptor introspection (oracle tests): list of per-group dicts
        py::list get_row_groups()
        {
                std::vector<gato::rows::RowGroupDesc<T>> h_groups(gato::rows::MAX_ROW_GROUPS);
                solver_.copy_row_groups_to_host(h_groups.data());
                py::list out;
                for (int32_t g = 0; g < solver_.num_row_groups(); ++g) {
                        const auto&    grp = h_groups[g];
                        py::dict       d;
                        py::array_t<T> lo({(py::ssize_t)grp.n_rows}), hi({(py::ssize_t)grp.n_rows});
                        memcpy(lo.request().ptr, grp.lo, grp.n_rows * sizeof(T));
                        memcpy(hi.request().ptr, grp.hi, grp.n_rows * sizeof(T));
                        d["kind"] = grp.kind;
                        d["block"] = grp.block;
                        d["mech"] = grp.mech;
                        d["n_rows"] = grp.n_rows;
                        d["knot_lo"] = grp.knot_lo;
                        d["knot_hi"] = grp.knot_hi;
                        d["sigma"] = grp.sigma;
                        d["rho"] = grp.mu;    // the mechanism weight: barrier mu / ADMM rho / AL rho
                        d["cone"] = grp.cone;
                        d["lo"] = lo;
                        d["hi"] = hi;
                        py::array_t<uint64_t> act({(py::ssize_t)KNOT_POINTS});
                        memcpy(act.request().ptr, grp.active, KNOT_POINTS * sizeof(uint64_t));
                        d["active"] = act;
                        if (grp.kind == gato::rows::CONTACT_POS) {
                                py::array_t<T> tg({(py::ssize_t)KNOT_POINTS, (py::ssize_t)grp.n_rows});
                                T* dst = static_cast<T*>(tg.request().ptr);
                                for (uint32_t k = 0; k < KNOT_POINTS; k++) { memcpy(dst + (size_t)k * grp.n_rows, grp.tgt + (size_t)k * gato::rows::MAX_ROWS_PER_GROUP, (size_t)grp.n_rows * sizeof(T)); }
                                d["tgt"] = tg;
                        }
                        if (grp.kind == gato::rows::LIN_U) {
                                py::array_t<T> Cm({(py::ssize_t)grp.n_rows, (py::ssize_t)CONTROL_SIZE});
                                py::array_t<T> dv({(py::ssize_t)grp.n_rows});
                                memcpy(Cm.request().ptr, grp.Cmat, (size_t)grp.n_rows * CONTROL_SIZE * sizeof(T));
                                memcpy(dv.request().ptr, grp.dvec, (size_t)grp.n_rows * sizeof(T));
                                d["C"] = Cm;
                                d["d"] = dv;
                        }
                        out.append(d);
                }
                return out;
        }

        void reset_dual() { solver_.reset_dual(); }
        void reset_rho() { solver_.reset_rho(); }
        void set_rho_adaptation(bool enabled) { solver_.set_rho_adaptation(enabled); }
        void set_collect_stats(bool enabled) { solver_.set_collect_stats(enabled); }
        void set_time_budget_us(double us) { solver_.set_time_budget_us(us); }
        void set_defect_telemetry(bool on) { solver_.set_defect_telemetry(on); }
        void set_defect_exit(T tol) { solver_.set_defect_exit(tol); }
        void set_linsys_mode(int mode) { solver_.set_linsys_mode(mode); }
        void set_admm_linsys_pcg(bool on) { solver_.set_admm_linsys_pcg(on); }
        void set_admm_exit(T eps_abs, T eps_rel, uint32_t check_every) { solver_.set_admm_exit(eps_abs, eps_rel, check_every); }
        bool admm_active() const { return solver_.admm_active(); }
        void set_exact_hessian(bool on) { solver_.set_exact_hessian(on); }
        bool exact_hessian() const { return solver_.exact_hessian(); }

        // debug/test: KKT setup only (no solve) on the given trajectory + block readback
        py::dict debug_setup_kkt(DenseArray xu_traj_batch, T timestep, DenseArray x_s_batch, DenseArray reference_traj_batch)
        {
                auto inputs = upload_problem(xu_traj_batch, timestep, x_s_batch, reference_traj_batch);
                solver_.debug_setup_kkt(d_xu_traj_batch_, inputs);

                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);
                py::array_t<T>    Q({B, (py::ssize_t)STATE_SQ_P_KNOTS});
                py::array_t<T>    R({B, (py::ssize_t)CONTROL_SQ_P_KNOTS});
                py::array_t<T>    q({B, (py::ssize_t)STATE_P_KNOTS});
                py::array_t<T>    r({B, (py::ssize_t)CONTROL_P_KNOTS});
                solver_.copy_kkt_blocks_to_host(static_cast<T*>(Q.request().ptr), static_cast<T*>(R.request().ptr), static_cast<T*>(q.request().ptr), static_cast<T*>(r.request().ptr));
                // dynamics linearization blocks (tangent A|B col-major per knot, defect c)
                py::array_t<T> A({B, (py::ssize_t)STATE_SQ_P_KNOTS});
                py::array_t<T> Bm({B, (py::ssize_t)STATE_P_CONTROL_P_KNOTS});
                py::array_t<T> c({B, (py::ssize_t)STATE_P_KNOTS});
                solver_.copy_kkt_dynamics_to_host(static_cast<T*>(A.request().ptr), static_cast<T*>(Bm.request().ptr), static_cast<T*>(c.request().ptr));
                py::dict out;
                out["Q"] = Q;
                out["R"] = R;
                out["q"] = q;
                out["r"] = r;
                out["A"] = A;
                out["B"] = Bm;
                out["c"] = c;
                return out;
        }
        // debug/oracle: contact-wrench chain at one (q, qd, u, f_c) sample (CL-3 prep).
        // Jacobians come back as (rows, cols) numpy arrays (strided from the device
        // col-major buffers): dqdd_dfc (NQ, 6*NUM_CONTACT_FRAMES), dqdd_dq /
        // dqdd_dq_corr (NQ, NQ). dqdd_dq holds f_ext FIXED; dqdd_dq_corr is the
        // dfext/dq chain term (total = dqdd_dq + dqdd_dq_corr).
        py::dict debug_contact_dynamics(DenseArray q, DenseArray qd, DenseArray u, DenseArray fc)
        {
#ifdef GRID_HAS_CONTACT_FRAMES
                constexpr py::ssize_t NQ = gato::plant::NQ;
                constexpr py::ssize_t FEXT = 6 * grid::NUM_BODIES;
                constexpr py::ssize_t NFCW = 6 * grid::NUM_CONTACT_FRAMES;
                py::buffer_info       q_buf = q.request(), qd_buf = qd.request(), u_buf = u.request(), fc_buf = fc.request();
                check_size(q_buf, (size_t)NQ, "q");
                check_size(qd_buf, (size_t)NQ, "qd");
                check_size(u_buf, (size_t)NQ, "u");
                check_size(fc_buf, (size_t)NFCW, "fc");
                std::vector<T> qdd(NQ), fext(FEXT), dfc(NQ * NFCW), dq(NQ * NQ), dq_corr(NQ * NQ);
                std::vector<T> dfc_adapter(NQ * NFCW), dq_adapter(NQ * NQ);
                solver_.debug_contact_dynamics(static_cast<T*>(q_buf.ptr), static_cast<T*>(qd_buf.ptr), static_cast<T*>(u_buf.ptr), static_cast<T*>(fc_buf.ptr),
                                               qdd.data(), fext.data(), dfc.data(), dq.data(), dq_corr.data(),
                                               dfc_adapter.data(), dq_adapter.data());
                const py::ssize_t sT = (py::ssize_t)sizeof(T);
                py::dict          out;
                out["qdd"] = py::array_t<T>({NQ}, qdd.data());
                out["fext"] = py::array_t<T>({FEXT}, fext.data());
                out["dqdd_dfc"] = py::array_t<T>({NQ, NFCW}, {sT, NQ * sT}, dfc.data());
                out["dqdd_dq"] = py::array_t<T>({NQ, NQ}, {sT, NQ * sT}, dq.data());
                out["dqdd_dq_corr"] = py::array_t<T>({NQ, NQ}, {sT, NQ * sT}, dq_corr.data());
#if GATO_CONTACT_FORCES
                // the ADAPTER's B-block fc columns (fc-build gate: same math as
                // dqdd_dfc through the solver's in-adapter code path)
                out["dqdd_dfc_adapter"] = py::array_t<T>({NQ, NFCW}, {sT, NQ * sT}, dfc_adapter.data());
                // the ADAPTER's A-block dq columns, W2 chain term INCLUDED (fc-build gate:
                // must equal dqdd_dq + dqdd_dq_corr, the independent oracle composition)
                out["dqdd_dq_adapter"] = py::array_t<T>({NQ, NQ}, {sT, NQ * sT}, dq_adapter.data());
#endif
                return out;
#else
                throw std::runtime_error("module generated without contact frames (GRID_HAS_CONTACT_FRAMES)");
#endif
        }

        py::array_t<T> get_lambda()
        {
                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);
                py::array_t<T>    lam({B, (py::ssize_t)VEC_SIZE_PADDED});
                solver_.copy_lambda_to_host(static_cast<T*>(lam.request().ptr));
                return lam;
        }

        void set_cost_weights(T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost)
        {
                solver_.set_cost_weights(q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
        }

        void set_q_pos_cost(T w) { solver_.set_q_pos_cost(w); }
        void set_fc_cost(T w) { solver_.set_fc_cost(w); }
        void set_fc_ref(DenseArray ref)
        {
                auto buf = ref.request();
                if (buf.size == 0) { solver_.set_fc_ref(nullptr); return; }  // empty -> reset to zeros
                constexpr py::ssize_t FC = (py::ssize_t)gato::constants::FC_SIZE;
                if (buf.size == FC) { solver_.set_fc_ref(static_cast<T*>(buf.ptr), /*per_knot=*/false); return; }
                if (buf.size == (py::ssize_t)KNOT_POINTS * FC) { solver_.set_fc_ref(static_cast<T*>(buf.ptr), /*per_knot=*/true); return; }
                throw std::runtime_error("fc_ref must have FC_SIZE or KNOT_POINTS*FC_SIZE entries (or be empty to reset)");
        }
        // per-joint weight / nominal vectors; an empty array resets to the scalar weight / zeros
        void set_u_cost_vec(DenseArray w) { solver_.set_u_cost_vec(optional_ptr(w, ACTUATED_SIZE, "u_cost_vec")); }
        void set_q_pos_cost_vec(DenseArray w) { solver_.set_q_pos_cost_vec(optional_ptr(w, grid::NUM_JOINTS, "q_pos_cost_vec")); }
        void set_q_nom(DenseArray q_nom) { solver_.set_q_nom(optional_ptr(q_nom, grid::NUM_JOINTS, "q_nom")); }

        void set_cost_weights_per_knot(DenseArray knot_weights)
        {
                py::buffer_info buf = knot_weights.request();
                if (static_cast<size_t>(buf.size) != 3 * KNOT_POINTS) {
                        throw py::value_error("knot_weights: expected " + std::to_string(3 * KNOT_POINTS) + " elements ((KNOT_POINTS, 3) [ee, qd, u] triples), got " + std::to_string(buf.size));
                }
                solver_.set_cost_weights_per_knot(static_cast<T*>(buf.ptr));
        }
        void clear_cost_weights_per_knot() { solver_.clear_cost_weights_per_knot(); }

      private:
        // (iters, B) array of one per-iteration stat: at(stat, b) -> V
        template<typename V, typename S, typename F>
        py::array_t<V> per_iter_array(const std::vector<S>& per_iter, F at)
        {
                std::vector<V> flat;
                flat.reserve(per_iter.size() * batch_size_);
                for (const auto& st : per_iter) {
                        for (size_t b = 0; b < batch_size_; ++b) { flat.push_back(static_cast<V>(at(st, b))); }
                }
                return py::array_t<V>({(py::ssize_t)per_iter.size(), (py::ssize_t)batch_size_}, flat.data());
        }

        ProblemInputs<T> upload_problem(const DenseArray& xu_traj_batch, T timestep,
                                       const DenseArray& x_s_batch, const DenseArray& reference_traj_batch)
        {
                py::buffer_info xu_buf = xu_traj_batch.request();
                py::buffer_info xs_buf = x_s_batch.request();
                py::buffer_info ref_buf = reference_traj_batch.request();
                check_size(xu_buf, (size_t)XU_TRAJ_SIZE * batch_size_, "xu_traj_batch");
                check_size(xs_buf, (size_t)XU_STATE_SIZE * batch_size_, "x_s_batch");
                check_size(ref_buf, (size_t)REFERENCE_TRAJ_SIZE * batch_size_, "reference_traj_batch");

                memcpy(h_xu_staging_, xu_buf.ptr, XU_TRAJ_SIZE * batch_size_ * sizeof(T));
                gpuErrchk(cudaMemcpy(d_xu_traj_batch_, h_xu_staging_, XU_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_x_s_batch_, xs_buf.ptr, XU_STATE_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(d_reference_traj_batch_, ref_buf.ptr, REFERENCE_TRAJ_SIZE * batch_size_ * sizeof(T), cudaMemcpyHostToDevice));

                ProblemInputs<T> inputs;
                inputs.timestep = timestep;
                inputs.d_x_s_batch = d_x_s_batch_;
                inputs.d_reference_traj_batch = d_reference_traj_batch_;

                return inputs;
        }

        std::pair<py::array_t<T>, py::array_t<T>> state_pair(bool admm, size_t offset, size_t width,
                                                                        const std::vector<py::ssize_t>& shape)
        {
                constexpr size_t stride = gato::rows::TOTAL_ROW_STATE_SIZE;
                py::array_t<T> a(shape), b(shape);
                std::vector<T> ha(stride * batch_size_), hb(stride * batch_size_);
                if (admm) solver_.copy_admm_state_to_host(ha.data(), hb.data());
                else solver_.copy_row_duals_to_host(ha.data(), hb.data());
                for (uint32_t s = 0; s < batch_size_; ++s) {
                        memcpy(a.mutable_data() + s * width, ha.data() + s * stride + offset, width * sizeof(T));
                        memcpy(b.mutable_data() + s * width, hb.data() + s * stride + offset, width * sizeof(T));
                }
                return {a, b};
        }

        py::dict state_dict(bool admm, bool collision)
        {
                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);
                const auto p = collision ? state_pair(admm, gato::rows::ROW_STATE_SIZE, gato::rows::COLLISION_ROW_STATE_SIZE, {B, KNOT_POINTS, gato::plant::NCC})
                                         : state_pair(admm, 0, gato::rows::ROW_STATE_SIZE, {B, gato::rows::MAX_ROW_GROUPS, KNOT_POINTS, gato::rows::MAX_ROWS_PER_GROUP});
                py::dict d;
                d[admm ? "z" : "lam_hi"] = p.first;
                d[admm ? "y" : "lam_lo"] = p.second;
                return d;
        }

        uint32_t       batch_size_;
        BSQP<T>        solver_;
        T*             h_xu_staging_;
        T*             d_xu_traj_batch_;
        T*             d_x_s_batch_;
        T*             d_reference_traj_batch_;

        // for sim_forward
        T *            d_xkp1_batch_, *d_xk_, *d_uk_;
        std::vector<T> h_xkp1_batch_;
        // resident warm start: one plan row (shift source) and the pred_err scalar
        T*             d_xu_row_;
        T*             d_scalar_;
};


// the plant name token is injected at compile time (-DGATO_PLANT_NAME=<name>)
#ifndef GATO_PLANT_NAME
#error "GATO_PLANT_NAME must be defined (plant name token for the module name)"
#endif

// bsqpN{N}_{plant}[_{variant}]: GATO_MODULE_VARIANT is "" (default), _fc or _eh —
// variant modules are separate ABIs and live side by side in python/gato/.
#ifndef GATO_MODULE_VARIANT
#define GATO_MODULE_VARIANT
#endif
#define MODULE_NAME_HELPER(knot, plant, variant) bsqpN##knot##_##plant##variant
#define MODULE_NAME(knot, plant, variant) MODULE_NAME_HELPER(knot, plant, variant)

// Register the runtime-batch-size PyBSQP class for the given precision type
#define REGISTER_BSQP_CLASS(Type)                                                                                                                                                                  \
        py::class_<PyBSQP<Type>>(m, "BSQP_" #Type, py::module_local()) /* every module defines the same C++ type; a global registration collides when two solver modules load in one process */                                                                                                                                                 \
            .def(py::init<const uint32_t, const uint32_t, const uint32_t, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, \
                          const Type, const Type, const Type>())                                                                                                                                    \
            .def("solve", &PyBSQP<Type>::solve)                                                                                                                                                    \
            .def("solve_device", &PyBSQP<Type>::solve_device, py::arg("xu_ptr"), py::arg("timestep"), py::arg("x_ptr"), py::arg("ref_ptr"))                                                      \
            .def("solve_resident", &PyBSQP<Type>::solve_resident, py::arg("x_ptr"), py::arg("x_batched"), py::arg("ref_ptr"), py::arg("ref_batched"), py::arg("timestep"))                     \
            .def("resident_set_device", &PyBSQP<Type>::resident_set_device, py::arg("xu_ptr"), py::arg("batched"))                                                                             \
            .def("resident_set_host", &PyBSQP<Type>::resident_set_host, py::arg("xu"))                                                                                                         \
            .def("resident_get_host", &PyBSQP<Type>::resident_get_host)                                                                                                                        \
            .def("resident_hold", &PyBSQP<Type>::resident_hold, py::arg("x_ptr"), py::arg("batched"))                                                                                          \
            .def("resident_pred_err", &PyBSQP<Type>::resident_pred_err, py::arg("x_ptr"))                                                                                                      \
            .def("resident_shift", &PyBSQP<Type>::resident_shift, py::arg("best"), py::arg("hold"))                                                                                            \
            .def("reset_dual", &PyBSQP<Type>::reset_dual)                                                                                                                                          \
            .def("set_f_ext_batch", &PyBSQP<Type>::set_f_ext_batch)                                                                                                                                \
            .def("set_f_ext_knot_batch", &PyBSQP<Type>::set_f_ext_knot_batch)                                                                                                                      \
            .def("set_rho_penalty_batch", &PyBSQP<Type>::set_rho_penalty_batch, py::arg("rho_batch"), py::arg("set_as_reset_default") = true)                                                      \
            .def("set_drho_batch", &PyBSQP<Type>::set_drho_batch, py::arg("drho_batch"), py::arg("set_as_reset_default") = true)                                                                   \
            .def("set_mu_batch", &PyBSQP<Type>::set_mu_batch)                                                                                                                                      \
            .def("set_pcg_tol_batch", &PyBSQP<Type>::set_pcg_tol_batch)                                                                                                                            \
            .def("sim_forward", &PyBSQP<Type>::sim_forward)                                                                                                                                        \
            .def("reset_rho", &PyBSQP<Type>::reset_rho)                                                                                                                                            \
            .def("set_rho_adaptation", &PyBSQP<Type>::set_rho_adaptation)                                                                                                                              \
            .def("set_collect_stats", &PyBSQP<Type>::set_collect_stats)                                                                                                                                \
            .def("set_time_budget_us", &PyBSQP<Type>::set_time_budget_us, py::arg("us"))                                                                                                                                \
            .def("set_defect_telemetry", &PyBSQP<Type>::set_defect_telemetry, py::arg("on"))                                                                                                                             \
            .def("set_defect_exit", &PyBSQP<Type>::set_defect_exit, py::arg("tol"))                                                                                                                                      \
            .def("set_linsys_mode", &PyBSQP<Type>::set_linsys_mode)                                                                                                                                    \
            .def("set_admm_linsys_pcg", &PyBSQP<Type>::set_admm_linsys_pcg, py::arg("on"))                                                                                                             \
            .def("set_admm_exit", &PyBSQP<Type>::set_admm_exit, py::arg("eps_abs"), py::arg("eps_rel"), py::arg("check_every"))                                                                       \
            .def("admm_active", &PyBSQP<Type>::admm_active)                                                                                                                                           \
            .def("set_exact_hessian", &PyBSQP<Type>::set_exact_hessian, py::arg("on"))                                                                                                                 \
            .def("exact_hessian", &PyBSQP<Type>::exact_hessian)                                                                                                                                        \
            .def("debug_setup_kkt", &PyBSQP<Type>::debug_setup_kkt)                                                                                                                                    \
            .def("debug_contact_dynamics", &PyBSQP<Type>::debug_contact_dynamics, py::arg("q"), py::arg("qd"), py::arg("u"), py::arg("fc"))                                                             \
            .def("get_lambda", &PyBSQP<Type>::get_lambda)                                                                                                                                              \
            .def("set_cost_weights", &PyBSQP<Type>::set_cost_weights)                                                                                                                                  \
            .def("set_q_pos_cost", &PyBSQP<Type>::set_q_pos_cost)                                                                                                                                 \
            .def("set_fc_cost", &PyBSQP<Type>::set_fc_cost)                                                                                                                                            \
            .def("set_fc_ref", &PyBSQP<Type>::set_fc_ref)                                                                                                                                              \
            .def("set_u_cost_vec", &PyBSQP<Type>::set_u_cost_vec)                                                                                                                                 \
            .def("set_q_pos_cost_vec", &PyBSQP<Type>::set_q_pos_cost_vec)                                                                                                                          \
            .def("set_q_nom", &PyBSQP<Type>::set_q_nom)                                                                                                                                                \
            .def("set_cost_weights_per_knot", &PyBSQP<Type>::set_cost_weights_per_knot)                                                                                                                \
            .def("clear_cost_weights_per_knot", &PyBSQP<Type>::clear_cost_weights_per_knot)                                                                                                            \
            .def("enable_limit_telemetry", &PyBSQP<Type>::enable_limit_telemetry)                                                                                                                      \
            .def("enable_limit_barrier", &PyBSQP<Type>::enable_limit_barrier, py::arg("mu"), py::arg("delta"))                                                                                         \
            .def("enable_limit_admm", &PyBSQP<Type>::enable_limit_admm, py::arg("rho"), py::arg("iters"))                                                                                              \
            .def("enable_limit_al", &PyBSQP<Type>::enable_limit_al, py::arg("rho"))                                                                                                                    \
            .def("enable_ee_terminal_equality", &PyBSQP<Type>::enable_ee_terminal_equality, py::arg("target"), py::arg("rho"))                                                                         \
            .def("disable_row_groups", &PyBSQP<Type>::disable_row_groups)                                                                                                                              \
            .def("get_row_groups", &PyBSQP<Type>::get_row_groups)                                                                                                                                      \
            .def("get_row_duals", &PyBSQP<Type>::get_row_duals)                                                                                                                                        \
            .def("get_admm_state", &PyBSQP<Type>::get_admm_state)                                                                                                                                      \
            .def("set_row_group_mask", &PyBSQP<Type>::set_row_group_mask)                                                                                                                                \
            .def("add_contact_pos_group", &PyBSQP<Type>::add_contact_pos_group, py::arg("mech"), py::arg("lo"), py::arg("hi"), py::arg("tgt"), py::arg("rho"), py::arg("delta"), py::arg("sigma"), py::arg("knot_lo"), py::arg("knot_hi"), py::arg("admm_iters")) \
            .def("set_row_group_targets", &PyBSQP<Type>::set_row_group_targets, py::arg("g"), py::arg("tgt")) \
            .def("set_row_group_bounds", &PyBSQP<Type>::set_row_group_bounds, py::arg("g"), py::arg("lo"), py::arg("hi"))                                                                            \
            .def("set_row_group_soft", &PyBSQP<Type>::set_row_group_soft, py::arg("g"), py::arg("sigma"))                                                                     \
            .def("set_row_group_rho", &PyBSQP<Type>::set_row_group_rho, py::arg("g"), py::arg("rho"))                                                                         \
            .def("set_admm_merit", &PyBSQP<Type>::set_admm_merit, py::arg("on"))                                                                                              \
            .def("set_admm_rho_adaptation", &PyBSQP<Type>::set_admm_rho_adaptation, py::arg("on"))                                                                            \
            .def("get_admm_rho_scale", &PyBSQP<Type>::get_admm_rho_scale)                                                                                                     \
            .def("add_lin_u_group", &PyBSQP<Type>::add_lin_u_group, py::arg("mech"), py::arg("C"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("cone"),               \
                 py::arg("rho"), py::arg("delta"), py::arg("sigma"), py::arg("knot_lo"), py::arg("knot_hi"), py::arg("admm_iters"), py::arg("equilibrate") = false)            \
            .def("set_collision_environment", &PyBSQP<Type>::set_collision_environment, py::arg("spheres"), py::arg("capsules"), py::arg("cuboids"), py::arg("planes"))       \
            .def("enable_collision", &PyBSQP<Type>::enable_collision, py::arg("mech"), py::arg("margin"), py::arg("rho"), py::arg("delta"), py::arg("sigma"),                 \
                 py::arg("knot_lo"), py::arg("admm_iters"))                                                                                                                   \
            .def("get_collision_row_duals", &PyBSQP<Type>::get_collision_row_duals)                                                                                          \
            .def("get_collision_admm_state", &PyBSQP<Type>::get_collision_admm_state)

PYBIND11_MODULE(MODULE_NAME(KNOT_POINTS, GATO_PLANT_NAME, GATO_MODULE_VARIANT), m)
{
        m.attr("NATIVE_SOURCE_ID") = GATO_NATIVE_SOURCE_ID;
        m.attr("KNOT_POINTS") = KNOT_POINTS;      // to check num knots for current module
        m.attr("NUM_BODIES") = grid::NUM_BODIES;  // body-major f_ext is 6*NUM_BODIES per (solve, knot)
        m.attr("NUM_COLLISION_SPHERES") = gato::plant::NCC;  // clearance rows per knot (CL-2)
        m.attr("EXACT_HESSIAN_AVAILABLE") = bool(USE_EXACT_HESSIAN);  // SO-SQP path compiled in?
        // control layout (CL-3a): CONTROL_SIZE = ACTUATED_SIZE + FC_SIZE; FC_SIZE > 0
        // only on GATO_CONTACT_FORCES builds (fc wrench slots appended after torques)
        m.attr("CONTROL_SIZE") = gato::constants::CONTROL_SIZE;
        m.attr("ACTUATED_SIZE") = gato::constants::ACTUATED_SIZE;
        m.attr("FC_SIZE") = gato::constants::FC_SIZE;
#ifdef GRID_HAS_CONTACT_FRAMES
        m.attr("NUM_CONTACT_FRAMES") = grid::NUM_CONTACT_FRAMES;   // baked wrench frames (fc slots = 6 each)
#else
        m.attr("NUM_CONTACT_FRAMES") = 0;
#endif
        // state layout (CL-3): stored xu packs [q(NQ); qd(NV); u], the KKT/dz
        // tangent state is STATE_SIZE = 2*NV. Fixed base: NQ == NV.
        m.attr("NQ") = gato::constants::NQ;
        m.attr("NV") = gato::constants::NV;
        m.attr("FLOATING_BASE") = gato::constants::FLOATING_BASE;
        // flat per-batch-row sizes the device path validates against
        m.attr("XU_TRAJ_SIZE") = XU_TRAJ_SIZE;
        m.attr("REFERENCE_TRAJ_SIZE") = REFERENCE_TRAJ_SIZE;

        // zero-copy device outputs (solve_device): torch.from_dlpack(a) / jax.dlpack.from_dlpack(a)
        py::class_<DeviceArray>(m, "DeviceArray", py::module_local())
            .def_property_readonly("shape", [](const DeviceArray& a) { return py::tuple(py::cast(a.shape)); })
            .def_property_readonly("data_ptr", &DeviceArray::data_ptr)
            .def("__dlpack__", [](const DeviceArray& a, py::kwargs) { return a.dlpack(); })
            .def("__dlpack_device__", &DeviceArray::dlpack_device)
            .def("numpy", &DeviceArray::to_numpy);

#ifdef USE_DOUBLES
        REGISTER_BSQP_CLASS(double);
#else
        REGISTER_BSQP_CLASS(float);
#endif
}
