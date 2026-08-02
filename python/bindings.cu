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
                result["linsys_mode"] = solver_.linsys_mode();

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
                        }
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

        // per-knot wrench band: (B, KNOT_POINTS, 6*NUM_BODIES) flattened
        void set_f_ext_knot_batch(py::array_t<T> f_ext_knot_batch)
        {
                py::buffer_info buf = f_ext_knot_batch.request();
                check_size(buf, (size_t)6 * grid::NUM_BODIES * KNOT_POINTS * batch_size_, "f_ext_knot_batch");
                solver_.set_f_ext_knot_batch(static_cast<T*>(buf.ptr));
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

        void enable_limit_telemetry() { solver_.enable_limit_telemetry(); }
        void enable_limit_barrier(T mu, T delta) { solver_.enable_limit_barrier(mu, delta); }
        void enable_limit_admm(T rho, uint32_t iters) { solver_.enable_limit_admm(rho, iters); }
        void enable_limit_al(T rho) { solver_.enable_limit_al(rho); }
        void enable_ee_terminal_equality(py::array_t<T> target, T rho)
        {
                py::buffer_info buf = target.request();
                if (buf.size != 3) { throw py::value_error("enable_ee_terminal_equality: target must be xyz (3 elements)"); }
                solver_.enable_ee_terminal_equality(static_cast<T*>(buf.ptr), rho);
        }
        void disable_row_groups() { solver_.disable_row_groups(); }

        // LIN_U row-group append (CL-2): C is (m, NU); d/lo/hi length m (d may be
        // empty -> zero offset; lo/hi ignored for cone rows). mech is the
        // rows::Mechanism enum value (0 telemetry, 1 barrier, 2 admm, 3 al).
        void add_lin_u_group(int32_t mech, py::array_t<T> C, py::array_t<T> d, py::array_t<T> lo, py::array_t<T> hi, bool cone, T rho, T delta, T sigma, int32_t knot_lo, int32_t knot_hi, uint32_t admm_iters, bool equilibrate)
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

        // per-solve row-state pair, dense row_state_index layout
        // (B, MAX_ROW_GROUPS, KNOT_POINTS, MAX_ROWS_PER_GROUP) per array:
        // AL duals {lam_hi, lam_lo} / ADMM state {z, y}
        py::dict get_row_duals()
        {
                auto p = row_state_pair(/*admm=*/false);
                py::dict d;
                d["lam_hi"] = p.first;
                d["lam_lo"] = p.second;
                return d;
        }
        py::dict get_admm_state()
        {
                auto p = row_state_pair(/*admm=*/true);
                py::dict d;
                d["z"] = p.first;
                d["y"] = p.second;
                return d;
        }

        py::dict get_collision_row_duals()
        {
                auto p = collision_state_pair(/*admm=*/false);
                py::dict d;
                d["lam_hi"] = p.first;
                d["lam_lo"] = p.second;
                return d;
        }
        py::dict get_collision_admm_state()
        {
                auto p = collision_state_pair(/*admm=*/true);
                py::dict d;
                d["z"] = p.first;
                d["y"] = p.second;
                return d;
        }

        // flat (n, k) float arrays per primitive: spheres (n,4) {x,y,z,r};
        // capsules (n,7) {a(3),b(3),r}; cuboids (n,15) {c(3),u(3),hu,v(3),hv,
        // w(3),hw}; planes (n,4) {n_unit(3),d} — geometry-header layouts
        void set_collision_environment(py::array_t<T> spheres, py::array_t<T> capsules, py::array_t<T> cuboids, py::array_t<T> planes)
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
        void set_admm_merit(bool on) { solver_.set_admm_merit(on); }
        void set_admm_rho_adaptation(bool on) { solver_.set_admm_rho_adaptation(on); }
        // per-solve adapted rho scale (B,): rho_effective = group rho * scale
        py::array_t<T> get_admm_rho_scale()
        {
                py::array_t<T> out((py::ssize_t)batch_size_);
                solver_.copy_admm_rho_scale_to_host(static_cast<T*>(out.request().ptr));
                return out;
        }

        void set_row_group_bounds(int32_t g, py::array_t<T> lo, py::array_t<T> hi)
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
                        d["cone"] = grp.cone;
                        d["lo"] = lo;
                        d["hi"] = hi;
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
        void set_linsys_mode(int mode) { solver_.set_linsys_mode(mode); }
        void set_admm_linsys_pcg(bool on) { solver_.set_admm_linsys_pcg(on); }
        void set_exact_hessian(bool on) { solver_.set_exact_hessian(on); }
        bool exact_hessian() const { return solver_.exact_hessian(); }

        // debug/test: KKT setup only (no solve) on the given trajectory + block readback
        py::dict debug_setup_kkt(py::array_t<T> xu_traj_batch, T timestep, py::array_t<T> x_s_batch, py::array_t<T> reference_traj_batch)
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
                solver_.debug_setup_kkt(d_xu_traj_batch_, inputs);

                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);
                py::array_t<T>    Q({B, (py::ssize_t)STATE_SQ_P_KNOTS});
                py::array_t<T>    R({B, (py::ssize_t)CONTROL_SQ_P_KNOTS});
                py::array_t<T>    q({B, (py::ssize_t)STATE_P_KNOTS});
                py::array_t<T>    r({B, (py::ssize_t)CONTROL_P_KNOTS});
                solver_.copy_kkt_blocks_to_host(static_cast<T*>(Q.request().ptr), static_cast<T*>(R.request().ptr), static_cast<T*>(q.request().ptr), static_cast<T*>(r.request().ptr));
                py::dict out;
                out["Q"] = Q;
                out["R"] = R;
                out["q"] = q;
                out["r"] = r;
                return out;
        }
        // debug/oracle: contact-wrench chain at one (q, qd, u, f_c) sample (CL-3 prep).
        // Jacobians come back as (rows, cols) numpy arrays (strided from the device
        // col-major buffers): dqdd_dfc (NQ, 6*NUM_CONTACT_FRAMES), dqdd_dq /
        // dqdd_dq_corr (NQ, NQ). dqdd_dq holds f_ext FIXED; dqdd_dq_corr is the
        // dfext/dq chain term (total = dqdd_dq + dqdd_dq_corr).
        py::dict debug_contact_dynamics(py::array_t<T> q, py::array_t<T> qd, py::array_t<T> u, py::array_t<T> fc)
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
                solver_.debug_contact_dynamics(static_cast<T*>(q_buf.ptr), static_cast<T*>(qd_buf.ptr), static_cast<T*>(u_buf.ptr), static_cast<T*>(fc_buf.ptr),
                                               qdd.data(), fext.data(), dfc.data(), dq.data(), dq_corr.data());
                const py::ssize_t sT = (py::ssize_t)sizeof(T);
                py::dict          out;
                out["qdd"] = py::array_t<T>({NQ}, qdd.data());
                out["fext"] = py::array_t<T>({FEXT}, fext.data());
                out["dqdd_dfc"] = py::array_t<T>({NQ, NFCW}, {sT, NQ * sT}, dfc.data());
                out["dqdd_dq"] = py::array_t<T>({NQ, NQ}, {sT, NQ * sT}, dq.data());
                out["dqdd_dq_corr"] = py::array_t<T>({NQ, NQ}, {sT, NQ * sT}, dq_corr.data());
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
        void set_q_nom(py::array_t<T> q_nom)
        {
                auto buf = q_nom.request();
                if (buf.size == 0) { solver_.set_q_nom(nullptr); return; }  // empty -> reset to zeros
                if (buf.size != (py::ssize_t)grid::NUM_JOINTS) throw std::runtime_error("q_nom must have NUM_JOINTS entries (or be empty to reset)");
                solver_.set_q_nom(static_cast<T*>(buf.ptr));
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
        std::pair<py::array_t<T>, py::array_t<T>> row_state_pair(bool admm)
        {
                // device buffers are TOTAL_ROW_STATE_SIZE-strided per solve (dense
                // per-group slots + the collision band); this view exposes the DENSE
                // prefix — the collision band has its own getter below
                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);
                const py::ssize_t G = (py::ssize_t)gato::rows::MAX_ROW_GROUPS;
                const py::ssize_t K = (py::ssize_t)KNOT_POINTS;
                const py::ssize_t R = (py::ssize_t)gato::rows::MAX_ROWS_PER_GROUP;
                const size_t      TOT = gato::rows::TOTAL_ROW_STATE_SIZE;
                const size_t      DENSE = gato::rows::ROW_STATE_SIZE;
                py::array_t<T>    a({B, G, K, R}), b({B, G, K, R});
                std::vector<T>    ha(TOT * batch_size_), hb(TOT * batch_size_);
                if (admm) {
                        solver_.copy_admm_state_to_host(ha.data(), hb.data());
                } else {
                        solver_.copy_row_duals_to_host(ha.data(), hb.data());
                }
                T* pa = static_cast<T*>(a.request().ptr);
                T* pb = static_cast<T*>(b.request().ptr);
                for (uint32_t s = 0; s < batch_size_; s++) {
                        memcpy(pa + (size_t)s * DENSE, ha.data() + (size_t)s * TOT, DENSE * sizeof(T));
                        memcpy(pb + (size_t)s * DENSE, hb.data() + (size_t)s * TOT, DENSE * sizeof(T));
                }
                return {a, b};
        }

        std::pair<py::array_t<T>, py::array_t<T>> collision_state_pair(bool admm)
        {
                // the collision band: (B, KNOT_POINTS, NCC) per array
                const py::ssize_t B = static_cast<py::ssize_t>(batch_size_);
                const py::ssize_t K = (py::ssize_t)KNOT_POINTS;
                const py::ssize_t S = (py::ssize_t)gato::plant::NCC;
                const size_t      TOT = gato::rows::TOTAL_ROW_STATE_SIZE;
                const size_t      DENSE = gato::rows::ROW_STATE_SIZE;
                const size_t      BAND = gato::rows::COLLISION_ROW_STATE_SIZE;
                py::array_t<T>    a({B, K, S}), b({B, K, S});
                std::vector<T>    ha(TOT * batch_size_), hb(TOT * batch_size_);
                if (admm) {
                        solver_.copy_admm_state_to_host(ha.data(), hb.data());
                } else {
                        solver_.copy_row_duals_to_host(ha.data(), hb.data());
                }
                T* pa = static_cast<T*>(a.request().ptr);
                T* pb = static_cast<T*>(b.request().ptr);
                for (uint32_t s = 0; s < batch_size_; s++) {
                        memcpy(pa + (size_t)s * BAND, ha.data() + (size_t)s * TOT + DENSE, BAND * sizeof(T));
                        memcpy(pb + (size_t)s * BAND, hb.data() + (size_t)s * TOT + DENSE, BAND * sizeof(T));
                }
                return {a, b};
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
};


// the plant name token is injected at compile time (-DGATO_PLANT_NAME=<name>)
#ifndef GATO_PLANT_NAME
#error "GATO_PLANT_NAME must be defined (plant name token for the module name)"
#endif

#define MODULE_NAME_HELPER(knot, plant) bsqpN##knot##_##plant
#define MODULE_NAME(knot, plant) MODULE_NAME_HELPER(knot, plant)

// Register the runtime-batch-size PyBSQP class for the given precision type
#define REGISTER_BSQP_CLASS(Type)                                                                                                                                                                  \
        py::class_<PyBSQP<Type>>(m, "BSQP_" #Type, py::module_local()) /* every module defines the same C++ type; a global registration collides when two solver modules load in one process */                                                                                                                                                 \
            .def(py::init<const uint32_t, const Type, const uint32_t, const Type, const uint32_t, const Type, const Type, const Type, const Type, const Type, const Type, const Type, const Type, \
                          const Type, const Type, const Type>())                                                                                                                                    \
            .def("solve", &PyBSQP<Type>::solve)                                                                                                                                                    \
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
            .def("set_linsys_mode", &PyBSQP<Type>::set_linsys_mode)                                                                                                                                    \
            .def("set_admm_linsys_pcg", &PyBSQP<Type>::set_admm_linsys_pcg, py::arg("on"))                                                                                                             \
            .def("set_exact_hessian", &PyBSQP<Type>::set_exact_hessian, py::arg("on"))                                                                                                                 \
            .def("exact_hessian", &PyBSQP<Type>::exact_hessian)                                                                                                                                        \
            .def("debug_setup_kkt", &PyBSQP<Type>::debug_setup_kkt)                                                                                                                                    \
            .def("debug_contact_dynamics", &PyBSQP<Type>::debug_contact_dynamics, py::arg("q"), py::arg("qd"), py::arg("u"), py::arg("fc"))                                                             \
            .def("get_lambda", &PyBSQP<Type>::get_lambda)                                                                                                                                              \
            .def("set_cost_weights", &PyBSQP<Type>::set_cost_weights)                                                                                                                                  \
            .def("set_q_pos_cost", &PyBSQP<Type>::set_q_pos_cost)                                                                                                                                 \
            .def("set_fc_cost", &PyBSQP<Type>::set_fc_cost)                                                                                                                                            \
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
            .def("set_row_group_bounds", &PyBSQP<Type>::set_row_group_bounds, py::arg("g"), py::arg("lo"), py::arg("hi"))                                                                            \
            .def("set_row_group_soft", &PyBSQP<Type>::set_row_group_soft, py::arg("g"), py::arg("sigma"))                                                                     \
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

PYBIND11_MODULE(MODULE_NAME(KNOT_POINTS, GATO_PLANT_NAME), m)
{
        m.attr("KNOT_POINTS") = KNOT_POINTS;      // to check num knots for current module
        m.attr("NUM_BODIES") = grid::NUM_BODIES;  // body-major f_ext is 6*NUM_BODIES per (solve, knot)
        m.attr("NUM_COLLISION_SPHERES") = gato::plant::NCC;  // clearance rows per knot (CL-2)
        m.attr("EXACT_HESSIAN_AVAILABLE") = bool(USE_EXACT_HESSIAN);  // SO-SQP path compiled in?

#ifdef USE_DOUBLES
        REGISTER_BSQP_CLASS(double);
#else
        REGISTER_BSQP_CLASS(float);
#endif
}
