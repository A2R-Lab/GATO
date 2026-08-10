#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)
#include "dynamics/integrator.cuh"
#include "bsqp/rowgroups.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;
// NOTE: no file-scope `using namespace gato::plant` — it leaks plant:: symbols into every
// other kernel header sharing this TU (caused a gato::plant::EE_POS_SIZE vs constants:: clash
// in merit.cuh). Qualify plant calls explicitly instead.

// shared-memory element count of the flag-independent kernel layout (the
// terminal-branch chain is the superset of the running-knot one): everything
// through r_dummy plus the larger of the two s_temp consumers. The exact-
// Hessian carve (below) starts at exactly this offset.
template<typename T>
__host__ __device__ constexpr uint32_t setupKKTBaseSMemCt()
{
        constexpr uint32_t temp_ct = gato::plant::trackingCostGradHess_TempMemCt<T>() > gato::plant::forwardDynamicsAndGradient_TempMemSize_Shared()
                                         ? gato::plant::trackingCostGradHess_TempMemCt<T>()
                                         : gato::plant::forwardDynamicsAndGradient_TempMemSize_Shared();
        return STATE_S_CONTROL + STATE_SIZE +  // xux_k
               2 * constants::EE_POS_SIZE +    // reference_traj_k
               STATE_SIZE_SQ +                 // Q_k
               CONTROL_SIZE_SQ +               // R_k
               STATE_SIZE +                    // q_k
               CONTROL_SIZE +                  // r_k
               STATE_SIZE_SQ +                 // A_k
               STATE_P_CONTROL +               // B_k
               STATE_SIZE +                    // c_k
               STATE_SIZE_SQ +                 // Q_last
               STATE_SIZE +                    // q_last
               CONTROL_SIZE_SQ +               // R_dummy (throwaway terminal R)
               CONTROL_SIZE +                  // r_dummy
               temp_ct;
}

#if USE_EXACT_HESSIAN
// Exact-Hessian (SO-SQP) stage-block PSD projection — PROJECT-only per the
// numpy prototype verdict (docs/open-tasks/so_sqp_prototype/RESULTS_2026-07-17).
// eps = 1e-5 * (1 + max|diag|) per block (the f32 clip floor; every thread
// computes it redundantly from shared diagonals — deterministic, no staging).
// __noinline__: these bodies land in the same TU that hit the cicc inlining
// cliff in CL-2a — keep them out of the kernel's inline blob.

// element count of the exact-Hessian shared carve: the assembled (nx+nu)^2
// stage block + a union region holding the fdsva_so arena during the
// contraction and the glass::psd_project scratch during the projection
// (the SO tensors are dead by projection time, so the regions alias)
template<typename T>
__host__ __device__ inline uint32_t setupKKTExactHessSMemCt()
{
        const uint32_t proj_ct = (uint32_t)(glass::psd_project_scratch_bytes<T, STATE_S_CONTROL>() / sizeof(T));
        const uint32_t so_ct = gato::plant::exactHessianSO_TempMemCt<T>();
        return STATE_S_CONTROL * STATE_S_CONTROL + (so_ct > proj_ct ? so_ct : proj_ct);
}

// running knot: assemble P = [[Q, 0], [0, R]] (column-major, nx+nu), add the
// lagged-lambda exact-Hessian term lam^T d2F (plant.cuh exactHessianContraction
// — fills the diagonal blocks AND the qu/qv cross blocks), project, scatter the
// diagonal blocks back. Entry: s_Q_k/s_R_k writes barrier-visible. Exit: ends
// on a barrier.
template<typename T, unsigned INTEGRATOR_TYPE>
__device__ __noinline__ void projectStageBlockExact(T* s_Q_k, T* s_R_k, const T* s_xux_k, const T* d_lambda_kp1, T dt, void* d_GRiD_mem, T* s_eh)
{
        constexpr uint32_t P_DIM = STATE_S_CONTROL;
        T*                 s_P = s_eh;
        T*                 s_union = s_eh + P_DIM * P_DIM;  // SO arena, then psd scratch
        for (uint32_t i = threadIdx.x; i < P_DIM * P_DIM; i += blockDim.x) {
                const uint32_t r = i % P_DIM, c = i / P_DIM;
                T              v = static_cast<T>(0);
                if (r < STATE_SIZE && c < STATE_SIZE) {
                        v = s_Q_k[c * STATE_SIZE + r];
                } else if (r >= STATE_SIZE && c >= STATE_SIZE) {
                        v = s_R_k[(c - STATE_SIZE) * CONTROL_SIZE + (r - STATE_SIZE)];
                }
                s_P[i] = v;
        }
        __syncthreads();
        gato::plant::exactHessianContraction<T, INTEGRATOR_TYPE>(s_P, s_xux_k, d_lambda_kp1, dt, s_union, d_GRiD_mem);  // ends on a barrier
        // eps = 1e-5 * (1 + max|diag|). ABS per the prototype's _eps_for: a
        // curvature-dominated negative diagonal must still give a POSITIVE
        // floor (a raw max went negative here and produced an indefinite
        // "projection" -> NaN cascade in the first device A/B). 1e-5 (not the
        // prototype's f64 1e-6): the clip must clear the f32 fixed-sweep Jacobi
        // reconstruction noise (<= 1.9e-6 relative, jacobi_study p95) or the
        // output is numerically indefinite at the bottom of the spectrum and
        // the bdsv factor non-PD-skips every iteration once lambda != 0
        // (measured 2026-07-30: rho 1e-5 froze, rho >= 1e-4 descended).
        T maxdiag = static_cast<T>(0);
        for (uint32_t j = 0; j < P_DIM; j++) {
                const T d = s_P[j * P_DIM + j];
                const T a = (d < static_cast<T>(0)) ? -d : d;
                maxdiag = (a > maxdiag) ? a : maxdiag;
        }
        const T eps = static_cast<T>(1e-5) * (static_cast<T>(1) + maxdiag);
        glass::psd_project<T, P_DIM>(s_P, eps, s_union);  // ends on a barrier
        for (uint32_t i = threadIdx.x; i < P_DIM * P_DIM; i += blockDim.x) {
                const uint32_t r = i % P_DIM, c = i / P_DIM;
                if (r < STATE_SIZE && c < STATE_SIZE) {
                        s_Q_k[c * STATE_SIZE + r] = s_P[i];
                } else if (r >= STATE_SIZE && c >= STATE_SIZE) {
                        s_R_k[(c - STATE_SIZE) * CONTROL_SIZE + (r - STATE_SIZE)] = s_P[i];
                }
                // off-diagonal (cross) blocks dropped: formSchur consumes Q/R only
                // (full-stage-block Schur = deferred step 1d, decide on data)
        }
        __syncthreads();
}

// terminal knot: the stage block IS Q_last (no control) — project in place.
// Entry: s_Q_last writes barrier-visible. Exit: ends on a barrier.
template<typename T>
__device__ __noinline__ void projectTerminalBlockExact(T* s_Q_last, T* s_eh)
{
        T maxdiag = static_cast<T>(0);  // eps floor on max|diag| (see the stage helper)
        for (uint32_t j = 0; j < STATE_SIZE; j++) {
                const T d = s_Q_last[j * STATE_SIZE + j];
                const T a = (d < static_cast<T>(0)) ? -d : d;
                maxdiag = (a > maxdiag) ? a : maxdiag;
        }
        const T eps = static_cast<T>(1e-5) * (static_cast<T>(1) + maxdiag);
        glass::psd_project<T, STATE_SIZE>(s_Q_last, eps, s_eh);  // ends on a barrier
}
#endif  // USE_EXACT_HESSIAN

template<typename T, uint32_t INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false>
__global__ __launch_bounds__(KKT_THREADS) void setupKKTSystemBatchedKernel(T*    d_Q_batch,
                                            T*    d_R_batch,
                                            T*    d_q_batch,
                                            T*    d_r_batch,
                                            T*    d_A_batch,
                                            T*    d_B_batch,
                                            T*    d_c_batch,
                                            T*    d_xu_traj_batch,         // X, U trajectories
                                            void* d_GRiD_mem,              // dynamics constraint TODO: can be const?
                                            T*    d_x_s_batch,             // initial state
                                            T*    d_reference_traj_batch,  // end effector position trajectory
                                            T*    d_f_ext_batch,
                                            T     timestep,
                                            T     q_cost,
                                            T     qd_cost,
                                            T     u_cost,
                                            T     N_cost,
                                            T     q_lim_cost,
                                            T     vel_lim_cost,
                                            T     ctrl_lim_cost,
                                            T     q_pos_cost,                                 // joint-posture anchor weight (0 = off)
                                            const T* __restrict__       d_q_nom,              // posture target (nullable = zeros)
                                            T     fc_cost,                                    // contact-force regularization (GATO_CONTACT_FORCES)
                                            const T* __restrict__       d_u_cost_vec,         // per-joint effort weights (nullable; overrides u_cost)
                                            const T* __restrict__       d_q_pos_w_vec,        // per-joint anchor weights (nullable; overrides q_pos_cost)
                                            const T* __restrict__       d_fc_ref,             // contact-wrench reference (nullable = zeros; GATO_CONTACT_FORCES)
                                            const int32_t* __restrict__ d_kkt_converged_batch,
                                            const T* __restrict__       d_knot_cost_weights,  // optional (nullable) per-knot [ee,qd,u] triples
                                            const gato::rows::RowGroupDesc<T>* __restrict__ d_row_groups,  // constraint row-groups (MECH_BARRIER_RELAXED /
                                            int32_t                     n_row_groups,                      // MECH_AL fold into the cost blocks; 0 -> untouched path)
                                            const T* __restrict__       d_lam_hi_batch,                    // per-solve AL duals (nullable; MECH_AL only)
                                            const T* __restrict__       d_lam_lo_batch,
                                            const grid_collision::Environment<T> env,                      // obstacle set (COLLISION rows; empty default = no-op)
                                            const T* __restrict__       d_admm_rho_scale_batch             // per-solve ADMM rho adaptation scale (nullable -> 1)
#if USE_EXACT_HESSIAN
                                            ,
                                            int32_t                     exact_hessian,   // runtime per-TASK toggle (0 = GN path)
                                            const T* __restrict__       d_lambda_batch   // lagged Schur multipliers (lam^T d2F term)
#endif
)
{
        // kernel launched with 2D grid: (knot_idx, solve_idx)
        const uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch[solve_idx]) return;  // converged solve: freeze (no KKT rebuild)

        const T* d_lam_hi = d_lam_hi_batch ? d_lam_hi_batch + (size_t)solve_idx * gato::rows::TOTAL_ROW_STATE_SIZE : nullptr;
        const T* d_lam_lo = d_lam_lo_batch ? d_lam_lo_batch + (size_t)solve_idx * gato::rows::TOTAL_ROW_STATE_SIZE : nullptr;
        const T admm_rho_scale = d_admm_rho_scale_batch ? d_admm_rho_scale_batch[solve_idx] : static_cast<T>(1);

        extern __shared__ T s_mem[];
        T*                  s_xux_k = s_mem;  // x_k, u_k, x_k+1
        T*                  s_reference_traj_k = s_xux_k + 2 * STATE_SIZE + CONTROL_SIZE;
        T*                  s_Q_k = s_reference_traj_k + 2 * constants::EE_POS_SIZE;
        T*                  s_R_k = s_Q_k + STATE_SIZE_SQ;
        T*                  s_q_k = s_R_k + CONTROL_SIZE_SQ;
        T*                  s_r_k = s_q_k + STATE_SIZE;
        T*                  s_A_k = s_r_k + CONTROL_SIZE;
        T*                  s_B_k = s_A_k + STATE_SIZE_SQ;
        T*                  s_c_k = s_B_k + STATE_P_CONTROL;  // integrator error
        T*                  s_temp = s_c_k + STATE_SIZE;
#if USE_EXACT_HESSIAN
        // past BOTH branch layouts (running s_temp AND the terminal Q_last..r_dummy+s_temp chain)
        T* s_eh = s_mem + setupKKTBaseSMemCt<T>();
        // collision carve sits past the exact carve when that is live; only
        // dereferenced when a COLLISION group is active (host sized it then)
        T* s_cc = s_eh + (exact_hessian ? setupKKTExactHessSMemCt<T>() : 0);
#else
        T* s_cc = s_mem + setupKKTBaseSMemCt<T>();
#endif


        for (uint32_t knot_idx = blockIdx.x; knot_idx < KNOT_POINTS - 1; knot_idx += gridDim.x) {

                // Input pointers
                T* d_xu_traj_k = getOffsetXU<T>(d_xu_traj_batch, solve_idx, knot_idx);
                T* d_reference_traj_k = getOffsetReferenceTraj<T>(d_reference_traj_batch, solve_idx, knot_idx);
                T* d_f_ext = getOffsetWrench<T>(d_f_ext_batch, solve_idx, knot_idx);

                // Output pointers
                T* d_Q_k = getOffsetStateSq<T>(d_Q_batch, solve_idx, knot_idx);
                T* d_R_k = getOffsetControlSq<T>(d_R_batch, solve_idx, knot_idx);
                T* d_q_k = getOffsetState<T>(d_q_batch, solve_idx, knot_idx);
                T* d_r_k = getOffsetControl<T>(d_r_batch, solve_idx, knot_idx);
                T* d_A_k = getOffsetStateSq<T>(d_A_batch, solve_idx, knot_idx);
                T* d_B_k = getOffsetStatePControl<T>(d_B_batch, solve_idx, knot_idx);
                T* d_c_k = getOffsetState<T>(d_c_batch, solve_idx, knot_idx + 1);  // c_k+1 = e_k

                // per-knot weight override (nullptr -> the scalar weights)
                const T ee_w_k  = d_knot_cost_weights ? d_knot_cost_weights[knot_idx * 3 + 0] : q_cost;
                const T qd_w_k  = d_knot_cost_weights ? d_knot_cost_weights[knot_idx * 3 + 1] : qd_cost;
                const T u_w_k   = d_knot_cost_weights ? d_knot_cost_weights[knot_idx * 3 + 2] : u_cost;
                const T ee_w_kp1 = d_knot_cost_weights ? d_knot_cost_weights[(knot_idx + 1) * 3 + 0] : N_cost;

                glass::copy<T, STATE_S_CONTROL + STATE_SIZE>(d_xu_traj_k, s_xux_k);
                glass::copy<T, 2 * constants::EE_POS_SIZE>(d_reference_traj_k, s_reference_traj_k);
                __syncthreads();

                gato::plant::compute_linearized_dynamics<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(s_xux_k, s_A_k, s_B_k, s_c_k, s_temp, d_GRiD_mem, timestep, d_f_ext);
                __syncthreads();

                glass::copy<T, STATE_SIZE_SQ>(s_A_k, d_A_k);
                glass::copy<T, STATE_P_CONTROL>(s_B_k, d_B_k);
                glass::copy<T, STATE_SIZE>(s_c_k, d_c_k);  // c_k+1 = e_k

                const grid::robotModel<T>* d_robotModel = (const grid::robotModel<T>*)d_GRiD_mem;
                if (knot_idx < KNOT_POINTS - 2) {

                        // running knot k: EE weight q_cost, at state x_k = s_xux_k.
                        gato::plant::trackingCostGradHess<T>(
                            s_xux_k, s_xux_k + STATE_SIZE, s_reference_traj_k,
                            s_Q_k, s_q_k, s_R_k, s_r_k, s_temp, d_robotModel,
                            qd_w_k, u_w_k, q_lim_cost, vel_lim_cost, ctrl_lim_cost, /*ee_weight=*/ee_w_k, q_pos_cost, d_q_nom, fc_cost, d_u_cost_vec, d_q_pos_w_vec, d_fc_ref);

                } else {  // compute Q_last, q_last, and c_0 as well for the last knot point

                        T* s_Q_last  = s_c_k + STATE_SIZE;
                        T* s_q_last  = s_Q_last + STATE_SIZE_SQ;
                        T* s_R_dummy = s_q_last + STATE_SIZE;          // throwaway terminal R (no control at x_{k+1})
                        T* s_r_dummy = s_R_dummy + CONTROL_SIZE_SQ;
                        s_temp       = s_r_dummy + CONTROL_SIZE;

                        // running knot k: EE weight q_cost, at state x_k.
                        gato::plant::trackingCostGradHess<T>(
                            s_xux_k, s_xux_k + STATE_SIZE, s_reference_traj_k,
                            s_Q_k, s_q_k, s_R_k, s_r_k, s_temp, d_robotModel,
                            qd_w_k, u_w_k, q_lim_cost, vel_lim_cost, ctrl_lim_cost, /*ee_weight=*/ee_w_k, q_pos_cost, d_q_nom, fc_cost, d_u_cost_vec, d_q_pos_w_vec, d_fc_ref);
                        __syncthreads();

                        // terminal knot k+1: EE weight N_cost, at state x_{k+1} (PR #17 fix:
                        // the OLD _lastblock used x_k + q_cost here, so N_cost had no effect).
                        // R block discarded (the terminal state has no control).
                        T* s_xkp1 = s_xux_k + STATE_SIZE + CONTROL_SIZE;
                        gato::plant::trackingCostGradHess<T>(
                            s_xkp1, s_xkp1, &s_reference_traj_k[constants::EE_POS_SIZE],
                            s_Q_last, s_q_last, s_R_dummy, s_r_dummy, s_temp, d_robotModel,
                            qd_w_k, u_w_k, q_lim_cost, vel_lim_cost, ctrl_lim_cost, /*ee_weight=*/ee_w_kp1, q_pos_cost, d_q_nom, fc_cost, d_u_cost_vec, d_q_pos_w_vec, d_fc_ref);

                        // constraint row-groups on the TERMINAL knot's state block
                        // (no control there; the c_0 __syncthreads below covers the writes)
                        if (n_row_groups > 0) {
                                gato::rows::apply_row_grad_hess<T>(d_row_groups, n_row_groups, (int32_t)(KNOT_POINTS - 1), s_xkp1, d_lam_hi, d_lam_lo, s_Q_last, s_q_last, s_R_dummy, s_r_dummy, /*has_control=*/false, admm_rho_scale);
                                // EE_POS rows (cooperative FK; dense J^T J fold — v1 installs
                                // them terminal-only, so this is their single fold site).
                                // s_temp is free here (trackingCostGradHess above is done) and
                                // trackingCostGradHess_TempMemCt >= the EE grad carve.
                                if (gato::rows::has_ee_rows<T>(d_row_groups, n_row_groups, (int32_t)(KNOT_POINTS - 1))) {
                                        __syncthreads();  // s_Q_last/s_q_last selection writes above
                                        gato::rows::apply_ee_row_grad_hess<T>(d_row_groups, n_row_groups, (int32_t)(KNOT_POINTS - 1), s_xkp1, d_lam_hi, d_lam_lo, s_Q_last, s_q_last, s_temp, d_robotModel, admm_rho_scale);
                                }
                                // COLLISION rows on the terminal state block (dense
                                // q-half fold; dedicated carve — s_temp is too small)
                                if (gato::rows::has_collision_rows<T>(d_row_groups, n_row_groups, (int32_t)(KNOT_POINTS - 1))) {
                                        __syncthreads();
                                        gato::rows::apply_collision_row_grad_hess<T>(d_row_groups, n_row_groups, (int32_t)(KNOT_POINTS - 1), s_xkp1, d_lam_hi, d_lam_lo, s_Q_last, s_q_last, s_cc, d_robotModel, env, admm_rho_scale);
                                }
                        }

                        // c_0 = x_0 - x_s (CL-3 floating: manifold difference — see SSOT)
                        T* d_c_0 = getOffsetState<T>(d_c_batch, solve_idx, 0);
                        T* d_xu_0 = getOffsetXU<T>(d_xu_traj_batch, solve_idx, 0);
                        glass::axpby<T, STATE_SIZE>(static_cast<T>(1), d_xu_0, static_cast<T>(-1), d_x_s_batch + solve_idx * STATE_SIZE, d_c_0);
                        __syncthreads();

#if USE_EXACT_HESSIAN
                        if (exact_hessian) { projectTerminalBlockExact<T>(s_Q_last, s_eh); }
#endif
                        glass::copy<T, STATE_SIZE_SQ>(s_Q_last, d_Q_k + STATE_SIZE_SQ);
                        glass::copy<T, STATE_SIZE>(s_q_last, d_q_k + STATE_SIZE);
                }

                // constraint row-groups fold into this knot's cost blocks
                // (both branches: trackingCostGradHess ended on a barrier, and the
                // copies below need one after our strided diagonal writes)
                if (n_row_groups > 0) {
                        gato::rows::apply_row_grad_hess<T>(d_row_groups, n_row_groups, (int32_t)knot_idx, s_xux_k, d_lam_hi, d_lam_lo, s_Q_k, s_q_k, s_R_k, s_r_k, /*has_control=*/true, admm_rho_scale);
                        __syncthreads();
                        // COLLISION rows fold at EVERY active knot (clearance is a
                        // per-knot state constraint, unlike the terminal-only EE rows)
                        if (gato::rows::has_collision_rows<T>(d_row_groups, n_row_groups, (int32_t)knot_idx)) {
                                gato::rows::apply_collision_row_grad_hess<T>(d_row_groups, n_row_groups, (int32_t)knot_idx, s_xux_k, d_lam_hi, d_lam_lo, s_Q_k, s_q_k, s_cc, d_robotModel, env, admm_rho_scale);
                        }
                }

#if USE_EXACT_HESSIAN
                if (exact_hessian) {
                        const T* d_lambda_kp1 = getOffsetStatePadded<T>(d_lambda_batch, solve_idx, knot_idx + 1);
                        projectStageBlockExact<T, INTEGRATOR_TYPE>(s_Q_k, s_R_k, s_xux_k, d_lambda_kp1, timestep, d_GRiD_mem, s_eh);
                }
#endif
                glass::copy<T, STATE_SIZE_SQ>(s_Q_k, d_Q_k);
                glass::copy<T, CONTROL_SIZE_SQ>(s_R_k, d_R_k);
                glass::copy<T, STATE_SIZE>(s_q_k, d_q_k);
                glass::copy<T, CONTROL_SIZE>(s_r_k, d_r_k);
        }
}

template<typename T>
__host__ size_t getSetupKKTSystemBatchedSMemSize(int exact_hessian = 0, int has_collision = 0)
{
        size_t size = sizeof(T) * setupKKTBaseSMemCt<T>();
#if USE_EXACT_HESSIAN
        // runtime-sized: the exact carve is only requested when the toggle is on,
        // so exact builds running the GN path keep the small launch (occupancy)
        if (exact_hessian) { size += sizeof(T) * setupKKTExactHessSMemCt<T>(); }
#else
        (void)exact_hessian;
#endif
        // runtime-sized like the exact carve: only when a COLLISION group is
        // registered (host-known); the default path keeps the small launch
        if (has_collision) { size += sizeof(T) * gato::rows::collision_rows_grad_scratch_ct<T>(); }
        return size;
}

template<typename T>
__host__ void setupKKTSystemBatched(uint32_t batch_size, KKTSystem<T> kkt, ProblemInputs<T> inputs, T* d_xu_traj_batch, T* d_f_ext_batch, void* d_GRiD_mem, T q_cost, T qd_cost, T u_cost, T N_cost, T q_lim_cost, T vel_lim_cost, T ctrl_lim_cost, const int32_t* d_kkt_converged_batch, const T* d_knot_cost_weights, const gato::rows::RowGroupDesc<T>* d_row_groups = nullptr, int32_t n_row_groups = 0, const T* d_lam_hi_batch = nullptr, const T* d_lam_lo_batch = nullptr, int32_t exact_hessian = 0, const T* d_lambda_batch = nullptr, int32_t has_collision = 0, const grid_collision::Environment<T>& env = grid_collision::Environment<T>{}, const T* d_admm_rho_scale_batch = nullptr, T q_pos_cost = 0, const T* d_q_nom = nullptr, T fc_cost = 0, const T* d_u_cost_vec = nullptr, const T* d_q_pos_w_vec = nullptr, const T* d_fc_ref = nullptr)
{
#if !USE_EXACT_HESSIAN
        (void)exact_hessian;  // path compiled out (settings.h USE_EXACT_HESSIAN)
        (void)d_lambda_batch;
#endif
        dim3   grid(KNOT_POINTS - 1, batch_size);  // kernel loop covers knots [0, K-2]; K blocks left one row idle
        dim3   block(KKT_THREADS);
        size_t s_mem_size = getSetupKKTSystemBatchedSMemSize<T>(exact_hessian, has_collision);  // TODO: why is MPCGPU launched with 2 * s_mem_size ?
        // the exact / collision carves can exceed the 48KB default dynamic-smem
        // ceiling — opt the kernel in once (harmless when already under)
        if (s_mem_size > 48 * 1024) {
                static bool attr_set = false;
                if (!attr_set) {
                        cudaFuncSetAttribute(setupKKTSystemBatchedKernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s_mem_size);
                        attr_set = true;
                }
        }

        setupKKTSystemBatchedKernel<T><<<grid, block, s_mem_size>>>(kkt.d_Q_batch,
                                                                               kkt.d_R_batch,
                                                                               kkt.d_q_batch,
                                                                               kkt.d_r_batch,
                                                                               kkt.d_A_batch,
                                                                               kkt.d_B_batch,
                                                                               kkt.d_c_batch,
                                                                               d_xu_traj_batch,
                                                                               d_GRiD_mem,
                                                                               inputs.d_x_s_batch,
                                                                               inputs.d_reference_traj_batch,
                                                                               d_f_ext_batch,
                                                                               inputs.timestep,
                                                                               q_cost,
                                                                               qd_cost,
                                                                               u_cost,
                                                                               N_cost,
                                                                               q_lim_cost,
                                                                               vel_lim_cost,
                                                                               ctrl_lim_cost,
                                                                               q_pos_cost,
                                                                               d_q_nom,
                                                                               fc_cost,
                                                                               d_u_cost_vec,
                                                                               d_q_pos_w_vec,
                                                                               d_fc_ref,
                                                                               d_kkt_converged_batch,
                                                                               d_knot_cost_weights,
                                                                               d_row_groups,
                                                                               n_row_groups,
                                                                               d_lam_hi_batch,
                                                                               d_lam_lo_batch,
                                                                               env,
                                                                               d_admm_rho_scale_batch
#if USE_EXACT_HESSIAN
                                                                               ,
                                                                               exact_hessian,
                                                                               d_lambda_batch
#endif
        );
}
