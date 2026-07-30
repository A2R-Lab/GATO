#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "types.cuh"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)
#include "bsqp/rowgroups.cuh"

// MECH_ADMM inner loop (constraint-layer arc CL-1) — OSQP-style splitting of
// the SQP subproblem's interval rows, with the dynamics equality handled
// EXACTLY inside every x-update via the existing Schur solve:
//
//   per SQP iteration (fixed rho = grp.mu):
//     setup_kkt folds rho*G^T*G into Q/R (rowgroups.cuh) -> formSchur once ->
//     bdsv factor ONCE, then K ADMM iterations of
//       q/r  <- base + G^T(y - rho*(z - sel(x)))      [admmGradientBatchedKernel]
//       gamma <- rebuild from stored Q^-1/R^-1          [computeGammaBatched]
//       lambda <- factored re-solve                     [solveBDSVFactoredBatched]
//       dz    <- recover                                [computeDzBatched]
//       w = sel(x + dz); z <- clip(w + y/rho, lo, hi); y += rho*(w - z)
//                                                       [admmProjectDualBatchedKernel]
//
// z/y live in ABSOLUTE row space (z tracks sel(x+dz), bounds are the
// descriptor's lo/hi directly); the numpy oracle for these exact updates is
// test/oracles/mechanisms.py::admm_interval (relative-space equivalent).
// (z, y) persist across SQP iterations and solves (dual warm start);
// reset via the init kernel (z = clip(sel(x_warm)), y = 0).
//
// EE_POS rows (constraint-layer arc CL-1, cooperative-FK kind) ride the same
// loop with g(x) in place of sel(x), LINEARIZED inside the inner loop:
//   init:     z = clip(g(x_warm))            [cooperative eePos]
//   gradient: q += J^T (y - rho*(z - g(x)))  [eePosGrad; the rho*J^T*J
//             Hessian half is folded once by setup_kkt (apply_ee_row_grad_hess)]
//   project:  w = g(x) + J*dz_q  (first-order; exact for selection rows),
//             then the identical clip / dual update / residuals.
// g and J are evaluated at the SQP linearization point x, which is CONSTANT
// across the inner loop — consistent with the frozen Schur factor.
//
// LIN_U rows (CL-2, g = C·u + d) ride the same loop through a dense C^T
// scatter in the gradient kernel (the rho*C^T*C Hessian half is folded once
// by setup_kkt). Interval LIN_U keeps the per-row clip; cone LIN_U
// (grp.cone) swaps the z-update for the SOC projection z = Π_K(w + y/rho)
// per knot vector (soft sigma: segment blend toward Π_K, the interval
// smoothed-projection analogue). Numpy twin: mechanisms.py::admm_soc.
//
// EQUALITY rows (lo == hi) reinitialize (z, y) at EVERY solve (the init
// kernel's eq_rows_only mode): z clips to the degenerate interval (z == lo,
// no information), so a cross-solve warm-started y is a pure violation
// integrator — MEASURED on the EE terminal equality: when the primal parks
// (line-search/linearization-limited pocket), persistent y grows linearly
// without bound (|y| ~ 120 after 11 solves at rho=10) and degrades the
// solution; per-solve reinit bounds y at within-solve scale and the
// violation stops drifting. The within-solve splitting is untouched;
// interval rows keep their dual warm start (they wind to a steady
// multiplier because the primal CAN satisfy them).
//
// v1 semantics (documented in the arc plan): fixed iteration budget
// (approximately-hard); residuals reported per solve, no in-loop early exit;
// the merit/line search is UNCHANGED (no constraint term for ADMM rows —
// telemetry reports the true violation of whatever the line search accepts).

using namespace sqp;

namespace gato {
namespace rows {

constexpr uint32_t ADMM_THREADS = 128;

// z-update of one row: hard clamp to [lo, hi], or (sigma > 0) the SOFT
// smoothed projection — argmin_z (rho/2)(z - v)^2 + (sigma/2) dist(z,[lo,hi])^2
// = a rho/(rho+sigma)-slope pull past the bound instead of a clamp
// (sigma -> inf recovers the clamp; the fixed-point y is bounded by sigma).
template<typename T>
__device__ __forceinline__ T admm_z_update(T v, T lo, T hi, T rho, T sigma)
{
        if (sigma > static_cast<T>(0)) {
                if (v > hi) return hi + rho * (v - hi) / (rho + sigma);
                if (v < lo) return lo + rho * (v - lo) / (rho + sigma);
                return v;
        }
        if (v > hi) return hi;
        if (v < lo) return lo;
        return v;
}

// value of one row at (xu + dz) — selection/LIN rows are affine, so eval plus
// the DIRECTIONAL part (eval_row_dir drops the LIN_U constant offset, which
// would otherwise double-count)
template<typename T>
__device__ __forceinline__ T eval_row_stepped(const RowGroupDesc<T>& grp, const T* xu_k, const T* dz_k, uint32_t i)
{
        return eval_row<T>(grp, xu_k, i) + eval_row_dir<T>(grp, dz_k, i);
}

// ---- z/y (re)initialization ---------------------------------------------
// z = clip(sel(x_warm), lo, hi), y = 0 — a neutral start: with y = 0 and
// z = sel(x) (feasible x) the first gradient modification vanishes.
// eq_rows_only: touch ONLY equality rows (lo == hi; z = lo, y = 0), leaving
// interval rows' warm-started state intact — fired every solve (see header).
template<typename T>
__global__ __launch_bounds__(ADMM_THREADS) void admmInitStateBatchedKernel(T* __restrict__ d_z_batch,
                                                                           T* __restrict__ d_y_batch,
                                                                           const T* __restrict__ d_xu_traj_batch,
                                                                           const RowGroupDesc<T>* __restrict__ d_groups,
                                                                           int32_t n_groups,
                                                                           const grid::robotModel<T>* __restrict__ d_robot_model,
                                                                           bool eq_rows_only)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        extern __shared__ char s_raw_init[];
        T* s_ee_scratch = reinterpret_cast<T*>(s_raw_init);  // ee_rows_scratch_ct
        T* d_z = d_z_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        T* d_y = d_y_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        if (eq_rows_only) {
                // z clips to the degenerate interval regardless of the primal —
                // no FK needed even for EE rows
                for (int32_t gi = 0; gi < n_groups; gi++) {
                        const RowGroupDesc<T>& grp = d_groups[gi];
                        if (grp.mech != MECH_ADMM) continue;
                        const int32_t n_elems = (grp.knot_hi - grp.knot_lo) * grp.n_rows;
                        for (int32_t e = rank; e < n_elems; e += size) {
                                const int32_t knot = grp.knot_lo + e / grp.n_rows;
                                const int32_t i = e % grp.n_rows;
                                if (grp.lo[i] != grp.hi[i]) continue;
                                const uint32_t idx = row_state_index(gi, knot, i);
                                d_z[idx] = grp.lo[i];
                                d_y[idx] = static_cast<T>(0);
                        }
                }
                return;
        }

        for (uint32_t e = rank; e < ROW_STATE_SIZE; e += size) { d_y[e] = static_cast<T>(0); d_z[e] = static_cast<T>(0); }
        __syncthreads();
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_ADMM) continue;
                if (grp.kind == EE_POS) {
                        // cooperative FK per knot (barriers inside — ALL threads)
                        for (int32_t knot = grp.knot_lo; knot < grp.knot_hi; knot++) {
                                const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                                const T* s_pose = ee_eval_pose<T>(xu_k, s_ee_scratch, d_robot_model);
                                for (int32_t i = rank; i < grp.n_rows; i += size) {
                                        T z = s_pose[i];
                                        if (z > grp.hi[i]) z = grp.hi[i];
                                        if (z < grp.lo[i]) z = grp.lo[i];
                                        d_z[row_state_index(gi, knot, i)] = z;
                                }
                                __syncthreads();  // s_ee_scratch reused next knot
                        }
                        continue;
                }
                if (grp.kind == LIN_U && grp.cone) {
                        // z = Π_K(g(x_warm)) per knot (one owner thread per knot)
                        const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                        for (int32_t k = rank; k < n_knots; k += size) {
                                const int32_t knot = grp.knot_lo + k;
                                const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                                T g[MAX_ROWS_PER_GROUP];
                                for (int32_t i = 0; i < grp.n_rows; i++) { g[i] = eval_row<T>(grp, xu_k, (uint32_t)i); }
                                glass::thread::soc_project<T>(g, g, grp.n_rows);
                                for (int32_t i = 0; i < grp.n_rows; i++) { d_z[row_state_index(gi, knot, i)] = g[i]; }
                        }
                        continue;
                }
                const int32_t n_elems = (grp.knot_hi - grp.knot_lo) * grp.n_rows;
                for (int32_t e = rank; e < n_elems; e += size) {
                        const int32_t knot = grp.knot_lo + e / grp.n_rows;
                        const int32_t i = e % grp.n_rows;
                        const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                        const T w = eval_row<T>(grp, xu_k, (uint32_t)i);
                        T z = w;
                        if (z > grp.hi[i]) z = grp.hi[i];
                        if (z < grp.lo[i]) z = grp.lo[i];
                        d_z[row_state_index(gi, knot, i)] = z;
                }
        }
}

// ---- per-iteration gradient rebuild --------------------------------------
// q/r <- base + scatter of (y - rho*(z - sel(x))) onto the row targets.
// (z - sel(x)) is the RELATIVE auxiliary the OSQP form penalizes; see header.
template<typename T>
__global__ __launch_bounds__(ADMM_THREADS) void admmGradientBatchedKernel(T* __restrict__ d_q_batch,
                                                                          T* __restrict__ d_r_batch,
                                                                          const T* __restrict__ d_q_base_batch,
                                                                          const T* __restrict__ d_r_base_batch,
                                                                          const T* __restrict__ d_xu_traj_batch,
                                                                          const T* __restrict__ d_z_batch,
                                                                          const T* __restrict__ d_y_batch,
                                                                          const RowGroupDesc<T>* __restrict__ d_groups,
                                                                          int32_t n_groups,
                                                                          const int32_t* __restrict__ d_kkt_converged_batch,
                                                                          const grid::robotModel<T>* __restrict__ d_robot_model)
{
        // grid (KNOT_POINTS, batch_size)
        const uint32_t knot_idx = blockIdx.x;
        const uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch && d_kkt_converged_batch[solve_idx]) return;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        extern __shared__ char s_raw_grad[];
        T* s_ee_scratch = reinterpret_cast<T*>(s_raw_grad);  // ee_rows_grad_scratch_ct

        T*       d_q_k = getOffsetState<T>(d_q_batch, solve_idx, knot_idx);
        T*       d_r_k = getOffsetControl<T>(d_r_batch, solve_idx, knot_idx);
        const T* d_q_base_k = getOffsetState<T>(d_q_base_batch, solve_idx, knot_idx);
        const T* d_r_base_k = getOffsetControl<T>(d_r_base_batch, solve_idx, knot_idx);
        const T* d_xu_k = getOffsetTraj<T>(d_xu_traj_batch, solve_idx, knot_idx);
        const T* d_z = d_z_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        const T* d_y = d_y_batch + (size_t)solve_idx * ROW_STATE_SIZE;

        for (uint32_t i = rank; i < STATE_SIZE; i += size) { d_q_k[i] = d_q_base_k[i]; }
        for (uint32_t i = rank; i < CONTROL_SIZE; i += size) { d_r_k[i] = d_r_base_k[i]; }
        __syncthreads();

        const bool has_control = (knot_idx < KNOT_POINTS - 1);
        constexpr int32_t NQ = constants::STATE_SIZE / 2;
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_ADMM) continue;
                if ((int32_t)knot_idx < grp.knot_lo || (int32_t)knot_idx >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                if (grp.kind == EE_POS) {
                        // cooperative pose + J, then the dense J^T scatter onto the q
                        // half: q += sum_i J_i * (y_i - rho*(z_i - g_i(x))). Same
                        // scratch carve as apply_ee_row_grad_hess (s_mod in the gr slot).
                        T* s_pose = s_ee_scratch;
                        T* s_grad = s_pose + 6 * gato::plant::NEE;
                        T* s_mod = s_grad + 6 * NQ * gato::plant::NEE;
                        T* s_arena = align16_ptr<T>(s_mod + 2 * MAX_ROWS_PER_GROUP);
                        gato::plant::eePosGrad<T>(s_pose, s_grad, d_xu_k, s_arena, d_robot_model);
                        for (int32_t i = rank; i < grp.n_rows; i += size) {
                                const uint32_t idx = row_state_index(gi, knot_idx, i);
                                s_mod[i] = d_y[idx] - grp.mu * (d_z[idx] - s_pose[i]);
                        }
                        __syncthreads();
                        for (int32_t qi = rank; qi < NQ; qi += size) {
                                T acc = static_cast<T>(0);
                                for (int32_t i = 0; i < grp.n_rows; i++) { acc += s_mod[i] * s_grad[6 * qi + i]; }
                                d_q_k[qi] += acc;
                        }
                        __syncthreads();
                        continue;
                }
                if (grp.kind == LIN_U) {
                        // dense C^T scatter onto the control gradient:
                        // r += sum_i C[i,:] * (y_i - rho*(z_i - g_i(x))) — identical
                        // splitting form for interval and cone rows (only the
                        // z-projection differs, in the project kernel)
                        for (int32_t a = rank; a < (int32_t)CONTROL_SIZE; a += size) {
                                T acc = static_cast<T>(0);
                                for (int32_t i = 0; i < grp.n_rows; i++) {
                                        const uint32_t idx = row_state_index(gi, knot_idx, (uint32_t)i);
                                        const T mod = d_y[idx] - grp.mu * (d_z[idx] - eval_row<T>(grp, d_xu_k, (uint32_t)i));
                                        acc += grp.Cmat[i * CONTROL_SIZE + a] * mod;
                                }
                                d_r_k[a] += acc;
                        }
                        __syncthreads();
                        continue;
                }
                for (int32_t i = rank; i < grp.n_rows; i += size) {
                        const uint32_t idx = row_state_index(gi, knot_idx, i);
                        const T z_rel = d_z[idx] - eval_row<T>(grp, d_xu_k, (uint32_t)i);
                        const T mod = d_y[idx] - grp.mu * z_rel;  // mu = rho
                        if (grp.block == BLOCK_X) {
                                d_q_k[x_index_of_row<T>(grp, i)] += mod;
                        } else {
                                d_r_k[i] += mod;
                        }
                }
                __syncthreads();  // groups may share targets in the future; order them
        }
}

// ---- projection + dual update + residuals --------------------------------
// One block per solve. Per active row: w = sel(x + dz); z <- clip(w + y/rho);
// y += rho*(w - z). Residuals (per solve): r_prim = max|w - z|,
// r_dual = rho * max|z - z_prev| — deterministic (scratch + rank-0 fold, same
// pattern as the telemetry kernel). Output d_resid[solve*2 + {0,1}].
template<typename T>
__global__ __launch_bounds__(ADMM_THREADS) void admmProjectDualBatchedKernel(T* __restrict__ d_z_batch,
                                                                             T* __restrict__ d_y_batch,
                                                                             T* __restrict__ d_resid_batch,
                                                                             const T* __restrict__ d_xu_traj_batch,
                                                                             const T* __restrict__ d_dz_batch,
                                                                             const RowGroupDesc<T>* __restrict__ d_groups,
                                                                             int32_t n_groups,
                                                                             const int32_t* __restrict__ d_kkt_converged_batch,
                                                                             const grid::robotModel<T>* __restrict__ d_robot_model)
{
        const uint32_t solve_idx = blockIdx.x;
        if (d_kkt_converged_batch && d_kkt_converged_batch[solve_idx]) return;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        constexpr int32_t NQ = constants::STATE_SIZE / 2;

        extern __shared__ char s_raw[];
        T* s_prim = reinterpret_cast<T*>(s_raw);              // KNOT_POINTS * MAX_ROWS_PER_GROUP
        T* s_dual = s_prim + KNOT_POINTS * MAX_ROWS_PER_GROUP;
        T* s_ee_scratch = s_dual + KNOT_POINTS * MAX_ROWS_PER_GROUP;  // ee_rows_grad_scratch_ct

        T*       d_z = d_z_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        T*       d_y = d_y_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;
        const T* d_dz = d_dz_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        T run_prim = static_cast<T>(0), run_dual = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_ADMM) continue;
                const int32_t n_elems = (grp.knot_hi - grp.knot_lo) * grp.n_rows;
                if (grp.kind == EE_POS) {
                        // linearized step value w = g(x) + J*dz_q (position rows -> the
                        // q half of dz); cooperative eePosGrad per knot, then the
                        // identical clip / dual update / residual writes
                        T* s_pose = s_ee_scratch;
                        T* s_grad = s_pose + 6 * gato::plant::NEE;
                        T* s_arena = align16_ptr<T>(s_grad + 6 * NQ * gato::plant::NEE + 2 * MAX_ROWS_PER_GROUP);
                        for (int32_t knot = grp.knot_lo; knot < grp.knot_hi; knot++) {
                                const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                                const T* dz_k = d_dz + (size_t)knot * constants::STATE_S_CONTROL;
                                gato::plant::eePosGrad<T>(s_pose, s_grad, xu_k, s_arena, d_robot_model);
                                for (int32_t i = rank; i < grp.n_rows; i += size) {
                                        const int32_t e = (knot - grp.knot_lo) * grp.n_rows + i;
                                        const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                        T w = s_pose[i];
                                        for (int32_t qi = 0; qi < NQ; qi++) { w += s_grad[6 * qi + i] * dz_k[qi]; }
                                        const T z_prev = d_z[idx];
                                        const T z = admm_z_update<T>(w + d_y[idx] / grp.mu, grp.lo[i], grp.hi[i], grp.mu, grp.sigma);
                                        d_y[idx] += grp.mu * (w - z);
                                        d_z[idx] = z;
                                        T dp = w - z;
                                        if (dp < 0) dp = -dp;
                                        T dd = grp.mu * (z - z_prev);
                                        if (dd < 0) dd = -dd;
                                        s_prim[e] = dp;
                                        s_dual[e] = dd;
                                }
                                __syncthreads();  // scratch reused next knot; writes visible below
                        }
                } else if (grp.cone) {  // LIN_U cone: SOC projection per knot vector
                        const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                        const int32_t m = grp.n_rows;
                        for (int32_t k = rank; k < n_knots; k += size) {  // one owner thread per knot
                                const int32_t knot = grp.knot_lo + k;
                                const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                                const T* dz_k = d_dz + (size_t)knot * constants::STATE_S_CONTROL;
                                T w[MAX_ROWS_PER_GROUP], v[MAX_ROWS_PER_GROUP], p[MAX_ROWS_PER_GROUP];
                                for (int32_t i = 0; i < m; i++) {
                                        const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                        w[i] = eval_row_stepped<T>(grp, xu_k, dz_k, (uint32_t)i);
                                        v[i] = w[i] + d_y[idx] / grp.mu;
                                }
                                glass::thread::soc_project<T>(v, p, m);
                                for (int32_t i = 0; i < m; i++) {
                                        const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                        // soft sigma > 0: quadratic slack == segment blend
                                        // toward the projection (interval-formula analogue:
                                        // z = (rho*v + sigma*Pi(v))/(rho+sigma); hard = Pi(v))
                                        const T z = (grp.sigma > static_cast<T>(0)) ? (grp.mu * v[i] + grp.sigma * p[i]) / (grp.mu + grp.sigma) : p[i];
                                        const T z_prev = d_z[idx];
                                        d_y[idx] += grp.mu * (w[i] - z);
                                        d_z[idx] = z;
                                        T dp = w[i] - z;
                                        if (dp < 0) dp = -dp;
                                        T dd = grp.mu * (z - z_prev);
                                        if (dd < 0) dd = -dd;
                                        s_prim[k * m + i] = dp;
                                        s_dual[k * m + i] = dd;
                                }
                        }
                } else {
                        for (int32_t e = rank; e < n_elems; e += size) {
                                const int32_t knot = grp.knot_lo + e / grp.n_rows;
                                const int32_t i = e % grp.n_rows;
                                const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                                const T* dz_k = d_dz + (size_t)knot * constants::STATE_S_CONTROL;
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);

                                const T w = eval_row_stepped<T>(grp, xu_k, dz_k, (uint32_t)i);
                                const T z_prev = d_z[idx];
                                const T z = admm_z_update<T>(w + d_y[idx] / grp.mu, grp.lo[i], grp.hi[i], grp.mu, grp.sigma);
                                d_y[idx] += grp.mu * (w - z);
                                d_z[idx] = z;
                                T dp = w - z;
                                if (dp < 0) dp = -dp;
                                T dd = grp.mu * (z - z_prev);
                                if (dd < 0) dd = -dd;
                                s_prim[e] = dp;
                                s_dual[e] = dd;
                        }
                }
                __syncthreads();
                if (rank == 0) {
                        for (int32_t e = 0; e < n_elems; e++) {  // fixed order: deterministic
                                if (s_prim[e] > run_prim) run_prim = s_prim[e];
                                if (s_dual[e] > run_dual) run_dual = s_dual[e];
                        }
                }
                __syncthreads();
        }
        if (rank == 0) {
                d_resid_batch[2 * solve_idx + 0] = run_prim;
                d_resid_batch[2 * solve_idx + 1] = run_dual;
        }
}

// ---- host wrappers --------------------------------------------------------

template<typename T>
__host__ void admmInitStateBatched(uint32_t batch_size, T* d_z_batch, T* d_y_batch, const T* d_xu_traj_batch, const RowGroupDesc<T>* d_groups, int32_t n_groups, const void* d_GRiD_mem, bool eq_rows_only = false)
{
        admmInitStateBatchedKernel<T><<<batch_size, ADMM_THREADS, sizeof(T) * ee_rows_scratch_ct<T>()>>>(d_z_batch, d_y_batch, d_xu_traj_batch, d_groups, n_groups, (const grid::robotModel<T>*)d_GRiD_mem, eq_rows_only);
}

template<typename T>
__host__ void admmGradientBatched(uint32_t batch_size, T* d_q_batch, T* d_r_batch, const T* d_q_base_batch, const T* d_r_base_batch, const T* d_xu_traj_batch, const T* d_z_batch, const T* d_y_batch,
                                  const RowGroupDesc<T>* d_groups, int32_t n_groups, const int32_t* d_kkt_converged_batch, const void* d_GRiD_mem)
{
        dim3 grid(KNOT_POINTS, batch_size);
        admmGradientBatchedKernel<T><<<grid, ADMM_THREADS, sizeof(T) * ee_rows_grad_scratch_ct<T>()>>>(d_q_batch, d_r_batch, d_q_base_batch, d_r_base_batch, d_xu_traj_batch, d_z_batch, d_y_batch, d_groups, n_groups, d_kkt_converged_batch, (const grid::robotModel<T>*)d_GRiD_mem);
}

template<typename T>
__host__ size_t getAdmmProjectDualSMemSize()
{
        return sizeof(T) * (2 * KNOT_POINTS * MAX_ROWS_PER_GROUP + ee_rows_grad_scratch_ct<T>());
}

template<typename T>
__host__ void admmProjectDualBatched(uint32_t batch_size, T* d_z_batch, T* d_y_batch, T* d_resid_batch, const T* d_xu_traj_batch, const T* d_dz_batch,
                                     const RowGroupDesc<T>* d_groups, int32_t n_groups, const int32_t* d_kkt_converged_batch, const void* d_GRiD_mem)
{
        admmProjectDualBatchedKernel<T><<<batch_size, ADMM_THREADS, getAdmmProjectDualSMemSize<T>()>>>(d_z_batch, d_y_batch, d_resid_batch, d_xu_traj_batch, d_dz_batch, d_groups, n_groups, d_kkt_converged_batch, (const grid::robotModel<T>*)d_GRiD_mem);
}

}  // namespace rows
}  // namespace gato
