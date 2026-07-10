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
// v1 semantics (documented in the arc plan): fixed iteration budget
// (approximately-hard); residuals reported per solve, no in-loop early exit;
// the merit/line search is UNCHANGED (no constraint term for ADMM rows —
// telemetry reports the true violation of whatever the line search accepts).

using namespace sqp;

namespace gato {
namespace rows {

constexpr uint32_t ADMM_THREADS = 128;

// value of one row at (xu + dz) — selection rows are linear, so eval on each
// and add
template<typename T>
__device__ __forceinline__ T eval_row_stepped(const RowGroupDesc<T>& grp, const T* xu_k, const T* dz_k, uint32_t i)
{
        return eval_row<T>(grp, xu_k, i) + eval_row<T>(grp, dz_k, i);
}

// ---- z/y (re)initialization ---------------------------------------------
// z = clip(sel(x_warm), lo, hi), y = 0 — a neutral start: with y = 0 and
// z = sel(x) (feasible x) the first gradient modification vanishes.
template<typename T>
__global__ __launch_bounds__(ADMM_THREADS) void admmInitStateBatchedKernel(T* __restrict__ d_z_batch,
                                                                           T* __restrict__ d_y_batch,
                                                                           const T* __restrict__ d_xu_traj_batch,
                                                                           const RowGroupDesc<T>* __restrict__ d_groups,
                                                                           int32_t n_groups)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        T* d_z = d_z_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        T* d_y = d_y_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        for (uint32_t e = rank; e < ROW_STATE_SIZE; e += size) { d_y[e] = static_cast<T>(0); d_z[e] = static_cast<T>(0); }
        __syncthreads();
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_ADMM) continue;
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
                                                                          const int32_t* __restrict__ d_kkt_converged_batch)
{
        // grid (KNOT_POINTS, batch_size)
        const uint32_t knot_idx = blockIdx.x;
        const uint32_t solve_idx = blockIdx.y;
        if (d_kkt_converged_batch && d_kkt_converged_batch[solve_idx]) return;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

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
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_ADMM) continue;
                if ((int32_t)knot_idx < grp.knot_lo || (int32_t)knot_idx >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
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
                                                                             const int32_t* __restrict__ d_kkt_converged_batch)
{
        const uint32_t solve_idx = blockIdx.x;
        if (d_kkt_converged_batch && d_kkt_converged_batch[solve_idx]) return;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

        extern __shared__ char s_raw[];
        T* s_prim = reinterpret_cast<T*>(s_raw);              // KNOT_POINTS * MAX_ROWS_PER_GROUP
        T* s_dual = s_prim + KNOT_POINTS * MAX_ROWS_PER_GROUP;

        T*       d_z = d_z_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        T*       d_y = d_y_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;
        const T* d_dz = d_dz_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        T run_prim = static_cast<T>(0), run_dual = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_ADMM) continue;
                const int32_t n_elems = (grp.knot_hi - grp.knot_lo) * grp.n_rows;
                for (int32_t e = rank; e < n_elems; e += size) {
                        const int32_t knot = grp.knot_lo + e / grp.n_rows;
                        const int32_t i = e % grp.n_rows;
                        const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                        const T* dz_k = d_dz + (size_t)knot * constants::STATE_S_CONTROL;
                        const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);

                        const T w = eval_row_stepped<T>(grp, xu_k, dz_k, (uint32_t)i);
                        const T z_prev = d_z[idx];
                        T z = w + d_y[idx] / grp.mu;
                        if (z > grp.hi[i]) z = grp.hi[i];
                        if (z < grp.lo[i]) z = grp.lo[i];
                        d_y[idx] += grp.mu * (w - z);
                        d_z[idx] = z;
                        T dp = w - z;
                        if (dp < 0) dp = -dp;
                        T dd = grp.mu * (z - z_prev);
                        if (dd < 0) dd = -dd;
                        s_prim[e] = dp;
                        s_dual[e] = dd;
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
__host__ void admmInitStateBatched(uint32_t batch_size, T* d_z_batch, T* d_y_batch, const T* d_xu_traj_batch, const RowGroupDesc<T>* d_groups, int32_t n_groups)
{
        admmInitStateBatchedKernel<T><<<batch_size, ADMM_THREADS>>>(d_z_batch, d_y_batch, d_xu_traj_batch, d_groups, n_groups);
}

template<typename T>
__host__ void admmGradientBatched(uint32_t batch_size, T* d_q_batch, T* d_r_batch, const T* d_q_base_batch, const T* d_r_base_batch, const T* d_xu_traj_batch, const T* d_z_batch, const T* d_y_batch,
                                  const RowGroupDesc<T>* d_groups, int32_t n_groups, const int32_t* d_kkt_converged_batch)
{
        dim3 grid(KNOT_POINTS, batch_size);
        admmGradientBatchedKernel<T><<<grid, ADMM_THREADS>>>(d_q_batch, d_r_batch, d_q_base_batch, d_r_base_batch, d_xu_traj_batch, d_z_batch, d_y_batch, d_groups, n_groups, d_kkt_converged_batch);
}

template<typename T>
__host__ size_t getAdmmProjectDualSMemSize()
{
        return sizeof(T) * 2 * KNOT_POINTS * MAX_ROWS_PER_GROUP;
}

template<typename T>
__host__ void admmProjectDualBatched(uint32_t batch_size, T* d_z_batch, T* d_y_batch, T* d_resid_batch, const T* d_xu_traj_batch, const T* d_dz_batch,
                                     const RowGroupDesc<T>* d_groups, int32_t n_groups, const int32_t* d_kkt_converged_batch)
{
        admmProjectDualBatchedKernel<T><<<batch_size, ADMM_THREADS, getAdmmProjectDualSMemSize<T>()>>>(d_z_batch, d_y_batch, d_resid_batch, d_xu_traj_batch, d_dz_batch, d_groups, n_groups, d_kkt_converged_batch);
}

}  // namespace rows
}  // namespace gato
