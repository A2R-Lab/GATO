#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"

// Constraint row-group layer (constraint-layer arc CL-0).
// Architecture: docs/open-tasks/constraint_layer_locomotion_arc_plan_2026-07-10.md.
//
// A row-group = {evaluator kind, target block, mechanism binding, knot mask,
// interval bounds}. Per the CL-0 cross-term audit (cl0_cross_term_audit_2026-07-10.md)
// every group is PURE-X or PURE-U within a knot — the `block` field states which,
// and formSchur's block-diagonal Hessian assumption depends on it. A row that
// couples (x_k, u_k) is a design error this layer rejects by construction.
//
// CL-0 ships the descriptor/evaluator seam plus TWO bindings:
//   MECH_TELEMETRY       — rows evaluated + true-violation reported, never enforced
//                          (zero solver-path impact; validates evaluators).
//   MECH_BARRIER_RELAXED — relaxed log-barrier folded into the KKT cost blocks
//                          (bounded Hessian, C² quadratic extension below `delta`;
//                          infeasible-start safe — the soft prior mode).
// ADMM-projection and AL/PHR bindings land in CL-1 on the same descriptors.

using namespace sqp;

namespace gato {
namespace rows {

constexpr uint32_t MAX_ROW_GROUPS = 8;
constexpr uint32_t MAX_ROWS_PER_GROUP = constants::STATE_SIZE;

enum Kind : int32_t {
        BOX_Q = 0,   // g_i = q_i      (block X, rows = NQ)
        BOX_QD = 1,  // g_i = qd_i     (block X, rows = NQ)
        BOX_U = 2,   // g_i = u_i      (block U, rows = NU; no terminal control)
};

enum Block : int32_t { BLOCK_X = 0, BLOCK_U = 1 };

enum Mechanism : int32_t {
        MECH_TELEMETRY = 0,
        MECH_BARRIER_RELAXED = 1,
        MECH_ADMM = 2,  // OSQP-style interval projection (kernels/admm.cuh); mu = the ADMM rho
        // MECH_AL: CL-1 (next)
};

// per-solve ADMM dual/slack state extent: one (z, y) pair per potential row
// slot, dense [group][knot][row] — indexable without per-group prefix sums
constexpr uint32_t ROW_STATE_SIZE = MAX_ROW_GROUPS * KNOT_POINTS * MAX_ROWS_PER_GROUP;

__device__ __forceinline__ uint32_t row_state_index(uint32_t grp_idx, uint32_t knot, uint32_t i)
{
        return (grp_idx * KNOT_POINTS + knot) * MAX_ROWS_PER_GROUP + i;
}

template<typename T>
struct RowGroupDesc {
        int32_t kind;
        int32_t block;    // audit rule: pure-X or pure-U, never both
        int32_t mech;
        int32_t n_rows;   // rows per active knot (<= MAX_ROWS_PER_GROUP)
        int32_t knot_lo;  // active knots [knot_lo, knot_hi)
        int32_t knot_hi;
        T       lo[MAX_ROWS_PER_GROUP];  // interval bounds (equalities: lo == hi)
        T       hi[MAX_ROWS_PER_GROUP];
        T       mu;     // MECH_BARRIER_RELAXED weight
        T       delta;  // relaxed-barrier switch distance (> 0)
};

// ---- evaluators --------------------------------------------------------

// g for row i of a group at one knot, from that knot's (x, u) slice of the
// trajectory. Boxes are selection rows; richer evaluators (EE equality, cones,
// collision) join this switch in CL-1/CL-2.
template<typename T>
__device__ __forceinline__ T eval_row(const RowGroupDesc<T>& grp, const T* xu_k, uint32_t i)
{
        switch (grp.kind) {
                case BOX_Q: return xu_k[i];
                case BOX_QD: return xu_k[constants::STATE_SIZE / 2 + i];
                case BOX_U: return xu_k[constants::STATE_SIZE + i];
        }
        return static_cast<T>(0);
}

// true violation of an interval row: max(0, g - hi) + max(0, lo - g)
template<typename T>
__device__ __forceinline__ T interval_violation(T g, T lo, T hi)
{
        T over = g - hi;
        T under = lo - g;
        T v = static_cast<T>(0);
        if (over > static_cast<T>(0)) v += over;
        if (under > static_cast<T>(0)) v += under;
        return v;
}

// ---- relaxed log-barrier scalars (MECH_BARRIER_RELAXED) ----------------
//
// One-sided barrier on the distance d to a bound (d = g - lo or hi - g):
//   d > delta : B(d)  = -mu * log(d)
//   d <= delta: B(d)  = -mu * (log(delta) - 3/2 + 2 d/delta - d^2/(2 delta^2))
// C² at d = delta; defined for ALL d (including d <= 0 — infeasible-start
// safe, unlike grid_plant's clamped log barrier); Hessian bounded by
// mu/delta^2. dB/dd and d²B/dd² below; the caller chains the sign of dd/dg
// (+1 for the lower bound, -1 for the upper — the Hessian is sign-free).

template<typename T>
__device__ __forceinline__ T rb_value(T d, T mu, T delta)
{
        if (d > delta) { return -mu * log(d); }
        T r = d / delta;
        return -mu * (log(delta) - static_cast<T>(1.5) + static_cast<T>(2) * r - static_cast<T>(0.5) * r * r);
}

template<typename T>
__device__ __forceinline__ T rb_grad(T d, T mu, T delta)
{
        if (d > delta) { return -mu / d; }
        return -mu * (static_cast<T>(2) - d / delta) / delta;
}

template<typename T>
__device__ __forceinline__ T rb_hess(T d, T mu, T delta)
{
        if (d > delta) { return mu / (d * d); }
        return mu / (delta * delta);
}

// two-sided interval helpers on g in [lo, hi] (infinite bound contributes 0,
// matching grid_plant's isfinite convention)
template<typename T>
__device__ __forceinline__ T rb_interval_value(T g, T lo, T hi, T mu, T delta)
{
        T v = static_cast<T>(0);
        if (isfinite(lo)) v += rb_value<T>(g - lo, mu, delta);
        if (isfinite(hi)) v += rb_value<T>(hi - g, mu, delta);
        return v;
}

template<typename T>
__device__ __forceinline__ T rb_interval_grad(T g, T lo, T hi, T mu, T delta)
{
        T v = static_cast<T>(0);
        if (isfinite(lo)) v += rb_grad<T>(g - lo, mu, delta);
        if (isfinite(hi)) v -= rb_grad<T>(hi - g, mu, delta);
        return v;
}

template<typename T>
__device__ __forceinline__ T rb_interval_hess(T g, T lo, T hi, T mu, T delta)
{
        T v = static_cast<T>(0);
        if (isfinite(lo)) v += rb_hess<T>(g - lo, mu, delta);
        if (isfinite(hi)) v += rb_hess<T>(hi - g, mu, delta);
        return v;
}

// ---- MECH_BARRIER_RELAXED cost contributions (the setup_kkt/merit seam) --
//
// Box rows have selection Jacobians, so the GN contribution is DIAGONAL on the
// target block: grad_i -> s_q/s_r[idx], hess_i -> s_Q/s_R[idx,idx] (the audit
// rule guarantees a group never touches both blocks). Strided over rows, no
// cross-thread writes to the same slot; the caller owns the following barrier.
// `has_control` gates U-groups off the terminal knot.

template<typename T>
__device__ __forceinline__ int32_t x_index_of_row(const RowGroupDesc<T>& grp, int32_t i)
{
        return (grp.kind == BOX_QD) ? constants::STATE_SIZE / 2 + i : i;
}

template<typename T>
__device__ void apply_rb_grad_hess(const RowGroupDesc<T>* __restrict__ groups,
                                   int32_t n_groups, int32_t knot, const T* xu_k,
                                   T* s_Q, T* s_q, T* s_R, T* s_r, bool has_control)
{
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_ADMM) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                for (int32_t i = rank; i < grp.n_rows; i += size) {
                        T gr, h;
                        if (grp.mech == MECH_ADMM) {
                                // constant rho*G^T G diagonal fold — the gradient half is
                                // per-ADMM-iteration (kernels/admm.cuh), NOT here
                                gr = static_cast<T>(0);
                                h = grp.mu;  // mu = ADMM rho
                        } else {
                                const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                gr = rb_interval_grad<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                                h = rb_interval_hess<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                        }
                        if (grp.block == BLOCK_X) {
                                const int32_t xi = x_index_of_row<T>(grp, i);
                                s_q[xi] += gr;
                                s_Q[xi * constants::STATE_SIZE + xi] += h;  // diagonal: layout-agnostic
                        } else {
                                s_r[i] += gr;
                                s_R[i * constants::CONTROL_SIZE + i] += h;
                        }
                }
        }
}

// scalar RB cost of one knot (merit seam). Every thread computes the same
// serial sum (a few dozen evals) — uniform, deterministic, thread-invariant.
template<typename T>
__device__ T rb_cost_value(const RowGroupDesc<T>* __restrict__ groups,
                           int32_t n_groups, int32_t knot, const T* xu_k, bool has_control)
{
        T total = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.mech != MECH_BARRIER_RELAXED) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                for (int32_t i = 0; i < grp.n_rows; i++) {
                        const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                        total += rb_interval_value<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                }
        }
        return total;
}

// ---- telemetry kernel --------------------------------------------------
//
// One block per solve. Per group: every thread strides over (knot, row) pairs
// writing violations into shared scratch, then rank 0 reduces max + sum in a
// FIXED serial order — deterministic and thread-count invariant by
// construction (violation sums may feed acceptance thresholds later; the
// atomicAdd+threshold pattern is a known irreproducibility hazard).
//
// Output layout: d_telemetry[solve * (2 * MAX_ROW_GROUPS) + 2 * grp + {0,1}]
// = {max, sum} true violation over the group's active rows.

constexpr uint32_t ROWGROUP_THREADS = 128;

template<typename T>
__global__ __launch_bounds__(ROWGROUP_THREADS) void rowGroupTelemetryBatchedKernel(T* __restrict__                      d_telemetry,
                                                                                   const RowGroupDesc<T>* __restrict__ d_groups,
                                                                                   int32_t                              n_groups,
                                                                                   const T* __restrict__                d_xu_traj_batch)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

        extern __shared__ char s_raw[];
        T* s_viol = reinterpret_cast<T*>(s_raw);  // KNOT_POINTS * MAX_ROWS_PER_GROUP

        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        for (int32_t grp_idx = 0; grp_idx < n_groups; grp_idx++) {
                const RowGroupDesc<T>& grp = d_groups[grp_idx];
                const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                const int32_t n_elems = n_knots * grp.n_rows;

                for (int32_t e = rank; e < n_elems; e += size) {
                        const int32_t knot = grp.knot_lo + e / grp.n_rows;
                        const int32_t i = e % grp.n_rows;
                        const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                        const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                        s_viol[e] = interval_violation<T>(g, grp.lo[i], grp.hi[i]);
                }
                __syncthreads();

                if (rank == 0) {
                        T vmax = static_cast<T>(0), vsum = static_cast<T>(0);
                        for (int32_t e = 0; e < n_elems; e++) {  // fixed order: deterministic
                                if (s_viol[e] > vmax) vmax = s_viol[e];
                                vsum += s_viol[e];
                        }
                        d_telemetry[(size_t)solve_idx * 2 * MAX_ROW_GROUPS + 2 * grp_idx + 0] = vmax;
                        d_telemetry[(size_t)solve_idx * 2 * MAX_ROW_GROUPS + 2 * grp_idx + 1] = vsum;
                }
                __syncthreads();
        }
}

template<typename T>
__host__ size_t getRowGroupTelemetrySMemSize()
{
        return sizeof(T) * KNOT_POINTS * MAX_ROWS_PER_GROUP;
}

template<typename T>
__host__ void rowGroupTelemetryBatched(uint32_t batch_size, T* d_telemetry, const RowGroupDesc<T>* d_groups, int32_t n_groups, const T* d_xu_traj_batch)
{
        if (n_groups <= 0) return;
        rowGroupTelemetryBatchedKernel<T><<<batch_size, ROWGROUP_THREADS, getRowGroupTelemetrySMemSize<T>()>>>(d_telemetry, d_groups, n_groups, d_xu_traj_batch);
}

// ---- limit row-group initialization ------------------------------------
//
// The vendored {JOINT,VEL,CTRL}_LIMITS tables are __device__ constexpr, so the
// canonical limit groups are built ON DEVICE (single block; rank 0 writes the
// scalar fields, all ranks stride the bound arrays). BOX_U's knot mask stops
// at KNOT_POINTS-1 (the terminal knot has no control).

template<typename T>
__global__ void initLimitRowGroupsKernel(RowGroupDesc<T>* d_groups, int32_t mech, T mu, T delta)
{
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        constexpr int32_t NQ = constants::STATE_SIZE / 2;
        constexpr int32_t NU = constants::CONTROL_SIZE;

        if (rank == 0) {
                d_groups[0].kind = BOX_Q;
                d_groups[0].block = BLOCK_X;
                d_groups[0].n_rows = NQ;
                d_groups[0].knot_lo = 0;
                d_groups[0].knot_hi = KNOT_POINTS;

                d_groups[1].kind = BOX_QD;
                d_groups[1].block = BLOCK_X;
                d_groups[1].n_rows = NQ;
                d_groups[1].knot_lo = 0;
                d_groups[1].knot_hi = KNOT_POINTS;

                d_groups[2].kind = BOX_U;
                d_groups[2].block = BLOCK_U;
                d_groups[2].n_rows = NU;
                d_groups[2].knot_lo = 0;
                d_groups[2].knot_hi = KNOT_POINTS - 1;

                for (int g = 0; g < 3; g++) {
                        d_groups[g].mech = mech;
                        d_groups[g].mu = mu;
                        d_groups[g].delta = delta;
                }
        }
        for (int32_t i = rank; i < NQ; i += size) {
                d_groups[0].lo[i] = gato::plant::JOINT_LIMITS<T>()[i][0];
                d_groups[0].hi[i] = gato::plant::JOINT_LIMITS<T>()[i][1];
                d_groups[1].lo[i] = gato::plant::VEL_LIMITS<T>()[i][0];
                d_groups[1].hi[i] = gato::plant::VEL_LIMITS<T>()[i][1];
        }
        for (int32_t i = rank; i < NU; i += size) {
                d_groups[2].lo[i] = gato::plant::CTRL_LIMITS<T>()[i][0];
                d_groups[2].hi[i] = gato::plant::CTRL_LIMITS<T>()[i][1];
        }
}

}  // namespace rows
}  // namespace gato
