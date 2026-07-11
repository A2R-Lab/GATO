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
// Mechanism bindings on the shared descriptors:
//   MECH_TELEMETRY       — rows evaluated + true-violation reported, never enforced
//                          (zero solver-path impact; validates evaluators).
//   MECH_BARRIER_RELAXED — relaxed log-barrier folded into the KKT cost blocks
//                          (bounded Hessian, C² quadratic extension below `delta`;
//                          infeasible-start safe — the soft prior mode).
//   MECH_ADMM            — OSQP-style interval projection as a fixed-budget inner
//                          loop on the reused bdsv factor (kernels/admm.cuh).
//   MECH_AL              — PHR augmented Lagrangian: grad/GN-Hessian + merit value
//                          folds here, outer dual update once per SOLVE
//                          (alDualUpdateBatchedKernel; the solve is the inner
//                          minimization); equality rows (lo == hi) always
//                          active. Default bindings decided by round R1.

using namespace sqp;

namespace gato {
namespace rows {

constexpr uint32_t MAX_ROW_GROUPS = 8;
constexpr uint32_t MAX_ROWS_PER_GROUP = constants::STATE_SIZE;

enum Kind : int32_t {
        BOX_Q = 0,   // g_i = q_i      (block X, rows = NQ)
        BOX_QD = 1,  // g_i = qd_i     (block X, rows = NQ)
        BOX_U = 2,   // g_i = u_i      (block U, rows = NU; no terminal control)
        EE_POS = 3,  // g_i = ee_pos_i(q_k), i < 3 (block X; equality when lo == hi).
                     // NOT a selection row: needs a block-COOPERATIVE FK eval
                     // (gato::plant::eePos[Grad]) — handled at dedicated sites,
                     // never through the per-thread eval_row switch. v1:
                     // terminal knot, MECH_AL / MECH_TELEMETRY, single EE (ee 0).
};

__device__ __forceinline__ bool is_selection_kind(int32_t kind)
{
        return kind <= BOX_U;
}

enum Block : int32_t { BLOCK_X = 0, BLOCK_U = 1 };

enum Mechanism : int32_t {
        MECH_TELEMETRY = 0,
        MECH_BARRIER_RELAXED = 1,
        MECH_ADMM = 2,  // OSQP-style interval projection (kernels/admm.cuh); mu = the ADMM rho
        MECH_AL = 3,    // PHR augmented Lagrangian; mu = the AL penalty rho, duals in the
                        // per-solve (lam_hi, lam_lo) buffers, outer update once per solve
};

// per-solve ADMM dual/slack state extent: one (z, y) pair per potential row
// slot, dense [group][knot][row] — indexable without per-group prefix sums
constexpr uint32_t ROW_STATE_SIZE = MAX_ROW_GROUPS * KNOT_POINTS * MAX_ROWS_PER_GROUP;

__host__ __device__ __forceinline__ uint32_t row_state_index(uint32_t grp_idx, uint32_t knot, uint32_t i)
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
        T       sigma;  // soft/slack toggle (TurboMPC delta_xi pattern):
                        // <= 0 = HARD (exact legacy path); > 0 = elastic
                        // weight — AL: L1 elastic slack == effective
                        // multiplier saturates at sigma; ADMM: quadratic
                        // slack == smoothed z-projection.
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

// ---- PHR augmented-Lagrangian scalars (MECH_AL) -------------------------
//
// Interval rows split into hinge sides with INDEPENDENT multipliers
// (lam_hi for g <= hi, lam_lo for g >= lo, both >= 0); lo == hi rows are
// ALWAYS-ACTIVE equalities whose signed multiplier lives in the lam_hi slot
// (lam_lo unused). With c the signed constraint value (c = g - hi resp.
// lo - g; equality c = g - hi):
//   inequality side: phi = (max(0, lam + rho c)^2 - lam^2) / (2 rho)
//   equality:        phi = lam c + rho c^2 / 2
// so dphi/dc = max(0, lam + rho c) (resp. lam + rho c) and the GN Hessian is
// rho on the active set — C^1 across activation INCLUDING the -lam^2 offset
// (dropping it puts a jump under the line search). The outer update is
// lam <- max(0, lam + rho c) (equalities unclamped) on the ACCEPTED
// trajectory. Numpy twin: test/oracles/mechanisms.py::al_phr.
//
// SOFT rows (sigma > 0, L1 elastic slack xi >= 0 with weight sigma*xi,
// minimized analytically): the activation a = lam + rho c SATURATES at
// sigma. a > sigma =>
//   phi = sigma c - (sigma - lam)^2 / (2 rho),  dphi/dc = sigma,  hess = 0
// (equalities symmetric: a < -sigma => phi = -sigma c - (sigma + lam)^2 /
// (2 rho)). C^1 at |a| == sigma against the quadratic region (checked
// analytically at both seams). The outer update caps the multiplier at
// sigma (|lam| <= sigma for equalities) — the elastic problem's multiplier
// bound. sigma <= 0 keeps the exact hard code path.

template<typename T>
__device__ __forceinline__ bool is_eq_row(T lo, T hi)
{
        return isfinite(lo) && lo == hi;
}

// one hinge side's value with optional elastic saturation (c signed, a = lam + rho c)
template<typename T>
__device__ __forceinline__ T al_hinge_value(T c, T lam, T rho, T sigma)
{
        T a = lam + rho * c;
        if (a < static_cast<T>(0)) a = static_cast<T>(0);
        if (sigma > static_cast<T>(0) && a > sigma) {
                const T d = sigma - lam;
                return sigma * c - d * d / (static_cast<T>(2) * rho);
        }
        return (a * a - lam * lam) / (static_cast<T>(2) * rho);
}

template<typename T>
__device__ __forceinline__ T al_interval_value(T g, T lo, T hi, T lam_hi, T lam_lo, T rho, T sigma)
{
        if (is_eq_row<T>(lo, hi)) {
                const T c = g - hi;
                const T a = lam_hi + rho * c;
                if (sigma > static_cast<T>(0) && a > sigma) {
                        const T d = sigma - lam_hi;
                        return sigma * c - d * d / (static_cast<T>(2) * rho);
                }
                if (sigma > static_cast<T>(0) && a < -sigma) {
                        const T d = sigma + lam_hi;
                        return -sigma * c - d * d / (static_cast<T>(2) * rho);
                }
                return lam_hi * c + static_cast<T>(0.5) * rho * c * c;
        }
        T v = static_cast<T>(0);
        if (isfinite(hi)) v += al_hinge_value<T>(g - hi, lam_hi, rho, sigma);
        if (isfinite(lo)) v += al_hinge_value<T>(lo - g, lam_lo, rho, sigma);
        return v;
}

template<typename T>
__device__ __forceinline__ void al_interval_grad_hess(T g, T lo, T hi, T lam_hi, T lam_lo, T rho, T sigma, T& gr, T& h)
{
        gr = static_cast<T>(0);
        h = static_cast<T>(0);
        const bool soft = sigma > static_cast<T>(0);
        if (is_eq_row<T>(lo, hi)) {
                const T a = lam_hi + rho * (g - hi);
                if (soft && a > sigma) { gr = sigma; return; }
                if (soft && a < -sigma) { gr = -sigma; return; }
                gr = a;
                h = rho;
                return;
        }
        if (isfinite(hi)) {
                const T a = lam_hi + rho * (g - hi);
                if (a > static_cast<T>(0)) {
                        if (soft && a > sigma) { gr += sigma; }
                        else { gr += a; h += rho; }
                }
        }
        if (isfinite(lo)) {
                const T a = lam_lo + rho * (lo - g);
                if (a > static_cast<T>(0)) {
                        if (soft && a > sigma) { gr -= sigma; }
                        else { gr -= a; h += rho; }
                }
        }
}

// ---- EE_POS rows (cooperative FK evaluator sites) ------------------------
//
// EE rows evaluate through gato::plant::eePos[Grad] (block-cooperative,
// caller-scratch), so they cannot ride the per-thread selection paths. The
// helpers below own the scratch carving; every consumer kernel adds
// EE_ROWS_SCRATCH_CT (value sites) or EE_ROWS_GRAD_SCRATCH_CT (fold site)
// elements of T to its shared budget. Alignment: the GRiD arena wants 16B.

template<typename T>
__device__ __forceinline__ T* align16_ptr(T* p)
{
        uintptr_t u = reinterpret_cast<uintptr_t>(p);
        return reinterpret_cast<T*>((u + 15) & ~static_cast<uintptr_t>(15));
}

template<typename T>
__host__ __device__ constexpr uint32_t ee_rows_scratch_ct()
{
        // pose + aligned eval arena (+16B slop for the align-up)
        return 6 * gato::plant::NEE + gato::plant::eePos_TempMemCt<T>() + 16 / sizeof(T) + 1;
}

template<typename T>
__host__ __device__ constexpr uint32_t ee_rows_grad_scratch_ct()
{
        // pose + jacobian + per-row (gr, h) scalars + aligned eval arena
        return 6 * gato::plant::NEE + 6 * (constants::STATE_SIZE / 2) * gato::plant::NEE + 2 * MAX_ROWS_PER_GROUP + gato::plant::eePosGrad_TempMemCt<T>() + 16 / sizeof(T) + 1;
}

// any EE_POS group active at `knot` with an enforcing/reporting mechanism?
template<typename T>
__device__ __forceinline__ bool has_ee_rows(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups, int32_t knot)
{
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind == EE_POS && knot >= grp.knot_lo && knot < grp.knot_hi) return true;
        }
        return false;
}

// cooperative pose eval into s_scratch; returns the pose pointer (position =
// entries 0..2 of EE 0). ALL threads must call (contains barriers).
template<typename T>
__device__ __forceinline__ const T* ee_eval_pose(const T* xu_k, T* s_scratch, const grid::robotModel<T>* d_robot_model)
{
        T* s_pose = s_scratch;
        T* s_arena = align16_ptr<T>(s_pose + 6 * gato::plant::NEE);
        gato::plant::eePos<T>(s_pose, xu_k, s_arena, d_robot_model);
        return s_pose;
}

// GN fold of EE_POS rows at one knot into the state cost blocks (dense:
// q += sum_i gr_i * J_i, Q += sum_i h_i * J_i^T J_i over the q half). Scalars
// per mechanism (AL / RB / ADMM — ADMM folds the constant rho*J^T*J Hessian
// only; its gradient half is per-ADMM-iteration, kernels/admm.cuh);
// MECH_TELEMETRY groups contribute nothing here. ALL threads call; ends on a
// barrier per group.
template<typename T>
__device__ void apply_ee_row_grad_hess(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups,
                                       int32_t knot, const T* xu_k,
                                       const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                                       T* s_Q, T* s_q, T* s_scratch, const grid::robotModel<T>* d_robot_model)
{
        constexpr int32_t NQ = constants::STATE_SIZE / 2;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind != EE_POS) continue;
                if (grp.mech != MECH_AL && grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_ADMM) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;

                T* s_pose = s_scratch;
                T* s_grad = s_pose + 6 * gato::plant::NEE;
                T* s_gr = s_grad + 6 * NQ * gato::plant::NEE;
                T* s_h = s_gr + MAX_ROWS_PER_GROUP;
                T* s_arena = align16_ptr<T>(s_h + MAX_ROWS_PER_GROUP);
                gato::plant::eePosGrad<T>(s_pose, s_grad, xu_k, s_arena, d_robot_model);

                for (int32_t i = rank; i < grp.n_rows; i += size) {
                        const T g = s_pose[i];
                        if (grp.mech == MECH_ADMM) {
                                // constant rho*J^T*J fold; gradient half per-ADMM-iteration
                                s_gr[i] = static_cast<T>(0);
                                s_h[i] = grp.mu;  // mu = ADMM rho
                        } else if (grp.mech == MECH_AL) {
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                al_interval_grad_hess<T>(g, grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma, s_gr[i], s_h[i]);
                        } else {
                                s_gr[i] = rb_interval_grad<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                                s_h[i] = rb_interval_hess<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                        }
                }
                __syncthreads();

                // J_i[qi] = s_grad[6*qi + i] (single-EE layout; row i, joint qi)
                for (int32_t e = rank; e < NQ * NQ; e += size) {
                        const int32_t qi = e / NQ, qj = e % NQ;
                        T acc = static_cast<T>(0);
                        for (int32_t i = 0; i < grp.n_rows; i++) { acc += s_h[i] * s_grad[6 * qi + i] * s_grad[6 * qj + i]; }
                        s_Q[qi * constants::STATE_SIZE + qj] += acc;
                }
                for (int32_t qi = rank; qi < NQ; qi += size) {
                        T acc = static_cast<T>(0);
                        for (int32_t i = 0; i < grp.n_rows; i++) { acc += s_gr[i] * s_grad[6 * qi + i]; }
                        s_q[qi] += acc;
                }
                __syncthreads();
        }
}

// scalar EE row cost of one knot (merit seam; mirrors apply_ee_row_grad_hess
// exactly). ALL threads call (cooperative FK); the sum itself is uniform.
template<typename T>
__device__ T ee_row_cost_value(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups,
                               int32_t knot, const T* xu_k,
                               const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                               T* s_scratch, const grid::robotModel<T>* d_robot_model)
{
        T total = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind != EE_POS) continue;
                if (grp.mech != MECH_AL && grp.mech != MECH_BARRIER_RELAXED) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                const T* s_pose = ee_eval_pose<T>(xu_k, s_scratch, d_robot_model);
                for (int32_t i = 0; i < grp.n_rows; i++) {
                        if (grp.mech == MECH_AL) {
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                total += al_interval_value<T>(s_pose[i], grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma);
                        } else {
                                total += rb_interval_value<T>(s_pose[i], grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                        }
                }
        }
        return total;
}

// ---- mechanism cost contributions (the setup_kkt/merit seam) -------------
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

// d_lam_hi/d_lam_lo: this SOLVE's AL dual base pointers (row_state_index
// layout; only dereferenced for MECH_AL groups — nullptr is fine without them)
template<typename T>
__device__ void apply_row_grad_hess(const RowGroupDesc<T>* __restrict__ groups,
                                    int32_t n_groups, int32_t knot, const T* xu_k,
                                    const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                                    T* s_Q, T* s_q, T* s_R, T* s_r, bool has_control)
{
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (!is_selection_kind(grp.kind)) continue;  // EE rows: apply_ee_row_grad_hess
                if (grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_ADMM && grp.mech != MECH_AL) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                for (int32_t i = rank; i < grp.n_rows; i += size) {
                        T gr, h;
                        if (grp.mech == MECH_ADMM) {
                                // constant rho*G^T G diagonal fold — the gradient half is
                                // per-ADMM-iteration (kernels/admm.cuh), NOT here
                                gr = static_cast<T>(0);
                                h = grp.mu;  // mu = ADMM rho
                        } else if (grp.mech == MECH_AL) {
                                const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                al_interval_grad_hess<T>(g, grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma, gr, h);
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

// scalar mechanism cost of one knot (merit seam): RB barrier value or AL
// value (must mirror apply_row_grad_hess or the line search accepts against a
// different objective; MECH_ADMM adds nothing here — v1 leaves the merit
// unchanged). Every thread computes the same serial sum (a few dozen evals)
// — uniform, deterministic, thread-invariant.
template<typename T>
__device__ T row_cost_value(const RowGroupDesc<T>* __restrict__ groups,
                            int32_t n_groups, int32_t knot, const T* xu_k,
                            const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                            bool has_control)
{
        T total = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (!is_selection_kind(grp.kind)) continue;  // EE rows: ee_row_cost_value
                if (grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_AL) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                for (int32_t i = 0; i < grp.n_rows; i++) {
                        const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                        if (grp.mech == MECH_AL) {
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                total += al_interval_value<T>(g, grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma);
                        } else {
                                total += rb_interval_value<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                        }
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
                                                                                   const T* __restrict__                d_xu_traj_batch,
                                                                                   const grid::robotModel<T>* __restrict__ d_robot_model)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

        extern __shared__ char s_raw[];
        T* s_viol = reinterpret_cast<T*>(s_raw);                       // KNOT_POINTS * MAX_ROWS_PER_GROUP
        T* s_ee_scratch = s_viol + KNOT_POINTS * MAX_ROWS_PER_GROUP;  // ee_rows_scratch_ct

        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        for (int32_t grp_idx = 0; grp_idx < n_groups; grp_idx++) {
                const RowGroupDesc<T>& grp = d_groups[grp_idx];
                const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                const int32_t n_elems = n_knots * grp.n_rows;

                if (grp.kind == EE_POS) {
                        // cooperative FK per active knot, then per-row violations
                        for (int32_t k = 0; k < n_knots; k++) {
                                const T* xu_k = d_xu + (size_t)(grp.knot_lo + k) * constants::STATE_S_CONTROL;
                                const T* s_pose = ee_eval_pose<T>(xu_k, s_ee_scratch, d_robot_model);
                                for (int32_t i = rank; i < grp.n_rows; i += size) {
                                        s_viol[k * grp.n_rows + i] = interval_violation<T>(s_pose[i], grp.lo[i], grp.hi[i]);
                                }
                                __syncthreads();
                        }
                } else {
                        for (int32_t e = rank; e < n_elems; e += size) {
                                const int32_t knot = grp.knot_lo + e / grp.n_rows;
                                const int32_t i = e % grp.n_rows;
                                const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                                const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                s_viol[e] = interval_violation<T>(g, grp.lo[i], grp.hi[i]);
                        }
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
        return sizeof(T) * (KNOT_POINTS * MAX_ROWS_PER_GROUP + ee_rows_scratch_ct<T>());
}

template<typename T>
__host__ void rowGroupTelemetryBatched(uint32_t batch_size, T* d_telemetry, const RowGroupDesc<T>* d_groups, int32_t n_groups, const T* d_xu_traj_batch, const void* d_GRiD_mem)
{
        if (n_groups <= 0) return;
        rowGroupTelemetryBatchedKernel<T><<<batch_size, ROWGROUP_THREADS, getRowGroupTelemetrySMemSize<T>()>>>(d_telemetry, d_groups, n_groups, d_xu_traj_batch,
                                                                                                               (const grid::robotModel<T>*)d_GRiD_mem);
}

// ---- AL outer dual update (MECH_AL) --------------------------------------
//
// One block per solve, launched ONCE PER SOLVE on the final trajectory (the
// solve is the inner minimization; per-SQP-iteration updates diverge — a
// damped or rejected step is not a converged inner solve): per active row,
//   lam_hi <- max(0, lam_hi + rho*(g - hi)),  lam_lo <- max(0, lam_lo + rho*(lo - g))
// (equality rows: lam_hi += rho*(g - hi), unclamped). Each (group, knot, row)
// slot is owned by exactly one thread — no barriers between row updates,
// deterministic.
//
// TRUE-VIOLATION ACCEPTANCE (the PDDP-settled PHR safeguard): duals update
// only when this solve's max true violation over the AL groups (read from
// the telemetry buffer — launch AFTER the telemetry kernel) is feasible or
// STRICTLY improved on the best accepted so far. A stalled primal (line
// search rejecting every step at this rho) therefore freezes lambda at its
// last accepted value instead of accumulating rho*viol every solve until an
// overshooting step lands (measured drift: viol 0.069 plateau -> 0.57).

constexpr float AL_ACCEPT_FACTOR = 0.99f;  // required relative improvement
constexpr float AL_FEAS_TOL = 1e-5f;       // feasible -> always accept

template<typename T>
__global__ __launch_bounds__(ROWGROUP_THREADS) void alDualUpdateBatchedKernel(T* __restrict__ d_lam_hi_batch,
                                                                              T* __restrict__ d_lam_lo_batch,
                                                                              T* __restrict__ d_prev_viol_batch,
                                                                              const T* __restrict__ d_telemetry,
                                                                              const T* __restrict__ d_xu_traj_batch,
                                                                              const RowGroupDesc<T>* __restrict__ d_groups,
                                                                              int32_t n_groups,
                                                                              const grid::robotModel<T>* __restrict__ d_robot_model)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

        extern __shared__ char s_raw[];
        T* s_ee_scratch = reinterpret_cast<T*>(s_raw);  // ee_rows_scratch_ct

        __shared__ int32_t s_accept;
        if (rank == 0) {
                T viol = static_cast<T>(0);
                for (int32_t gi = 0; gi < n_groups; gi++) {
                        if (d_groups[gi].mech != MECH_AL) continue;
                        const T v = d_telemetry[(size_t)solve_idx * 2 * MAX_ROW_GROUPS + 2 * gi];
                        if (v > viol) viol = v;
                }
                const T prev = d_prev_viol_batch[solve_idx];
                s_accept = (viol < static_cast<T>(AL_FEAS_TOL) || viol < static_cast<T>(AL_ACCEPT_FACTOR) * prev) ? 1 : 0;
                if (s_accept) d_prev_viol_batch[solve_idx] = viol;
        }
        __syncthreads();
        if (!s_accept) return;

        T*       d_lam_hi = d_lam_hi_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        T*       d_lam_lo = d_lam_lo_batch + (size_t)solve_idx * ROW_STATE_SIZE;
        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::TRAJ_SIZE;

        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_AL) continue;
                const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                for (int32_t k = 0; k < n_knots; k++) {
                        const int32_t knot = grp.knot_lo + k;
                        const T* xu_k = d_xu + (size_t)knot * constants::STATE_S_CONTROL;
                        const T* s_pose = nullptr;
                        if (grp.kind == EE_POS) {
                                s_pose = ee_eval_pose<T>(xu_k, s_ee_scratch, d_robot_model);  // all threads
                        }
                        const bool soft = grp.sigma > static_cast<T>(0);
                        for (int32_t i = rank; i < grp.n_rows; i += size) {
                                const T g = (grp.kind == EE_POS) ? s_pose[i] : eval_row<T>(grp, xu_k, (uint32_t)i);
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                if (is_eq_row<T>(grp.lo[i], grp.hi[i])) {
                                        T a = d_lam_hi[idx] + grp.mu * (g - grp.hi[i]);
                                        if (soft) {  // elastic multiplier bound |lam| <= sigma
                                                if (a > grp.sigma) a = grp.sigma;
                                                if (a < -grp.sigma) a = -grp.sigma;
                                        }
                                        d_lam_hi[idx] = a;
                                } else {
                                        if (isfinite(grp.hi[i])) {
                                                T a = d_lam_hi[idx] + grp.mu * (g - grp.hi[i]);
                                                if (a < static_cast<T>(0)) a = static_cast<T>(0);
                                                if (soft && a > grp.sigma) a = grp.sigma;
                                                d_lam_hi[idx] = a;
                                        }
                                        if (isfinite(grp.lo[i])) {
                                                T a = d_lam_lo[idx] + grp.mu * (grp.lo[i] - g);
                                                if (a < static_cast<T>(0)) a = static_cast<T>(0);
                                                if (soft && a > grp.sigma) a = grp.sigma;
                                                d_lam_lo[idx] = a;
                                        }
                                }
                        }
                        if (grp.kind == EE_POS) { __syncthreads(); }  // s_ee_scratch reused next knot
                }
        }
}

template<typename T>
__host__ void alDualUpdateBatched(uint32_t batch_size, T* d_lam_hi_batch, T* d_lam_lo_batch, T* d_prev_viol_batch, const T* d_telemetry, const T* d_xu_traj_batch,
                                  const RowGroupDesc<T>* d_groups, int32_t n_groups, const void* d_GRiD_mem)
{
        alDualUpdateBatchedKernel<T><<<batch_size, ROWGROUP_THREADS, sizeof(T) * ee_rows_scratch_ct<T>()>>>(d_lam_hi_batch, d_lam_lo_batch, d_prev_viol_batch, d_telemetry, d_xu_traj_batch, d_groups,
                                                                                                            n_groups, (const grid::robotModel<T>*)d_GRiD_mem);
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
                        d_groups[g].sigma = static_cast<T>(0);  // hard by default
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
