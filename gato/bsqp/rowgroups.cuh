#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "glass.cuh"

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
        BOX_U = 2,   // g_i = u_i      (block U, rows = ACTUATED_SIZE — the URDF torque
                     // limits; fc slots (contact-force builds) have no table row and
                     // are bounded via user LIN_U rows instead; no terminal control)
        EE_POS = 3,  // g_i = ee_pos_i(q_k), i < 3 (block X; equality when lo == hi).
                     // NOT a selection row: needs a block-COOPERATIVE FK eval
                     // (gato::plant::eePos[Grad]) — handled at dedicated sites,
                     // never through the per-thread eval_row switch. v1:
                     // terminal knot, MECH_AL / MECH_TELEMETRY, single EE (ee 0).
        LIN_U = 4,   // g_i = C[i,:]·u_k + d_i (block U, host-supplied map; CL-2).
                     // Per-thread evaluable like the boxes, but the Jacobian is
                     // C (dense fold onto R/r, not diagonal). With grp.cone the
                     // row VECTOR carries second-order-cone semantics
                     // (row 0 = axis t, rows 1.. = x̄; g ∈ K ⟺ ‖x̄‖ ≤ t);
                     // interval bounds are unused (∓inf). The audit rule keeps
                     // it pure-U: config-dependent maps (EE wrench cones) are
                     // FROZEN at a host-chosen configuration — contact-frame
                     // parameterization per the CL-0 cross-term audit.
        COLLISION = 5,  // g_i = d_i(q_k): per-sphere min environment clearance
                     // (block X; CL-2 collision half). Cooperative evaluator
                     // (gato::plant::collisionDist[Grad] over grid_collision::)
                     // at dedicated sites like EE_POS — never the per-thread
                     // eval_row switch. n_rows = plant::NCC and MAY EXCEED
                     // MAX_ROWS_PER_GROUP: bounds are UNIFORM one-sided
                     // (lo[0] = clearance margin, hi[0] = +inf; the per-row
                     // lo/hi/Cmat arrays are NOT indexed past 0) and the
                     // dual/slack state lives in the dedicated collision band
                     // (collision_row_state_index below), not the dense
                     // per-group slots. AT MOST ONE collision group (the band
                     // is single; the environment is global per solve batch).
                     // Active-obstacle argmin: re-picked at each linearization
                     // point — constant within one QP/ADMM inner loop by
                     // construction; relinearized per SQP iteration (see the
                     // plant.cuh adapter note; pairs-ABI = fallback).
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

// COLLISION rows have n_rows = plant::NCC > MAX_ROWS_PER_GROUP, so their
// (z, y) / (lam_hi, lam_lo) state lives in a dedicated band APPENDED after the
// dense per-group slots (one collision group max — see the Kind doc). Every
// per-solve row-state buffer is TOTAL_ROW_STATE_SIZE-strided.
constexpr uint32_t COLLISION_ROW_STATE_SIZE = KNOT_POINTS * (uint32_t)gato::plant::NCC;
constexpr uint32_t TOTAL_ROW_STATE_SIZE = ROW_STATE_SIZE + COLLISION_ROW_STATE_SIZE;

__host__ __device__ __forceinline__ uint32_t row_state_index(uint32_t grp_idx, uint32_t knot, uint32_t i)
{
        return (grp_idx * KNOT_POINTS + knot) * MAX_ROWS_PER_GROUP + i;
}

__host__ __device__ __forceinline__ uint32_t collision_row_state_index(uint32_t knot, uint32_t i)
{
        return ROW_STATE_SIZE + knot * (uint32_t)gato::plant::NCC + i;
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
                        // slack == smoothed z-projection (cone: segment blend
                        // toward the SOC projection). Cone-AL rows are hard-only.
        // LIN_U payload (unread for other kinds)
        int32_t cone;   // 0 = interval rows; 1 = SOC on the row vector
        T       Cmat[MAX_ROWS_PER_GROUP * constants::CONTROL_SIZE];  // row-major map
        T       dvec[MAX_ROWS_PER_GROUP];                            // constant offset
};

// ---- evaluators --------------------------------------------------------

// g for row i of a group at one knot, from that knot's (x, u) slice of the
// trajectory. Boxes are selection rows; LIN_U is the dense-map row (CL-2);
// EE/collision kinds stay at their cooperative sites.
template<typename T>
__device__ __forceinline__ T eval_row(const RowGroupDesc<T>& grp, const T* xu_k, uint32_t i)
{
        switch (grp.kind) {
                case BOX_Q: return xu_k[i];
                case BOX_QD: return xu_k[constants::STATE_SIZE / 2 + i];
                case BOX_U: return xu_k[constants::STATE_SIZE + i];
                case LIN_U: {
                        T acc = grp.dvec[i];
                        for (int32_t j = 0; j < (int32_t)constants::CONTROL_SIZE; j++) { acc += grp.Cmat[i * constants::CONTROL_SIZE + j] * xu_k[constants::STATE_SIZE + j]; }
                        return acc;
                }
        }
        return static_cast<T>(0);
}

// DIRECTIONAL row value (no constant offset): g(x + dz) = eval_row(x) +
// eval_row_dir(dz). Identical to eval_row for selection rows; LIN_U must
// drop dvec or the offset double-counts on stepped evals.
template<typename T>
__device__ __forceinline__ T eval_row_dir(const RowGroupDesc<T>& grp, const T* dz_k, uint32_t i)
{
        if (grp.kind == LIN_U) {
                T acc = static_cast<T>(0);
                for (int32_t j = 0; j < (int32_t)constants::CONTROL_SIZE; j++) { acc += grp.Cmat[i * constants::CONTROL_SIZE + j] * dz_k[constants::STATE_SIZE + j]; }
                return acc;
        }
        return eval_row<T>(grp, dz_k, i);
}

// ---- constraint scalars: promoted to GLASS (2026-07-28 robotics ops) -------
//
// The SOC cone kit (soc_tail_norm/soc_violation/soc_project/al_soc_value),
// interval/PHR AL scalars (interval_violation/al_is_eq_row/al_hinge_value/
// al_interval_value/al_interval_grad_hess), and the relaxed log-barrier
// family (relaxed_barrier_[interval_]value/grad/hess) now live in
// glass proj/{cone,interval}.cuh -- promoted verbatim from this file (numpy
// twins: test/oracles/mechanisms.py). Call sites here use glass:: directly;
// soc_project sites use glass::thread:: (each (group,knot) vector is owned
// by exactly ONE thread and w/p may alias, which the thread tier permits).
// al_soc_value is the bufferless GLASS form (case-split norm, no
// MAX_ROWS_PER_GROUP locals) -- same math, summation order differs by ULPs.

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
                                       T* s_Q, T* s_q, T* s_scratch, const grid::robotModel<T>* d_robot_model,
                                       T admm_rho_scale = static_cast<T>(1))
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
                                s_h[i] = grp.mu * admm_rho_scale;  // mu = ADMM rho (x per-solve adaptation scale)
                        } else if (grp.mech == MECH_AL) {
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                glass::al_interval_grad_hess<T>(g, grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma, s_gr[i], s_h[i]);
                        } else {
                                s_gr[i] = glass::relaxed_barrier_interval_grad<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                                s_h[i] = glass::relaxed_barrier_interval_hess<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
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
__device__ __noinline__ T ee_row_cost_value(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups,
                               int32_t knot, const T* xu_k,
                               const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                               T* s_scratch, const grid::robotModel<T>* d_robot_model,
                               const T* __restrict__ d_z_admm = nullptr,
                               const T* __restrict__ d_y_admm = nullptr,
                               T admm_rho_scale = static_cast<T>(1))
{
        T total = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind != EE_POS) continue;
                const bool admm_term = grp.mech == MECH_ADMM && d_z_admm != nullptr;
                if (grp.mech != MECH_AL && grp.mech != MECH_BARRIER_RELAXED && !admm_term) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                const T* s_pose = ee_eval_pose<T>(xu_k, s_scratch, d_robot_model);
                for (int32_t i = 0; i < grp.n_rows; i++) {
                        if (admm_term) {
                                // true-nonlinear pose against the CURRENT (z, y) row
                                // state (set_admm_merit; see row_cost_value)
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                const T r = s_pose[i] - d_z_admm[idx];
                                total += d_y_admm[idx] * r + static_cast<T>(0.5) * grp.mu * admm_rho_scale * r * r;
                        } else if (grp.mech == MECH_AL) {
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                total += glass::al_interval_value<T>(s_pose[i], grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma);
                        } else {
                                total += glass::relaxed_barrier_interval_value<T>(s_pose[i], grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                        }
                }
        }
        return total;
}

// ---- COLLISION rows (cooperative clearance evaluator sites) --------------
//
// Clearance rows evaluate through gato::plant::collisionDist[Grad]
// (block-cooperative, caller-scratch), like the EE rows above. n_rows =
// plant::NCC exceeds MAX_ROWS_PER_GROUP, so bounds are uniform (margin =
// grp.lo[0], hi = +inf) and dual/slack state uses collision_row_state_index.
// Consumers add collision_rows[_grad]_scratch_ct<T>() elements to their
// shared budget WHEN a collision group is registered (host-known — the
// sizers take a flag; the carve is untouched otherwise).

template<typename T>
__host__ __device__ constexpr uint32_t collision_rows_scratch_ct()
{
        // dist + aligned eval arena (+16B slop for the align-up)
        return gato::plant::NCC + gato::plant::collisionDist_TempMemCt<T>() + 16 / sizeof(T) + 1;
}

template<typename T>
__host__ __device__ constexpr uint32_t collision_rows_grad_scratch_ct()
{
        // dist + jacobian + per-row (gr, h) scalars + aligned eval arena
        return gato::plant::NCC + gato::plant::NCC * (constants::STATE_SIZE / 2) + 2 * gato::plant::NCC + gato::plant::collisionDistGrad_TempMemCt<T>() + 16 / sizeof(T) + 1;
}

// any COLLISION group active at `knot`?
template<typename T>
__device__ __forceinline__ bool has_collision_rows(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups, int32_t knot)
{
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind == COLLISION && knot >= grp.knot_lo && knot < grp.knot_hi) return true;
        }
        return false;
}

// GN fold of COLLISION rows at one knot into the state cost blocks (dense on
// the q half: q += sum_i gr_i * J_i, Q += sum_i h_i * J_i^T J_i with J_i =
// the clearance Jacobian row). Scalars per mechanism exactly like the EE
// fold (ADMM folds the constant rho*J^T*J only). ALL threads call; ends on a
// barrier. __noinline__ ON PURPOSE: this fold lands in the same setup_kkt TU
// as the LIN_U folds that sent cicc superlinear (2026-07-19); same defense.
template<typename T>
__device__ __noinline__ void apply_collision_row_grad_hess(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups,
                                                           int32_t knot, const T* xu_k,
                                                           const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                                                           T* s_Q, T* s_q, T* s_scratch,
                                                           const grid::robotModel<T>* d_robot_model,
                                                           const grid_collision::Environment<T>& env,
                                                           T admm_rho_scale = static_cast<T>(1))
{
        constexpr int32_t NQ = constants::STATE_SIZE / 2;
        constexpr int32_t NS = gato::plant::NCC;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind != COLLISION) continue;
                if (grp.mech != MECH_AL && grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_ADMM) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;

                T* s_dist = s_scratch;
                T* s_ddist = s_dist + NS;
                T* s_gr = s_ddist + NS * NQ;
                T* s_h = s_gr + NS;
                T* s_arena = align16_ptr<T>(s_h + NS);
                gato::plant::collisionDistGrad<T>(s_dist, s_ddist, xu_k, s_arena, d_robot_model, env);

                const T margin = grp.lo[0];
                for (int32_t i = rank; i < NS; i += size) {
                        if (grp.mech == MECH_ADMM) {
                                // constant rho*J^T*J fold; gradient half per-ADMM-iteration
                                s_gr[i] = static_cast<T>(0);
                                s_h[i] = grp.mu * admm_rho_scale;  // mu = ADMM rho (x per-solve adaptation scale)
                        } else if (grp.mech == MECH_AL) {
                                const uint32_t idx = collision_row_state_index((uint32_t)knot, (uint32_t)i);
                                glass::al_interval_grad_hess<T>(s_dist[i], margin, grp.hi[0], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma, s_gr[i], s_h[i]);
                        } else {
                                s_gr[i] = glass::relaxed_barrier_interval_grad<T>(s_dist[i], margin, grp.hi[0], grp.mu, grp.delta);
                                s_h[i] = glass::relaxed_barrier_interval_hess<T>(s_dist[i], margin, grp.hi[0], grp.mu, grp.delta);
                        }
                }
                __syncthreads();

                for (int32_t e = rank; e < NQ * NQ; e += size) {
                        const int32_t qi = e / NQ, qj = e % NQ;
                        T acc = static_cast<T>(0);
                        for (int32_t i = 0; i < NS; i++) { acc += s_h[i] * s_ddist[i * NQ + qi] * s_ddist[i * NQ + qj]; }
                        s_Q[qi * constants::STATE_SIZE + qj] += acc;
                }
                for (int32_t qi = rank; qi < NQ; qi += size) {
                        T acc = static_cast<T>(0);
                        for (int32_t i = 0; i < NS; i++) { acc += s_gr[i] * s_ddist[i * NQ + qi]; }
                        s_q[qi] += acc;
                }
                __syncthreads();
        }
}

// scalar COLLISION row cost of one knot (merit seam; mirrors
// apply_collision_row_grad_hess). ALL threads call (cooperative FK); the
// serial sum itself is uniform. Value-only path (no Jacobian).
template<typename T>
__device__ __noinline__ T collision_row_cost_value(const RowGroupDesc<T>* __restrict__ groups, int32_t n_groups,
                                                   int32_t knot, const T* xu_k,
                                                   const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                                                   T* s_scratch, const grid::robotModel<T>* d_robot_model,
                                                   const grid_collision::Environment<T>& env,
                                                   const T* __restrict__ d_z_admm = nullptr,
                                                   const T* __restrict__ d_y_admm = nullptr,
                                                   T admm_rho_scale = static_cast<T>(1))
{
        constexpr int32_t NS = gato::plant::NCC;
        T total = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.kind != COLLISION) continue;
                const bool admm_term = grp.mech == MECH_ADMM && d_z_admm != nullptr;
                if (grp.mech != MECH_AL && grp.mech != MECH_BARRIER_RELAXED && !admm_term) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                T* s_dist = s_scratch;
                T* s_arena = align16_ptr<T>(s_dist + NS);
                gato::plant::collisionDist<T>(s_dist, xu_k, s_arena, d_robot_model, env);
                const T margin = grp.lo[0];
                for (int32_t i = 0; i < NS; i++) {
                        const uint32_t idx = collision_row_state_index((uint32_t)knot, (uint32_t)i);
                        if (admm_term) {
                                const T r = s_dist[i] - d_z_admm[idx];
                                total += d_y_admm[idx] * r + static_cast<T>(0.5) * grp.mu * admm_rho_scale * r * r;
                        } else if (grp.mech == MECH_AL) {
                                total += glass::al_interval_value<T>(s_dist[i], margin, grp.hi[0], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma);
                        } else {
                                total += glass::relaxed_barrier_interval_value<T>(s_dist[i], margin, grp.hi[0], grp.mu, grp.delta);
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

// ---- LIN_U dense-fold helpers (one per mechanism x cone case) -------------
//
// __noinline__ ON PURPOSE (build-time, not perf): inlined into the setup_kkt
// TU this fold sends cicc into a superlinear optimizer blowup on the big
// (plant,N) configs (measured 2026-07-19, N32_iiwa14: 14s -> >65min/26GB RSS
// with the fold inlined, 13s with it stubbed; the two group-boundary barriers
// and every other CL-2 hunk are innocent). As standalone functions each body
// compiles once and the TU stays at seconds. They run only when LIN_U groups
// are active (never on the default path) and the call overhead is noise next
// to the dense math. No barriers inside: the call site owns the
// group-boundary __syncthreads(), and every early return below is
// block-uniform (computed from data all threads read identically).

template<typename T>
__device__ __noinline__ void fold_lin_u_conic_al(const RowGroupDesc<T>& grp, int32_t gi, int32_t knot, const T* xu_k,
                                                 const T* __restrict__ d_lam_hi, T* s_R, T* s_r, uint32_t rank, uint32_t size)
{
        constexpr int32_t NU = constants::CONTROL_SIZE;
        const int32_t m = grp.n_rows;
        T g[MAX_ROWS_PER_GROUP];
        for (int32_t i = 0; i < m; i++) { g[i] = eval_row<T>(grp, xu_k, (uint32_t)i); }
        // conic PHR: grad_g = -p, H_g = rho*Jpi at w = lam - rho*g
        T w[MAX_ROWS_PER_GROUP], p[MAX_ROWS_PER_GROUP];
        for (int32_t i = 0; i < m; i++) { w[i] = d_lam_hi[row_state_index(gi, (uint32_t)knot, (uint32_t)i)] - grp.mu * g[i]; }
        glass::thread::soc_project<T>(w, p, m);
        for (int32_t a = rank; a < NU; a += size) {
                T acc = static_cast<T>(0);
                for (int32_t i = 0; i < m; i++) { acc -= p[i] * grp.Cmat[i * NU + a]; }
                s_r[a] += acc;
        }
        const T r = glass::soc_tail_norm<T>(w, m);
        // polar case (w in -K deg): Pi_K(w) = 0, so p = 0 above added nothing to
        // s_r and Jpi = 0 - there is no Hessian contribution; skip the e-loop.
        if (r <= -w[0]) return;  // Jpi = 0
        const bool inside = (r <= w[0]);
        // outside both cones: Jpi = 0.5*[[1, xh^T],[xh, ((w0+r)/r)I - (w0/r) xh xh^T]]
        const T inv_r = inside ? static_cast<T>(0) : static_cast<T>(1) / r;
        const T beta = inside ? static_cast<T>(0) : static_cast<T>(0.5) * (w[0] + r) * inv_r;
        const T gam = inside ? static_cast<T>(0) : static_cast<T>(0.5) * w[0] * inv_r;
        for (int32_t e = rank; e < NU * NU; e += size) {
                const int32_t a = e / NU, b = e % NU;
                T acc = static_cast<T>(0);
                if (inside) {  // Jpi = I -> rho*C^T C
                        for (int32_t i = 0; i < m; i++) { acc += grp.Cmat[i * NU + a] * grp.Cmat[i * NU + b]; }
                } else {
                        // column cb = C[:, b]; s1 = xh . cb-bar with xh = w-bar/r
                        T s1 = static_cast<T>(0);
                        for (int32_t i = 1; i < m; i++) { s1 += w[i] * inv_r * grp.Cmat[i * NU + b]; }
                        // (Jpi cb)_0 = 0.5*(cb0 + s1); (Jpi cb)_i = 0.5*xh_i*cb0 + beta*cb_i - gam*s1*xh_i
                        acc += grp.Cmat[0 * NU + a] * static_cast<T>(0.5) * (grp.Cmat[0 * NU + b] + s1);
                        for (int32_t i = 1; i < m; i++) {
                                const T xh = w[i] * inv_r;
                                acc += grp.Cmat[i * NU + a] * (static_cast<T>(0.5) * xh * grp.Cmat[0 * NU + b] + beta * grp.Cmat[i * NU + b] - gam * s1 * xh);
                        }
                }
                s_R[a * NU + b] += grp.mu * acc;
        }
}

template<typename T>
__device__ __noinline__ void fold_lin_u_rb_cone(const RowGroupDesc<T>& grp, const T* xu_k, T* s_R, T* s_r, uint32_t rank, uint32_t size)
{
        constexpr int32_t NU = constants::CONTROL_SIZE;
        const int32_t m = grp.n_rows;
        T g[MAX_ROWS_PER_GROUP];
        for (int32_t i = 0; i < m; i++) { g[i] = eval_row<T>(grp, xu_k, (uint32_t)i); }
        // relaxed barrier on the margin s = g0 - ||g-bar|| (GN: rank-1
        // hessian rb_hess(s) * (C^T n)(C^T n)^T, n = (1, -g-bar/||g-bar||))
        const T rt = glass::soc_tail_norm<T>(g, m);
        const T s = g[0] - rt;
        const T gr = glass::relaxed_barrier_grad<T>(s, grp.mu, grp.delta);
        const T hh = glass::relaxed_barrier_hess<T>(s, grp.mu, grp.delta);
        const T inv_rt = rt > static_cast<T>(1e-9) ? static_cast<T>(1) / rt : static_cast<T>(0);
        T u[NU];  // u = C^T n
        for (int32_t a = 0; a < NU; a++) {
                T acc = grp.Cmat[0 * NU + a];
                for (int32_t i = 1; i < m; i++) { acc -= g[i] * inv_rt * grp.Cmat[i * NU + a]; }
                u[a] = acc;
        }
        for (int32_t a = rank; a < NU; a += size) { s_r[a] += gr * u[a]; }
        for (int32_t e = rank; e < NU * NU; e += size) { s_R[(e / NU) * NU + (e % NU)] += hh * u[e / NU] * u[e % NU]; }
}

template<typename T>
__device__ __noinline__ void fold_lin_u_admm_cone(const RowGroupDesc<T>& grp, T* s_R, uint32_t rank, uint32_t size, T admm_rho_scale)
{
        constexpr int32_t NU = constants::CONTROL_SIZE;
        const int32_t m = grp.n_rows;
        // constant rho*C^T C fold; gradient half per-ADMM-iteration
        for (int32_t e = rank; e < NU * NU; e += size) {
                const int32_t a = e / NU, b = e % NU;
                T acc = static_cast<T>(0);
                for (int32_t i = 0; i < m; i++) { acc += grp.Cmat[i * NU + a] * grp.Cmat[i * NU + b]; }
                s_R[a * NU + b] += grp.mu * admm_rho_scale * acc;
        }
}

template<typename T>
__device__ __noinline__ void fold_lin_u_interval(const RowGroupDesc<T>& grp, int32_t gi, int32_t knot, const T* xu_k,
                                                 const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                                                 T* s_R, T* s_r, uint32_t rank, uint32_t size, T admm_rho_scale)
{
        constexpr int32_t NU = constants::CONTROL_SIZE;
        const int32_t m = grp.n_rows;
        T g[MAX_ROWS_PER_GROUP];
        for (int32_t i = 0; i < m; i++) { g[i] = eval_row<T>(grp, xu_k, (uint32_t)i); }
        // interval LIN_U rows (pyramid facets): per-row scalars
        // through the identical mechanism math, dense fold
        T grs[MAX_ROWS_PER_GROUP], hs[MAX_ROWS_PER_GROUP];
        for (int32_t i = 0; i < m; i++) {
                if (grp.mech == MECH_ADMM) {
                        grs[i] = static_cast<T>(0);
                        hs[i] = grp.mu * admm_rho_scale;
                } else if (grp.mech == MECH_AL) {
                        const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                        glass::al_interval_grad_hess<T>(g[i], grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma, grs[i], hs[i]);
                } else {
                        grs[i] = glass::relaxed_barrier_interval_grad<T>(g[i], grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                        hs[i] = glass::relaxed_barrier_interval_hess<T>(g[i], grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                }
        }
        for (int32_t a = rank; a < NU; a += size) {
                T acc = static_cast<T>(0);
                for (int32_t i = 0; i < m; i++) { acc += grs[i] * grp.Cmat[i * NU + a]; }
                s_r[a] += acc;
        }
        for (int32_t e = rank; e < NU * NU; e += size) {
                const int32_t a = e / NU, b = e % NU;
                T acc = static_cast<T>(0);
                for (int32_t i = 0; i < m; i++) { acc += hs[i] * grp.Cmat[i * NU + a] * grp.Cmat[i * NU + b]; }
                s_R[a * NU + b] += acc;
        }
}

// d_lam_hi/d_lam_lo: this SOLVE's AL dual base pointers (row_state_index
// layout; only dereferenced for MECH_AL groups — nullptr is fine without them)
template<typename T>
__device__ void apply_row_grad_hess(const RowGroupDesc<T>* __restrict__ groups,
                                    int32_t n_groups, int32_t knot, const T* xu_k,
                                    const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                                    T* s_Q, T* s_q, T* s_R, T* s_r, bool has_control,
                                    T admm_rho_scale = static_cast<T>(1))
{
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_ADMM && grp.mech != MECH_AL) continue;
                if (grp.kind == LIN_U) {
                        // dense fold onto the U block: r += C^T grad_g, R += C^T H_g C.
                        // Row scalars/vectors are recomputed per thread (uniform, tiny:
                        // m <= MAX_ROWS_PER_GROUP, NU-dim rows) - no shared scratch;
                        // strided writes own disjoint R/r entries. Bodies live in the
                        // __noinline__ fold_lin_u_* helpers above (cicc build-time cliff).
                        // MECH_ADMM's fold is gradient-free, so below knot_lo it extends
                        // as a pure proximal term - same f32 strip-homogeneity requirement
                        // as the selection-row admm_prox (R1 y-windup lesson).
                        const bool lin_admm_prox = grp.mech == MECH_ADMM && (int32_t)knot < grp.knot_lo;
                        if (((int32_t)knot < grp.knot_lo && !lin_admm_prox) || (int32_t)knot >= grp.knot_hi) continue;
                        if (!has_control) continue;  // audit rule: LIN_U is pure-U
                        // BARRIER: the dense e-strided writes in the helpers assign s_R
                        // entry (i,i) to thread (i*NU+i) % size, but a selection U-group's
                        // diagonal fold assigned it to thread i % size - without a
                        // barrier between group iterations that is a cross-thread
                        // write-write race (all guards above are block-uniform, so
                        // the barrier is uniformly reached).
                        __syncthreads();
                        if (grp.cone && grp.mech == MECH_AL) {
                                fold_lin_u_conic_al<T>(grp, gi, (int32_t)knot, xu_k, d_lam_hi, s_R, s_r, rank, size);
                        } else if (grp.cone && grp.mech == MECH_BARRIER_RELAXED) {
                                fold_lin_u_rb_cone<T>(grp, xu_k, s_R, s_r, rank, size);
                        } else if (grp.cone && grp.mech == MECH_ADMM) {
                                fold_lin_u_admm_cone<T>(grp, s_R, rank, size, admm_rho_scale);
                        } else {
                                fold_lin_u_interval<T>(grp, gi, (int32_t)knot, xu_k, d_lam_hi, d_lam_lo, s_R, s_r, rank, size, admm_rho_scale);
                        }
                        __syncthreads();  // later groups may re-target these slots (see above)
                        continue;
                }
                if (!is_selection_kind(grp.kind)) continue;  // EE rows: apply_ee_row_grad_hess
                // MECH_ADMM's fold is gradient-free (gr = 0, h = rho), so extending
                // it BELOW knot_lo is a pure proximal term (rho/2)|dz_rows|^2 on the
                // dz-QP — unbiased, no z/y row state. Required for the f32 factor:
                // without it the Schur strip is rho-heterogeneous at the excluded
                // knots and the bdsv factor error feeds the y-integrator (measured:
                // exponential y windup, x2.2/inner-iter, once the fold at knot 0
                // was correctly dropped from the ROW semantics).
                const bool admm_prox = grp.mech == MECH_ADMM && knot < grp.knot_lo;
                if ((knot < grp.knot_lo && !admm_prox) || knot >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                for (int32_t i = rank; i < grp.n_rows; i += size) {
                        T gr, h;
                        if (grp.mech == MECH_ADMM) {
                                // constant rho*G^T G diagonal fold — the gradient half is
                                // per-ADMM-iteration (kernels/admm.cuh), NOT here
                                gr = static_cast<T>(0);
                                h = grp.mu * admm_rho_scale;  // mu = ADMM rho (x per-solve adaptation scale)
                        } else if (grp.mech == MECH_AL) {
                                const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                glass::al_interval_grad_hess<T>(g, grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma, gr, h);
                        } else {
                                const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                gr = glass::relaxed_barrier_interval_grad<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
                                h = glass::relaxed_barrier_interval_hess<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
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
__device__ __noinline__ T row_cost_value(const RowGroupDesc<T>* __restrict__ groups,
                            int32_t n_groups, int32_t knot, const T* xu_k,
                            const T* __restrict__ d_lam_hi, const T* __restrict__ d_lam_lo,
                            bool has_control,
                            const T* __restrict__ d_z_admm = nullptr,
                            const T* __restrict__ d_y_admm = nullptr,
                            T admm_rho_scale = static_cast<T>(1))
{
        T total = static_cast<T>(0);
        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = groups[gi];
                if (!is_selection_kind(grp.kind) && grp.kind != LIN_U) continue;  // EE rows: ee_row_cost_value
                const bool admm_term = grp.mech == MECH_ADMM && d_z_admm != nullptr;
                if (grp.mech != MECH_BARRIER_RELAXED && grp.mech != MECH_AL && !admm_term) continue;
                if (knot < grp.knot_lo || knot >= grp.knot_hi) continue;
                if (grp.block == BLOCK_U && !has_control) continue;
                if (grp.kind == LIN_U && grp.cone && !admm_term) {
                        // vector value terms (the admm merit term below is per-row
                        // summable and needs no special case)
                        const int32_t m = grp.n_rows;
                        T g[MAX_ROWS_PER_GROUP];
                        for (int32_t i = 0; i < m; i++) { g[i] = eval_row<T>(grp, xu_k, (uint32_t)i); }
                        if (grp.mech == MECH_AL) {
                                T lam[MAX_ROWS_PER_GROUP];
                                for (int32_t i = 0; i < m; i++) { lam[i] = d_lam_hi[row_state_index(gi, (uint32_t)knot, (uint32_t)i)]; }
                                total += glass::al_soc_value<T>(g, lam, grp.mu, m);
                        } else {  // MECH_BARRIER_RELAXED: margin barrier
                                total += glass::relaxed_barrier_value<T>(g[0] - glass::soc_tail_norm<T>(g, m), grp.mu, grp.delta);
                        }
                        continue;
                }
                for (int32_t i = 0; i < grp.n_rows; i++) {
                        const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                        if (admm_term) {
                                // AL-form value on the CURRENT (z, y) row state — the
                                // set_admm_merit ablation (R1): lets the line search see
                                // the feasibility progress the ADMM-modified QP encodes
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                const T r = g - d_z_admm[idx];
                                total += d_y_admm[idx] * r + static_cast<T>(0.5) * grp.mu * admm_rho_scale * r * r;
                        } else if (grp.mech == MECH_AL) {
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                total += glass::al_interval_value<T>(g, grp.lo[i], grp.hi[i], d_lam_hi[idx], d_lam_lo[idx], grp.mu, grp.sigma);
                        } else {
                                total += glass::relaxed_barrier_interval_value<T>(g, grp.lo[i], grp.hi[i], grp.mu, grp.delta);
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

// COLLISION groups have n_rows = plant::NCC (> MAX_ROWS_PER_GROUP), so the
// telemetry / dual-update scratch carves size for the max unconditionally
// (these are tiny once-per-solve kernels; a few KB of smem is not a
// bottleneck and it keeps the sizers flag-free).
constexpr uint32_t TELEMETRY_MAX_ROWS = (uint32_t)gato::plant::NCC > MAX_ROWS_PER_GROUP ? (uint32_t)gato::plant::NCC : MAX_ROWS_PER_GROUP;

template<typename T>
__host__ __device__ constexpr uint32_t rowgroup_eval_scratch_ct()
{
        return ee_rows_scratch_ct<T>() > collision_rows_scratch_ct<T>() ? ee_rows_scratch_ct<T>() : collision_rows_scratch_ct<T>();
}

template<typename T>
__host__ __device__ constexpr uint32_t rowgroup_eval_grad_scratch_ct()
{
        return ee_rows_grad_scratch_ct<T>() > collision_rows_grad_scratch_ct<T>() ? ee_rows_grad_scratch_ct<T>() : collision_rows_grad_scratch_ct<T>();
}

template<typename T>
__global__ __launch_bounds__(ROWGROUP_THREADS) void rowGroupTelemetryBatchedKernel(T* __restrict__                      d_telemetry,
                                                                                   const RowGroupDesc<T>* __restrict__ d_groups,
                                                                                   int32_t                              n_groups,
                                                                                   const T* __restrict__                d_xu_traj_batch,
                                                                                   const grid::robotModel<T>* __restrict__ d_robot_model,
                                                                                   const grid_collision::Environment<T> env)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

        extern __shared__ char s_raw[];
        T* s_viol = reinterpret_cast<T*>(s_raw);                    // KNOT_POINTS * TELEMETRY_MAX_ROWS
        T* s_ee_scratch = s_viol + KNOT_POINTS * TELEMETRY_MAX_ROWS;  // rowgroup_eval_scratch_ct

        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::XU_TRAJ_SIZE;

        for (int32_t grp_idx = 0; grp_idx < n_groups; grp_idx++) {
                const RowGroupDesc<T>& grp = d_groups[grp_idx];
                const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                const int32_t n_elems = n_knots * grp.n_rows;

                if (grp.kind == EE_POS) {
                        // cooperative FK per active knot, then per-row violations
                        for (int32_t k = 0; k < n_knots; k++) {
                                const T* xu_k = d_xu + (size_t)(grp.knot_lo + k) * constants::XU_KNOT_STRIDE;
                                const T* s_pose = ee_eval_pose<T>(xu_k, s_ee_scratch, d_robot_model);
                                for (int32_t i = rank; i < grp.n_rows; i += size) {
                                        s_viol[k * grp.n_rows + i] = glass::interval_violation<T>(s_pose[i], grp.lo[i], grp.hi[i]);
                                }
                                __syncthreads();
                        }
                } else if (grp.kind == COLLISION) {
                        // cooperative clearance per active knot; TRUE min-clearance
                        // violation vs the uniform margin (one-sided)
                        T* s_dist = s_ee_scratch;
                        T* s_arena = align16_ptr<T>(s_dist + gato::plant::NCC);
                        for (int32_t k = 0; k < n_knots; k++) {
                                const T* xu_k = d_xu + (size_t)(grp.knot_lo + k) * constants::XU_KNOT_STRIDE;
                                gato::plant::collisionDist<T>(s_dist, xu_k, s_arena, d_robot_model, env);
                                for (int32_t i = rank; i < grp.n_rows; i += size) {
                                        s_viol[k * grp.n_rows + i] = glass::interval_violation<T>(s_dist[i], grp.lo[0], grp.hi[0]);
                                }
                                __syncthreads();
                        }
                } else if (grp.kind == LIN_U && grp.cone) {
                        // one margin violation per knot (slot i == 0; others 0)
                        for (int32_t e = rank; e < n_elems; e += size) {
                                const int32_t knot = grp.knot_lo + e / grp.n_rows;
                                const int32_t i = e % grp.n_rows;
                                if (i != 0) {
                                        s_viol[e] = static_cast<T>(0);
                                        continue;
                                }
                                const T* xu_k = d_xu + (size_t)knot * constants::XU_KNOT_STRIDE;
                                T g[MAX_ROWS_PER_GROUP];
                                for (int32_t j = 0; j < grp.n_rows; j++) { g[j] = eval_row<T>(grp, xu_k, (uint32_t)j); }
                                s_viol[e] = glass::soc_violation<T>(g, grp.n_rows);
                        }
                } else {
                        for (int32_t e = rank; e < n_elems; e += size) {
                                const int32_t knot = grp.knot_lo + e / grp.n_rows;
                                const int32_t i = e % grp.n_rows;
                                const T* xu_k = d_xu + (size_t)knot * constants::XU_KNOT_STRIDE;
                                const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                s_viol[e] = glass::interval_violation<T>(g, grp.lo[i], grp.hi[i]);
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
        return sizeof(T) * (KNOT_POINTS * TELEMETRY_MAX_ROWS + rowgroup_eval_scratch_ct<T>());
}

template<typename T>
__host__ void rowGroupTelemetryBatched(uint32_t batch_size, T* d_telemetry, const RowGroupDesc<T>* d_groups, int32_t n_groups, const T* d_xu_traj_batch, const void* d_GRiD_mem,
                                       const grid_collision::Environment<T>& env = grid_collision::Environment<T>{})
{
        if (n_groups <= 0) return;
        rowGroupTelemetryBatchedKernel<T><<<batch_size, ROWGROUP_THREADS, getRowGroupTelemetrySMemSize<T>()>>>(d_telemetry, d_groups, n_groups, d_xu_traj_batch,
                                                                                                               (const grid::robotModel<T>*)d_GRiD_mem, env);
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
                                                                              const grid::robotModel<T>* __restrict__ d_robot_model,
                                                                              const grid_collision::Environment<T> env)
{
        const uint32_t solve_idx = blockIdx.x;
        const uint32_t rank = threadIdx.x;
        const uint32_t size = blockDim.x;

        extern __shared__ char s_raw[];
        T* s_ee_scratch = reinterpret_cast<T*>(s_raw);  // rowgroup_eval_scratch_ct

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

        T*       d_lam_hi = d_lam_hi_batch + (size_t)solve_idx * TOTAL_ROW_STATE_SIZE;
        T*       d_lam_lo = d_lam_lo_batch + (size_t)solve_idx * TOTAL_ROW_STATE_SIZE;
        const T* d_xu = d_xu_traj_batch + (size_t)solve_idx * constants::XU_TRAJ_SIZE;

        for (int32_t gi = 0; gi < n_groups; gi++) {
                const RowGroupDesc<T>& grp = d_groups[gi];
                if (grp.mech != MECH_AL) continue;
                const int32_t n_knots = grp.knot_hi - grp.knot_lo;
                if (grp.kind == COLLISION) {
                        // one-sided rows (lo = margin, hi = +inf): only lam_lo
                        // accumulates, against the TRUE min clearance. Band-indexed.
                        T* s_dist = s_ee_scratch;
                        T* s_arena = align16_ptr<T>(s_dist + gato::plant::NCC);
                        const T margin = grp.lo[0];
                        const bool soft = grp.sigma > static_cast<T>(0);
                        for (int32_t k = 0; k < n_knots; k++) {
                                const int32_t knot = grp.knot_lo + k;
                                const T* xu_k = d_xu + (size_t)knot * constants::XU_KNOT_STRIDE;
                                gato::plant::collisionDist<T>(s_dist, xu_k, s_arena, d_robot_model, env);  // all threads
                                for (int32_t i = rank; i < grp.n_rows; i += size) {
                                        const uint32_t idx = collision_row_state_index((uint32_t)knot, (uint32_t)i);
                                        T a = d_lam_lo[idx] + grp.mu * (margin - s_dist[i]);
                                        if (a < static_cast<T>(0)) a = static_cast<T>(0);
                                        if (soft && a > grp.sigma) a = grp.sigma;
                                        d_lam_lo[idx] = a;
                                }
                                __syncthreads();  // s_dist reused next knot
                        }
                        continue;
                }
                if (grp.kind == LIN_U && grp.cone) {
                        // conic dual update lam <- Π_K(lam - rho*g): the row vector is
                        // coupled, so each KNOT is owned by one thread (deterministic)
                        for (int32_t k = rank; k < n_knots; k += size) {
                                const int32_t knot = grp.knot_lo + k;
                                const T* xu_k = d_xu + (size_t)knot * constants::XU_KNOT_STRIDE;
                                T w[MAX_ROWS_PER_GROUP];
                                for (int32_t i = 0; i < grp.n_rows; i++) {
                                        const T g = eval_row<T>(grp, xu_k, (uint32_t)i);
                                        w[i] = d_lam_hi[row_state_index(gi, (uint32_t)knot, (uint32_t)i)] - grp.mu * g;
                                }
                                glass::thread::soc_project<T>(w, w, grp.n_rows);
                                for (int32_t i = 0; i < grp.n_rows; i++) { d_lam_hi[row_state_index(gi, (uint32_t)knot, (uint32_t)i)] = w[i]; }
                        }
                        continue;
                }
                for (int32_t k = 0; k < n_knots; k++) {
                        const int32_t knot = grp.knot_lo + k;
                        const T* xu_k = d_xu + (size_t)knot * constants::XU_KNOT_STRIDE;
                        const T* s_pose = nullptr;
                        if (grp.kind == EE_POS) {
                                s_pose = ee_eval_pose<T>(xu_k, s_ee_scratch, d_robot_model);  // all threads
                        }
                        const bool soft = grp.sigma > static_cast<T>(0);
                        for (int32_t i = rank; i < grp.n_rows; i += size) {
                                const T g = (grp.kind == EE_POS) ? s_pose[i] : eval_row<T>(grp, xu_k, (uint32_t)i);
                                const uint32_t idx = row_state_index(gi, (uint32_t)knot, (uint32_t)i);
                                if (glass::al_is_eq_row<T>(grp.lo[i], grp.hi[i])) {
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
                                  const RowGroupDesc<T>* d_groups, int32_t n_groups, const void* d_GRiD_mem,
                                  const grid_collision::Environment<T>& env = grid_collision::Environment<T>{})
{
        alDualUpdateBatchedKernel<T><<<batch_size, ROWGROUP_THREADS, sizeof(T) * rowgroup_eval_scratch_ct<T>()>>>(d_lam_hi_batch, d_lam_lo_batch, d_prev_viol_batch, d_telemetry, d_xu_traj_batch, d_groups,
                                                                                                                  n_groups, (const grid::robotModel<T>*)d_GRiD_mem, env);
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
        // CTRL_LIMITS has ACTUATED_SIZE rows — on contact-force builds
        // (CONTROL_SIZE > ACTUATED_SIZE) the fc slots have no limit table row.
        constexpr int32_t NA = constants::ACTUATED_SIZE;

        if (rank == 0) {
                // State boxes start at knot 1: x_0 is DATA (pinned to the
                // measurement by the initial-state constraint), so knot-0
                // state rows are unsatisfiable whenever the measured state
                // violates a limit — AL multipliers wind up (or the
                // acceptance gate freezes them) on rows no step can fix and
                // closed-loop MPC destabilizes (R1 measured: indy7 reach
                // cascade). Same class as the ADMM equality-row reinit.
                d_groups[0].kind = BOX_Q;
                d_groups[0].block = BLOCK_X;
                d_groups[0].n_rows = NQ;
                d_groups[0].knot_lo = 1;
                d_groups[0].knot_hi = KNOT_POINTS;

                d_groups[1].kind = BOX_QD;
                d_groups[1].block = BLOCK_X;
                d_groups[1].n_rows = NQ;
                d_groups[1].knot_lo = 1;
                d_groups[1].knot_hi = KNOT_POINTS;

                d_groups[2].kind = BOX_U;
                d_groups[2].block = BLOCK_U;
                d_groups[2].n_rows = NA;
                d_groups[2].knot_lo = 0;
                d_groups[2].knot_hi = KNOT_POINTS - 1;

                for (int g = 0; g < 3; g++) {
                        d_groups[g].mech = mech;
                        d_groups[g].mu = mu;
                        d_groups[g].delta = delta;
                        d_groups[g].sigma = static_cast<T>(0);  // hard by default
                        d_groups[g].cone = 0;
                }
        }
        for (int32_t i = rank; i < NQ; i += size) {
                d_groups[0].lo[i] = gato::plant::JOINT_LIMITS<T>()[i][0];
                d_groups[0].hi[i] = gato::plant::JOINT_LIMITS<T>()[i][1];
                d_groups[1].lo[i] = gato::plant::VEL_LIMITS<T>()[i][0];
                d_groups[1].hi[i] = gato::plant::VEL_LIMITS<T>()[i][1];
        }
        for (int32_t i = rank; i < NA; i += size) {
                d_groups[2].lo[i] = gato::plant::CTRL_LIMITS<T>()[i][0];
                d_groups[2].hi[i] = gato::plant::CTRL_LIMITS<T>()[i][1];
        }
}

}  // namespace rows
}  // namespace gato
