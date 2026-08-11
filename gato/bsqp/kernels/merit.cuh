#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "settings.h"
#include "constants.h"
#include "utils/cuda.cuh"
#include "utils/linalg.cuh"
#include "glass.cuh"  // top-level GLASS (global glass::, distinct from grid.cuh's grid::glass)
#include "dynamics/integrator.cuh"
#include "dynamics/manifold.cuh"
#include "dynamics/grid_plant_step.cuh"  // floating twins + GATO_FLOATING_STEP
#include "bsqp/rowgroups.cuh"

using namespace sqp;
using namespace gato;
using namespace gato::constants;

// element count of the merit kernel's s_temp tail (the larger consumer of the
// cost-value and integrator-error scratch). The COLLISION value carve overlays
// this region when it fits — s_temp is dead between those two consumers.
template<typename T>
__host__ __device__ constexpr size_t computeMeritTempMemCt()
{
        constexpr size_t a = gato::plant::trackingCostValue_TempMemCt<T>();
#if GATO_FLOATING_STEP
        // floating: the integrator-error scratch is the grid value-step arena
        // (also covers the terminal knot's trial-x0 + tangent-gap carve:
        // XU_STATE_SIZE + STATE_SIZE << the arena)
        constexpr size_t b = gato::plant::stepValueFloating_TempMemCt<T>();
#else
        constexpr size_t b = gato::plant::forwardDynamics_TempMemSize_Shared();
#endif
        return a > b ? a : b;
}

template<typename T>
__host__ __device__ constexpr size_t computeMeritBaseSMemCt()
{
        // xux_k + reference_traj_k + s_temp (used by BOTH the kernel's carve and
        // the host sizer — the collision tail carve offsets from this)
        return constants::XUX_SIZE + constants::EE_POS_SIZE + computeMeritTempMemCt<T>();
}

// __launch_bounds__ min-2-blocks: this kernel sat at 95 regs after the P4.3/P4.4
// wave — 2 registers past the 2-blocks/SM bound at MAX_PERF_LEVEL_THREADS=352 on
// sm_120 (65536/(2*352) = 93). The halved occupancy doubled the kernel at
// saturated B (+62% whole-solve, the 2026-08-09 bisect's entire regression);
// capping regs trades a ~2-slot spill for 2x occupancy.
template<typename T, unsigned INTEGRATOR_TYPE = gato::constants::INTEGRATOR_TYPE_DEFAULT, bool ANGLE_WRAP = false>
__global__ void __launch_bounds__(grid::MAX_PERF_LEVEL_THREADS, 2)
computeMeritBatchedKernel(T* __restrict__       d_merit_partial_batch,  // per-(solve,alpha,knot) partials [merit_index * KNOT_POINTS + knot]
                                          T* __restrict__       d_dz_batch,
                                          T* __restrict__       d_xu_traj_batch,
                                          T* __restrict__       d_x_initial_batch,
                                          T* __restrict__       d_reference_traj_batch,
                                          void*                 d_GRiD_mem,
                                          const T* __restrict__ d_mu_batch,
                                          T* __restrict__       d_f_ext_batch,
                                          T                     timestep,
                                          T                     q_cost,
                                          T                     qd_cost,
                                          T                     u_cost,
                                          T                     N_cost,
                                          T                     q_lim_cost,
                                          T                     vel_lim_cost,
                                          T                     ctrl_lim_cost,
                                          T                     q_pos_cost,
                                          const T* __restrict__ d_q_nom,
                                          T                     fc_cost,
                                          const T* __restrict__ d_u_cost_vec,
                                          const T* __restrict__ d_q_pos_w_vec,
                                          const T* __restrict__ d_fc_ref,
                                          const int32_t* __restrict__ d_kkt_converged_batch,
                                          const T* __restrict__       d_knot_cost_weights,  // optional (nullable) per-knot [ee,qd,u] triples
                                          const gato::rows::RowGroupDesc<T>* __restrict__ d_row_groups,  // MECH_BARRIER_RELAXED / MECH_AL value terms
                                          int32_t                     n_row_groups,                      // (must match setup_kkt's fold; 0 -> untouched)
                                          const T* __restrict__       d_lam_hi_batch,                    // per-solve AL duals (nullable; MECH_AL only)
                                          const T* __restrict__       d_lam_lo_batch,
                                          const T* __restrict__       d_z_admm_batch,                    // per-solve ADMM row state (nullable; only with
                                          const T* __restrict__       d_y_admm_batch,                    // set_admm_merit — adds the AL-form ADMM value term)
                                          const grid_collision::Environment<T> env,                      // obstacle set (COLLISION rows; empty default = no-op)
                                          const T* __restrict__       d_admm_rho_scale_batch)            // per-solve ADMM rho adaptation scale (nullable -> 1)
{
        // launched with 3D grid (KNOT_POINTS, batch_size, num_alphas)

        grid::robotModel<T>* d_robot_model = (grid::robotModel<T>*)d_GRiD_mem;
        const uint32_t       solve_idx = blockIdx.y;
        if (d_kkt_converged_batch != nullptr && d_kkt_converged_batch[solve_idx]) return;  // converged: skip (line search skips too)
        const uint32_t       knot_idx = blockIdx.x;
        const uint32_t       alpha_idx = blockIdx.z;
        T                    alpha = 1.0 / (1 << alpha_idx);
        T                    mu = d_mu_batch[solve_idx];

        T cost_k, constraint_k;  // cost function, constraint error, per-point merit

        extern __shared__ T s_mem[];
        T*                  s_xux_k = s_mem;  // current state, control, and next state (STORED format)
        T*                  s_reference_traj_k = s_xux_k + constants::XUX_SIZE;
        T*                  s_temp = s_reference_traj_k + constants::EE_POS_SIZE;


        T* d_xu_k = getOffsetXU<T>(d_xu_traj_batch, solve_idx, knot_idx);
        T* d_dz_k = getOffsetDz<T>(d_dz_batch, solve_idx, knot_idx);
        T* d_x_initial_k = d_x_initial_batch + solve_idx * XU_STATE_SIZE;  // STORED format
        T* d_f_ext = getOffsetWrench<T>(d_f_ext_batch, solve_idx, knot_idx);

        // line-search trial step: s_xux_k = d_xu_k ⊞ alpha * d_dz_k
        // (fixed base: plain axpby; floating: per-knot state retract + control
        // axpy — s_xux_k is STORED format [q(NQ); qd(NV); u; x_next]).
        if constexpr (!FLOATING_BASE) {
                if (knot_idx == KNOT_POINTS - 1) {
                        glass::axpby<T, STATE_SIZE>(static_cast<T>(1), d_xu_k, alpha, d_dz_k, s_xux_k);
                } else {
                        glass::axpby<T, STATE_SIZE + STATE_S_CONTROL>(static_cast<T>(1), d_xu_k, alpha, d_dz_k, s_xux_k);
                }
        } else {
                gato::plant::state_retract<T>(s_xux_k, d_xu_k, d_dz_k, alpha);
                if (knot_idx < KNOT_POINTS - 1) {
                        for (uint32_t i = threadIdx.x; i < CONTROL_SIZE; i += blockDim.x)
                                s_xux_k[XU_STATE_SIZE + i] = d_xu_k[XU_STATE_SIZE + i] + alpha * d_dz_k[STATE_SIZE + i];
                        gato::plant::state_retract<T>(s_xux_k + XU_KNOT_STRIDE,
                                                      d_xu_k + XU_KNOT_STRIDE,
                                                      d_dz_k + DZ_KNOT_STRIDE, alpha);
                }
        }

        T* d_reference_traj_k = getOffsetReferenceTraj<T>(d_reference_traj_batch, solve_idx, knot_idx);
        glass::copy<T, constants::EE_POS_SIZE>(d_reference_traj_k, s_reference_traj_k);
        __syncthreads();

        // per-knot weight override (nullptr -> the scalar weights); with per-knot weights
        // the terminal EE weight is just the last knot's entry, so pass it as both the
        // running (q_cost) and terminal (N_cost) arguments.
        const T ee_w_k = d_knot_cost_weights ? d_knot_cost_weights[knot_idx * 3 + 0] : q_cost;
        const T qd_w_k = d_knot_cost_weights ? d_knot_cost_weights[knot_idx * 3 + 1] : qd_cost;
        const T u_w_k  = d_knot_cost_weights ? d_knot_cost_weights[knot_idx * 3 + 2] : u_cost;
        const T eeN_w  = d_knot_cost_weights ? ee_w_k : N_cost;

        // cost function (grid_plant::tracking_cost via the adapter; terminal knot picks
        // N_cost EE weight + drops the control reg/barrier, matching the old trackingcost).
        cost_k =
            plant::trackingCostValue<T>(s_xux_k, s_xux_k + XU_STATE_SIZE, s_reference_traj_k, s_temp, d_robot_model, ee_w_k, qd_w_k, u_w_k, eeN_w, q_lim_cost, vel_lim_cost, ctrl_lim_cost, /*is_terminal=*/(knot_idx == KNOT_POINTS - 1), q_pos_cost, d_q_nom, fc_cost, d_u_cost_vec, d_q_pos_w_vec, d_fc_ref);
        __syncthreads();

        // row-group mechanism value terms (RB barrier / AL — must mirror setup_kkt's
        // grad/hess fold or the line search accepts against a different objective)
        if (n_row_groups > 0) {
                const T* d_lam_hi = d_lam_hi_batch ? d_lam_hi_batch + (size_t)solve_idx * gato::rows::TOTAL_ROW_STATE_SIZE : nullptr;
                const T* d_lam_lo = d_lam_lo_batch ? d_lam_lo_batch + (size_t)solve_idx * gato::rows::TOTAL_ROW_STATE_SIZE : nullptr;
                const T* d_z_admm = d_z_admm_batch ? d_z_admm_batch + (size_t)solve_idx * gato::rows::TOTAL_ROW_STATE_SIZE : nullptr;
                const T* d_y_admm = d_y_admm_batch ? d_y_admm_batch + (size_t)solve_idx * gato::rows::TOTAL_ROW_STATE_SIZE : nullptr;
                const T admm_rho_scale = d_admm_rho_scale_batch ? d_admm_rho_scale_batch[solve_idx] : static_cast<T>(1);
                cost_k += gato::rows::row_cost_value<T>(d_row_groups, n_row_groups, (int32_t)knot_idx, s_xux_k, d_lam_hi, d_lam_lo, /*has_control=*/(knot_idx < KNOT_POINTS - 1), d_z_admm, d_y_admm, admm_rho_scale);
                // EE_POS rows: cooperative FK at the CANDIDATE state (true nonlinear
                // value — the fold linearizes, the merit must not). s_temp is free
                // between trackingCostValue and the constraint-error section, and
                // trackingCostValue_TempMemCt >= the EE value carve.
                if (gato::rows::has_ee_rows<T>(d_row_groups, n_row_groups, (int32_t)knot_idx)) {
                        __syncthreads();
                        cost_k += gato::rows::ee_row_cost_value<T>(d_row_groups, n_row_groups, (int32_t)knot_idx, s_xux_k, d_lam_hi, d_lam_lo, s_temp, d_robot_model, d_z_admm, d_y_admm, admm_rho_scale);
                }
                // COLLISION rows: true nonlinear clearance value at the CANDIDATE
                // state (same mirroring requirement). s_temp is dead here (like
                // the EE carve above), so the collision carve OVERLAYS it when it
                // fits — the constraint-error section below re-uses s_temp, hence
                // the trailing barrier; only the excess over the temp tail (small
                // fixed-base arenas) extends the host-sized launch.
                if (gato::rows::has_collision_rows<T>(d_row_groups, n_row_groups, (int32_t)knot_idx)) {
                        constexpr size_t cc_ct = gato::rows::collision_rows_scratch_ct<T>();
                        constexpr size_t tail_ct = computeMeritTempMemCt<T>();
                        T* s_cc = (cc_ct <= tail_ct) ? s_temp : s_mem + computeMeritBaseSMemCt<T>();
                        __syncthreads();
                        cost_k += gato::rows::collision_row_cost_value<T>(d_row_groups, n_row_groups, (int32_t)knot_idx, s_xux_k, d_lam_hi, d_lam_lo, s_cc, d_robot_model, env, d_z_admm, d_y_admm, admm_rho_scale);
                        __syncthreads();
                }
        }

        // constraint error
        if (knot_idx < KNOT_POINTS - 1) {  // not last knot
#if GATO_FLOATING_STEP
                constraint_k = gato::plant::compute_integrator_error_floating<T, INTEGRATOR_TYPE>(s_xux_k, s_xux_k + XU_KNOT_STRIDE, s_temp, d_robot_model, timestep, d_f_ext);
#else
                constraint_k = gato::plant::compute_integrator_error<T, INTEGRATOR_TYPE, ANGLE_WRAP>(s_xux_k, s_xux_k + XU_KNOT_STRIDE, s_temp, d_robot_model, timestep, d_f_ext);
#endif
        } else {
                d_xu_k = getOffsetXU<T>(d_xu_traj_batch, solve_idx, 0);
                d_dz_k = getOffsetDz<T>(d_dz_batch, solve_idx, 0);
#if GATO_FLOATING_STEP
                // trial x_0 = xu_0 ⊞ α·dz_0, then the TANGENT gap to the stored
                // initial state: |x_0^trial ⊟ x_s|_1
                T* s_x0  = s_temp;                  // stored trial state (NQ+NV)
                T* s_gap = s_temp + XU_STATE_SIZE;  // tangent gap (2*NV)
                gato::plant::state_retract<T>(s_x0, d_xu_k, d_dz_k, alpha);
                __syncthreads();
                gato::plant::state_difference<T>(s_gap, /*from=*/d_x_initial_k, /*to=*/s_x0);
                __syncthreads();
                for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) { s_gap[i] = abs(s_gap[i]); }
                __syncthreads();
                glass::reduce<T, STATE_SIZE>(s_gap);
                __syncthreads();
                constraint_k = s_gap[0];
#else
                for (uint32_t i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
                        s_temp[i] = abs(d_xu_k[i] + alpha * d_dz_k[i] - d_x_initial_k[i]);  // initial state constraint error
                }
                __syncthreads();
                // block-wide GLASS reduce (sum -> s_temp[0]); NOT glass::warp::reduce: this block
                // runs MAX_PERF_LEVEL_THREADS (multi-warp), and warp::reduce's contract is one
                // 32-lane warp owning the reduction, so it is unsafe across >1 warp here.
                glass::reduce<T, STATE_SIZE>(s_temp);
                __syncthreads();
                constraint_k = s_temp[0];
#endif
        }
        __syncthreads();

        // per-knot partial write (one block owns one slot — no atomics). The
        // final merit is summed in FIXED knot order by reduceMeritPartialsKernel:
        // the old atomicAdd accumulation summed in schedule order, whose ±1ulp
        // run-to-run jitter could flip line-search ties and break trajectory
        // bit-determinism (the 2026-08-01 P4.4 class) — two-pass is exact.
        if (threadIdx.x == 0) {
                uint32_t merit_index = solve_idx * gridDim.z + alpha_idx;
                d_merit_partial_batch[(size_t)merit_index * KNOT_POINTS + knot_idx] = cost_k + mu * constraint_k;
        }
}

// fixed-order per-slot reduction of the per-knot merit partials (deterministic
// by construction; one thread per (solve, alpha) slot, serial over knots).
// Converged solves are skipped — their d_merit_batch slots stay untouched,
// mirroring the merit kernel's own converged-skip (line search skips them too).
template<typename T>
__global__ void reduceMeritPartialsKernel(T* __restrict__ d_merit_batch,
                                          const T* __restrict__ d_merit_partial_batch,
                                          uint32_t n_slots,
                                          uint32_t num_alphas,
                                          const int32_t* __restrict__ d_kkt_converged_batch)
{
        const uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
        if (m >= n_slots) return;
        if (d_kkt_converged_batch != nullptr && d_kkt_converged_batch[m / num_alphas]) return;
        const T* p = d_merit_partial_batch + (size_t)m * KNOT_POINTS;
        T acc = static_cast<T>(0);
        for (uint32_t k = 0; k < KNOT_POINTS; k++) { acc += p[k]; }
        d_merit_batch[m] = acc;
}

template<typename T>
__host__ size_t getComputeMeritBatchedSMemSize(int has_collision = 0)
{
        size_t size = sizeof(T) * computeMeritBaseSMemCt<T>();
        // runtime-sized: only when a COLLISION group is registered (host-known)
        // AND its carve exceeds the dead s_temp tail it overlays in the kernel
        if (has_collision) {
                constexpr size_t cc_ct = gato::rows::collision_rows_scratch_ct<T>();
                constexpr size_t tail_ct = computeMeritTempMemCt<T>();
                size += sizeof(T) * (cc_ct > tail_ct ? cc_ct - tail_ct : (size_t)0);
        }
        return size;
}

template<typename T, uint32_t NumAlphas>
__host__ void computeMeritBatched(uint32_t                    batch_size,
                                  const int32_t*              d_kkt_converged_batch,  // nullptr => no converged-skip (initial merit)
                                  const T*                    d_knot_cost_weights,    // nullptr => scalar weights
                                  T*                          d_merit_batch,
                                  T*                          d_merit_partial_batch,  // scratch: >= batch * NumAlphas * KNOT_POINTS
                                  T*                          d_dz_batch,
                                  T*                          d_xu_traj_batch,
                                  T*                          d_f_ext_batch,
                                  ProblemInputs<T> inputs,
                                  T*                          d_mu_batch,
                                  void*                       d_GRiD_mem,
                                  T                           q_cost,
                                  T                           qd_cost,
                                  T                           u_cost,
                                  T                           N_cost,
                                  T                           q_lim_cost,
                                  T                           vel_lim_cost,
                                  T                           ctrl_lim_cost,
                                  const gato::rows::RowGroupDesc<T>* d_row_groups = nullptr,
                                  int32_t                     n_row_groups = 0,
                                  const T*                    d_lam_hi_batch = nullptr,
                                  const T*                    d_lam_lo_batch = nullptr,
                                  const T*                    d_z_admm_batch = nullptr,
                                  const T*                    d_y_admm_batch = nullptr,
                                  int32_t                     has_collision = 0,
                                  const grid_collision::Environment<T>& env = grid_collision::Environment<T>{},
                                  const T*                    d_admm_rho_scale_batch = nullptr,
                                  T                           q_pos_cost = 0,
                                  const T*                    d_q_nom = nullptr,
                                  T                           fc_cost = 0,
                                  const T*                    d_u_cost_vec = nullptr,
                                  const T*                    d_q_pos_w_vec = nullptr,
                                  const T*                    d_fc_ref = nullptr)
{
        dim3   grid(KNOT_POINTS, batch_size, NumAlphas);
        dim3   thread_block(grid::MAX_PERF_LEVEL_THREADS);  // regen removed grid::SUGGESTED_THREADS
        size_t s_mem_size = getComputeMeritBatchedSMemSize<T>(has_collision);
        // mirror setup_kkt: opt in past the 48KB default once, FAIL LOUD on both
        // the attribute set and the launch (silent launch failures leave the
        // merit buffers unwritten while the line search "runs")
        if (s_mem_size > 48 * 1024) {
                static bool attr_set = false;
                if (!attr_set) {
                        gpuErrchk(cudaFuncSetAttribute(computeMeritBatchedKernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s_mem_size));
                        attr_set = true;
                }
        }

        computeMeritBatchedKernel<T><<<grid, thread_block, s_mem_size>>>(d_merit_partial_batch,
                                                                                    d_dz_batch,
                                                                                    d_xu_traj_batch,
                                                                                    inputs.d_x_s_batch,
                                                                                    inputs.d_reference_traj_batch,
                                                                                    d_GRiD_mem,
                                                                                    d_mu_batch,
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
                                                                                    d_z_admm_batch,
                                                                                    d_y_admm_batch,
                                                                                    env,
                                                                                    d_admm_rho_scale_batch);
        gpuErrchk(cudaGetLastError());  // launch-config failures must not pass silently
        const uint32_t n_slots = batch_size * NumAlphas;
        reduceMeritPartialsKernel<T><<<(n_slots + 127) / 128, 128>>>(d_merit_batch, d_merit_partial_batch, n_slots, NumAlphas, d_kkt_converged_batch);
}
