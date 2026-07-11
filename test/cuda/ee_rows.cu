// Standalone gate for the EE_POS row machinery (constraint-layer arc CL-1):
//
//   1. J-read: the row Jacobian used by apply_ee_row_grad_hess
//      (J_i[qi] = s_grad[6*qi + i]) must match central finite differences of
//      the SAME device pose evaluator — catches any layout/transpose mistake
//      independent of external references (the pose itself is separately
//      gated against pinocchio in test_rowgroups.py to ~1e-7).
//   2. Fold: the (Q, q) increments produced by apply_ee_row_grad_hess must
//      equal the host-recomputed dense GN fold from the dumped (pose, J) and
//      the AL scalars.
//
// Build (REAL headers; -arch=native REQUIRED — default-arch JIT miscompiles,
// see test/cuda/bdsv_factor_solve.cu):
//   nvcc -std=c++17 -O2 -DNDEBUG -arch=native -DKNOT_POINTS=8 \
//        -DGATO_PLANT_HEADER='"dynamics/plant.cuh"' \
//        -I gato -I gato/dynamics/indy7 -I external/GLASS -I test/cuda \
//        test/cuda/ee_rows.cu -o /tmp/ee_rows && /tmp/ee_rows

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "settings.h"
#include "constants.h"
#include "bsqp/rowgroups.cuh"

using namespace gato;
using T = float;

constexpr int NQ = constants::STATE_SIZE / 2;
constexpr int NROWS = 3;

// one block: pose + J at q, plus fold increments on zeroed (Q, q) blocks
__global__ void eeProbeKernel(T* d_pose, T* d_J, T* d_Qinc, T* d_qinc,
                              const T* d_xu, const rows::RowGroupDesc<T>* d_groups,
                              const T* d_lam_hi, const T* d_lam_lo,
                              const grid::robotModel<T>* drm)
{
        extern __shared__ T s_mem[];
        T* s_Q = s_mem;                                    // STATE_SIZE^2
        T* s_q = s_Q + constants::STATE_SIZE_SQ;           // STATE_SIZE
        T* s_scratch = s_q + constants::STATE_SIZE;

        for (uint32_t i = threadIdx.x; i < constants::STATE_SIZE_SQ; i += blockDim.x) s_Q[i] = 0;
        for (uint32_t i = threadIdx.x; i < constants::STATE_SIZE; i += blockDim.x) s_q[i] = 0;
        __syncthreads();

        // pose + J dump (the same carve apply_ee_row_grad_hess uses)
        T* s_pose = s_scratch;
        T* s_grad = s_pose + 6 * gato::plant::NEE;
        T* s_arena = rows::align16_ptr<T>(s_grad + 6 * NQ * gato::plant::NEE + 2 * rows::MAX_ROWS_PER_GROUP);
        gato::plant::eePosGrad<T>(s_pose, s_grad, d_xu, s_arena, drm);
        for (uint32_t i = threadIdx.x; i < 6 * gato::plant::NEE; i += blockDim.x) d_pose[i] = s_pose[i];
        for (uint32_t i = threadIdx.x; i < 6u * NQ; i += blockDim.x) d_J[i] = s_grad[i];
        __syncthreads();

        rows::apply_ee_row_grad_hess<T>(d_groups, 1, KNOT_POINTS - 1, d_xu, d_lam_hi, d_lam_lo, s_Q, s_q, s_scratch, drm);

        for (uint32_t i = threadIdx.x; i < constants::STATE_SIZE_SQ; i += blockDim.x) d_Qinc[i] = s_Q[i];
        for (uint32_t i = threadIdx.x; i < constants::STATE_SIZE; i += blockDim.x) d_qinc[i] = s_q[i];
}

__global__ void poseOnlyKernel(T* d_pose, const T* d_q, const grid::robotModel<T>* drm)
{
        extern __shared__ T s_mem[];
        const T* s_pose = rows::ee_eval_pose<T>(d_q, s_mem, drm);
        for (uint32_t i = threadIdx.x; i < 6 * gato::plant::NEE; i += blockDim.x) d_pose[i] = s_pose[i];
}

static T pose_at(T* d_pose, T* d_q, const std::vector<T>& q, const grid::robotModel<T>* drm, int row)
{
        cudaMemcpy(d_q, q.data(), q.size() * sizeof(T), cudaMemcpyHostToDevice);
        poseOnlyKernel<<<1, 128, sizeof(T) * rows::ee_rows_scratch_ct<T>()>>>(d_pose, d_q, drm);
        T h_pose[6 * gato::plant::NEE];
        cudaMemcpy(h_pose, d_pose, sizeof(h_pose), cudaMemcpyDeviceToHost);
        return h_pose[row];
}

int main()
{
        grid::robotModel<T>* drm = grid::init_robotModel<T>();

        std::vector<T> q = {0.3f, -0.5f, 0.8f, -0.2f, 0.6f, 0.1f, 0.4f};  // first NQ used
        q.resize(constants::STATE_S_CONTROL, 0.0f);

        // one MECH_AL EE_POS equality group with nonzero duals
        rows::RowGroupDesc<T> h_grp;
        memset(&h_grp, 0, sizeof(h_grp));
        h_grp.kind = rows::EE_POS;
        h_grp.block = rows::BLOCK_X;
        h_grp.mech = rows::MECH_AL;
        h_grp.n_rows = NROWS;
        h_grp.knot_lo = KNOT_POINTS - 1;
        h_grp.knot_hi = KNOT_POINTS;
        h_grp.mu = 7.0f;
        for (int i = 0; i < NROWS; i++) { h_grp.lo[i] = 0.1f * (i + 1); h_grp.hi[i] = h_grp.lo[i]; }

        std::vector<T> h_lam_hi(rows::ROW_STATE_SIZE, 0.0f), h_lam_lo(rows::ROW_STATE_SIZE, 0.0f);
        for (int i = 0; i < NROWS; i++) h_lam_hi[rows::row_state_index(0, KNOT_POINTS - 1, i)] = 0.5f * (i + 1) - 1.0f;  // signed eq multipliers

        T *d_xu, *d_pose, *d_J, *d_Qinc, *d_qinc, *d_lam_hi, *d_lam_lo;
        rows::RowGroupDesc<T>* d_groups;
        cudaMalloc(&d_xu, constants::STATE_S_CONTROL * sizeof(T));
        cudaMalloc(&d_pose, 6 * gato::plant::NEE * sizeof(T));
        cudaMalloc(&d_J, 6 * NQ * sizeof(T));
        cudaMalloc(&d_Qinc, constants::STATE_SIZE_SQ * sizeof(T));
        cudaMalloc(&d_qinc, constants::STATE_SIZE * sizeof(T));
        cudaMalloc(&d_groups, sizeof(h_grp));
        cudaMalloc(&d_lam_hi, rows::ROW_STATE_SIZE * sizeof(T));
        cudaMalloc(&d_lam_lo, rows::ROW_STATE_SIZE * sizeof(T));
        cudaMemcpy(d_xu, q.data(), constants::STATE_S_CONTROL * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_groups, &h_grp, sizeof(h_grp), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lam_hi, h_lam_hi.data(), rows::ROW_STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lam_lo, h_lam_lo.data(), rows::ROW_STATE_SIZE * sizeof(T), cudaMemcpyHostToDevice);

        size_t smem = sizeof(T) * (constants::STATE_SIZE_SQ + constants::STATE_SIZE + rows::ee_rows_grad_scratch_ct<T>());
        eeProbeKernel<<<1, 128, smem>>>(d_pose, d_J, d_Qinc, d_qinc, d_xu, d_groups, d_lam_hi, d_lam_lo, drm);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("FAIL kernel: %s\n", cudaGetErrorString(err)); return 1; }

        T h_pose[6 * gato::plant::NEE];
        std::vector<T> h_J(6 * NQ), h_Qinc(constants::STATE_SIZE_SQ), h_qinc(constants::STATE_SIZE);
        cudaMemcpy(h_pose, d_pose, sizeof(h_pose), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_J.data(), d_J, 6 * NQ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Qinc.data(), d_Qinc, constants::STATE_SIZE_SQ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_qinc.data(), d_qinc, constants::STATE_SIZE * sizeof(T), cudaMemcpyDeviceToHost);

        int fails = 0;

        // gate 1: J-read vs central FD of the device pose (rows 0..2)
        const T eps = 1e-3f;
        T max_rel = 0;
        for (int j = 0; j < NQ; j++) {
                for (int i = 0; i < NROWS; i++) {
                        std::vector<T> qp = q, qm = q;
                        qp[j] += eps;
                        qm[j] -= eps;
                        const T fd = (pose_at(d_pose, d_xu, qp, drm, i) - pose_at(d_pose, d_xu, qm, drm, i)) / (2 * eps);
                        const T an = h_J[6 * j + i];  // the read the fold uses
                        const T rel = fabsf(fd - an) / fmaxf(fabsf(fd), 1e-3f);
                        if (rel > max_rel) max_rel = rel;
                        if (rel > 5e-2f) {
                                printf("FAIL J[%d][q%d]: analytic %.6f fd %.6f\n", i, j, an, fd);
                                fails++;
                        }
                }
        }
        printf("gate 1 (J-read vs device FD): max rel %.3e %s\n", max_rel, fails ? "FAIL" : "PASS");

        // gate 2: fold increments vs host recompute from (pose, J)
        cudaMemcpy(d_xu, q.data(), constants::STATE_S_CONTROL * sizeof(T), cudaMemcpyHostToDevice);  // restore
        T gr[NROWS], hh[NROWS];
        for (int i = 0; i < NROWS; i++) {
                const T c = h_pose[i] - h_grp.hi[i];
                gr[i] = h_lam_hi[rows::row_state_index(0, KNOT_POINTS - 1, i)] + h_grp.mu * c;  // equality: always active
                hh[i] = h_grp.mu;
        }
        T worst = 0;
        for (int qi = 0; qi < NQ; qi++) {
                T acc = 0;
                for (int i = 0; i < NROWS; i++) acc += gr[i] * h_J[6 * qi + i];
                worst = fmaxf(worst, fabsf(acc - h_qinc[qi]));
                for (int qj = 0; qj < NQ; qj++) {
                        T accQ = 0;
                        for (int i = 0; i < NROWS; i++) accQ += hh[i] * h_J[6 * qi + i] * h_J[6 * qj + i];
                        worst = fmaxf(worst, fabsf(accQ - h_Qinc[qi * constants::STATE_SIZE + qj]));
                }
        }
        // untouched slots (qd block, off-q rows) must be exactly zero
        for (int r = 0; r < (int)constants::STATE_SIZE; r++) {
                if (r >= NQ && h_qinc[r] != 0) { printf("FAIL qinc[%d] nonzero\n", r); fails++; }
                for (int cidx = 0; cidx < (int)constants::STATE_SIZE; cidx++) {
                        if ((r >= NQ || cidx >= NQ) && h_Qinc[r * constants::STATE_SIZE + cidx] != 0) {
                                printf("FAIL Qinc[%d][%d] nonzero\n", r, cidx);
                                fails++;
                        }
                }
        }
        printf("gate 2 (fold vs host GN):    worst abs %.3e %s\n", worst, (worst < 1e-4f) ? "PASS" : (fails++, "FAIL"));

        printf(fails ? "EE_ROWS: FAIL (%d)\n" : "EE_ROWS: ALL PASS\n", fails);
        return fails ? 1 : 0;
}
