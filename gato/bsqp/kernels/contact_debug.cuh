#pragma once

// ===================================================================
// Contact-wrench dynamics evaluation (CL-3 prep, debug/oracle surface)
//
// Evaluates the full contact-frame chain the CL-3 forces-as-controls
// wave will consume in setup_kkt:
//
//   f_ext(q, f_c)      = grid::f_ext_body        (world wrench at the baked
//                        contact frame -> joint-local per-body wrenches)
//   dqdd/df_c          = dqdd/dfext . dfext/df_c   (the future B-block columns)
//   dqdd/dq correction = dqdd/dfext . dfext/dq     (the chain-rule A-block term
//                        a solver drops if it treats f_ext as q-independent)
//
// plus qdd and dqdd/d[q,qd] at the mapped f_ext (the existing first-order
// path), so a host-side finite-difference gate can validate BOTH jacobians
// against the project's own dynamics:
//   FD over f_c            ==  dqdd_dfc
//   FD over q (f_c fixed)  ==  dqdd_dq(fixed f_ext) + dqdd_dq_corr
//
// Single (state, control, wrench) sample, one block — this is an oracle
// kernel (test/test_f_ext.py), not a hot path. The grid *_device wrappers
// each claim the extern dynamic-smem arena at offset 0; calls are strictly
// sequential with barriers between, so the arena reuse is safe HERE (a
// consumer kernel with its own dynamic-smem layout must keep using the
// caller-scratch _inner pattern instead — see plant.cuh).
// ===================================================================

#ifdef GRID_HAS_CONTACT_FRAMES

namespace gato {

template<typename T>
__global__ void debugContactDynamicsKernel(T*       d_qdd,          // NQ
                                           T*       d_fext,         // 6*NUM_BODIES
                                           T*       d_dqdd_dfc,     // NQ x 6*NFC col-major (oracle composition)
                                           T*       d_dqdd_dq,      // NQ x NQ col-major (f_ext held FIXED)
                                           T*       d_dqdd_dq_corr, // NQ x NQ col-major (the dfext/dq chain term)
                                           T*       d_dqdd_dfc_adapter, // NQ x 6*NFC (ADAPTER's B-block fc
                                                                        // columns; written on fc builds only)
                                           const T* d_q,
                                           const T* d_qd,
                                           const T* d_u,
                                           const T* d_fc,           // 6*NFC world-aligned [n_w; f_w]
                                           void*    d_GRiD_mem)
{
        constexpr int NQ = gato::plant::NQ;
        constexpr int NB = grid::NUM_BODIES;
        constexpr int NFC = grid::NUM_CONTACT_FRAMES;
        constexpr int FEXT = 6 * NB;

        const grid::robotModel<T>* d_robotModel = (const grid::robotModel<T>*)d_GRiD_mem;

        // On GATO_CONTACT_FORCES builds the adapter reads fc as the control tail
        // (s_u[NQ..)) and appends its dqdd/dfc block at s_df_du[3*NQ*NQ..) — size
        // both for the wide layout unconditionally (debug kernel; smem is cheap).
        __shared__ T s_q[NQ], s_qd[NQ], s_fc[6 * NFC];
        __shared__ T s_u[NQ + 6 * NFC];
        __shared__ T s_fext[FEXT];
        __shared__ T s_qdd[NQ];
        __shared__ T s_df_du[3 * NQ * NQ + NQ * 6 * NFC]; // [dq | dqd | Minv | dfc(fc builds)]
        __shared__ T s_dtau_dfext[NQ * FEXT];           // -J^T, col-major [v + NQ*(6b+k)]
        __shared__ T s_dqdd_dfext[NQ * FEXT];           // Minv J^T, same layout
        __shared__ T s_dfext_dfc[FEXT * 6 * NFC];       // col-major [row + FEXT*col]
        __shared__ T s_dfext_dq[FEXT * NQ];             // col-major [row + FEXT*v]

        extern __shared__ T s_mem[];                    // sequential grid arena / plant scratch
        T*                  s_scratch = s_mem;

        const int tid = threadIdx.x, nth = blockDim.x;
        for (int i = tid; i < NQ; i += nth) { s_q[i] = d_q[i]; s_qd[i] = d_qd[i]; s_u[i] = d_u[i]; }
        for (int i = tid; i < 6 * NFC; i += nth) { s_fc[i] = d_fc[i]; }
        __syncthreads();

        // f_c -> joint-local per-body wrenches (zeroes non-contact bodies)
        grid::f_ext_body_device<T>(s_fext, s_fc, s_q, d_robotModel);
        __syncthreads();

#if GATO_CONTACT_FORCES
        // fc build: hand fc to the ADAPTER as the control tail and let it map the
        // wrench itself (band = nullptr) — same f_ext as the oracle's, and its
        // dqdd/dfc block lands at s_df_du[3*NQ*NQ..) for the adapter-vs-oracle gate.
        for (int i = tid; i < 6 * NFC; i += nth) { s_u[NQ + i] = s_fc[i]; }
        __syncthreads();
        gato::plant::forwardDynamicsAndGradient<T, true>(s_df_du, s_qdd, s_q, s_qd, s_u, s_scratch, (void*)d_robotModel, nullptr);
#else
        // qdd + dqdd/d[q,qd] at the mapped (held-fixed) f_ext
        gato::plant::forwardDynamicsAndGradient<T, true>(s_df_du, s_qdd, s_q, s_qd, s_u, s_scratch, (void*)d_robotModel, s_fext);
#endif

        // dtau/dfext (-J^T) and dqdd/dfext (Minv J^T)
        grid::f_ext_gradient_device<T>(s_dtau_dfext, s_dqdd_dfext, s_q, d_robotModel);
        __syncthreads();

        // dfext/df_c (f_c-independent) and dfext/dq (linear in f_c)
        grid::f_ext_body_jacobian_dfc_device<T>(s_dfext_dfc, s_q, d_robotModel);
        __syncthreads();
        grid::f_ext_body_jacobian_dq_device<T>(s_dfext_dq, s_fc, s_dtau_dfext, s_q, d_robotModel);
        __syncthreads();

        // compose + write back
        for (int i = tid; i < NQ; i += nth) { d_qdd[i] = s_qdd[i]; }
        for (int i = tid; i < FEXT; i += nth) { d_fext[i] = s_fext[i]; }
        for (int ind = tid; ind < NQ * NQ; ind += nth) { d_dqdd_dq[ind] = s_df_du[ind]; }
        for (int ind = tid; ind < NQ * 6 * NFC; ind += nth) {
                const int v = ind % NQ, c = ind / NQ;
                T acc = static_cast<T>(0);
                for (int r = 0; r < FEXT; r++) { acc += s_dqdd_dfext[v + NQ * r] * s_dfext_dfc[r + FEXT * c]; }
                d_dqdd_dfc[ind] = acc;
        }
        for (int ind = tid; ind < NQ * NQ; ind += nth) {
                const int v = ind % NQ, j = ind / NQ;
                T acc = static_cast<T>(0);
                for (int r = 0; r < FEXT; r++) { acc += s_dqdd_dfext[v + NQ * r] * s_dfext_dq[r + FEXT * j]; }
                d_dqdd_dq_corr[ind] = acc;
        }
#if GATO_CONTACT_FORCES
        for (int ind = tid; ind < NQ * 6 * NFC; ind += nth) {
                d_dqdd_dfc_adapter[ind] = s_df_du[3 * NQ * NQ + ind];
        }
#endif
}

template<typename T>
__host__ void debugContactDynamics(T* d_qdd, T* d_fext, T* d_dqdd_dfc, T* d_dqdd_dq, T* d_dqdd_dq_corr,
                                   T* d_dqdd_dfc_adapter,
                                   const T* d_q, const T* d_qd, const T* d_u, const T* d_fc, void* d_GRiD_mem)
{
        // Arena must cover every sequential grid arena claim + the plant FD-gradient
        // scratch. FD_DU_MAX (minv + id + id-gradient + vaf temps) dominates the
        // f_ext_gradient arena (XImats + minv/jacobianT temp) and the hom-transform
        // family (~288 + EE linalg bytes) on every generated robot. fc builds append
        // the adapter's persistent fc block after the FD_DU arena.
        const size_t smem = static_cast<size_t>(grid::FD_DU_MAX_SHARED_MEM_COUNT + gato::plant::FC_PERSIST_COUNT) * sizeof(T)
                            + static_cast<size_t>(grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
        debugContactDynamicsKernel<T><<<1, 128, smem>>>(d_qdd, d_fext, d_dqdd_dfc, d_dqdd_dq, d_dqdd_dq_corr,
                                                        d_dqdd_dfc_adapter, d_q, d_qd, d_u, d_fc, d_GRiD_mem);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
}

}  // namespace gato

#endif  // GRID_HAS_CONTACT_FRAMES
