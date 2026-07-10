// Gate for computeGammaBatchedKernel (gato/bsqp/kernels/schur_linsys.cuh):
// the CL-1 ADMM inner loop rebuilds ONLY gamma (new q/r each iteration) from
// the Q^-1/R^-1 blocks formSchur left behind — this harness proves the
// recompute matches formSchur's own gamma on the same data.
//
// Gates (real headers via plant_shim.cuh, batch of B synthetic KKT systems):
//   1. knots 1..K-1: BITWISE equal to formSchur's gamma (same glass ops in
//      the same order).
//   2. knot 0: 1e-6 rel (formSchur's last block re-inverts Q_0 fresh with the
//      SINGLE glass::inv; the stored inverse came from the FUSED 3-matrix inv
//      in block 0 — same matrix, different interleaving, near-ulp diff).
//   3. thread-count sweep {32, 64, 128}: bitwise invariant.
//
// Build-on-demand (not part of pytest):
//
//   nvcc -std=c++17 -O2 -DNDEBUG -arch=native -DKNOT_POINTS=16 \
//        -DGATO_PLANT_HEADER='"plant_shim.cuh"' \
//        -I gato -I external/GLASS -I test/cuda \
//        test/cuda/gamma_parity.cu -o build/gamma_parity && ./build/gamma_parity
//
// -DNDEBUG and -arch=native are REQUIRED (see bdsv_factor_solve.cu header:
// default-arch JIT miscompiles GATO kernels on CUDA 13.2/RTX 5090).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "types.cuh"
#include "bsqp/kernels/schur_linsys.cuh"

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

static constexpr int D = gato::constants::STATE_SIZE;
static constexpr int C = gato::constants::CONTROL_SIZE;
static constexpr int K = KNOT_POINTS;
static constexpr int B = 4;
static constexpr int NPAD = gato::constants::VEC_SIZE_PADDED;

int main()
{
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.f, 1.f);

    // synthetic KKT batch: SPD-ish Q/R (M M^T + dim*I), random q/r/A/B/c
    const size_t nQ = (size_t)B * K * D * D, nR = (size_t)B * K * C * C;
    const size_t nq = (size_t)B * K * D, nr = (size_t)B * K * C;
    const size_t nA = nQ, nB_ = (size_t)B * K * D * C, nc = nq;
    std::vector<float> hQ(nQ, 0.f), hR(nR, 0.f), hq(nq), hr(nr), hA(nA), hB(nB_), hc(nc);
    auto spd_fill = [&](float* M, int n) {
        std::vector<float> W(n * n);
        for (auto& v : W) v = nd(rng);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                float s = (i == j) ? (float)n : 0.f;
                for (int m = 0; m < n; m++) s += W[i * n + m] * W[j * n + m];
                M[i * n + j] = s;
            }
    };
    for (int b = 0; b < B; b++)
        for (int k = 0; k < K; k++) {
            spd_fill(&hQ[((size_t)b * K + k) * D * D], D);
            spd_fill(&hR[((size_t)b * K + k) * C * C], C);
        }
    for (auto& v : hq) v = nd(rng);
    for (auto& v : hr) v = nd(rng);
    for (auto& v : hA) v = nd(rng);
    for (auto& v : hB) v = nd(rng);
    for (auto& v : hc) v = nd(rng);

    KKTSystem<float>   kkt;
    SchurSystem<float> schur;
    float*             d_rho;
    int32_t*           d_conv;
    CHECK_CUDA(cudaMalloc(&kkt.d_Q_batch, nQ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&kkt.d_R_batch, nR * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&kkt.d_q_batch, nq * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&kkt.d_r_batch, nr * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&kkt.d_A_batch, nA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&kkt.d_B_batch, nB_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&kkt.d_c_batch, nc * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&schur.d_S_batch, (size_t)B * gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&schur.d_P_inv_batch, (size_t)B * gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&schur.d_gamma_batch, (size_t)B * NPAD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rho, B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv, B * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(kkt.d_Q_batch, hQ.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kkt.d_R_batch, hR.data(), nR * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kkt.d_q_batch, hq.data(), nq * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kkt.d_r_batch, hr.data(), nr * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kkt.d_A_batch, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kkt.d_B_batch, hB.data(), nB_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kkt.d_c_batch, hc.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> hrho(B, 1e-3f);
    CHECK_CUDA(cudaMemcpy(d_rho, hrho.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_conv, 0, B * sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(schur.d_S_batch, 0, (size_t)B * gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float)));
    CHECK_CUDA(cudaMemset(schur.d_P_inv_batch, 0, (size_t)B * gato::constants::B3D_MATRIX_SIZE_PADDED * sizeof(float)));
    CHECK_CUDA(cudaMemset(schur.d_gamma_batch, 0, (size_t)B * NPAD * sizeof(float)));

    // reference: formSchur (also leaves Q^-1/R^-1 in kkt.d_Q/d_R)
    formSchurSystemBatched<float>(B, schur, kkt, d_rho, d_conv);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> gamma_ref((size_t)B * NPAD);
    CHECK_CUDA(cudaMemcpy(gamma_ref.data(), schur.d_gamma_batch, gamma_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int fails = 0;
    std::vector<float> got((size_t)B * NPAD), ref_bits;
    for (uint32_t threads : {32u, 64u, 128u}) {
        CHECK_CUDA(cudaMemset(schur.d_gamma_batch, 0, (size_t)B * NPAD * sizeof(float)));
        computeGammaBatchedKernel<float><<<dim3(K, B), threads, getComputeGammaBatchedSMemSize<float>()>>>(
            schur.d_gamma_batch, kkt.d_Q_batch, kkt.d_R_batch, kkt.d_q_batch, kkt.d_r_batch,
            kkt.d_A_batch, kkt.d_B_batch, kkt.d_c_batch, d_conv);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(got.data(), schur.d_gamma_batch, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

        bool bits_k1 = true;   // knots 1..K-1 bitwise vs formSchur
        double g0_rel = 0.0;   // knot 0 near-ulp
        for (int b = 0; b < B; b++) {
            const float* r = &gamma_ref[(size_t)b * NPAD];
            const float* g = &got[(size_t)b * NPAD];
            if (memcmp(&r[2 * D], &g[2 * D], (size_t)(K - 1) * D * sizeof(float)) != 0) bits_k1 = false;
            double num = 0, den = 0;
            for (int i = 0; i < D; i++) {
                double dd = g[D + i] - r[D + i];
                num += dd * dd;
                den += (double)r[D + i] * r[D + i];
            }
            g0_rel = std::max(g0_rel, std::sqrt(num / (den > 0 ? den : 1.0)));
        }
        bool sweep_ok = true;
        if (ref_bits.empty()) ref_bits = got;
        else sweep_ok = (memcmp(ref_bits.data(), got.data(), got.size() * sizeof(float)) == 0);

        bool ok = bits_k1 && g0_rel < 1e-6 && sweep_ok;
        printf("gamma threads=%3u  k>=1:%s  g0_rel=%.2e  sweep:%s  %s\n", threads,
               bits_k1 ? "bitwise" : "DIFF", g0_rel, sweep_ok ? "same" : "DIFF",
               ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
