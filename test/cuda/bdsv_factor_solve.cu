// Gates for the bdsv factor/solve SPLIT path (gato/bsqp/kernels/bdsv.cuh:
// factorBDSVBatchedKernel + solveBDSVFactoredBatchedKernel) — compiled against
// the REAL kernel header via test/cuda/plant_shim.cuh (no grid.cuh), so there
// is no mirror to drift. CL-0 gate of
// docs/open-tasks/constraint_layer_locomotion_arc_plan_2026-07-10.md.
//
// Gates:
//   1. factor(γ-system) + solve(γ) is BITWISE identical to the monolithic
//      solveBDSVBatchedKernel result (same ops, same order).
//   2. factor REUSE: a second solve on the SAME factored strips with a fresh
//      rhs matches a CPU dense double reference (<1e-4 rel) — the property the
//      CL-1 ADMM inner loop depends on.
//   3. thread-count sweep {32, 64, 128, 256} on both kernels: BITWISE
//      invariant (direct solve — identical output required, not just close).
//   4. non-PD problem in the batch: factor status = NON_PD, factored-solve
//      leaves x untouched and reports iterations = 2; the other problems in
//      the batch are unaffected.
//   5. converged skip: status = SKIPPED, solve reports iterations = 0, x
//      untouched. mask = 0: nothing written at all (sentinels intact).
//
// Build-on-demand (not part of pytest):
//
//   nvcc -std=c++17 -O2 -DNDEBUG -arch=native -DKNOT_POINTS=64 \
//        -DGATO_PLANT_HEADER='"plant_shim.cuh"' \
//        -I gato -I external/GLASS -I test/cuda \
//        test/cuda/bdsv_factor_solve.cu -o build/bdsv_factor_solve && \
//   ./build/bdsv_factor_solve
//
// -DNDEBUG is required (see CLAUDE.md: gpuAssert ODR gotcha); no fast-math.
// -arch=native (or sm_120) is REQUIRED: without it (default-arch PTX + JIT,
// observed on CUDA 13.2 / RTX 5090) these kernels produce deterministic,
// sanitizer-clean, thread-invariant WRONG solves (rel err ~7) while a
// launch_bounds/restrict-free mirror of the same sequence in the same TU is
// correct — a JIT codegen artifact, not a kernel bug. The ship modules always
// compile with CMAKE_CUDA_ARCHITECTURES=120, so they are unaffected; just
// never benchmark or gate GATO kernels from a default-arch nvcc line.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

#include "bsqp/kernels/bdsv.cuh"

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

using gato::constants::STATE_SIZE;
using gato::constants::VEC_SIZE_PADDED;
using gato::constants::BLOCK_ROW_SIZE;
using gato::constants::B3D_MATRIX_SIZE_PADDED;

static constexpr int D = STATE_SIZE;
static constexpr int K = KNOT_POINTS;
static constexpr int B = 8;  // batch
static constexpr int NPAD = VEC_SIZE_PADDED;
static constexpr int STRIP = B3D_MATRIX_SIZE_PADDED;

// dense n x n Gaussian elimination with partial pivoting (double)
static std::vector<double> dense_solve(std::vector<double> A, std::vector<double> b, int n)
{
    for (int c = 0; c < n; c++) {
        int piv = c;
        for (int r = c + 1; r < n; r++)
            if (std::fabs(A[r * n + c]) > std::fabs(A[piv * n + c])) piv = r;
        for (int j = 0; j < n; j++) std::swap(A[c * n + j], A[piv * n + j]);
        std::swap(b[c], b[piv]);
        for (int r = c + 1; r < n; r++) {
            double f = A[r * n + c] / A[c * n + c];
            for (int j = c; j < n; j++) A[r * n + j] -= f * A[c * n + j];
            b[r] -= f * b[c];
        }
    }
    std::vector<double> x(n);
    for (int r = n - 1; r >= 0; r--) {
        double s = b[r];
        for (int j = r + 1; j < n; j++) s -= A[r * n + j] * x[j];
        x[r] = s / A[r * n + r];
    }
    return x;
}

// invert a d x d SPD block (double, Gauss-Jordan)
static void invert_block(const double* M, double* Minv, int d)
{
    std::vector<double> a(M, M + d * d), inv(d * d, 0.0);
    for (int i = 0; i < d; i++) inv[i * d + i] = 1.0;
    for (int c = 0; c < d; c++) {
        double p = a[c * d + c];
        for (int j = 0; j < d; j++) { a[c * d + j] /= p; inv[c * d + j] /= p; }
        for (int r = 0; r < d; r++) {
            if (r == c) continue;
            double f = a[r * d + c];
            for (int j = 0; j < d; j++) { a[r * d + j] -= f * a[c * d + j]; inv[r * d + j] -= f * inv[c * d + j]; }
        }
    }
    for (int i = 0; i < d * d; i++) Minv[i] = inv[i];
}

// One problem's SPD block-tridiagonal system in GATO's NEGATED-store
// convention (same generator family as bdsv_vs_pcg.cu, SPD-guarded there).
struct Problem {
    std::vector<double> Dblk, Oblk, bd;   // true (un-negated) system, double
    std::vector<float>  hS, hP, hb;       // stored strips/Pinv/γ (negated), float
    std::vector<double> Adense;           // dense true system for the CPU reference
};

static Problem make_problem(uint32_t seed)
{
    Problem P;
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    P.Dblk.resize(K * D * D);
    P.Oblk.resize((K - 1) * D * D);
    for (int k = 0; k < K; k++) {
        std::vector<double> A(D * D);
        for (auto& v : A) v = nd(rng);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                double s = (i == j) ? (double)D : 0.0;
                for (int m = 0; m < D; m++) s += A[i * D + m] * A[j * D + m];
                P.Dblk[k * D * D + i * D + j] = s;
            }
    }
    for (auto& v : P.Oblk) v = 0.5 * nd(rng);

    const int BRL = 3 * D;
    P.hS.assign(STRIP, 0.f);
    P.hP.assign(STRIP, 0.f);
    std::vector<double> negD(D * D), Dinv(D * D);
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < D * D; i++) negD[i] = -P.Dblk[k * D * D + i];
        invert_block(negD.data(), Dinv.data(), D);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                float* row = &P.hS[k * BRL * D + i * BRL];
                if (k > 0)     row[0 * D + j] = (float)-P.Oblk[(k - 1) * D * D + j * D + i]; // L = −O^T
                row[1 * D + j] = (float)-P.Dblk[k * D * D + i * D + j];                      // D = −D_k
                if (k < K - 1) row[2 * D + j] = (float)-P.Oblk[k * D * D + i * D + j];       // R = −O
                P.hP[k * BRL * D + i * BRL + 1 * D + j] = (float)Dinv[i * D + j];
            }
    }

    P.bd.resize(K * D);
    for (auto& v : P.bd) v = nd(rng);
    P.hb.assign(NPAD, 0.f);
    for (int i = 0; i < K * D; i++) P.hb[D + i] = (float)-P.bd[i];  // stored γ is negated

    const int n = K * D;
    P.Adense.assign((size_t)n * n, 0.0);
    for (int k = 0; k < K; k++)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                P.Adense[(size_t)(k * D + i) * n + k * D + j] = P.Dblk[k * D * D + i * D + j];
                if (k < K - 1) {
                    P.Adense[(size_t)(k * D + i) * n + (k + 1) * D + j] = P.Oblk[k * D * D + i * D + j];
                    P.Adense[(size_t)((k + 1) * D + i) * n + k * D + j] = P.Oblk[k * D * D + j * D + i];
                }
            }
    return P;
}

static double rel_err_vs(const float* got_pad, const std::vector<double>& xref)
{
    double err = 0, nrm = 0;
    for (size_t i = 0; i < xref.size(); i++) {
        double dd = got_pad[D + i] - xref[i];
        err += dd * dd;
        nrm += xref[i] * xref[i];
    }
    return std::sqrt(err) / std::sqrt(nrm);
}

int main()
{
    int fails = 0;

    // batch of B problems; problem NONPD_IDX made indefinite (flip one diag block's sign)
    const int NONPD_IDX = 3;
    std::vector<Problem> probs;
    for (int b = 0; b < B; b++) probs.push_back(make_problem(777 + b));
    {
        // make problem NONPD_IDX non-PD on the STORED convention: −S must fail
        // Cholesky → store +D_k (instead of −D_k) in one knot's MAIN slot.
        const int BRL = 3 * D;
        Problem& P = probs[NONPD_IDX];
        const int kbad = K / 2;
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                P.hS[kbad * BRL * D + i * BRL + 1 * D + j] = (float)P.Dblk[kbad * D * D + i * D + j];
    }

    // flat batch uploads
    std::vector<float> hS_all(B * STRIP), hP_all(B * STRIP), hb_all(B * NPAD);
    for (int b = 0; b < B; b++) {
        memcpy(&hS_all[(size_t)b * STRIP], probs[b].hS.data(), STRIP * sizeof(float));
        memcpy(&hP_all[(size_t)b * STRIP], probs[b].hP.data(), STRIP * sizeof(float));
        memcpy(&hb_all[(size_t)b * NPAD], probs[b].hb.data(), NPAD * sizeof(float));
    }

    float *dS, *dP, *db, *dx, *drhs2;
    uint32_t* dit;
    int32_t *dconv, *dstat, *dmask;
    CHECK_CUDA(cudaMalloc(&dS, hS_all.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP, hP_all.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, hb_all.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dx, (size_t)B * NPAD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&drhs2, (size_t)B * NPAD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dit, B * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&dconv, B * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dstat, B * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dmask, B * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dP, hP_all.data(), hP_all.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, hb_all.data(), hb_all.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dconv, 0, B * sizeof(int32_t)));

    SchurSystem<float> schur{dS, dP, db};

    const size_t smem_mono = getSolveBDSVBatchedSMemSize<float>();
    const size_t smem_factor = getFactorBDSVBatchedSMemSize<float>();
    const size_t smem_fsolve = getSolveBDSVFactoredBatchedSMemSize<float>();

    // ---------- reference: monolithic kernel on the PD problems ----------
    std::vector<float> x_mono((size_t)B * NPAD);
    std::vector<uint32_t> it_mono(B);
    {
        CHECK_CUDA(cudaMemcpy(dS, hS_all.data(), hS_all.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dx, 0, (size_t)B * NPAD * sizeof(float)));
        solveBDSVBatched<float>(B, dx, schur, dconv, dit);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(x_mono.data(), dx, x_mono.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(it_mono.data(), dit, B * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int b = 0; b < B; b++) {
            uint32_t want = (b == NONPD_IDX) ? 2u : 1u;
            if (it_mono[b] != want) {
                printf("mono b=%d iters=%u (want %u)  FAIL\n", b, it_mono[b], want);
                fails++;
            }
        }
    }

    // ---------- gate 1+3: factor+solve(γ) bitwise == monolithic, thread sweep ----------
    // (sweep ≤ BDSV_THREADS: the ship kernels carry __launch_bounds__(BDSV_THREADS))
    std::vector<float> x_split((size_t)B * NPAD);
    std::vector<uint32_t> it_split(B);
    std::vector<int32_t> st_split(B);
    std::vector<float> ref_bits;
    for (uint32_t threads : {32u, 64u, 128u, 256u}) {
        CHECK_CUDA(cudaMemcpy(dS, hS_all.data(), hS_all.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dx, 0, (size_t)B * NPAD * sizeof(float)));
        factorBDSVBatchedKernel<float><<<B, threads, smem_factor>>>(dstat, dS, dconv, nullptr);
        CHECK_CUDA(cudaGetLastError());
        solveBDSVFactoredBatchedKernel<float><<<B, threads, smem_fsolve>>>(dit, dx, dS, db, dstat, nullptr);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(x_split.data(), dx, x_split.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(it_split.data(), dit, B * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(st_split.data(), dstat, B * sizeof(int32_t), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int b = 0; b < B; b++) {
            const bool nonpd = (b == NONPD_IDX);
            const int32_t want_st = nonpd ? bdsv_status::NON_PD : bdsv_status::OK;
            const uint32_t want_it = nonpd ? 2u : 1u;
            if (st_split[b] != want_st || it_split[b] != want_it) ok = false;
            // bitwise vs monolithic λ (PD problems; non-PD: both untouched → zeros)
            if (memcmp(&x_split[(size_t)b * NPAD], &x_mono[(size_t)b * NPAD], NPAD * sizeof(float)) != 0) ok = false;
        }
        bool bits_ok = true;
        if (ref_bits.empty()) ref_bits = x_split;
        else bits_ok = (memcmp(ref_bits.data(), x_split.data(), x_split.size() * sizeof(float)) == 0);
        printf("split threads=%4u  vs-mono=%s  sweep-bits=%s  %s\n", threads,
               ok ? "same" : "DIFF", bits_ok ? "same" : "DIFF",
               (ok && bits_ok) ? "PASS" : "FAIL");
        fails += !(ok && bits_ok);
    }

    // ---------- gate 2: factor REUSE — second rhs on the already-factored strips ----------
    // strips still hold the factor from the last sweep iteration above
    {
        std::vector<float> hrhs2_all((size_t)B * NPAD, 0.f);
        std::vector<std::vector<double>> b2(B);
        std::mt19937 rng(4242);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int bb = 0; bb < B; bb++) {
            b2[bb].resize(K * D);
            for (auto& v : b2[bb]) v = nd(rng);
            for (int i = 0; i < K * D; i++) hrhs2_all[(size_t)bb * NPAD + D + i] = (float)-b2[bb][i];
        }
        CHECK_CUDA(cudaMemcpy(drhs2, hrhs2_all.data(), hrhs2_all.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dx, 0, (size_t)B * NPAD * sizeof(float)));
        solveBDSVFactoredBatchedKernel<float><<<B, 256, smem_fsolve>>>(dit, dx, dS, drhs2, dstat, nullptr);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(x_split.data(), dx, x_split.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int bb = 0; bb < B; bb++) {
            if (bb == NONPD_IDX) continue;
            std::vector<double> xref = dense_solve(probs[bb].Adense, b2[bb], K * D);
            double err = rel_err_vs(&x_split[(size_t)bb * NPAD], xref);
            bool ok = err < 1e-4;
            printf("reuse b=%d rel_err=%.3e  %s\n", bb, err, ok ? "PASS" : "FAIL");
            fails += !ok;
        }
    }

    // ---------- gate 4+5: skip semantics (converged, mask) ----------
    {
        std::vector<int32_t> hconv(B, 0), hmask(B, 1);
        hconv[1] = 1;   // converged solve
        hmask[2] = 0;   // masked-out solve (PCG owns it)
        CHECK_CUDA(cudaMemcpy(dconv, hconv.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dmask, hmask.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice));

        // sentinels: status = −7, iterations = 99, x = 0.5 everywhere
        std::vector<int32_t> hstat_init(B, -7);
        std::vector<uint32_t> hit_init(B, 99);
        std::vector<float> hx_init((size_t)B * NPAD, 0.5f);
        CHECK_CUDA(cudaMemcpy(dstat, hstat_init.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dit, hit_init.data(), B * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dx, hx_init.data(), hx_init.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dS, hS_all.data(), hS_all.size() * sizeof(float), cudaMemcpyHostToDevice));

        factorBDSVBatchedKernel<float><<<B, 256, smem_factor>>>(dstat, dS, dconv, dmask);
        CHECK_CUDA(cudaGetLastError());
        solveBDSVFactoredBatchedKernel<float><<<B, 256, smem_fsolve>>>(dit, dx, dS, db, dstat, dmask);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int32_t> hstat(B);
        std::vector<uint32_t> hit(B);
        std::vector<float> hx((size_t)B * NPAD);
        CHECK_CUDA(cudaMemcpy(hstat.data(), dstat, B * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hit.data(), dit, B * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hx.data(), dx, hx.size() * sizeof(float), cudaMemcpyDeviceToHost));

        auto x_untouched = [&](int b) {
            for (int i = 0; i < NPAD; i++)
                if (hx[(size_t)b * NPAD + i] != 0.5f) return false;
            return true;
        };
        bool ok = true;
        // b=1 converged: SKIPPED + iterations 0 + x untouched
        if (hstat[1] != bdsv_status::SKIPPED || hit[1] != 0 || !x_untouched(1)) ok = false;
        // b=2 masked: sentinels fully intact
        if (hstat[2] != -7 || hit[2] != 99 || !x_untouched(2)) ok = false;
        // b=NONPD: NON_PD + iterations 2 + x untouched
        if (hstat[NONPD_IDX] != bdsv_status::NON_PD || hit[NONPD_IDX] != 2 || !x_untouched(NONPD_IDX)) ok = false;
        // a normal problem still solves
        if (hstat[0] != bdsv_status::OK || hit[0] != 1 || x_untouched(0)) ok = false;
        printf("skip semantics (converged/mask/non-PD)  %s\n", ok ? "PASS" : "FAIL");
        fails += !ok;
    }

    CHECK_CUDA(cudaFree(dS)); CHECK_CUDA(cudaFree(dP)); CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(drhs2)); CHECK_CUDA(cudaFree(dit));
    CHECK_CUDA(cudaFree(dconv)); CHECK_CUDA(cudaFree(dstat)); CHECK_CUDA(cudaFree(dmask));
    printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
