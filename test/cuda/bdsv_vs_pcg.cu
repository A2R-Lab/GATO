// glass::bdsv (via GATO's negate-strips path) vs glass::pcg vs a CPU dense
// double reference, on random SPD block-tridiagonal systems stored the way
// GATO stores its Schur complement: NEGATED strips + matching-sign rhs
// (schur_linsys.cuh stores −theta_k on the main diagonal, so −S_stored is SPD
// and the bdsv path must solve (−S)λ = (−γ) — same λ). Gate §5.3 of
// docs/open-tasks/hybrid_pcg_bdsv_plan_2026-07-07.md.
//
// The bdsv kernel below mirrors gato/bsqp/kernels/bdsv.cuh's sequence exactly
// (eta0 guard → in-place negate → bdsv_factor<CHECK> → bdsv_solve), so its
// thread-count sweep also checks the GATO-side loops' invariance BITWISE
// (bdsv is direct — unlike pcg, identical output across thread counts is
// required, not just convergence).
//
// Build-on-demand (not part of pytest):
//
//   nvcc -std=c++17 -O2 -DNDEBUG -arch=native -I external/GLASS test/cuda/bdsv_vs_pcg.cu \
//        -o build/bdsv_vs_pcg && ./build/bdsv_vs_pcg
//
// -DNDEBUG is required (see CLAUDE.md: gpuAssert ODR gotcha); no fast-math.
// Prefer -arch=native: a default-arch (PTX JIT) build of the REAL gato bdsv
// kernels produced deterministic wrong solves on CUDA 13.2/RTX 5090 (see
// bdsv_factor_solve.cu header); this mirror harness happened to be immune,
// but don't rely on that.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>

#include "glass.cuh"

#define CHECK(call)                                                              \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

template <int d, int k>
__global__ void pcg_kernel(float* x, float* S, float* Pinv, float* b,
                           uint32_t max_it, float rel_tol, float abs_tol,
                           uint32_t* iters)
{
    extern __shared__ float smem[];
    glass::pcg<float, d, k>(x, S, Pinv, b, smem, max_it, rel_tol, abs_tol, iters);
}

// Mirror of solveBDSVBatchedKernel (gato/bsqp/kernels/bdsv.cuh) for one
// problem: eta0 converged-start guard on the UN-negated stored system → negate
// strips → CHECKED factor → (success only) x ← −b + solve. iters: 0 = guard
// fired, 1 = solved, 2 = non-PD pivot → update skipped, x untouched (matches
// the ship kernel; the generator here is SPD-guarded so 2 is a FAIL below).
template <int d, int k>
__global__ void bdsv_kernel(float* x, float* S, float* Pinv, float* b, uint32_t* iters)
{
    constexpr uint32_t VEC = (k + 2) * d;
    extern __shared__ float smem[];
    float* s_r = smem;
    float* s_z = s_r + VEC;
    float* s_scr = s_z + VEC;
    __shared__ float s_rho;
    __shared__ int s_fail;
    const uint32_t rank = threadIdx.x;
    const uint32_t size = blockDim.x;

    glass::set_const<float, 2 * VEC>(0.f, s_r);
    if (rank == 0) s_fail = 0;
    __syncthreads();
    glass::bdmv<float, k, d>(s_r, S, x);                    // r = S·x_warm
    __syncthreads();
    glass::axpby<float, VEC>(1.f, b, -1.f, s_r, s_r);       // r = b − S·x
    __syncthreads();
    glass::bdmv<float, k, d>(s_z, Pinv, s_r);               // z = Pinv·r
    __syncthreads();
    glass::dot_fast<float, VEC>(s_r, s_z, &s_rho, s_scr);
    __syncthreads();
    float arho = (s_rho < 0.f) ? -s_rho : s_rho;
    if (arho < 1e-6f) {
        if (rank == 0) *iters = 0;
        return;
    }
    for (uint32_t i = rank; i < 3 * d * d * k; i += size) S[i] = -S[i];
    __syncthreads();
    glass::bdsv_factor<float, k, d, /*CHECK=*/true>(S, smem, &s_fail);
    __syncthreads();
    if (s_fail) {
        glass::set_const<float>(2 * d * d, 0.f, smem);  // scrub NaN staging (smem persists on the SM)
        __syncthreads();
        if (rank == 0) *iters = 2;
        return;
    }
    for (uint32_t i = rank; i < VEC; i += size) x[i] = -b[i];
    __syncthreads();
    glass::bdsv_solve<float, k, d>(S, x, smem);
    __syncthreads();
    if (rank == 0) *iters = 1;
}

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

template <int D, int K>
static int run_shape()
{
    constexpr int NPAD = (K + 2) * D;
    std::mt19937 rng(1234 + D * 1000 + K);
    std::normal_distribution<double> nd(0.0, 1.0);

    // SPD block-tridiagonal: D_k = A_k A_k^T + D*I, off-diag O_k at 0.5 coupling.
    // NOTE: pcg_vs_cpu.cu's generator (0.5*D boost, 1.5 coupling) is INDEFINITE
    // (min eig ≈ −3 at (12,16)) — glass::pcg tolerates that via |rho|, but a
    // Cholesky factor rightly refuses. The SPD guard below enforces the premise;
    // if it fires, fix the generator — never loosen the gate.
    std::vector<double> Dblk(K * D * D), Oblk((K - 1) * D * D);
    for (int k = 0; k < K; k++) {
        std::vector<double> A(D * D);
        for (auto& v : A) v = nd(rng);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                double s = (i == j) ? (double)D : 0.0;
                for (int m = 0; m < D; m++) s += A[i * D + m] * A[j * D + m];
                Dblk[k * D * D + i * D + j] = s;
            }
    }
    for (auto& v : Oblk) v = 0.5 * nd(rng);

    // GATO-convention storage: NEGATED strips + NEGATED rhs. Pinv strips are the
    // block-Jacobi inverses of the STORED (negated) diagonal blocks, matching
    // formSchur's fused invert of what it stores.
    const int BRL = 3 * D;
    std::vector<float> hS(K * BRL * D, 0.f), hP(K * BRL * D, 0.f);
    std::vector<double> negD(D * D), Dinv(D * D);
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < D * D; i++) negD[i] = -Dblk[k * D * D + i];
        invert_block(negD.data(), Dinv.data(), D);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                float* row = &hS[k * BRL * D + i * BRL];
                if (k > 0)     row[0 * D + j] = (float)-Oblk[(k - 1) * D * D + j * D + i]; // L = −O^T
                row[1 * D + j] = (float)-Dblk[k * D * D + i * D + j];                      // D = −D_k
                if (k < K - 1) row[2 * D + j] = (float)-Oblk[k * D * D + i * D + j];       // R = −O
                hP[k * BRL * D + i * BRL + 1 * D + j] = (float)Dinv[i * D + j];
            }
    }

    std::vector<double> bd(K * D);
    for (auto& v : bd) v = nd(rng);
    std::vector<float> hb(NPAD, 0.f), hx0(NPAD, 0.f);
    for (int i = 0; i < K * D; i++) hb[D + i] = (float)-bd[i];  // stored γ is negated too

    // CPU reference on the TRUE (un-negated) SPD system: S λ = b ⇔ (−S)λ = (−b)
    const int n = K * D;
    std::vector<double> Adense(n * n, 0.0);
    for (int k = 0; k < K; k++)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                Adense[(k * D + i) * n + k * D + j] = Dblk[k * D * D + i * D + j];
                if (k < K - 1) {
                    Adense[(k * D + i) * n + (k + 1) * D + j] = Oblk[k * D * D + i * D + j];
                    Adense[((k + 1) * D + i) * n + k * D + j] = Oblk[k * D * D + j * D + i];
                }
            }
    // SPD guard: double Cholesky on the dense system must succeed (see generator note)
    {
        std::vector<double> Achk(Adense);
        for (int c = 0; c < n; c++) {
            double dpiv = Achk[c * n + c];
            for (int m = 0; m < c; m++) dpiv -= Achk[c * n + m] * Achk[c * n + m];
            if (dpiv <= 0.0) {
                printf("D=%2d K=%3d  GENERATOR NOT SPD (pivot %d) — fix the harness\n", D, K, c);
                return 1;
            }
            dpiv = std::sqrt(dpiv);
            Achk[c * n + c] = dpiv;
            for (int r = c + 1; r < n; r++) {
                double s = Achk[r * n + c];
                for (int m = 0; m < c; m++) s -= Achk[r * n + m] * Achk[c * n + m];
                Achk[r * n + c] = s / dpiv;
            }
        }
    }
    std::vector<double> xref = dense_solve(Adense, bd, n);
    double xref_norm = 0;
    for (double v : xref) xref_norm += v * v;
    xref_norm = std::sqrt(xref_norm);

    float *dS, *dP, *db, *dx;
    uint32_t* dit;
    CHECK(cudaMalloc(&dS, hS.size() * sizeof(float)));
    CHECK(cudaMalloc(&dP, hP.size() * sizeof(float)));
    CHECK(cudaMalloc(&db, NPAD * sizeof(float)));
    CHECK(cudaMalloc(&dx, NPAD * sizeof(float)));
    CHECK(cudaMalloc(&dit, sizeof(uint32_t)));
    CHECK(cudaMemcpy(dP, hP.data(), hP.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb.data(), NPAD * sizeof(float), cudaMemcpyHostToDevice));

    auto rel_err = [&](const std::vector<float>& got) {
        double err = 0;
        for (int i = 0; i < n; i++) {
            double dd = got[D + i] - xref[i];
            err += dd * dd;
        }
        return std::sqrt(err) / xref_norm;
    };

    int fails = 0;

    // --- pcg on the stored (negated) system, tight tol ---
    CHECK(cudaMemcpy(dS, hS.data(), hS.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx0.data(), NPAD * sizeof(float), cudaMemcpyHostToDevice));
    {
        size_t smem = glass::pcg_scratch_bytes<float, D, K>(256);
        pcg_kernel<D, K><<<1, 256, smem>>>(dx, dS, dP, db, 2000, 1e-7f, 1e-8f, dit);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<float> got(NPAD);
        uint32_t iters = 0;
        CHECK(cudaMemcpy(got.data(), dx, NPAD * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&iters, dit, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        double err = rel_err(got);
        bool ok = err < 1e-3 && iters > 0 && iters < 2000;
        printf("D=%2d K=%3d  pcg(negated store)   iters=%4u rel_err=%.3e  %s\n",
               D, K, iters, err, ok ? "PASS" : "FAIL");
        fails += !ok;
    }

    // --- bdsv path (GATO sequence), thread sweep: accuracy + BITWISE invariance ---
    const size_t smem_guard = (2 * (size_t)NPAD + (1024 + 31) / 32) * sizeof(float);
    const size_t smem_bdsv = std::max(smem_guard, glass::bdsv_scratch_bytes<float, D>());
    std::vector<float> ref_bits;
    for (uint32_t threads : {32u, 128u, 256u, 1024u}) {
        // fresh strips each run: the kernel negates + factors S in place
        CHECK(cudaMemcpy(dS, hS.data(), hS.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dx, hx0.data(), NPAD * sizeof(float), cudaMemcpyHostToDevice));
        bdsv_kernel<D, K><<<1, threads, smem_bdsv>>>(dx, dS, dP, db, dit);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<float> got(NPAD);
        uint32_t iters = 0;
        CHECK(cudaMemcpy(got.data(), dx, NPAD * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&iters, dit, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        double err = rel_err(got);
        bool bits_ok = true;
        if (ref_bits.empty()) ref_bits = got;
        else bits_ok = (memcmp(ref_bits.data(), got.data(), NPAD * sizeof(float)) == 0);
        // f32 direct solve vs double reference; 1e-4 rel per plan §5.3 (serial
        // factor chain accumulates — watch K=128, risk §7.2)
        bool ok = err < 1e-4 && iters == 1 && bits_ok;
        printf("D=%2d K=%3d  bdsv threads=%4u    iters=%4u rel_err=%.3e bits=%s  %s\n",
               D, K, threads, iters, err, bits_ok ? "same" : "DIFF",
               ok ? "PASS" : "FAIL");
        fails += !ok;
    }

    CHECK(cudaFree(dS)); CHECK(cudaFree(dP)); CHECK(cudaFree(db));
    CHECK(cudaFree(dx)); CHECK(cudaFree(dit));
    return fails;
}

int main()
{
    int fails = 0;
    fails += run_shape<12, 16>();   // pcg_vs_cpu's shape (indy7-ish)
    fails += run_shape<14, 16>();   // GATO iiwa14/indy7 STATE_SIZE at fig7 N
    fails += run_shape<14, 64>();   // fig3/fig8 N
    fails += run_shape<14, 128>();  // long-chain f32 accumulation check (§7.2)
    printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
