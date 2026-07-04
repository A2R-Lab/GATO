// Single-block glass::pcg vs a CPU dense reference on a random SPD
// block-tridiagonal system (the exact [L|D|R] strip + padded-vector layout the
// BSQP Schur solve uses). Build-on-demand (not part of pytest):
//
//   nvcc -std=c++17 -O2 -DNDEBUG -I external/GLASS test/cuda/pcg_vs_cpu.cu \
//        -o build/pcg_vs_cpu && ./build/pcg_vs_cpu
//
// -DNDEBUG is required (see CLAUDE.md: gpuAssert ODR gotcha); no fast-math so
// the convergence tolerances are meaningful. Checks convergence at several
// block sizes (PCG dot-reduction order varies with thread count, so runs are
// compared to the CPU solution, not to each other).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
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

constexpr int D = 12;   // BlockSize (indy7 state size)
constexpr int K = 16;   // NumBlockRows (knot points)
constexpr int NPAD = (K + 2) * D;

template <int d, int k>
__global__ void pcg_kernel(float* x, float* S, float* Pinv, float* b,
                           uint32_t max_it, float rel_tol, float abs_tol,
                           uint32_t* iters)
{
    extern __shared__ float smem[];
    glass::pcg<float, d, k>(x, S, Pinv, b, smem, max_it, rel_tol, abs_tol, iters);
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

int main()
{
    std::mt19937 rng(1234);
    std::normal_distribution<double> nd(0.0, 1.0);

    // SPD block-tridiagonal S: D_k = A_k A_k^T + d/2*I, R_k = O_k, L_k = O_{k-1}^T
    // (strong off-diagonal coupling so PCG needs a real iteration count)
    std::vector<double> Dblk(K * D * D), Oblk((K - 1) * D * D);
    for (int k = 0; k < K; k++) {
        std::vector<double> A(D * D);
        for (auto& v : A) v = nd(rng);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                double s = (i == j) ? 0.5 * D : 0.0;
                for (int m = 0; m < D; m++) s += A[i * D + m] * A[j * D + m];
                Dblk[k * D * D + i * D + j] = s;
            }
    }
    for (auto& v : Oblk) v = 1.5 * nd(rng);

    // strips: [L|D|R] row-major per block-row; Pinv = blockdiag(D_k^{-1})
    const int BRL = 3 * D;
    std::vector<float> hS(K * BRL * D, 0.f), hP(K * BRL * D, 0.f);
    std::vector<double> Dinv(D * D);
    for (int k = 0; k < K; k++) {
        invert_block(&Dblk[k * D * D], Dinv.data(), D);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                float* row = &hS[k * BRL * D + i * BRL];
                if (k > 0)     row[0 * D + j] = (float)Oblk[(k - 1) * D * D + j * D + i]; // L = O^T
                row[1 * D + j] = (float)Dblk[k * D * D + i * D + j];
                if (k < K - 1) row[2 * D + j] = (float)Oblk[k * D * D + i * D + j];       // R = O
                hP[k * BRL * D + i * BRL + 1 * D + j] = (float)Dinv[i * D + j];
            }
    }

    // rhs + padded vectors (pads zero); x seeded 0
    std::vector<double> bd(K * D);
    for (auto& v : bd) v = nd(rng);
    std::vector<float> hb(NPAD, 0.f), hx(NPAD, 0.f);
    for (int i = 0; i < K * D; i++) hb[D + i] = (float)bd[i];

    // CPU reference: assemble dense from the SAME blocks and solve in double
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
    CHECK(cudaMemcpy(dS, hS.data(), hS.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dP, hP.data(), hP.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb.data(), NPAD * sizeof(float), cudaMemcpyHostToDevice));

    int fails = 0;
    for (uint32_t threads : {32u, 128u, 1024u}) {
        CHECK(cudaMemcpy(dx, hx.data(), NPAD * sizeof(float), cudaMemcpyHostToDevice));
        size_t smem = glass::pcg_scratch_bytes<float, D, K>(threads);
        pcg_kernel<D, K><<<1, threads, smem>>>(dx, dS, dP, db, 500, 1e-7f, 1e-8f, dit);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        std::vector<float> got(NPAD);
        uint32_t iters = 0;
        CHECK(cudaMemcpy(got.data(), dx, NPAD * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&iters, dit, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        double err = 0;
        for (int i = 0; i < n; i++) {
            double d = got[D + i] - xref[i];
            err += d * d;
        }
        err = std::sqrt(err) / xref_norm;
        // float32 PCG against a double reference: 1e-3 relative catches layout /
        // transpose / preconditioner bugs (those give O(1) errors or no convergence)
        bool ok = err < 1e-3 && iters > 3 && iters < 500;
        printf("threads=%4u  pcg_iters=%3u  rel_err_vs_cpu=%.3e  %s\n",
               threads, iters, err, ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails ? 1 : 0;
}
