// Gate for the fused merit launch (gato/bsqp/kernels/merit.cuh): one block per
// knot evaluating all NUM_ALPHAS candidates (FUSED) must produce the same
// per-knot partials and per-candidate merits, bit for bit, as one block per
// (knot, candidate). Real indy7 headers at N8, a batch of B random problems,
// both paths forced through the launcher's `fused` argument.
//
// Run by test_kernel_gates.py:
//   nvcc -std=c++17 -O2 -DNDEBUG -arch=native -DKNOT_POINTS=8 \
//        -DGATO_PLANT_HEADER='"dynamics/plant.cuh"' \
//        -I gato -I gato/dynamics/indy7 -I <GLASS> -I <collision> -I test/cuda \
//        test/cuda/merit_fusion_parity.cu -o /tmp/merit_fusion_parity && /tmp/merit_fusion_parity
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "settings.h"
#include "constants.h"
#include "types.cuh"
#include "bsqp/kernels/merit.cuh"

using T = float;

#define CHECK_CUDA(call)                                                                           \
        do {                                                                                       \
                cudaError_t err = (call);                                                          \
                if (err != cudaSuccess) {                                                          \
                        fprintf(stderr, "FAIL CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
                        exit(1);                                                                   \
                }                                                                                  \
        } while (0)

int main()
{
        using namespace gato::constants;
        constexpr int      B = 3;
        constexpr uint32_t NA = NUM_ALPHAS;
        std::mt19937                     rng(7);
        std::normal_distribution<float>  nd(0.f, 1.f);

        std::vector<T> xu(B * XU_TRAJ_SIZE), dz(B * TRAJ_SIZE), xs(B * XU_STATE_SIZE), ref(B * REFERENCE_TRAJ_SIZE), mu(B);
        for (auto& v : xu) v = 0.3f * nd(rng);
        for (auto& v : dz) v = 0.05f * nd(rng);
        for (int b = 0; b < B; b++) {
                for (uint32_t i = 0; i < XU_STATE_SIZE; i++) xs[b * XU_STATE_SIZE + i] = xu[b * XU_TRAJ_SIZE + i] + 0.01f * nd(rng);
                mu[b] = 1.0f + 0.5f * b;
        }
        for (auto& v : ref) v = 0.3f * nd(rng);

        T *d_xu, *d_dz, *d_xs, *d_ref, *d_mu, *d_merit[2], *d_partial[2];
        CHECK_CUDA(cudaMalloc(&d_xu, xu.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_dz, dz.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_xs, xs.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_ref, ref.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&d_mu, mu.size() * sizeof(T)));
        for (int f = 0; f < 2; f++) {
                CHECK_CUDA(cudaMalloc(&d_merit[f], B * NA * sizeof(T)));
                CHECK_CUDA(cudaMalloc(&d_partial[f], (size_t)B * NA * KNOT_POINTS * sizeof(T)));
                CHECK_CUDA(cudaMemset(d_merit[f], 0, B * NA * sizeof(T)));
                CHECK_CUDA(cudaMemset(d_partial[f], 0, (size_t)B * NA * KNOT_POINTS * sizeof(T)));
        }
        CHECK_CUDA(cudaMemcpy(d_xu, xu.data(), xu.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_dz, dz.data(), dz.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_xs, xs.data(), xs.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ref, ref.data(), ref.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_mu, mu.data(), mu.size() * sizeof(T), cudaMemcpyHostToDevice));

        grid::robotModel<T>* drm = grid::init_robotModel<T>();
        ProblemInputs<T>     in{0.01f, d_xs, d_ref};

        for (int f = 0; f < 2; f++) {
                compute_merit_batched<T, NA>(B, nullptr, nullptr, d_merit[f], d_partial[f], d_dz, d_xu, nullptr, in, d_mu, (void*)drm,
                                             1.0f, 1e-2f, 1e-4f, 5.0f, 0.1f, 0.1f, 0.1f,
                                             nullptr, 0, nullptr, nullptr, nullptr, nullptr, 0, grid_collision::Environment<T>{}, nullptr,
                                             0.0f, nullptr, 0.0f, nullptr, nullptr, nullptr, /*fused=*/f);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
        }

        std::vector<T> merit[2], partial[2];
        for (int f = 0; f < 2; f++) {
                merit[f].resize(B * NA);
                partial[f].resize((size_t)B * NA * KNOT_POINTS);
                CHECK_CUDA(cudaMemcpy(merit[f].data(), d_merit[f], merit[f].size() * sizeof(T), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(partial[f].data(), d_partial[f], partial[f].size() * sizeof(T), cudaMemcpyDeviceToHost));
        }

        int  fails = 0;
        bool finite = true;
        for (T v : merit[0]) finite = finite && std::isfinite(v);
        if (!finite) { printf("FAIL unfused merits not finite\n"); fails++; }
        if (memcmp(partial[0].data(), partial[1].data(), partial[0].size() * sizeof(T)) != 0) {
                printf("FAIL per-knot partials differ between unfused and fused launches\n");
                fails++;
        }
        if (memcmp(merit[0].data(), merit[1].data(), merit[0].size() * sizeof(T)) != 0) {
                printf("FAIL per-candidate merits differ between unfused and fused launches\n");
                fails++;
        }
        T spread = 0;
        for (int b = 0; b < B; b++)
                for (uint32_t a = 1; a < NA; a++) spread = fmaxf(spread, fabsf(merit[0][b * NA + a] - merit[0][b * NA]));
        if (!(spread > 0)) { printf("FAIL candidates all equal: the trial step did not move the merit\n"); fails++; }

        const size_t smem = get_compute_merit_batched_smem_size<T>(0);
        printf("merit fusion parity: B=%d N=%d alphas=%u threads=%u fused-by-rule=%d: %s\n", B, KNOT_POINTS, NA, MERIT_THREADS,
               (int)merit_fused_launch<T, NA>(B, smem, -1), fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
}
