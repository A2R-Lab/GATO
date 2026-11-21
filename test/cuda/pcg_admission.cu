// Gate for the PCG shared-memory admission check (gato/bsqp/kernels/pcg.cuh),
// compiled against the real launcher via test/cuda/plant_shim.cuh at
// KNOT_POINTS = 512, STATE_SIZE = 14: 144,068 bytes of block shared memory.
//
// Gates:
//   1. pcg_smem_fit reports the kernel's own request plus glass::pcg's five
//      static scalars, against the device's opt-in maximum.
//   2. Where the request passes the limit (sm_120: 101,376 bytes),
//      solve_pcg_batched throws std::runtime_error naming both numbers and the
//      direct solve, before any launch: the lambda buffer keeps its seed and
//      the CUDA error state stays clean. Before the check the attribute set
//      failed with "invalid argument" and the process exited (issue #19 sweep
//      on the RTX 5090, GATO 0.2.1).
//   3. Where the device allows the request, the launch goes through and solves
//      an identity system (lambda = gamma), so the check refuses nothing a
//      device can run.
//   4. opt_in_dynamic_smem (gato/utils/cuda.cuh), the check every other kernel
//      launch goes through: a request four bytes past the device maximum
//      throws, names the stage it was given and leaves the latch untouched;
//      a request at the maximum is set.
//
//   nvcc -std=c++17 -O2 -DNDEBUG -arch=native -DKNOT_POINTS=512 \
//        -DGATO_PLANT_HEADER='"plant_shim.cuh"' -I gato -I <GLASS> -I test/cuda \
//        test/cuda/pcg_admission.cu -o build/pcg_admission && ./build/pcg_admission
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "bsqp/kernels/pcg.cuh"

#define CHECK_CUDA(call)                                                                                       \
    do {                                                                                                       \
        cudaError_t err = (call);                                                                              \
        if (err != cudaSuccess) {                                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);          \
            exit(1);                                                                                           \
        }                                                                                                      \
    } while (0)

using gato::constants::STATE_SIZE;
using gato::constants::VEC_SIZE_PADDED;
using gato::constants::B3D_MATRIX_SIZE_PADDED;
using gato::constants::BLOCK_ROW_SIZE;

__global__ void probe_kernel(float* d_out)
{
    extern __shared__ float s_mem[];
    if (threadIdx.x == 0) { s_mem[0] = 1.f; d_out[0] = s_mem[0]; }
}

int main()
{
    int fails = 0;
    const uint32_t B = 2;
    const PcgSmemFit fit = pcg_smem_fit<float>();
    int device = 0, limit = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const size_t want = get_solve_pcg_batched_smem_size<float>() + 5 * sizeof(float);
    printf("need %zu bytes, limit %zu bytes, fits %d\n", fit.need, fit.limit, (int)fit.fits());
    if (fit.need != want || fit.need != 144068 || fit.limit != (size_t)limit) { printf("FAIL gate 1: fit report\n"); fails++; }

    // identity system: S = P_inv = I on the diagonal strip, gamma = k-dependent values, lambda seeded at 7
    const size_t        strip = B3D_MATRIX_SIZE_PADDED, vec = VEC_SIZE_PADDED;
    std::vector<float> hS(B * strip, 0.f), hG(B * vec, 0.f), hL(B * vec, 7.f), hEps(B, 1e-6f);
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t k = 0; k < KNOT_POINTS; k++) {
            for (uint32_t i = 0; i < STATE_SIZE; i++) {
                hS[b * strip + k * BLOCK_ROW_SIZE + i * 3 * STATE_SIZE + STATE_SIZE + i] = 1.f;
                hG[b * vec + (k + 1) * STATE_SIZE + i] = 0.25f * (float)((k + i + b) % 9) - 1.f;
            }
        }
        for (uint32_t i = 0; i < STATE_SIZE; i++) { hL[b * vec + i] = 0.f; hL[b * vec + vec - STATE_SIZE + i] = 0.f; }
    }
    float *dS, *dP, *dG, *dL, *dEps;
    int32_t*  dConv;
    uint32_t* dIt;
    CHECK_CUDA(cudaMalloc(&dS, hS.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP, hS.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dG, hG.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dL, hL.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dEps, B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dConv, B * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dIt, B * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(dS, hS.data(), hS.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dP, hS.data(), hS.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dG, hG.data(), hG.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dL, hL.data(), hL.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dEps, hEps.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dConv, 0, B * sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(dIt, 0, B * sizeof(uint32_t)));

    SchurSystem<float> schur{dS, dP, dG};
    bool               threw = false;
    std::string        what;
    try {
        solve_pcg_batched<float>(B, dL, schur, dEps, 50, dConv, dIt);
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> out(hL.size());
    CHECK_CUDA(cudaMemcpy(out.data(), dL, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (!fit.fits()) {
        const bool msg = what.find(std::to_string(fit.need)) != std::string::npos && what.find(std::to_string(fit.limit)) != std::string::npos
                         && what.find("bdsv") != std::string::npos && what.find("N = 512") != std::string::npos;
        if (!threw || !msg) { printf("FAIL gate 2: refusal (threw %d): %s\n", (int)threw, what.c_str()); fails++; }
        if (out != hL) { printf("FAIL gate 2: lambda changed under a refused solve\n"); fails++; }
        printf("refused: %s\n", what.c_str());
    } else {
        double worst = 0.0;
        for (size_t i = 0; i < out.size(); i++) { worst = std::fmax(worst, std::fabs((double)out[i] - (double)hG[i])); }
        if (threw || worst > 1e-5) { printf("FAIL gate 3: admitted solve (threw %d, worst %.3e)\n", (int)threw, worst); fails++; }
        printf("admitted: worst |lambda - gamma| %.3e\n", worst);
    }
    {
        size_t      latch = 0;
        bool        refused = false;
        std::string msg;
        try {
            opt_in_dynamic_smem(probe_kernel, fit.limit + 4, latch, "The probe stage");
        } catch (const std::runtime_error& e) {
            refused = true;
            msg = e.what();
        }
        const bool named = msg.find("The probe stage") != std::string::npos && msg.find(std::to_string(fit.limit + 4)) != std::string::npos
                           && msg.find(std::to_string(fit.limit)) != std::string::npos;
        if (!refused || !named || latch != 0) { printf("FAIL gate 4: over-limit opt-in (refused %d, latch %zu): %s\n", (int)refused, latch, msg.c_str()); fails++; }
        opt_in_dynamic_smem(probe_kernel, fit.limit, latch, "The probe stage");
        float* dOut;
        CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
        probe_kernel<<<1, 32, fit.limit>>>(dOut);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        float one = 0.f;
        CHECK_CUDA(cudaMemcpy(&one, dOut, sizeof(float), cudaMemcpyDeviceToHost));
        if (latch != fit.limit || one != 1.f) { printf("FAIL gate 4: opt-in at the maximum (latch %zu)\n", latch); fails++; }
    }
    printf(fails ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return fails ? 1 : 0;
}
