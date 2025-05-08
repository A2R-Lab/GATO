#pragma once

#include <cstdint>
#include "settings.h"
#include "constants.h"

using namespace sqp;
using namespace gato::constants;

// TODO: cholesky matrix inverse
namespace block {  // utils for block-level operations

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t FULL_MASK = 0xffffffff;

// set shared memory to zero
template<typename T, uint32_t size>
__device__ __forceinline__ void zeroSharedMemory(T* mem)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { mem[i] = T(0); }
}

template<typename T, uint32_t size>
__device__ __forceinline__ void copy(T* dst, T* src)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { dst[i] = src[i]; }
}

// overloaded copy with scaling
template<typename T, uint32_t size>
__device__ __forceinline__ void copy(T* dst, T* src, T alpha)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { dst[i] = alpha * src[i]; }
}

// vector sum with output parameter
template<typename T, uint32_t size>
__device__ __forceinline__ void vecSum(T* out, T* a, T* b)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { out[i] = a[i] + b[i]; }
}

// vector sum in-place
template<typename T, uint32_t size>
__device__ __forceinline__ void vecSum(T* a, T* b)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { a[i] += b[i]; }
}

// vector difference with output parameter
template<typename T, uint32_t size>
__device__ __forceinline__ void vecSub(T* out, T* a, T* b)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { out[i] = a[i] - b[i]; }
}

// vector difference in-place
template<typename T, uint32_t size>
__device__ __forceinline__ void vecSub(T* a, T* b)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) { a[i] -= b[i]; }
}

// load identity into memory (column-major)
template<typename T, uint32_t dim>
__device__ __forceinline__ void loadIdentity(T* A)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < dim * dim; i += blockDim.x) {
                uint32_t x, y;
                x = i / dim;
                y = i % dim;
                A[i] = static_cast<T>(x == y);
        }
}

// add a scaled identity matrix to a square matrix in column-major order
template<typename T, uint32_t dim>
__device__ __forceinline__ void addScaledIdentity(T* A, T alpha)
{
#pragma unroll
        for (uint32_t i = threadIdx.x; i < dim * dim; i += blockDim.x) {
                uint32_t x = i / dim;
                uint32_t y = i % dim;
                if (x == y) {
                        // For column-major, diagonal elements are at col*dim + col
                        A[x * dim + y] += alpha;
                }
        }
}

// C = A * B
// A is (m x n), B is (n x k), C is (m x k)
// A, B, C are assumed to be in column-major order
template<typename T, uint32_t m, uint32_t n, uint32_t k>
__device__ __forceinline__ void matMul(T* C, T* A, T* B, bool negate = false)
{
        // threads each compute an element of C
        uint32_t x, y;
        T        sum;
#pragma unroll
        for (uint32_t i = threadIdx.x; i < m * k; i += blockDim.x) {
                y = i % m;  // row index
                x = i / m;  // col index
                sum = static_cast<T>(0);
                for (uint32_t j = 0; j < n; j++) { sum += A[j * m + y] * B[x * n + j]; }
                C[x * m + y] = negate ? -sum : sum;
        }
}

// C += A * B
// A is (m x n), B is (n x k), C is (m x k)
// A, B, C are assumed to be in column-major order
template<typename T, uint32_t m, uint32_t n, uint32_t k>
__device__ __forceinline__ void matMulSum(T* C, T* A, T* B, bool negate = false)
{
        // threads each compute an element of C
        uint32_t x, y;
        T        sum;
#pragma unroll
        for (uint32_t i = threadIdx.x; i < m * k; i += blockDim.x) {
                y = i % m;  // row index
                x = i / m;  // col index
                sum = static_cast<T>(0);
                for (uint32_t j = 0; j < n; j++) { sum += A[j * m + y] * B[x * n + j]; }
                C[x * m + y] += negate ? -sum : sum;
        }
}

// C = A * B^T
// A is (m x n), B is (k x n), C is (m x k)
// A, B, C are assumed to be in column-major order
template<typename T, uint32_t m, uint32_t n, uint32_t k>
__device__ __forceinline__ void matMulTranspose(T* C, T* A, T* B, bool negate = false)
{
        // threads each compute an element of C
        uint32_t x, y;
        T        sum;
#pragma unroll
        for (uint32_t i = threadIdx.x; i < m * k; i += blockDim.x) {
                y = i % m;  // row index
                x = i / m;  // col index
                sum = static_cast<T>(0);
                for (uint32_t j = 0; j < n; j++) { sum += A[j * m + y] * B[j * k + x]; }
                C[x * m + y] = negate ? -sum : sum;
        }
}

// C += A * B^T
// A is (m x n), B is (k x n), C is (m x k)
// A, B, C are assumed to be in column-major order
template<typename T, uint32_t m, uint32_t n, uint32_t k>
__device__ __forceinline__ void matMulTransposeSum(T* C, T* A, T* B, bool negate = false)
{
        // threads each compute an element of C
        uint32_t x, y;
        T        sum;
#pragma unroll
        for (uint32_t i = threadIdx.x; i < m * k; i += blockDim.x) {
                y = i % m;  // row index
                x = i / m;  // col index
                sum = static_cast<T>(0);
                for (uint32_t j = 0; j < n; j++) { sum += A[j * m + y] * B[j * k + x]; }
                C[x * m + y] += negate ? -sum : sum;
        }
}

template<typename T, uint32_t NumBlockRows, uint32_t BlockSize>
__device__ __forceinline__ void btdMatrixVectorProduct(T* s_output, const T* s_matrix, const T* s_vector)
{
        const uint32_t BlockRowLength = 3 * BlockSize;

        // Get warp and lane indices
        const uint32_t lane_idx = threadIdx.x & 31;        // threadIdx.x % 32
        const uint32_t warp_idx = threadIdx.x >> 5;        // threadIdx.x / 32
        const uint32_t warps_per_block = blockDim.x >> 5;  // blockDim.x / 32

// each warp handles a block row matrix
#pragma unroll
        for (uint32_t block_row = warp_idx; block_row < NumBlockRows; block_row += warps_per_block) {
                const T* block = s_matrix + block_row * BlockRowLength * BlockSize;
                const T* vec = s_vector + block_row * BlockSize;

                // temp storage for each thread's sums
                T thread_sums[BlockSize] = {T(0.0)};

                T vec_val;

// Each lane handles a column
#pragma unroll
                for (uint32_t col = lane_idx; col < BlockRowLength; col += WARP_SIZE) {
                        // broadcast vector value to all currently active threads in warp
                        vec_val = vec[col];

// compute contribution to each row
#pragma unroll
                        for (uint32_t row = 0; row < BlockSize; row++) { thread_sums[row] += block[row * BlockRowLength + col] * vec_val; }
                }
                // __syncwarp();

// warp-level reduction for each row
#pragma unroll
                for (uint32_t row = 0; row < BlockSize; row++) {
                        T sum = thread_sums[row];

                        // Warp-wide reduction using shuffle with active mask
                        const unsigned active_mask = __activemask();
#pragma unroll
                        for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) { sum += __shfl_down_sync(active_mask, sum, offset); }

                        // First lane writes result
                        if (lane_idx == 0) { s_output[(block_row + 1) * BlockSize + row] = sum; }
                }
        }
}

// overloaded for 2 output vectors, otherwise same as above
template<typename T, uint32_t NumBlockRows, uint32_t BlockSize>
__device__ __forceinline__ void btdMatrixVectorProduct(T* s_output_1, T* s_output_2, const T* s_matrix, const T* s_vector)
{
        const uint32_t BlockRowLength = 3 * BlockSize;

        // Get warp and lane indices
        const uint32_t lane_idx = threadIdx.x & 31;        // threadIdx.x % 32
        const uint32_t warp_idx = threadIdx.x >> 5;        // threadIdx.x / 32
        const uint32_t warps_per_block = blockDim.x >> 5;  // blockDim.x / 32

// each warp handles a block row matrix
#pragma unroll
        for (uint32_t block_row = warp_idx; block_row < NumBlockRows; block_row += warps_per_block) {
                const T* block = s_matrix + block_row * BlockRowLength * BlockSize;
                const T* vec = s_vector + block_row * BlockSize;

                // temp storage for each thread's sums
                T thread_sums[BlockSize] = {T(0.0)};
                T vec_val;

// Each lane handles a column
#pragma unroll
                for (uint32_t col = lane_idx; col < BlockRowLength; col += WARP_SIZE) {
                        // broadcast vector value to all currently active threads in warp
                        vec_val = vec[col];

// compute contribution to each row
#pragma unroll
                        for (uint32_t row = 0; row < BlockSize; row++) { thread_sums[row] += block[row * BlockRowLength + col] * vec_val; }
                }
                // __syncwarp();

// warp-level reduction for each row
#pragma unroll
                for (uint32_t row = 0; row < BlockSize; row++) {
                        T sum = thread_sums[row];

                        // Warp-wide reduction using shuffle with active mask
                        const unsigned active_mask = __activemask();
#pragma unroll
                        for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) { sum += __shfl_down_sync(active_mask, sum, offset); }

                        // First lane writes result
                        if (lane_idx == 0) {
                                s_output_1[(block_row + 1) * BlockSize + row] = sum;
                                s_output_2[(block_row + 1) * BlockSize + row] = sum;
                        }
                }
        }
}

// /**
// * @brief Computes the dot product of two vectors within one thread block
// *
// * @tparam T Data type
// *
// * @param[out] result Output scalar result
// * @param[in] s_a First input vector in shared memory
// * @param[in] s_b Second input vector in shared memory
// * @param[in] s_scratch Shared memory workspace for warp-level reduction
// *                    (size: ceil(blockDim.x/32) elements)
// * @param[in] size Length of input vectors
// *
// * @note Requires blockDim.x to be a multiple of 32 (warp size)
// * @note s_scratch array must be in shared memory
// */
template<typename T>
__device__ __forceinline__ void dot(T* result, T* s_a, T* s_b, T* s_scratch, uint32_t size)
{
#ifndef NDEBUG
        assert(blockDim.x % WARP_SIZE == 0 && "blockDim.x must be a multiple of 32 (warp size)");
#endif

        const uint32_t tid = threadIdx.x;
        const uint32_t lane_idx = tid & 31;                // threadIdx.x % 32 (thread in warp)
        const uint32_t warp_idx = tid >> 5;                // threadIdx.x / 32
        const uint32_t warps_per_block = blockDim.x >> 5;  // blockDim.x / 32

        T sum = T(0.0);

// partial sum for each thread
#pragma unroll
        for (uint32_t i = tid; i < size; i += blockDim.x) { sum += s_a[i] * s_b[i]; }
        __syncthreads();

// sum within each warp
#pragma unroll
        for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) { sum += __shfl_down_sync(__activemask(), sum, offset); }

        // lane 0 of each warp writes to shared memory
        if (lane_idx == 0) { s_scratch[warp_idx] = sum; }
        __syncthreads();

        // copy warp sums to first warp, and reduce again
        if (tid < WARP_SIZE) {
                // only use valid warp sums (blockDim.x / 32) valid warps
                sum = (tid < warps_per_block) ? s_scratch[tid] : T(0.0);

#pragma unroll
                for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) { sum += __shfl_down_sync(__activemask(), sum, offset); }

                if (tid == 0) { *result = sum; }
        }
}

template<typename T>
__device__ __forceinline__ void reduce(const uint32_t n, T* x)
{
        uint32_t idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
        unsigned size_left = n;
        bool     odd_flag;

        // loop until only a few values left
        while (size_left > 3) {
                // determine if odd_adjust needed and update size
                odd_flag = size_left % 2;
                size_left = (size_left - odd_flag) / 2;
                // reduce in half
                for (uint32_t i = idx; i < size_left; i += stride) { x[i] += x[i + size_left]; }
                // add the odd size adjust if needed
                if (idx == 0 && odd_flag) { x[0] += x[2 * size_left]; }
                // sync and repeat
                __syncthreads();
        }
        // when we get really small sum up what is left
        if (idx == 0) {
                for (uint32_t i = 1; i < size_left; i++) { x[0] += x[i]; }
        }
}


/**
 * @brief Inverts a square matrix using Gaussian elimination
 *
 * @tparam T The data type of the matrix elements
 * @param DIM The dimension of the square matrix
 * @param A The input/output matrix
 * @param s_temp Temporary shared memory for calculations
 */
template<typename T>
__device__ void invertMatrix(uint32_t DIM, T* A, T* s_temp)
{
        // we are going to guassian elimination walking down the matrix (assuming no leading 0s)
        // we therefore use the columns in order as the pivot column for each pivot we need to rescale
        // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple
        // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
        // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
        for (unsigned pivRC = 0; pivRC < DIM; pivRC++) {
                unsigned pivColOffset = pivRC * DIM;
                // save the pivot and pivot column and row
                T pvInv = static_cast<T>(1) / A[pivRC + pivColOffset];
                for (unsigned ind = threadIdx.x; ind < 2 * DIM + 1; ind++) {
                        unsigned AInd;
                        if (ind < DIM) {
                                AInd = ind + pivColOffset;
                        } else {
                                AInd = pivRC + pivColOffset + (ind - DIM) * DIM;
                        }
                        s_temp[ind] = A[AInd];
                }
                __syncthreads();  //----------------------
                // make the pivot update
                for (unsigned ind = threadIdx.x; ind < DIM * (DIM + 1); ind += blockDim.x) {
                        unsigned row = ind % DIM;
                        unsigned col = ind / DIM;
                        unsigned colOffset = ind - row;
                        // s_temp = orpcvs|prvOld
                        if (row == pivRC) {
                                A[row + pivColOffset + colOffset] *= pvInv;
                        } else {
                                A[row + pivColOffset + colOffset] -= s_temp[row] * pvInv * s_temp[DIM + col];
                        }
                }
                __syncthreads();  //----------------------
        }
}


template<typename T>
__device__ void invertMatrix(unsigned DIMA, unsigned DIMB, unsigned MAX_DIM, T* A, T* B, T* s_temp)
{

        // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
        // we therefore use the columns in order as the pivot column for each pivot we need to rescale
        // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple
        // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
        // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
        T* s_memA = s_temp;
        T* s_memB = &s_memA[2 * DIMA + 1];
        for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++) {
                bool     AActive = pivRC < DIMA;
                bool     BActive = pivRC < DIMB;
                unsigned pivColOffsetA = pivRC * DIMA;
                unsigned pivColOffsetB = pivRC * DIMB;
                // save the pivot column and row
                for (unsigned ind = threadIdx.x; ind < MAX_DIM; ind++) {
                        if (AActive && ind < DIMA) { s_memA[ind] = A[ind + pivColOffsetA]; }
                        if (BActive && ind < DIMB) { s_memB[ind] = B[ind + pivColOffsetB]; }
                }
                for (unsigned ind = threadIdx.x; ind < MAX_DIM + 1; ind++) {
                        if (AActive && ind < DIMA + 1) { s_memA[ind + DIMA] = A[ind * DIMA + pivRC + pivColOffsetA]; }
                        if (BActive && ind < DIMB + 1) { s_memB[ind + DIMB] = B[ind * DIMB + pivRC + pivColOffsetB]; }
                }
                __syncthreads();  //----------------------
                // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
                for (unsigned ind = threadIdx.x; ind < MAX_DIM * (MAX_DIM + 1); ind += blockDim.x) {
                        if (AActive && ind < DIMA * (DIMA + 1)) {
                                unsigned row = ind % DIMA;
                                unsigned col = ind / DIMA;
                                if (row == pivRC) {
                                        A[pivColOffsetA + ind] /= s_memA[pivRC];
                                } else {
                                        A[pivColOffsetA + ind] -= s_memA[row] / s_memA[pivRC] * s_memA[DIMA + col];
                                }
                        }
                        if (BActive && ind < DIMB * (DIMB + 1)) {
                                unsigned row = ind % DIMB;
                                unsigned col = ind / DIMB;
                                if (row == pivRC) {
                                        B[pivColOffsetB + ind] /= s_memB[pivRC];
                                } else {
                                        B[pivColOffsetB + ind] -= s_memB[row] / s_memB[pivRC] * s_memB[DIMB + col];
                                }
                        }
                }
                __syncthreads();  //----------------------
        }
}

// invert A,B,C assume memory for all is [V | VInv] where both are DIMxDIM and continguous
// relies on s_temp of size [2*DIMA + 2*DIMB + 2*DIMC + 3]
template<typename T>
__device__ void invertMatrix(unsigned DIMA, unsigned DIMB, unsigned DIMC, unsigned MAX_DIM, T* A, T* B, T* C, T* s_temp)
{

        // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
        // we therefore use the columns in order as the pivot column for each pivot we need to rescale
        // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple
        // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
        // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
        T* s_memA = s_temp;
        T* s_memB = &s_memA[2 * DIMA + 1];
        T* s_memC = &s_memB[2 * DIMB + 1];
        for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++) {
                bool     AActive = pivRC < DIMA;
                bool     BActive = pivRC < DIMB;
                bool     CActive = pivRC < DIMC;
                unsigned pivColOffsetA = pivRC * DIMA;
                unsigned pivColOffsetB = pivRC * DIMB;
                unsigned pivColOffsetC = pivRC * DIMC;
                // save the pivot column and row
                for (unsigned ind = threadIdx.x; ind < MAX_DIM; ind++) {
                        if (AActive && ind < DIMA) { s_memA[ind] = A[ind + pivColOffsetA]; }
                        if (BActive && ind < DIMB) { s_memB[ind] = B[ind + pivColOffsetB]; }
                        if (CActive && ind < DIMC) { s_memC[ind] = C[ind + pivColOffsetC]; }
                }
                for (unsigned ind = threadIdx.x; ind < MAX_DIM + 1; ind++) {
                        if (AActive && ind < DIMA + 1) { s_memA[ind + DIMA] = A[ind * DIMA + pivRC + pivColOffsetA]; }
                        if (BActive && ind < DIMB + 1) { s_memB[ind + DIMB] = B[ind * DIMB + pivRC + pivColOffsetB]; }
                        if (CActive && ind < DIMC + 1) { s_memC[ind + DIMC] = C[ind * DIMC + pivRC + pivColOffsetC]; }
                }
                __syncthreads();  //----------------------
                // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
                for (unsigned ind = threadIdx.x; ind < MAX_DIM * (MAX_DIM + 1); ind += blockDim.x) {
                        if (AActive && ind < DIMA * (DIMA + 1)) {
                                unsigned row = ind % DIMA;
                                unsigned col = ind / DIMA;
                                if (row == pivRC) {
                                        A[pivColOffsetA + ind] /= s_memA[pivRC];
                                } else {
                                        A[pivColOffsetA + ind] -= s_memA[row] / s_memA[pivRC] * s_memA[DIMA + col];
                                }
                        }
                        if (BActive && ind < DIMB * (DIMB + 1)) {
                                unsigned row = ind % DIMB;
                                unsigned col = ind / DIMB;
                                if (row == pivRC) {
                                        B[pivColOffsetB + ind] /= s_memB[pivRC];
                                } else {
                                        B[pivColOffsetB + ind] -= s_memB[row] / s_memB[pivRC] * s_memB[DIMB + col];
                                }
                        }
                        if (CActive && ind < DIMC * (DIMC + 1)) {
                                unsigned row = ind % DIMC;
                                unsigned col = ind / DIMC;
                                if (row == pivRC) {
                                        C[pivColOffsetC + ind] /= s_memC[pivRC];
                                } else {
                                        C[pivColOffsetC + ind] -= s_memC[row] / s_memC[pivRC] * s_memC[DIMC + col];
                                }
                        }
                }
                __syncthreads();  //----------------------
        }
}

// print m x n matrix (column-major)
template<typename T, uint32_t m, uint32_t n>
__host__ __device__ void printMatrix(T* A)
{
        for (uint32_t y = 0; y < m; y++) {
                for (uint32_t x = 0; x < n; x++) { printf("%.4f ", A[x * m + y]); }
                printf("\n");
        }
}

// print m x n matrix (row-major)
template<typename T, uint32_t m, uint32_t n>
__host__ __device__ void printMatrixRowMajor(T* A)
{
        for (uint32_t y = 0; y < m; y++) {
                for (uint32_t x = 0; x < n; x++) { printf("%.4f ", A[y * n + x]); }
                printf("\n");
        }
}


}  // namespace block


namespace gato {

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetWrench(T* batch, uint32_t solve_idx)
{
        return batch + solve_idx * 6;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetState(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_KNOTS + knot_idx * STATE_SIZE;
}

// compute pointer to a (CONTROL_SIZE) vector from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetControl(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_P_KNOTS + knot_idx * CONTROL_SIZE;
}

// compute pointer to a (STATE_SIZE x STATE_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetStateSq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_SQ_P_KNOTS + knot_idx * STATE_SIZE_SQ;
}

// compute pointer to a (CONTROL_SIZE x CONTROL_SIZE) matrix from a batch (BATCH_SIZE X KNOT_POINTS)
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetControlSq(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * CONTROL_SQ_P_KNOTS + knot_idx * CONTROL_SIZE_SQ;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetStatePControl(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * STATE_P_CONTROL_P_KNOTS + knot_idx * STATE_P_CONTROL;
}

// compute offset for accessing a knot point of a batch of trajectories (BATCH_SIZE X ((STATE_SIZE + CONTROL_SIZE) X KNOT_POINTS - CONTROL_SIZE))
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetTraj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * TRAJ_SIZE + knot_idx * STATE_S_CONTROL;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetReferenceTraj(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * REFERENCE_TRAJ_SIZE + knot_idx * grid::EE_POS_SIZE;
}

// compute pointer to a (STATE_SIZE) vector from a batch (BATCH_SIZE X (KNOT_POINTS + 2))
// each solve batch is padded to (KNOT_POINTS + 2) * STATE_SIZE for the PCG solver
template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetStatePadded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * VEC_SIZE_PADDED + (knot_idx + 1) * STATE_SIZE;
}

template<typename T, uint32_t BatchSize>
__device__ __forceinline__ T* getOffsetBlockRowPadded(T* batch, uint32_t solve_idx, uint32_t knot_idx)
{
        return batch + solve_idx * B3D_MATRIX_SIZE_PADDED + knot_idx * BLOCK_ROW_SIZE;
}

}  // namespace gato
