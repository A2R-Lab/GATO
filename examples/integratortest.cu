#include <iostream>
#include <vector>

#define KNOT_POINTS 16

#include "bsqp/bsqp.cuh"
#include "types.cuh"
#include "utils/cuda.cuh"

// Define a simple type for testing
typedef float T;

// Simple kernel to test compute_integrator_error
__global__ void test_compute_integrator_error_kernel(T* d_error, void* d_dynMem_const) {
    // Allocate shared memory for the test
    extern __shared__ T s_mem[];
    
    // Set up pointers to different regions of shared memory
    T* s_xuk = s_mem;                             // [q, qd, u]
    T* s_xkp1 = s_xuk + STATE_SIZE + CONTROL_SIZE; // [q_next, qd_next]
    T* s_temp = s_xkp1 + STATE_SIZE;               // Temporary workspace
    
    // Initialize test data with meaningful values
    // Initialize position, velocity, and control values
    for (int i = threadIdx.x; i < STATE_SIZE + CONTROL_SIZE; i += blockDim.x) {
        s_xuk[i] = 1.0; 
    }
    
    // Initialize next state with slightly different values
    for (int i = threadIdx.x; i < STATE_SIZE; i += blockDim.x) {
        s_xkp1[i] = 1.0;
    }
    __syncthreads();
    
    // Call the function being tested
    T error = gato::plant::compute_integrator_error<T, 0, false>(s_xuk, s_xkp1, s_temp, d_dynMem_const, 0.01f);

    __syncthreads();    
    // Store the result
    if (threadIdx.x == 0) {
        *d_error = error;
    }
}

int main() {
    // Initialize dynamics memory (following bsqp.cu example)
    void* d_dynMem_const = gato::plant::initializeDynamicsConstMem<T>();
    
    // Allocate device memory for the error result
    T* d_error;
    cudaMalloc(&d_error, sizeof(T));
    
    // Calculate sufficient shared memory size 
    // Need space for:
    // - s_xuk: STATE_SIZE + CONTROL_SIZE
    // - s_xkp1: STATE_SIZE
    // - Temporary workspace for dynamics calculations and error computation
    size_t sharedMemSize = 10 * 1024;
    
    // Launch the test kernel
    test_compute_integrator_error_kernel<<<1, 256, sharedMemSize>>>(d_error, d_dynMem_const);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_error);
        gato::plant::freeDynamicsConstMem<T>(d_dynMem_const);
        return 1;
    }
    
    // Copy result back to host
    T h_error;
    cudaMemcpy(&h_error, d_error, sizeof(T), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_error);
    gato::plant::freeDynamicsConstMem<T>(d_dynMem_const);
    
    // Print result
    std::cout << "Integrator error: " << h_error << std::endl;
    
    return 0;
}