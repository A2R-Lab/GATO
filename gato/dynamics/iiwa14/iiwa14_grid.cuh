/**
 * This instance of grid.cuh is optimized for the urdf: KUKAiiwa14
 *
 * Notes:
 *   Interface is:
 *       __host__   robotModel<T> *d_robotModel = init_robotModel<T>()
 *       __host__   cudaStream_t streams = init_grid<T>()
 *       __host__   gridData<T> *hd_ata = init_gridData<T,NUM_TIMESTEPS>();    __host__   close_grid<T>(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data)
 *   
 *       __device__ inverse_dynamics_inner<T>(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_inner<T>(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_inner_vaf<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_inner_vaf<T>(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *   
 *       __device__ direct_minv_inner<T>(T *s_Minv, const T *s_q, T *s_XImats, int *s_topology_helpers, T *s_temp)
 *       __device__ direct_minv_device<T>(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ direct_minv_Kernel<T>(T *d_Minv, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   direct_minv<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_inner<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ forward_dynamics_device<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_kernel<T>(T *d_qdd, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_gradient_inner<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity)
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics_gradient<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics_gradient<T,USE_QDD_MINV_FLAG=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_inner<T>(T *s_eePos, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_device<T>(T *s_eePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_kernel<T>(T *d_eePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_gradient_inner<T>(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_gradient_device<T>(T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_gradient_kernel<T>(T *d_deePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose_gradient<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_pose_gradient_hessian_inner<T>(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_pose_gradient_hessian_device<T>(T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_pose_gradient_hessian_kernel<T>(T *d_deePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_pose_gradient_hessian<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ idsva_so_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, T *s_XImats, T *s_mem, const T gravity)
 *       __global__ idsva_so_kernel(T *d_idsva_so, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   idsva_so_host<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ fdsva_so_inner(T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_q, T *s_qd, const T *s_qdd, const T *s_tau, T *s_XImats, T *s_temp, const T gravity)
 *       __device__ fdsva_so_device(T *s_df2, T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ fdsva_so_kernel(T *d_df2, const T *d_q_qd_qdd_tau, const int stride_q_qd_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   fdsva_so<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *   
 *   Suggested Type T is float
 *   
 *   Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file
 *   
 *   By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [ID, MINV, FD, ID_DU, FD_DU]
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils/cuda.cuh"

// single kernel timing helper code
#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

#define XIMAT_SIZE 36

template <typename T, int M, int N>
__host__ __device__
void printMat(T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

template <typename T, int M, int N>
__host__ __device__
void printMat(const T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

/**
 * All functions are kept in this namespace
 *
 */
namespace grid {
    constexpr int NUM_JOINTS = 7;
    constexpr int NUM_VEL = 7;
    constexpr int NUM_EES = 1;
    constexpr int NQ = 7;
    constexpr int NX = 14;
    constexpr int NU = 7;
    constexpr int EE_POS_SIZE = 6;
    constexpr int NEE = 6;
    constexpr int ID_DYNAMIC_SHARED_MEM_COUNT = 882;
    constexpr int MINV_DYNAMIC_SHARED_MEM_COUNT = 1507;
    constexpr int FD_DYNAMIC_SHARED_MEM_COUNT = 1731;
    constexpr int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 2562;
    constexpr int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 2562;
    constexpr int ABA_DYNAMIC_SHARED_MEM_COUNT = 1820;
    constexpr int CRBA_SHARED_MEM_COUNT = 1820;
    constexpr int ID_DU_MAX_SHARED_MEM_COUNT = 2807;
    constexpr int FD_DU_MAX_SHARED_MEM_COUNT = 2961;
    constexpr int EE_POS_DYNAMIC_SHARED_MEM_COUNT = 144;
    constexpr int DEE_POS_DYNAMIC_SHARED_MEM_COUNT = 672;
    constexpr int D2EE_POS_DYNAMIC_SHARED_MEM_COUNT = 2160;
    constexpr int IDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = 0;
    constexpr int FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT = 2562;
    constexpr int SUGGESTED_THREADS = 352;
    // Define custom structs
    template <typename T>
    struct robotModel {
        T *d_XImats;
        int *d_topology_helpers;
    };
    template <typename T>
    struct gridData {
        // GPU INPUTS
        T *d_q_qd_u;
        T *d_q_qd;
        T *d_q;
        // CPU INPUTS
        T *h_q_qd_u;
        T *h_q_qd;
        T *h_q;
        // GPU OUTPUTS
        T *d_c;
        T *d_Minv;
        T *d_qdd;
        T *d_M;
        T *d_dc_du;
        T *d_df_du;
        T *d_eePos;
        T *d_deePos;
        T *d_d2eePos;
        T *d_idsva_so;
        T *d_df2;
        // CPU OUTPUTS
        T *h_c;
        T *h_Minv;
        T *h_qdd;
        T *h_M;
        T *h_dc_du;
        T *h_df_du;
        T *h_eePos;
        T *h_deePos;
        T *h_d2eePos;
        T *h_idsva_so;
        T *h_df2;
    };
    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, const T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, const T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = s_vec[2];
        s_vecX[2] = -s_vec[1];
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[5];
        s_vecX[5] = -s_vec[4];
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0_peq(T *s_vecX, const T *s_vec) {
        s_vecX[1] += s_vec[2];
        s_vecX[2] += -s_vec[1];
        s_vecX[4] += s_vec[5];
        s_vecX[5] += -s_vec[4];
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = s_vec[2]*alpha;
        s_vecX[2] = -s_vec[1]*alpha;
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[5]*alpha;
        s_vecX[5] = -s_vec[4]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[1] += s_vec[2]*alpha;
        s_vecX[2] += -s_vec[1]*alpha;
        s_vecX[4] += s_vec[5]*alpha;
        s_vecX[5] += -s_vec[4]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1(T *s_vecX, const T *s_vec) {
        s_vecX[0] = -s_vec[2];
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = s_vec[0];
        s_vecX[3] = -s_vec[5];
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[3];
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1_peq(T *s_vecX, const T *s_vec) {
        s_vecX[0] += -s_vec[2];
        s_vecX[2] += s_vec[0];
        s_vecX[3] += -s_vec[5];
        s_vecX[5] += s_vec[3];
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = -s_vec[2]*alpha;
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = s_vec[0]*alpha;
        s_vecX[3] = -s_vec[5]*alpha;
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[3]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] += -s_vec[2]*alpha;
        s_vecX[2] += s_vec[0]*alpha;
        s_vecX[3] += -s_vec[5]*alpha;
        s_vecX[5] += s_vec[3]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2(T *s_vecX, const T *s_vec) {
        s_vecX[0] = s_vec[1];
        s_vecX[1] = -s_vec[0];
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[4];
        s_vecX[4] = -s_vec[3];
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2_peq(T *s_vecX, const T *s_vec) {
        s_vecX[0] += s_vec[1];
        s_vecX[1] += -s_vec[0];
        s_vecX[3] += s_vec[4];
        s_vecX[4] += -s_vec[3];
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = s_vec[1]*alpha;
        s_vecX[1] = -s_vec[0]*alpha;
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[4]*alpha;
        s_vecX[4] = -s_vec[3]*alpha;
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] += s_vec[1]*alpha;
        s_vecX[1] += -s_vec[0]*alpha;
        s_vecX[3] += s_vec[4]*alpha;
        s_vecX[4] += -s_vec[3]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[2];
        s_vecX[5] = -s_vec[1];
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3_peq(T *s_vecX, const T *s_vec) {
        s_vecX[4] += s_vec[2];
        s_vecX[5] += -s_vec[1];
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[2]*alpha;
        s_vecX[5] = -s_vec[1]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[4] += s_vec[2]*alpha;
        s_vecX[5] += -s_vec[1]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = -s_vec[2];
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[0];
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4_peq(T *s_vecX, const T *s_vec) {
        s_vecX[3] += -s_vec[2];
        s_vecX[5] += s_vec[0];
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = -s_vec[2]*alpha;
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[0]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[3] += -s_vec[2]*alpha;
        s_vecX[5] += s_vec[0]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[1];
        s_vecX[4] = -s_vec[0];
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5_peq(T *s_vecX, const T *s_vec) {
        s_vecX[3] += s_vec[1];
        s_vecX[4] += -s_vec[0];
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[1]*alpha;
        s_vecX[4] = -s_vec[0]*alpha;
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[3] += s_vec[1]*alpha;
        s_vecX[4] += -s_vec[0]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0<T>(s_vecX, s_vec); break;
            case 1: mx1<T>(s_vecX, s_vec); break;
            case 2: mx2<T>(s_vecX, s_vec); break;
            case 3: mx3<T>(s_vecX, s_vec); break;
            case 4: mx4<T>(s_vecX, s_vec); break;
            case 5: mx5<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX_peq(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq<T>(s_vecX, s_vec); break;
            case 1: mx1_peq<T>(s_vecX, s_vec); break;
            case 2: mx2_peq<T>(s_vecX, s_vec); break;
            case 3: mx3_peq<T>(s_vecX, s_vec); break;
            case 4: mx4_peq<T>(s_vecX, s_vec); break;
            case 5: mx5_peq<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_peq_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_peq_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_matX is the destination matrix
     * @param s_vecX is the source vector
     */
    template <typename T>
    __device__
    void fx(T *s_matX, const T *s_vecX) {
        s_matX[6*0 + 0] = static_cast<T>(0);
        s_matX[6*0 + 1] = s_vecX[2];
        s_matX[6*0 + 2] = -s_vecX[1];
        s_matX[6*0 + 3] = static_cast<T>(0);
        s_matX[6*0 + 4] = static_cast<T>(0);
        s_matX[6*0 + 5] = static_cast<T>(0);
        s_matX[6*1 + 0] = -s_vecX[2];
        s_matX[6*1 + 1] = static_cast<T>(0);
        s_matX[6*1 + 2] = s_vecX[0];
        s_matX[6*1 + 3] = static_cast<T>(0);
        s_matX[6*1 + 4] = static_cast<T>(0);
        s_matX[6*1 + 5] = static_cast<T>(0);
        s_matX[6*2 + 0] = s_vecX[1];
        s_matX[6*2 + 1] = -s_vecX[0];
        s_matX[6*2 + 2] = static_cast<T>(0);
        s_matX[6*2 + 3] = static_cast<T>(0);
        s_matX[6*2 + 4] = static_cast<T>(0);
        s_matX[6*2 + 5] = static_cast<T>(0);
        s_matX[6*3 + 0] = static_cast<T>(0);
        s_matX[6*3 + 1] = s_vecX[5];
        s_matX[6*3 + 2] = -s_vecX[4];
        s_matX[6*3 + 3] = static_cast<T>(0);
        s_matX[6*3 + 4] = s_vecX[2];
        s_matX[6*3 + 5] = -s_vecX[1];
        s_matX[6*4 + 0] = -s_vecX[5];
        s_matX[6*4 + 1] = static_cast<T>(0);
        s_matX[6*4 + 2] = s_vecX[3];
        s_matX[6*4 + 3] = -s_vecX[2];
        s_matX[6*4 + 4] = static_cast<T>(0);
        s_matX[6*4 + 5] = s_vecX[0];
        s_matX[6*5 + 0] = s_vecX[4];
        s_matX[6*5 + 1] = -s_vecX[3];
        s_matX[6*5 + 2] = static_cast<T>(0);
        s_matX[6*5 + 3] = s_vecX[1];
        s_matX[6*5 + 4] = -s_vecX[0];
        s_matX[6*5 + 5] = static_cast<T>(0);
    }

    /**
     * Generates the motion vector cross product matrix for a pre-zeroed destination
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *   Assumes destination is zeroed
     *
     * @param s_matX is the destination matrix
     * @param s_vecX is the source vector
     */
    template <typename T>
    __device__
    void fx_zeroed(T *s_matX, const T *s_vecX) {
        s_matX[6*0 + 1] = s_vecX[2];
        s_matX[6*0 + 2] = -s_vecX[1];
        s_matX[6*1 + 0] = -s_vecX[2];
        s_matX[6*1 + 2] = s_vecX[0];
        s_matX[6*2 + 0] = s_vecX[1];
        s_matX[6*2 + 1] = -s_vecX[0];
        s_matX[6*3 + 1] = s_vecX[5];
        s_matX[6*3 + 2] = -s_vecX[4];
        s_matX[6*3 + 4] = s_vecX[2];
        s_matX[6*3 + 5] = -s_vecX[1];
        s_matX[6*4 + 0] = -s_vecX[5];
        s_matX[6*4 + 2] = s_vecX[3];
        s_matX[6*4 + 3] = -s_vecX[2];
        s_matX[6*4 + 5] = s_vecX[0];
        s_matX[6*5 + 0] = s_vecX[4];
        s_matX[6*5 + 1] = -s_vecX[3];
        s_matX[6*5 + 3] = s_vecX[1];
        s_matX[6*5 + 4] = -s_vecX[0];
    }

    /**
     * Generates the motion vector cross product matrix and multiples by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        s_result[0] = -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];
        s_result[1] =  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];
        s_result[2] = -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];
        s_result[3] =                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];
        s_result[4] =                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];
        s_result[5] =                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];
    }

    /**
     * Adds the motion vector cross product matrix multiplied by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v_peq(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        s_result[0] += -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];
        s_result[1] +=  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];
        s_result[2] += -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];
        s_result[3] +=                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];
        s_result[4] +=                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];
        s_result[5] +=                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];
    }
    template <typename T>
    __device__
    void vcross(T *dest, T *v){
        dest[0] = static_cast<T>(0);
        dest[1] = v[2];
        dest[2] = -1*v[1];
        dest[3] = static_cast<T>(0);
        dest[4] = v[5];
        dest[5] = -1*v[4];
        dest[6] = -1*v[2];
        dest[7] = static_cast<T>(0);
        dest[8] = v[0];
        dest[9] = -1*v[5];
        dest[10] = static_cast<T>(0);
        dest[11] = v[3];
        dest[12] = v[1];
        dest[13] = -1*v[0];
        dest[14] = static_cast<T>(0);
        dest[15] = v[4];
        dest[16] = -1*v[3];
        dest[17] = static_cast<T>(0);
        dest[18] = static_cast<T>(0);
        dest[19] = static_cast<T>(0);
        dest[20] = static_cast<T>(0);
        dest[21] = static_cast<T>(0);
        dest[22] = v[2];
        dest[23] = -1*v[1];
        dest[24] = static_cast<T>(0);
        dest[25] = static_cast<T>(0);
        dest[26] = static_cast<T>(0);
        dest[27] = -1*v[2];
        dest[28] = static_cast<T>(0);
        dest[29] = v[0];
        dest[30] = static_cast<T>(0);
        dest[31] = static_cast<T>(0);
        dest[32] = static_cast<T>(0);
        dest[33] = v[1];
        dest[34] = -1*v[0];
        dest[35] = static_cast<T>(0);
    }
    /**
     * Compute the inverse force cross product matrix of a 6-vector, v Returns the entry at the index.
     *
     * Notes:
     *   ICRF is the operation defined such that v crf f = f icrf v
     *
     * @param index is the index of the result matirx to compute
     * @param v is the 6-vector to take the cross product matrix of
     */
    template <typename T>
    __device__
    T icrf(int index, T *v) {
        T result;
        if (index == 0) result = static_cast<T>(0);
        if (index == 1) result = v[2];
        if (index == 2) result = -v[1];
        if (index == 3) result = static_cast<T>(0);
        if (index == 4) result = v[5];
        if (index == 5) result = -v[4];
        if (index == 6) result = -v[2];
        if (index == 7) result = static_cast<T>(0);
        if (index == 8) result = v[0];
        if (index == 9) result = -v[5];
        if (index == 10) result = static_cast<T>(0);
        if (index == 11) result = v[3];
        if (index == 12) result = v[1];
        if (index == 13) result = -v[0];
        if (index == 14) result = static_cast<T>(0);
        if (index == 15) result = v[4];
        if (index == 16) result = -v[3];
        if (index == 17) result = static_cast<T>(0);if (index == 18) result = static_cast<T>(0);
        if (index == 19) result = v[5];
        if (index == 20) result = -v[4];
        if (index == 21) result = static_cast<T>(0);
        if (index == 22) result = static_cast<T>(0);
        if (index == 23) result = static_cast<T>(0);
        if (index == 24) result = -v[5];
        if (index == 25) result = static_cast<T>(0);
        if (index == 26) result = v[3];
        if (index == 27) result = static_cast<T>(0);
        if (index == 28) result = static_cast<T>(0);
        if (index == 29) result = static_cast<T>(0);
        if (index == 30) result = v[4];
        if (index == 31) result = -v[3];
        if (index == 32) result = static_cast<T>(0);
        if (index == 33) result = static_cast<T>(0);
        if (index == 34) result = static_cast<T>(0);
        if (index == 35) result = static_cast<T>(0);
        return -result;
    }

    /**
     * Compute the motion cross product matrix of a 6-vector, v Returns the entry at the index.
     *
     * Notes:
     *   The force cross product matrix is just the negative transpose of this matrix
     *
     * @param index is the index of the result matirx to compute
     * @param v is the 6-vector to take the cross product matrix of
     */
    template <typename T>
    __device__
    T crm(int index, T *v) {
        T result;
        if (index == 0) result = static_cast<T>(0);
        if (index == 1) result = v[2];
        if (index == 2) result = -v[1];
        if (index == 3) result = static_cast<T>(0);
        if (index == 4) result = v[5];
        if (index == 5) result = -v[4];
        if (index == 6) result = -v[2];
        if (index == 7) result = static_cast<T>(0);
        if (index == 8) result = v[0];
        if (index == 9) result = -v[5];
        if (index == 10) result = static_cast<T>(0);
        if (index == 11) result = v[3];
        if (index == 12) result = v[1];
        if (index == 13) result = -v[0];
        if (index == 14) result = static_cast<T>(0);
        if (index == 15) result = v[4];
        if (index == 16) result = -v[3];
        if (index == 17) result = static_cast<T>(0);if (index == 18) result = static_cast<T>(0);
        if (index == 19) result = static_cast<T>(0);
        if (index == 20) result = static_cast<T>(0);
        if (index == 21) result = static_cast<T>(0);
        if (index == 22) result = v[2];
        if (index == 23) result = -v[1];
        if (index == 24) result = static_cast<T>(0);
        if (index == 25) result = static_cast<T>(0);
        if (index == 26) result = static_cast<T>(0);
        if (index == 27) result = -v[2];
        if (index == 28) result = static_cast<T>(0);
        if (index == 29) result = v[0];
        if (index == 30) result = static_cast<T>(0);
        if (index == 31) result = static_cast<T>(0);
        if (index == 32) result = static_cast<T>(0);
        if (index == 33) result = v[1];
        if (index == 34) result = -v[0];
        if (index == 35) result = static_cast<T>(0);
        return result;
    }

    /**
     * Compute the motion cross product multiplication of a 6-vector, v_crm, with a second 6-vector v
     *
     * @param index is the index of the result vector to compute
     * @param v_crm is the 6-vector to take the cross product matrix of
     * @param v is the 6-vector to multiply with v_crm
     */
    template <typename T>
    __device__
    T crm_mul(int index, T *v_crm, T *v) {
        T result;
        if (index == 0) result = -v_crm[2] * v[1] + v_crm[1] * v[2];
        if (index == 1) result = v_crm[2] * v[0] - v_crm[0] * v[2];
        if (index == 2) result = -v_crm[1] * v[0] + v_crm[0] * v[1];
        if (index == 3) result = -v_crm[5] * v[1] + v_crm[4] * v[2] - v_crm[2] * v[4] + v_crm[1] * v[5];
        if (index == 4) result = v_crm[5] * v[0] - v_crm[3] * v[2] + v_crm[2] * v[3] - v_crm[0] * v[5];
        if (index == 5) result = -v_crm[4] * v[0] + v_crm[3] * v[1] - v_crm[1] * v[3] + v_crm[0] * v[4];
        return result;
    }

    /**
     * Compute the inverse of a matrix
     *
     * Notes:
     *   Uses gaussian elimination
     *
     * @param dimA is number of rows in A
     * @param A is a pointer to the original invertible matrix. It is turned into an identity matrix
     * @param Ainv is a pointer to an identity matrix that will be transformed into the inverse of A
     * @param s_temp is a pointer to temporary memory of size 4*dimA
     */
    template <typename T>
    __device__
    void invert_matrix(uint32_t dimA, T *A, T *Ainv, T *s_temp) {
        for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
            unsigned pivColOffset = pivRC*dimA;
            T pvInv = static_cast<T>(1)/A[pivRC + pivColOffset];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < dimA; ind += blockDim.x*blockDim.y){
                s_temp[ind] = static_cast<T>(A[pivRC * dimA + ind]);
                s_temp[ind+dimA] = static_cast<T>(Ainv[pivRC * dimA + ind]);
                s_temp[ind+dimA*2] = static_cast<T>(A[pivRC + dimA * ind]);
                s_temp[ind+dimA*3] = static_cast<T>(Ainv[pivRC + dimA * ind]);
            }
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < dimA*dimA; ind += blockDim.x*blockDim.y){
                unsigned row = ind % dimA, col = ind / dimA;
                if (row == pivRC) {
                    A[row * dimA + col] *= pvInv;
                    Ainv[row * dimA + col] *= pvInv;
                }
                else {
                    T multiplier = s_temp[row+dimA*2] / s_temp[pivRC];
                    A[row * dimA + col] -= multiplier * s_temp[col];
                    Ainv[row * dimA + col] -= multiplier * s_temp[col+dimA];
                }
            }
            __syncthreads();
        }
    }

    /**
     * Matrix multiplication helper function of AB
     *
     * @param index - the index of the result vector
     * @param A - pointer to the first matrix
     * @param B - pointer to the second matrix
     * @param dest - pointer to the destination matrix
     * @param num - 36 or 6 depending on the indexing scheme
     * @param t - true => multiply with the transpose of B
     */
    template <typename T>
    __device__
    void matmul(int index, T *A, T *B, T *dest, int num, bool t) {
        int cur = 36*((index/num)%NUM_JOINTS);
        T *vec1 = &B[cur + (t*5+1)*(index%6)];
        T *vec2 = &A[6*(index/6)];
        dest[index] = dot_prod<T,6, 6, 1>(vec1, vec2);
    }

    /**
     * Matrix multiplication helper function where one of the matrices is tranposed.
     *
     * @param index - the index of the result vector
     * @param A - pointer to the first 6x6 matrix
     * @param B - pointer to the second 6x6 matrix
     * @param dest - pointer to the destination matrix
     * @param char trans_mat - a for A^TB, b for AB^T
     */
    template <typename T>
    __device__
    void matmul_trans(int index, T *A, T *B, T *dest, char trans_mat) {
        T *vec1;
        T *vec2;
        if (trans_mat == 'a'){
            vec1 = &A[6*(index%6)];
            vec2 = &B[6*(index/6)];
            dest[index] = dot_prod<T,6,1,1>(vec1, vec2);
        }
        if (trans_mat == 'b'){
            vec1 = &A[index%6];
            vec2 = &B[index/6];
            dest[index] = dot_prod<T,6,6,6>(vec1, vec2);
        }
    }

    /**
     * Compute the outer product between two vectors: dest = ab^T
     *
     * Notes:
     *   Function assumes it is called by a single thread.
     *
     * @param a - first vector
     * @param b - second vector
     * @param dest - destination matrix
     * @param aLength - length of a
     * @param bLength - length of b
     * @param idx - index of resulting matrix to be computed by this thread
     */
    template <typename T>
    __device__
    void outerProduct(T *a, T *b, T *dest, int aLength, int bLength, int idx) {
        int row = idx / bLength;
        int col = idx % bLength;
        if (row < aLength && col < bLength) dest[col * aLength + row] = a[row] * b[col];
    }

    //
    // Topology Helpers not needed!
    //
    template <typename T>
    __host__
    int *init_topology_helpers(){return nullptr;}
    /**
     * Initializes the Xmats and Imats in GPU memory
     *
     * Notes:
     *   Memory order is X[0...N], I[0...N], Xhom[0...N]
     *
     * @return A pointer to the XI memory in the GPU
     */
    template <typename T>
    __host__
    T* init_XImats() {
        T *h_XImats = (T *)malloc(840*sizeof(T));
        // X[0]
        h_XImats[0] = static_cast<T>(0);
        h_XImats[1] = static_cast<T>(0);
        h_XImats[2] = static_cast<T>(0);
        h_XImats[3] = static_cast<T>(0);
        h_XImats[4] = static_cast<T>(0);
        h_XImats[5] = static_cast<T>(0);
        h_XImats[6] = static_cast<T>(0);
        h_XImats[7] = static_cast<T>(0);
        h_XImats[8] = static_cast<T>(0);
        h_XImats[9] = static_cast<T>(0);
        h_XImats[10] = static_cast<T>(0);
        h_XImats[11] = static_cast<T>(0);
        h_XImats[12] = static_cast<T>(0);
        h_XImats[13] = static_cast<T>(0);
        h_XImats[14] = static_cast<T>(1.00000000000000);
        h_XImats[15] = static_cast<T>(0);
        h_XImats[16] = static_cast<T>(0);
        h_XImats[17] = static_cast<T>(0);
        h_XImats[18] = static_cast<T>(0);
        h_XImats[19] = static_cast<T>(0);
        h_XImats[20] = static_cast<T>(0);
        h_XImats[21] = static_cast<T>(0);
        h_XImats[22] = static_cast<T>(0);
        h_XImats[23] = static_cast<T>(0);
        h_XImats[24] = static_cast<T>(0);
        h_XImats[25] = static_cast<T>(0);
        h_XImats[26] = static_cast<T>(0);
        h_XImats[27] = static_cast<T>(0);
        h_XImats[28] = static_cast<T>(0);
        h_XImats[29] = static_cast<T>(0);
        h_XImats[30] = static_cast<T>(0);
        h_XImats[31] = static_cast<T>(0);
        h_XImats[32] = static_cast<T>(0);
        h_XImats[33] = static_cast<T>(0);
        h_XImats[34] = static_cast<T>(0);
        h_XImats[35] = static_cast<T>(1.00000000000000);
        // X[1]
        h_XImats[36] = static_cast<T>(0);
        h_XImats[37] = static_cast<T>(0);
        h_XImats[38] = static_cast<T>(0);
        h_XImats[39] = static_cast<T>(0);
        h_XImats[40] = static_cast<T>(0);
        h_XImats[41] = static_cast<T>(-0.202500000000000);
        h_XImats[42] = static_cast<T>(0);
        h_XImats[43] = static_cast<T>(0);
        h_XImats[44] = static_cast<T>(1.00000000000000);
        h_XImats[45] = static_cast<T>(0);
        h_XImats[46] = static_cast<T>(0);
        h_XImats[47] = static_cast<T>(0);
        h_XImats[48] = static_cast<T>(0);
        h_XImats[49] = static_cast<T>(0);
        h_XImats[50] = static_cast<T>(0);
        h_XImats[51] = static_cast<T>(0);
        h_XImats[52] = static_cast<T>(0);
        h_XImats[53] = static_cast<T>(0);
        h_XImats[54] = static_cast<T>(0);
        h_XImats[55] = static_cast<T>(0);
        h_XImats[56] = static_cast<T>(0);
        h_XImats[57] = static_cast<T>(0);
        h_XImats[58] = static_cast<T>(0);
        h_XImats[59] = static_cast<T>(0);
        h_XImats[60] = static_cast<T>(0);
        h_XImats[61] = static_cast<T>(0);
        h_XImats[62] = static_cast<T>(0);
        h_XImats[63] = static_cast<T>(0);
        h_XImats[64] = static_cast<T>(0);
        h_XImats[65] = static_cast<T>(1.00000000000000);
        h_XImats[66] = static_cast<T>(0);
        h_XImats[67] = static_cast<T>(0);
        h_XImats[68] = static_cast<T>(0);
        h_XImats[69] = static_cast<T>(0);
        h_XImats[70] = static_cast<T>(0);
        h_XImats[71] = static_cast<T>(0);
        // X[2]
        h_XImats[72] = static_cast<T>(0);
        h_XImats[73] = static_cast<T>(0);
        h_XImats[74] = static_cast<T>(1.00000000000000);
        h_XImats[75] = static_cast<T>(0);
        h_XImats[76] = static_cast<T>(0);
        h_XImats[77] = static_cast<T>(0);
        h_XImats[78] = static_cast<T>(0);
        h_XImats[79] = static_cast<T>(0);
        h_XImats[80] = static_cast<T>(0);
        h_XImats[81] = static_cast<T>(0);
        h_XImats[82] = static_cast<T>(0);
        h_XImats[83] = static_cast<T>(0);
        h_XImats[84] = static_cast<T>(0);
        h_XImats[85] = static_cast<T>(0);
        h_XImats[86] = static_cast<T>(0);
        h_XImats[87] = static_cast<T>(0);
        h_XImats[88] = static_cast<T>(0);
        h_XImats[89] = static_cast<T>(0);
        h_XImats[90] = static_cast<T>(0);
        h_XImats[91] = static_cast<T>(0);
        h_XImats[92] = static_cast<T>(0);
        h_XImats[93] = static_cast<T>(0);
        h_XImats[94] = static_cast<T>(0);
        h_XImats[95] = static_cast<T>(1.00000000000000);
        h_XImats[96] = static_cast<T>(0);
        h_XImats[97] = static_cast<T>(0);
        h_XImats[98] = static_cast<T>(0);
        h_XImats[99] = static_cast<T>(0);
        h_XImats[100] = static_cast<T>(0);
        h_XImats[101] = static_cast<T>(0);
        h_XImats[102] = static_cast<T>(0);
        h_XImats[103] = static_cast<T>(0);
        h_XImats[104] = static_cast<T>(0);
        h_XImats[105] = static_cast<T>(0);
        h_XImats[106] = static_cast<T>(0);
        h_XImats[107] = static_cast<T>(0);
        // X[3]
        h_XImats[108] = static_cast<T>(0);
        h_XImats[109] = static_cast<T>(0);
        h_XImats[110] = static_cast<T>(0);
        h_XImats[111] = static_cast<T>(0);
        h_XImats[112] = static_cast<T>(0);
        h_XImats[113] = static_cast<T>(0.215500000000000);
        h_XImats[114] = static_cast<T>(0);
        h_XImats[115] = static_cast<T>(0);
        h_XImats[116] = static_cast<T>(-1.00000000000000);
        h_XImats[117] = static_cast<T>(0);
        h_XImats[118] = static_cast<T>(0);
        h_XImats[119] = static_cast<T>(0);
        h_XImats[120] = static_cast<T>(0);
        h_XImats[121] = static_cast<T>(0);
        h_XImats[122] = static_cast<T>(0);
        h_XImats[123] = static_cast<T>(0);
        h_XImats[124] = static_cast<T>(0);
        h_XImats[125] = static_cast<T>(0);
        h_XImats[126] = static_cast<T>(0);
        h_XImats[127] = static_cast<T>(0);
        h_XImats[128] = static_cast<T>(0);
        h_XImats[129] = static_cast<T>(0);
        h_XImats[130] = static_cast<T>(0);
        h_XImats[131] = static_cast<T>(0);
        h_XImats[132] = static_cast<T>(0);
        h_XImats[133] = static_cast<T>(0);
        h_XImats[134] = static_cast<T>(0);
        h_XImats[135] = static_cast<T>(0);
        h_XImats[136] = static_cast<T>(0);
        h_XImats[137] = static_cast<T>(-1.00000000000000);
        h_XImats[138] = static_cast<T>(0);
        h_XImats[139] = static_cast<T>(0);
        h_XImats[140] = static_cast<T>(0);
        h_XImats[141] = static_cast<T>(0);
        h_XImats[142] = static_cast<T>(0);
        h_XImats[143] = static_cast<T>(0);
        // X[4]
        h_XImats[144] = static_cast<T>(0);
        h_XImats[145] = static_cast<T>(0);
        h_XImats[146] = static_cast<T>(0);
        h_XImats[147] = static_cast<T>(0);
        h_XImats[148] = static_cast<T>(0);
        h_XImats[149] = static_cast<T>(0);
        h_XImats[150] = static_cast<T>(0);
        h_XImats[151] = static_cast<T>(0);
        h_XImats[152] = static_cast<T>(1.00000000000000);
        h_XImats[153] = static_cast<T>(0);
        h_XImats[154] = static_cast<T>(0);
        h_XImats[155] = static_cast<T>(0);
        h_XImats[156] = static_cast<T>(0);
        h_XImats[157] = static_cast<T>(0);
        h_XImats[158] = static_cast<T>(0);
        h_XImats[159] = static_cast<T>(0);
        h_XImats[160] = static_cast<T>(0);
        h_XImats[161] = static_cast<T>(0);
        h_XImats[162] = static_cast<T>(0);
        h_XImats[163] = static_cast<T>(0);
        h_XImats[164] = static_cast<T>(0);
        h_XImats[165] = static_cast<T>(0);
        h_XImats[166] = static_cast<T>(0);
        h_XImats[167] = static_cast<T>(0);
        h_XImats[168] = static_cast<T>(0);
        h_XImats[169] = static_cast<T>(0);
        h_XImats[170] = static_cast<T>(0);
        h_XImats[171] = static_cast<T>(0);
        h_XImats[172] = static_cast<T>(0);
        h_XImats[173] = static_cast<T>(1.00000000000000);
        h_XImats[174] = static_cast<T>(0);
        h_XImats[175] = static_cast<T>(0);
        h_XImats[176] = static_cast<T>(0);
        h_XImats[177] = static_cast<T>(0);
        h_XImats[178] = static_cast<T>(0);
        h_XImats[179] = static_cast<T>(0);
        // X[5]
        h_XImats[180] = static_cast<T>(0);
        h_XImats[181] = static_cast<T>(0);
        h_XImats[182] = static_cast<T>(0);
        h_XImats[183] = static_cast<T>(0);
        h_XImats[184] = static_cast<T>(0);
        h_XImats[185] = static_cast<T>(-0.215499997286605);
        h_XImats[186] = static_cast<T>(0);
        h_XImats[187] = static_cast<T>(0);
        h_XImats[188] = static_cast<T>(1.00000000000000);
        h_XImats[189] = static_cast<T>(0);
        h_XImats[190] = static_cast<T>(0);
        h_XImats[191] = static_cast<T>(0);
        h_XImats[192] = static_cast<T>(0);
        h_XImats[193] = static_cast<T>(0);
        h_XImats[194] = static_cast<T>(0);
        h_XImats[195] = static_cast<T>(0);
        h_XImats[196] = static_cast<T>(0);
        h_XImats[197] = static_cast<T>(0);
        h_XImats[198] = static_cast<T>(0);
        h_XImats[199] = static_cast<T>(0);
        h_XImats[200] = static_cast<T>(0);
        h_XImats[201] = static_cast<T>(0);
        h_XImats[202] = static_cast<T>(0);
        h_XImats[203] = static_cast<T>(0);
        h_XImats[204] = static_cast<T>(0);
        h_XImats[205] = static_cast<T>(0);
        h_XImats[206] = static_cast<T>(0);
        h_XImats[207] = static_cast<T>(0);
        h_XImats[208] = static_cast<T>(0);
        h_XImats[209] = static_cast<T>(1.00000000000000);
        h_XImats[210] = static_cast<T>(0);
        h_XImats[211] = static_cast<T>(0);
        h_XImats[212] = static_cast<T>(0);
        h_XImats[213] = static_cast<T>(0);
        h_XImats[214] = static_cast<T>(0);
        h_XImats[215] = static_cast<T>(0);
        // X[6]
        h_XImats[216] = static_cast<T>(0);
        h_XImats[217] = static_cast<T>(0);
        h_XImats[218] = static_cast<T>(1.00000000000000);
        h_XImats[219] = static_cast<T>(0);
        h_XImats[220] = static_cast<T>(0);
        h_XImats[221] = static_cast<T>(0);
        h_XImats[222] = static_cast<T>(0);
        h_XImats[223] = static_cast<T>(0);
        h_XImats[224] = static_cast<T>(0);
        h_XImats[225] = static_cast<T>(0);
        h_XImats[226] = static_cast<T>(0);
        h_XImats[227] = static_cast<T>(0.0607000000000000);
        h_XImats[228] = static_cast<T>(0);
        h_XImats[229] = static_cast<T>(0);
        h_XImats[230] = static_cast<T>(0);
        h_XImats[231] = static_cast<T>(0);
        h_XImats[232] = static_cast<T>(0);
        h_XImats[233] = static_cast<T>(0);
        h_XImats[234] = static_cast<T>(0);
        h_XImats[235] = static_cast<T>(0);
        h_XImats[236] = static_cast<T>(0);
        h_XImats[237] = static_cast<T>(0);
        h_XImats[238] = static_cast<T>(0);
        h_XImats[239] = static_cast<T>(1.00000000000000);
        h_XImats[240] = static_cast<T>(0);
        h_XImats[241] = static_cast<T>(0);
        h_XImats[242] = static_cast<T>(0);
        h_XImats[243] = static_cast<T>(0);
        h_XImats[244] = static_cast<T>(0);
        h_XImats[245] = static_cast<T>(0);
        h_XImats[246] = static_cast<T>(0);
        h_XImats[247] = static_cast<T>(0);
        h_XImats[248] = static_cast<T>(0);
        h_XImats[249] = static_cast<T>(0);
        h_XImats[250] = static_cast<T>(0);
        h_XImats[251] = static_cast<T>(0);
        // I[0]
        h_XImats[252] = static_cast<T>(0.00455);
        h_XImats[253] = static_cast<T>(0.0);
        h_XImats[254] = static_cast<T>(0.0);
        h_XImats[255] = static_cast<T>(0.0);
        h_XImats[256] = static_cast<T>(0.0);
        h_XImats[257] = static_cast<T>(0.0);
        h_XImats[258] = static_cast<T>(0.0);
        h_XImats[259] = static_cast<T>(0.00454);
        h_XImats[260] = static_cast<T>(-1e-05);
        h_XImats[261] = static_cast<T>(0.0);
        h_XImats[262] = static_cast<T>(0.0);
        h_XImats[263] = static_cast<T>(0.0);
        h_XImats[264] = static_cast<T>(0.0);
        h_XImats[265] = static_cast<T>(-1e-05);
        h_XImats[266] = static_cast<T>(0.00029);
        h_XImats[267] = static_cast<T>(0.0);
        h_XImats[268] = static_cast<T>(0.0);
        h_XImats[269] = static_cast<T>(0.0);
        h_XImats[270] = static_cast<T>(0.0);
        h_XImats[271] = static_cast<T>(0.0);
        h_XImats[272] = static_cast<T>(0.0);
        h_XImats[273] = static_cast<T>(3.94781);
        h_XImats[274] = static_cast<T>(0.0);
        h_XImats[275] = static_cast<T>(0.0);
        h_XImats[276] = static_cast<T>(0.0);
        h_XImats[277] = static_cast<T>(0.0);
        h_XImats[278] = static_cast<T>(0.0);
        h_XImats[279] = static_cast<T>(0.0);
        h_XImats[280] = static_cast<T>(3.94781);
        h_XImats[281] = static_cast<T>(0.0);
        h_XImats[282] = static_cast<T>(0.0);
        h_XImats[283] = static_cast<T>(0.0);
        h_XImats[284] = static_cast<T>(0.0);
        h_XImats[285] = static_cast<T>(0.0);
        h_XImats[286] = static_cast<T>(0.0);
        h_XImats[287] = static_cast<T>(3.94781);
        // I[1]
        h_XImats[288] = static_cast<T>(0.02393692375);
        h_XImats[289] = static_cast<T>(-7.969867499999998e-05);
        h_XImats[290] = static_cast<T>(-5.6734649999999994e-05);
        h_XImats[291] = static_cast<T>(0.0);
        h_XImats[292] = static_cast<T>(-0.1891155);
        h_XImats[293] = static_cast<T>(0.26566225);
        h_XImats[294] = static_cast<T>(-7.9698675e-05);
        h_XImats[295] = static_cast<T>(0.0080432562475);
        h_XImats[296] = static_cast<T>(-0.0111578145);
        h_XImats[297] = static_cast<T>(0.1891155);
        h_XImats[298] = static_cast<T>(0.0);
        h_XImats[299] = static_cast<T>(-0.0013508249999999997);
        h_XImats[300] = static_cast<T>(-5.6734649999999994e-05);
        h_XImats[301] = static_cast<T>(-0.011157814499999998);
        h_XImats[302] = static_cast<T>(0.0160944779975);
        h_XImats[303] = static_cast<T>(-0.26566225);
        h_XImats[304] = static_cast<T>(0.0013508249999999997);
        h_XImats[305] = static_cast<T>(0.0);
        h_XImats[306] = static_cast<T>(0.0);
        h_XImats[307] = static_cast<T>(0.1891155);
        h_XImats[308] = static_cast<T>(-0.26566225);
        h_XImats[309] = static_cast<T>(4.50275);
        h_XImats[310] = static_cast<T>(0.0);
        h_XImats[311] = static_cast<T>(0.0);
        h_XImats[312] = static_cast<T>(-0.1891155);
        h_XImats[313] = static_cast<T>(0.0);
        h_XImats[314] = static_cast<T>(0.0013508249999999997);
        h_XImats[315] = static_cast<T>(0.0);
        h_XImats[316] = static_cast<T>(4.50275);
        h_XImats[317] = static_cast<T>(0.0);
        h_XImats[318] = static_cast<T>(0.26566225);
        h_XImats[319] = static_cast<T>(-0.0013508249999999997);
        h_XImats[320] = static_cast<T>(0.0);
        h_XImats[321] = static_cast<T>(0.0);
        h_XImats[322] = static_cast<T>(0.0);
        h_XImats[323] = static_cast<T>(4.50275);
        // I[2]
        h_XImats[324] = static_cast<T>(0.045932560000000004);
        h_XImats[325] = static_cast<T>(-5e-05);
        h_XImats[326] = static_cast<T>(7e-05);
        h_XImats[327] = static_cast<T>(0.0);
        h_XImats[328] = static_cast<T>(-0.319176);
        h_XImats[329] = static_cast<T>(0.073656);
        h_XImats[330] = static_cast<T>(-5e-05);
        h_XImats[331] = static_cast<T>(0.04368288);
        h_XImats[332] = static_cast<T>(-0.00950528);
        h_XImats[333] = static_cast<T>(0.319176);
        h_XImats[334] = static_cast<T>(0.0);
        h_XImats[335] = static_cast<T>(0.0);
        h_XImats[336] = static_cast<T>(7e-05);
        h_XImats[337] = static_cast<T>(-0.00950528);
        h_XImats[338] = static_cast<T>(0.00293968);
        h_XImats[339] = static_cast<T>(-0.073656);
        h_XImats[340] = static_cast<T>(0.0);
        h_XImats[341] = static_cast<T>(0.0);
        h_XImats[342] = static_cast<T>(0.0);
        h_XImats[343] = static_cast<T>(0.319176);
        h_XImats[344] = static_cast<T>(-0.073656);
        h_XImats[345] = static_cast<T>(2.4552);
        h_XImats[346] = static_cast<T>(0.0);
        h_XImats[347] = static_cast<T>(0.0);
        h_XImats[348] = static_cast<T>(-0.319176);
        h_XImats[349] = static_cast<T>(0.0);
        h_XImats[350] = static_cast<T>(0.0);
        h_XImats[351] = static_cast<T>(0.0);
        h_XImats[352] = static_cast<T>(2.4552);
        h_XImats[353] = static_cast<T>(0.0);
        h_XImats[354] = static_cast<T>(0.073656);
        h_XImats[355] = static_cast<T>(0.0);
        h_XImats[356] = static_cast<T>(0.0);
        h_XImats[357] = static_cast<T>(0.0);
        h_XImats[358] = static_cast<T>(0.0);
        h_XImats[359] = static_cast<T>(2.4552);
        // I[3]
        h_XImats[360] = static_cast<T>(0.053182199750000006);
        h_XImats[361] = static_cast<T>(0.00088);
        h_XImats[362] = static_cast<T>(-0.00112);
        h_XImats[363] = static_cast<T>(0.0);
        h_XImats[364] = static_cast<T>(-0.0887927);
        h_XImats[365] = static_cast<T>(0.17497385);
        h_XImats[366] = static_cast<T>(0.00088);
        h_XImats[367] = static_cast<T>(0.014458951800000001);
        h_XImats[368] = static_cast<T>(-0.0070591109);
        h_XImats[369] = static_cast<T>(0.0887927);
        h_XImats[370] = static_cast<T>(0.0);
        h_XImats[371] = static_cast<T>(0.0);
        h_XImats[372] = static_cast<T>(-0.00112);
        h_XImats[373] = static_cast<T>(-0.0070591109);
        h_XImats[374] = static_cast<T>(0.06130324795);
        h_XImats[375] = static_cast<T>(-0.17497385);
        h_XImats[376] = static_cast<T>(0.0);
        h_XImats[377] = static_cast<T>(0.0);
        h_XImats[378] = static_cast<T>(0.0);
        h_XImats[379] = static_cast<T>(0.0887927);
        h_XImats[380] = static_cast<T>(-0.17497385);
        h_XImats[381] = static_cast<T>(2.61155);
        h_XImats[382] = static_cast<T>(0.0);
        h_XImats[383] = static_cast<T>(0.0);
        h_XImats[384] = static_cast<T>(-0.0887927);
        h_XImats[385] = static_cast<T>(0.0);
        h_XImats[386] = static_cast<T>(0.0);
        h_XImats[387] = static_cast<T>(0.0);
        h_XImats[388] = static_cast<T>(2.61155);
        h_XImats[389] = static_cast<T>(0.0);
        h_XImats[390] = static_cast<T>(0.17497385);
        h_XImats[391] = static_cast<T>(0.0);
        h_XImats[392] = static_cast<T>(0.0);
        h_XImats[393] = static_cast<T>(0.0);
        h_XImats[394] = static_cast<T>(0.0);
        h_XImats[395] = static_cast<T>(2.61155);
        // I[4]
        h_XImats[396] = static_cast<T>(0.02396997);
        h_XImats[397] = static_cast<T>(-1.7161e-05);
        h_XImats[398] = static_cast<T>(-1.5916e-05);
        h_XImats[399] = static_cast<T>(0.0);
        h_XImats[400] = static_cast<T>(-0.25916);
        h_XImats[401] = static_cast<T>(0.07161000000000001);
        h_XImats[402] = static_cast<T>(-1.7161e-05);
        h_XImats[403] = static_cast<T>(0.0225361941);
        h_XImats[404] = static_cast<T>(-0.00544236);
        h_XImats[405] = static_cast<T>(0.25916);
        h_XImats[406] = static_cast<T>(0.0);
        h_XImats[407] = static_cast<T>(-0.00034100000000000005);
        h_XImats[408] = static_cast<T>(-1.5916e-05);
        h_XImats[409] = static_cast<T>(-0.00544236);
        h_XImats[410] = static_cast<T>(0.0016238441000000002);
        h_XImats[411] = static_cast<T>(-0.07161000000000001);
        h_XImats[412] = static_cast<T>(0.00034100000000000005);
        h_XImats[413] = static_cast<T>(0.0);
        h_XImats[414] = static_cast<T>(0.0);
        h_XImats[415] = static_cast<T>(0.25916);
        h_XImats[416] = static_cast<T>(-0.07161000000000001);
        h_XImats[417] = static_cast<T>(3.41);
        h_XImats[418] = static_cast<T>(0.0);
        h_XImats[419] = static_cast<T>(0.0);
        h_XImats[420] = static_cast<T>(-0.25916);
        h_XImats[421] = static_cast<T>(0.0);
        h_XImats[422] = static_cast<T>(0.00034100000000000005);
        h_XImats[423] = static_cast<T>(0.0);
        h_XImats[424] = static_cast<T>(3.41);
        h_XImats[425] = static_cast<T>(0.0);
        h_XImats[426] = static_cast<T>(0.07161000000000001);
        h_XImats[427] = static_cast<T>(-0.00034100000000000005);
        h_XImats[428] = static_cast<T>(0.0);
        h_XImats[429] = static_cast<T>(0.0);
        h_XImats[430] = static_cast<T>(0.0);
        h_XImats[431] = static_cast<T>(3.41);
        // I[5]
        h_XImats[432] = static_cast<T>(0.000501761734);
        h_XImats[433] = static_cast<T>(-5e-05);
        h_XImats[434] = static_cast<T>(-3e-05);
        h_XImats[435] = static_cast<T>(0.0);
        h_XImats[436] = static_cast<T>(-0.00135518);
        h_XImats[437] = static_cast<T>(0.0020327699999999997);
        h_XImats[438] = static_cast<T>(-5e-05);
        h_XImats[439] = static_cast<T>(0.002810542072);
        h_XImats[440] = static_cast<T>(-4.0813108000000006e-05);
        h_XImats[441] = static_cast<T>(0.00135518);
        h_XImats[442] = static_cast<T>(0.0);
        h_XImats[443] = static_cast<T>(0.0);
        h_XImats[444] = static_cast<T>(-3e-05);
        h_XImats[445] = static_cast<T>(-4.0813108000000006e-05);
        h_XImats[446] = static_cast<T>(0.002321219662);
        h_XImats[447] = static_cast<T>(-0.0020327699999999997);
        h_XImats[448] = static_cast<T>(0.0);
        h_XImats[449] = static_cast<T>(0.0);
        h_XImats[450] = static_cast<T>(0.0);
        h_XImats[451] = static_cast<T>(0.00135518);
        h_XImats[452] = static_cast<T>(-0.0020327699999999997);
        h_XImats[453] = static_cast<T>(3.38795);
        h_XImats[454] = static_cast<T>(0.0);
        h_XImats[455] = static_cast<T>(0.0);
        h_XImats[456] = static_cast<T>(-0.00135518);
        h_XImats[457] = static_cast<T>(0.0);
        h_XImats[458] = static_cast<T>(0.0);
        h_XImats[459] = static_cast<T>(0.0);
        h_XImats[460] = static_cast<T>(3.38795);
        h_XImats[461] = static_cast<T>(0.0);
        h_XImats[462] = static_cast<T>(0.0020327699999999997);
        h_XImats[463] = static_cast<T>(0.0);
        h_XImats[464] = static_cast<T>(0.0);
        h_XImats[465] = static_cast<T>(0.0);
        h_XImats[466] = static_cast<T>(0.0);
        h_XImats[467] = static_cast<T>(3.38795);
        // I[6]
        h_XImats[468] = static_cast<T>(0.008218328);
        h_XImats[469] = static_cast<T>(0.00022);
        h_XImats[470] = static_cast<T>(-0.00029);
        h_XImats[471] = static_cast<T>(0.0);
        h_XImats[472] = static_cast<T>(-0.0093664);
        h_XImats[473] = static_cast<T>(0.0);
        h_XImats[474] = static_cast<T>(0.00022);
        h_XImats[475] = static_cast<T>(0.011098328);
        h_XImats[476] = static_cast<T>(-0.00029);
        h_XImats[477] = static_cast<T>(0.0093664);
        h_XImats[478] = static_cast<T>(0.0);
        h_XImats[479] = static_cast<T>(0.0);
        h_XImats[480] = static_cast<T>(-0.00029);
        h_XImats[481] = static_cast<T>(-0.00029);
        h_XImats[482] = static_cast<T>(0.0029754);
        h_XImats[483] = static_cast<T>(0.0);
        h_XImats[484] = static_cast<T>(0.0);
        h_XImats[485] = static_cast<T>(0.0);
        h_XImats[486] = static_cast<T>(0.0);
        h_XImats[487] = static_cast<T>(0.0093664);
        h_XImats[488] = static_cast<T>(0.0);
        h_XImats[489] = static_cast<T>(0.41132);
        h_XImats[490] = static_cast<T>(0.0);
        h_XImats[491] = static_cast<T>(0.0);
        h_XImats[492] = static_cast<T>(-0.0093664);
        h_XImats[493] = static_cast<T>(0.0);
        h_XImats[494] = static_cast<T>(0.0);
        h_XImats[495] = static_cast<T>(0.0);
        h_XImats[496] = static_cast<T>(0.41132);
        h_XImats[497] = static_cast<T>(0.0);
        h_XImats[498] = static_cast<T>(0.0);
        h_XImats[499] = static_cast<T>(0.0);
        h_XImats[500] = static_cast<T>(0.0);
        h_XImats[501] = static_cast<T>(0.0);
        h_XImats[502] = static_cast<T>(0.0);
        h_XImats[503] = static_cast<T>(0.41132);
        // Xhom[0]
        h_XImats[504] = static_cast<T>(0);
        h_XImats[505] = static_cast<T>(0);
        h_XImats[506] = static_cast<T>(0);
        h_XImats[507] = static_cast<T>(0);
        h_XImats[508] = static_cast<T>(0);
        h_XImats[509] = static_cast<T>(0);
        h_XImats[510] = static_cast<T>(0);
        h_XImats[511] = static_cast<T>(0);
        h_XImats[512] = static_cast<T>(0);
        h_XImats[513] = static_cast<T>(0);
        h_XImats[514] = static_cast<T>(1.00000000000000);
        h_XImats[515] = static_cast<T>(0);
        h_XImats[516] = static_cast<T>(0);
        h_XImats[517] = static_cast<T>(0);
        h_XImats[518] = static_cast<T>(0.157500000000000);
        h_XImats[519] = static_cast<T>(1.00000000000000);
        // Xhom[1]
        h_XImats[520] = static_cast<T>(0);
        h_XImats[521] = static_cast<T>(0);
        h_XImats[522] = static_cast<T>(0);
        h_XImats[523] = static_cast<T>(0);
        h_XImats[524] = static_cast<T>(0);
        h_XImats[525] = static_cast<T>(0);
        h_XImats[526] = static_cast<T>(0);
        h_XImats[527] = static_cast<T>(0);
        h_XImats[528] = static_cast<T>(0);
        h_XImats[529] = static_cast<T>(1.00000000000000);
        h_XImats[530] = static_cast<T>(0);
        h_XImats[531] = static_cast<T>(0);
        h_XImats[532] = static_cast<T>(0);
        h_XImats[533] = static_cast<T>(0);
        h_XImats[534] = static_cast<T>(0.202500000000000);
        h_XImats[535] = static_cast<T>(1.00000000000000);
        // Xhom[2]
        h_XImats[536] = static_cast<T>(0);
        h_XImats[537] = static_cast<T>(0);
        h_XImats[538] = static_cast<T>(0);
        h_XImats[539] = static_cast<T>(0);
        h_XImats[540] = static_cast<T>(0);
        h_XImats[541] = static_cast<T>(0);
        h_XImats[542] = static_cast<T>(0);
        h_XImats[543] = static_cast<T>(0);
        h_XImats[544] = static_cast<T>(1.00000000000000);
        h_XImats[545] = static_cast<T>(0);
        h_XImats[546] = static_cast<T>(0);
        h_XImats[547] = static_cast<T>(0);
        h_XImats[548] = static_cast<T>(0.204500000000000);
        h_XImats[549] = static_cast<T>(0);
        h_XImats[550] = static_cast<T>(0);
        h_XImats[551] = static_cast<T>(1.00000000000000);
        // Xhom[3]
        h_XImats[552] = static_cast<T>(0);
        h_XImats[553] = static_cast<T>(0);
        h_XImats[554] = static_cast<T>(0);
        h_XImats[555] = static_cast<T>(0);
        h_XImats[556] = static_cast<T>(0);
        h_XImats[557] = static_cast<T>(0);
        h_XImats[558] = static_cast<T>(0);
        h_XImats[559] = static_cast<T>(0);
        h_XImats[560] = static_cast<T>(0);
        h_XImats[561] = static_cast<T>(-1.00000000000000);
        h_XImats[562] = static_cast<T>(0);
        h_XImats[563] = static_cast<T>(0);
        h_XImats[564] = static_cast<T>(0);
        h_XImats[565] = static_cast<T>(0);
        h_XImats[566] = static_cast<T>(0.215500000000000);
        h_XImats[567] = static_cast<T>(1.00000000000000);
        // Xhom[4]
        h_XImats[568] = static_cast<T>(0);
        h_XImats[569] = static_cast<T>(0);
        h_XImats[570] = static_cast<T>(0);
        h_XImats[571] = static_cast<T>(0);
        h_XImats[572] = static_cast<T>(0);
        h_XImats[573] = static_cast<T>(0);
        h_XImats[574] = static_cast<T>(0);
        h_XImats[575] = static_cast<T>(0);
        h_XImats[576] = static_cast<T>(0);
        h_XImats[577] = static_cast<T>(1.00000000000000);
        h_XImats[578] = static_cast<T>(0);
        h_XImats[579] = static_cast<T>(0);
        h_XImats[580] = static_cast<T>(0);
        h_XImats[581] = static_cast<T>(0.184500000000000);
        h_XImats[582] = static_cast<T>(0);
        h_XImats[583] = static_cast<T>(1.00000000000000);
        // Xhom[5]
        h_XImats[584] = static_cast<T>(0);
        h_XImats[585] = static_cast<T>(0);
        h_XImats[586] = static_cast<T>(0);
        h_XImats[587] = static_cast<T>(0);
        h_XImats[588] = static_cast<T>(0);
        h_XImats[589] = static_cast<T>(0);
        h_XImats[590] = static_cast<T>(0);
        h_XImats[591] = static_cast<T>(0);
        h_XImats[592] = static_cast<T>(0);
        h_XImats[593] = static_cast<T>(1.00000000000000);
        h_XImats[594] = static_cast<T>(0);
        h_XImats[595] = static_cast<T>(0);
        h_XImats[596] = static_cast<T>(0);
        h_XImats[597] = static_cast<T>(-0.0607000000000000);
        h_XImats[598] = static_cast<T>(0.215500000000000);
        h_XImats[599] = static_cast<T>(1.00000000000000);
        // Xhom[6]
        h_XImats[600] = static_cast<T>(0);
        h_XImats[601] = static_cast<T>(0);
        h_XImats[602] = static_cast<T>(0);
        h_XImats[603] = static_cast<T>(0);
        h_XImats[604] = static_cast<T>(0);
        h_XImats[605] = static_cast<T>(0);
        h_XImats[606] = static_cast<T>(0);
        h_XImats[607] = static_cast<T>(0);
        h_XImats[608] = static_cast<T>(1.00000000000000);
        h_XImats[609] = static_cast<T>(0);
        h_XImats[610] = static_cast<T>(0);
        h_XImats[611] = static_cast<T>(0);
        h_XImats[612] = static_cast<T>(0.0810000000000000);
        h_XImats[613] = static_cast<T>(0);
        h_XImats[614] = static_cast<T>(0.0607000000000000);
        h_XImats[615] = static_cast<T>(1.00000000000000);
        // dXhom[0]
        h_XImats[616] = static_cast<T>(0);
        h_XImats[617] = static_cast<T>(0);
        h_XImats[618] = static_cast<T>(0);
        h_XImats[619] = static_cast<T>(0);
        h_XImats[620] = static_cast<T>(0);
        h_XImats[621] = static_cast<T>(0);
        h_XImats[622] = static_cast<T>(0);
        h_XImats[623] = static_cast<T>(0);
        h_XImats[624] = static_cast<T>(0);
        h_XImats[625] = static_cast<T>(0);
        h_XImats[626] = static_cast<T>(0);
        h_XImats[627] = static_cast<T>(0);
        h_XImats[628] = static_cast<T>(0);
        h_XImats[629] = static_cast<T>(0);
        h_XImats[630] = static_cast<T>(0);
        h_XImats[631] = static_cast<T>(0);
        // dXhom[1]
        h_XImats[632] = static_cast<T>(0);
        h_XImats[633] = static_cast<T>(0);
        h_XImats[634] = static_cast<T>(0);
        h_XImats[635] = static_cast<T>(0);
        h_XImats[636] = static_cast<T>(0);
        h_XImats[637] = static_cast<T>(0);
        h_XImats[638] = static_cast<T>(0);
        h_XImats[639] = static_cast<T>(0);
        h_XImats[640] = static_cast<T>(0);
        h_XImats[641] = static_cast<T>(0);
        h_XImats[642] = static_cast<T>(0);
        h_XImats[643] = static_cast<T>(0);
        h_XImats[644] = static_cast<T>(0);
        h_XImats[645] = static_cast<T>(0);
        h_XImats[646] = static_cast<T>(0);
        h_XImats[647] = static_cast<T>(0);
        // dXhom[2]
        h_XImats[648] = static_cast<T>(0);
        h_XImats[649] = static_cast<T>(0);
        h_XImats[650] = static_cast<T>(0);
        h_XImats[651] = static_cast<T>(0);
        h_XImats[652] = static_cast<T>(0);
        h_XImats[653] = static_cast<T>(0);
        h_XImats[654] = static_cast<T>(0);
        h_XImats[655] = static_cast<T>(0);
        h_XImats[656] = static_cast<T>(0);
        h_XImats[657] = static_cast<T>(0);
        h_XImats[658] = static_cast<T>(0);
        h_XImats[659] = static_cast<T>(0);
        h_XImats[660] = static_cast<T>(0);
        h_XImats[661] = static_cast<T>(0);
        h_XImats[662] = static_cast<T>(0);
        h_XImats[663] = static_cast<T>(0);
        // dXhom[3]
        h_XImats[664] = static_cast<T>(0);
        h_XImats[665] = static_cast<T>(0);
        h_XImats[666] = static_cast<T>(0);
        h_XImats[667] = static_cast<T>(0);
        h_XImats[668] = static_cast<T>(0);
        h_XImats[669] = static_cast<T>(0);
        h_XImats[670] = static_cast<T>(0);
        h_XImats[671] = static_cast<T>(0);
        h_XImats[672] = static_cast<T>(0);
        h_XImats[673] = static_cast<T>(0);
        h_XImats[674] = static_cast<T>(0);
        h_XImats[675] = static_cast<T>(0);
        h_XImats[676] = static_cast<T>(0);
        h_XImats[677] = static_cast<T>(0);
        h_XImats[678] = static_cast<T>(0);
        h_XImats[679] = static_cast<T>(0);
        // dXhom[4]
        h_XImats[680] = static_cast<T>(0);
        h_XImats[681] = static_cast<T>(0);
        h_XImats[682] = static_cast<T>(0);
        h_XImats[683] = static_cast<T>(0);
        h_XImats[684] = static_cast<T>(0);
        h_XImats[685] = static_cast<T>(0);
        h_XImats[686] = static_cast<T>(0);
        h_XImats[687] = static_cast<T>(0);
        h_XImats[688] = static_cast<T>(0);
        h_XImats[689] = static_cast<T>(0);
        h_XImats[690] = static_cast<T>(0);
        h_XImats[691] = static_cast<T>(0);
        h_XImats[692] = static_cast<T>(0);
        h_XImats[693] = static_cast<T>(0);
        h_XImats[694] = static_cast<T>(0);
        h_XImats[695] = static_cast<T>(0);
        // dXhom[5]
        h_XImats[696] = static_cast<T>(0);
        h_XImats[697] = static_cast<T>(0);
        h_XImats[698] = static_cast<T>(0);
        h_XImats[699] = static_cast<T>(0);
        h_XImats[700] = static_cast<T>(0);
        h_XImats[701] = static_cast<T>(0);
        h_XImats[702] = static_cast<T>(0);
        h_XImats[703] = static_cast<T>(0);
        h_XImats[704] = static_cast<T>(0);
        h_XImats[705] = static_cast<T>(0);
        h_XImats[706] = static_cast<T>(0);
        h_XImats[707] = static_cast<T>(0);
        h_XImats[708] = static_cast<T>(0);
        h_XImats[709] = static_cast<T>(0);
        h_XImats[710] = static_cast<T>(0);
        h_XImats[711] = static_cast<T>(0);
        // dXhom[6]
        h_XImats[712] = static_cast<T>(0);
        h_XImats[713] = static_cast<T>(0);
        h_XImats[714] = static_cast<T>(0);
        h_XImats[715] = static_cast<T>(0);
        h_XImats[716] = static_cast<T>(0);
        h_XImats[717] = static_cast<T>(0);
        h_XImats[718] = static_cast<T>(0);
        h_XImats[719] = static_cast<T>(0);
        h_XImats[720] = static_cast<T>(0);
        h_XImats[721] = static_cast<T>(0);
        h_XImats[722] = static_cast<T>(0);
        h_XImats[723] = static_cast<T>(0);
        h_XImats[724] = static_cast<T>(0);
        h_XImats[725] = static_cast<T>(0);
        h_XImats[726] = static_cast<T>(0);
        h_XImats[727] = static_cast<T>(0);
        // d2Xhom[0]
        h_XImats[728] = static_cast<T>(0);
        h_XImats[729] = static_cast<T>(0);
        h_XImats[730] = static_cast<T>(0);
        h_XImats[731] = static_cast<T>(0);
        h_XImats[732] = static_cast<T>(0);
        h_XImats[733] = static_cast<T>(0);
        h_XImats[734] = static_cast<T>(0);
        h_XImats[735] = static_cast<T>(0);
        h_XImats[736] = static_cast<T>(0);
        h_XImats[737] = static_cast<T>(0);
        h_XImats[738] = static_cast<T>(0);
        h_XImats[739] = static_cast<T>(0);
        h_XImats[740] = static_cast<T>(0);
        h_XImats[741] = static_cast<T>(0);
        h_XImats[742] = static_cast<T>(0);
        h_XImats[743] = static_cast<T>(0);
        // d2Xhom[1]
        h_XImats[744] = static_cast<T>(0);
        h_XImats[745] = static_cast<T>(0);
        h_XImats[746] = static_cast<T>(0);
        h_XImats[747] = static_cast<T>(0);
        h_XImats[748] = static_cast<T>(0);
        h_XImats[749] = static_cast<T>(0);
        h_XImats[750] = static_cast<T>(0);
        h_XImats[751] = static_cast<T>(0);
        h_XImats[752] = static_cast<T>(0);
        h_XImats[753] = static_cast<T>(0);
        h_XImats[754] = static_cast<T>(0);
        h_XImats[755] = static_cast<T>(0);
        h_XImats[756] = static_cast<T>(0);
        h_XImats[757] = static_cast<T>(0);
        h_XImats[758] = static_cast<T>(0);
        h_XImats[759] = static_cast<T>(0);
        // d2Xhom[2]
        h_XImats[760] = static_cast<T>(0);
        h_XImats[761] = static_cast<T>(0);
        h_XImats[762] = static_cast<T>(0);
        h_XImats[763] = static_cast<T>(0);
        h_XImats[764] = static_cast<T>(0);
        h_XImats[765] = static_cast<T>(0);
        h_XImats[766] = static_cast<T>(0);
        h_XImats[767] = static_cast<T>(0);
        h_XImats[768] = static_cast<T>(0);
        h_XImats[769] = static_cast<T>(0);
        h_XImats[770] = static_cast<T>(0);
        h_XImats[771] = static_cast<T>(0);
        h_XImats[772] = static_cast<T>(0);
        h_XImats[773] = static_cast<T>(0);
        h_XImats[774] = static_cast<T>(0);
        h_XImats[775] = static_cast<T>(0);
        // d2Xhom[3]
        h_XImats[776] = static_cast<T>(0);
        h_XImats[777] = static_cast<T>(0);
        h_XImats[778] = static_cast<T>(0);
        h_XImats[779] = static_cast<T>(0);
        h_XImats[780] = static_cast<T>(0);
        h_XImats[781] = static_cast<T>(0);
        h_XImats[782] = static_cast<T>(0);
        h_XImats[783] = static_cast<T>(0);
        h_XImats[784] = static_cast<T>(0);
        h_XImats[785] = static_cast<T>(0);
        h_XImats[786] = static_cast<T>(0);
        h_XImats[787] = static_cast<T>(0);
        h_XImats[788] = static_cast<T>(0);
        h_XImats[789] = static_cast<T>(0);
        h_XImats[790] = static_cast<T>(0);
        h_XImats[791] = static_cast<T>(0);
        // d2Xhom[4]
        h_XImats[792] = static_cast<T>(0);
        h_XImats[793] = static_cast<T>(0);
        h_XImats[794] = static_cast<T>(0);
        h_XImats[795] = static_cast<T>(0);
        h_XImats[796] = static_cast<T>(0);
        h_XImats[797] = static_cast<T>(0);
        h_XImats[798] = static_cast<T>(0);
        h_XImats[799] = static_cast<T>(0);
        h_XImats[800] = static_cast<T>(0);
        h_XImats[801] = static_cast<T>(0);
        h_XImats[802] = static_cast<T>(0);
        h_XImats[803] = static_cast<T>(0);
        h_XImats[804] = static_cast<T>(0);
        h_XImats[805] = static_cast<T>(0);
        h_XImats[806] = static_cast<T>(0);
        h_XImats[807] = static_cast<T>(0);
        // d2Xhom[5]
        h_XImats[808] = static_cast<T>(0);
        h_XImats[809] = static_cast<T>(0);
        h_XImats[810] = static_cast<T>(0);
        h_XImats[811] = static_cast<T>(0);
        h_XImats[812] = static_cast<T>(0);
        h_XImats[813] = static_cast<T>(0);
        h_XImats[814] = static_cast<T>(0);
        h_XImats[815] = static_cast<T>(0);
        h_XImats[816] = static_cast<T>(0);
        h_XImats[817] = static_cast<T>(0);
        h_XImats[818] = static_cast<T>(0);
        h_XImats[819] = static_cast<T>(0);
        h_XImats[820] = static_cast<T>(0);
        h_XImats[821] = static_cast<T>(0);
        h_XImats[822] = static_cast<T>(0);
        h_XImats[823] = static_cast<T>(0);
        // d2Xhom[6]
        h_XImats[824] = static_cast<T>(0);
        h_XImats[825] = static_cast<T>(0);
        h_XImats[826] = static_cast<T>(0);
        h_XImats[827] = static_cast<T>(0);
        h_XImats[828] = static_cast<T>(0);
        h_XImats[829] = static_cast<T>(0);
        h_XImats[830] = static_cast<T>(0);
        h_XImats[831] = static_cast<T>(0);
        h_XImats[832] = static_cast<T>(0);
        h_XImats[833] = static_cast<T>(0);
        h_XImats[834] = static_cast<T>(0);
        h_XImats[835] = static_cast<T>(0);
        h_XImats[836] = static_cast<T>(0);
        h_XImats[837] = static_cast<T>(0);
        h_XImats[838] = static_cast<T>(0);
        h_XImats[839] = static_cast<T>(0);
        T *d_XImats; gpuErrchk(cudaMalloc((void**)&d_XImats,840*sizeof(T)));
        gpuErrchk(cudaMemcpy(d_XImats,h_XImats,840*sizeof(T),cudaMemcpyHostToDevice));
        free(h_XImats);
        return d_XImats;
    }

    /**
     * Initializes the robotModel helpers in GPU memory
     *
     * @return A pointer to the robotModel struct
     */
    template <typename T>
    __host__
    robotModel<T>* init_robotModel() {
        robotModel<T> h_robotModel;
        h_robotModel.d_XImats = init_XImats<T>();
        h_robotModel.d_topology_helpers = init_topology_helpers<T>();
        robotModel<T> *d_robotModel; gpuErrchk(cudaMalloc((void**)&d_robotModel,sizeof(robotModel<T>)));
        gpuErrchk(cudaMemcpy(d_robotModel,&h_robotModel,sizeof(robotModel<T>),cudaMemcpyHostToDevice));
        return d_robotModel;
    }

    template<typename T>
    __host__ void free_robotModel(robotModel<T>* d_robotModel)
    {
        gpuErrchk(cudaFree(d_robotModel));
    }
    /**
     * Allocated device and host memory for all computations
     *
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T, int NUM_TIMESTEPS>
    __host__
    gridData<T> *init_gridData(){
        gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));// first the input variables on the GPU
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        // then the GPU outputs
        gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_M, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_eePos, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_deePos, 6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_d2eePos, 6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_idsva_so, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df2, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_M = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_eePos = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_deePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_d2eePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_idsva_so = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df2 = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        return hd_data;
    }

    /**
     * Allocated device and host memory for all computations
     *
     * @param Max number of timesteps in the trajectory
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T>
    __host__
    gridData<T> *init_gridData(int NUM_TIMESTEPS){
        gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));// first the input variables on the GPU
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        // then the GPU outputs
        gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_M, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_eePos, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_deePos, 6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_d2eePos, 6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_idsva_so, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df2, 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_M = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_eePos = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_deePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_d2eePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_idsva_so = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df2 = (T *)malloc(4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        return hd_data;
    }

    /**
     * Updates the Xmats in (shared) GPU memory acording to the configuration
     *
     * @param s_XImats is the (shared) memory destination location for the XImats
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XImats_helpers(T *s_XImats, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 504; ind += blockDim.x*blockDim.y){
            s_XImats[ind] = d_robotModel->d_XImats[ind];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X[0]
            s_XImats[0] = static_cast<T>(1.0*s_temp[7]);
            s_XImats[1] = static_cast<T>(-1.0*s_temp[0]);
            s_XImats[3] = static_cast<T>(-0.1575*s_temp[0]);
            s_XImats[4] = static_cast<T>(-0.1575*s_temp[7]);
            s_XImats[6] = static_cast<T>(1.0*s_temp[0]);
            s_XImats[7] = static_cast<T>(1.0*s_temp[7]);
            s_XImats[9] = static_cast<T>(0.1575*s_temp[7]);
            s_XImats[10] = static_cast<T>(-0.1575*s_temp[0]);
            // X[1]
            s_XImats[36] = static_cast<T>(s_temp[1]);
            s_XImats[37] = static_cast<T>(s_temp[8]);
            s_XImats[45] = static_cast<T>(0.2025*s_temp[1]);
            s_XImats[46] = static_cast<T>(0.2025*s_temp[8]);
            s_XImats[48] = static_cast<T>(s_temp[8]);
            s_XImats[49] = static_cast<T>(-s_temp[1]);
            // X[2]
            s_XImats[78] = static_cast<T>(s_temp[9]);
            s_XImats[79] = static_cast<T>(-s_temp[2]);
            s_XImats[81] = static_cast<T>(-0.2045*s_temp[2]);
            s_XImats[82] = static_cast<T>(-0.2045*s_temp[9]);
            s_XImats[84] = static_cast<T>(s_temp[2]);
            s_XImats[85] = static_cast<T>(s_temp[9]);
            s_XImats[87] = static_cast<T>(0.2045*s_temp[9]);
            s_XImats[88] = static_cast<T>(-0.2045*s_temp[2]);
            // X[3]
            s_XImats[108] = static_cast<T>(s_temp[10]);
            s_XImats[109] = static_cast<T>(-s_temp[3]);
            s_XImats[117] = static_cast<T>(0.2155*s_temp[10]);
            s_XImats[118] = static_cast<T>(-0.2155*s_temp[3]);
            s_XImats[120] = static_cast<T>(s_temp[3]);
            s_XImats[121] = static_cast<T>(s_temp[10]);
            // X[4]
            s_XImats[144] = static_cast<T>(s_temp[11]);
            s_XImats[145] = static_cast<T>(-s_temp[4]);
            s_XImats[147] = static_cast<T>(-0.1845*s_temp[4]);
            s_XImats[148] = static_cast<T>(-0.1845*s_temp[11]);
            s_XImats[156] = static_cast<T>(-s_temp[4]);
            s_XImats[157] = static_cast<T>(-s_temp[11]);
            s_XImats[159] = static_cast<T>(-0.1845*s_temp[11]);
            s_XImats[160] = static_cast<T>(0.1845*s_temp[4]);
            // X[5]
            s_XImats[180] = static_cast<T>(s_temp[5]);
            s_XImats[181] = static_cast<T>(s_temp[12]);
            s_XImats[183] = static_cast<T>(-0.0607000096320555*s_temp[12]);
            s_XImats[184] = static_cast<T>(0.0607000096320555*s_temp[5]);
            s_XImats[189] = static_cast<T>(0.2155*s_temp[5]);
            s_XImats[190] = static_cast<T>(0.2155*s_temp[12]);
            s_XImats[192] = static_cast<T>(s_temp[12]);
            s_XImats[193] = static_cast<T>(-s_temp[5]);
            s_XImats[195] = static_cast<T>(0.0607*s_temp[5]);
            s_XImats[196] = static_cast<T>(0.0607*s_temp[12]);
            // X[6]
            s_XImats[219] = static_cast<T>(-0.0607*s_temp[13]);
            s_XImats[220] = static_cast<T>(0.0607*s_temp[6]);
            s_XImats[222] = static_cast<T>(s_temp[13]);
            s_XImats[223] = static_cast<T>(-s_temp[6]);
            s_XImats[225] = static_cast<T>(-0.081*s_temp[6]);
            s_XImats[226] = static_cast<T>(-0.081*s_temp[13]);
            s_XImats[228] = static_cast<T>(s_temp[6]);
            s_XImats[229] = static_cast<T>(s_temp[13]);
            s_XImats[231] = static_cast<T>(0.081*s_temp[13]);
            s_XImats[232] = static_cast<T>(-0.081*s_temp[6]);
        }
        __syncthreads();
        for(int kcr = threadIdx.x + threadIdx.y*blockDim.x; kcr < 63; kcr += blockDim.x*blockDim.y){
            int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;
            int srcInd = k*36 + c*6 + r; int dstInd = srcInd + 21; // 3 more rows and cols
            s_XImats[dstInd] = s_XImats[srcInd];
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_dXmatsHom is the (shared) memory destination location for the dXmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_dXmatsHom, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
            s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+616];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
            // dX_hom[0]
            s_dXmatsHom[0] = static_cast<T>(-s_temp[0]);
            s_dXmatsHom[1] = static_cast<T>(s_temp[7]);
            s_dXmatsHom[4] = static_cast<T>(-s_temp[7]);
            s_dXmatsHom[5] = static_cast<T>(-s_temp[0]);
            // dX_hom[1]
            s_dXmatsHom[16] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[18] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[20] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[22] = static_cast<T>(-s_temp[8]);
            // dX_hom[2]
            s_dXmatsHom[33] = static_cast<T>(-s_temp[2]);
            s_dXmatsHom[34] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[37] = static_cast<T>(-s_temp[9]);
            s_dXmatsHom[38] = static_cast<T>(-s_temp[2]);
            // dX_hom[3]
            s_dXmatsHom[48] = static_cast<T>(-s_temp[3]);
            s_dXmatsHom[50] = static_cast<T>(s_temp[10]);
            s_dXmatsHom[52] = static_cast<T>(-s_temp[10]);
            s_dXmatsHom[54] = static_cast<T>(-s_temp[3]);
            // dX_hom[4]
            s_dXmatsHom[64] = static_cast<T>(-s_temp[4]);
            s_dXmatsHom[66] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[68] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[70] = static_cast<T>(s_temp[4]);
            // dX_hom[5]
            s_dXmatsHom[80] = static_cast<T>(s_temp[12]);
            s_dXmatsHom[82] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[86] = static_cast<T>(-s_temp[12]);
            // dX_hom[6]
            s_dXmatsHom[97] = static_cast<T>(-s_temp[6]);
            s_dXmatsHom[98] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[101] = static_cast<T>(-s_temp[13]);
            s_dXmatsHom[102] = static_cast<T>(-s_temp[6]);
        }
        __syncthreads();
    }

    /**
     * Updates the (d)XmatsHom in (shared) GPU memory acording to the configuration
     *
     * @param s_XmatsHom is the (shared) memory destination location for the XmatsHom
     * @param s_d2XmatsHom is the (shared) memory destination location for the d2XmatsHom
     * @param s_dXmatsHom is the (shared) memory destination location for the dXmatsHom
     * @param s_q is the (shared) memory location of the current configuration
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 14
     */
    template <typename T>
    __device__
    void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_dXmatsHom, T *s_d2XmatsHom, const T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            s_XmatsHom[ind] = d_robotModel->d_XImats[ind+504];
            s_dXmatsHom[ind] = d_robotModel->d_XImats[ind+616];
            s_d2XmatsHom[ind] = d_robotModel->d_XImats[ind+728];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 7; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+7] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X_hom[0]
            s_XmatsHom[0] = static_cast<T>(s_temp[7]);
            s_XmatsHom[1] = static_cast<T>(s_temp[0]);
            s_XmatsHom[4] = static_cast<T>(-s_temp[0]);
            s_XmatsHom[5] = static_cast<T>(s_temp[7]);
            // X_hom[1]
            s_XmatsHom[16] = static_cast<T>(s_temp[1]);
            s_XmatsHom[18] = static_cast<T>(s_temp[8]);
            s_XmatsHom[20] = static_cast<T>(s_temp[8]);
            s_XmatsHom[22] = static_cast<T>(-s_temp[1]);
            // X_hom[2]
            s_XmatsHom[33] = static_cast<T>(s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[37] = static_cast<T>(-s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(-s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(-s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[5]);
            s_XmatsHom[82] = static_cast<T>(s_temp[12]);
            s_XmatsHom[84] = static_cast<T>(s_temp[12]);
            s_XmatsHom[86] = static_cast<T>(-s_temp[5]);
            // X_hom[6]
            s_XmatsHom[97] = static_cast<T>(s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[101] = static_cast<T>(-s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
            // dX_hom[0]
            s_dXmatsHom[0] = static_cast<T>(-s_temp[0]);
            s_dXmatsHom[1] = static_cast<T>(s_temp[7]);
            s_dXmatsHom[4] = static_cast<T>(-s_temp[7]);
            s_dXmatsHom[5] = static_cast<T>(-s_temp[0]);
            // dX_hom[1]
            s_dXmatsHom[16] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[18] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[20] = static_cast<T>(-s_temp[1]);
            s_dXmatsHom[22] = static_cast<T>(-s_temp[8]);
            // dX_hom[2]
            s_dXmatsHom[33] = static_cast<T>(-s_temp[2]);
            s_dXmatsHom[34] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[37] = static_cast<T>(-s_temp[9]);
            s_dXmatsHom[38] = static_cast<T>(-s_temp[2]);
            // dX_hom[3]
            s_dXmatsHom[48] = static_cast<T>(-s_temp[3]);
            s_dXmatsHom[50] = static_cast<T>(s_temp[10]);
            s_dXmatsHom[52] = static_cast<T>(-s_temp[10]);
            s_dXmatsHom[54] = static_cast<T>(-s_temp[3]);
            // dX_hom[4]
            s_dXmatsHom[64] = static_cast<T>(-s_temp[4]);
            s_dXmatsHom[66] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[68] = static_cast<T>(-s_temp[11]);
            s_dXmatsHom[70] = static_cast<T>(s_temp[4]);
            // dX_hom[5]
            s_dXmatsHom[80] = static_cast<T>(s_temp[12]);
            s_dXmatsHom[82] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[86] = static_cast<T>(-s_temp[12]);
            // dX_hom[6]
            s_dXmatsHom[97] = static_cast<T>(-s_temp[6]);
            s_dXmatsHom[98] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[101] = static_cast<T>(-s_temp[13]);
            s_dXmatsHom[102] = static_cast<T>(-s_temp[6]);
            // d2X_hom[0]
            s_d2XmatsHom[0] = static_cast<T>(-s_temp[7]);
            s_d2XmatsHom[1] = static_cast<T>(-s_temp[0]);
            s_d2XmatsHom[4] = static_cast<T>(s_temp[0]);
            s_d2XmatsHom[5] = static_cast<T>(-s_temp[7]);
            // d2X_hom[1]
            s_d2XmatsHom[16] = static_cast<T>(-s_temp[1]);
            s_d2XmatsHom[18] = static_cast<T>(-s_temp[8]);
            s_d2XmatsHom[20] = static_cast<T>(-s_temp[8]);
            s_d2XmatsHom[22] = static_cast<T>(s_temp[1]);
            // d2X_hom[2]
            s_d2XmatsHom[33] = static_cast<T>(-s_temp[9]);
            s_d2XmatsHom[34] = static_cast<T>(-s_temp[2]);
            s_d2XmatsHom[37] = static_cast<T>(s_temp[2]);
            s_d2XmatsHom[38] = static_cast<T>(-s_temp[9]);
            // d2X_hom[3]
            s_d2XmatsHom[48] = static_cast<T>(-s_temp[10]);
            s_d2XmatsHom[50] = static_cast<T>(-s_temp[3]);
            s_d2XmatsHom[52] = static_cast<T>(s_temp[3]);
            s_d2XmatsHom[54] = static_cast<T>(-s_temp[10]);
            // d2X_hom[4]
            s_d2XmatsHom[64] = static_cast<T>(-s_temp[11]);
            s_d2XmatsHom[66] = static_cast<T>(s_temp[4]);
            s_d2XmatsHom[68] = static_cast<T>(s_temp[4]);
            s_d2XmatsHom[70] = static_cast<T>(s_temp[11]);
            // d2X_hom[5]
            s_d2XmatsHom[80] = static_cast<T>(-s_temp[5]);
            s_d2XmatsHom[82] = static_cast<T>(-s_temp[12]);
            s_d2XmatsHom[84] = static_cast<T>(-s_temp[12]);
            s_d2XmatsHom[86] = static_cast<T>(s_temp[5]);
            // d2X_hom[6]
            s_d2XmatsHom[97] = static_cast<T>(-s_temp[13]);
            s_d2XmatsHom[98] = static_cast<T>(-s_temp[6]);
            s_d2XmatsHom[101] = static_cast<T>(s_temp[6]);
            s_d2XmatsHom[102] = static_cast<T>(-s_temp[13]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * Notes:
     *   Assumes the Xhom matricies have already been updated for the given q
     *
     * @param s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 32
     */
    template <typename T>
    __device__
    void end_effector_pose_inner(T *s_eePos, const T *s_q, const T *s_Xhom, T *s_temp) {
        //
        // For each branch in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        // Serial chain manipulator so optimize as parent is jid-1
        // First set to leaf transform
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            s_temp[ind] = s_Xhom[16*6 + ind];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_temp[0 + 4*col]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 16; ind += blockDim.x*blockDim.y){
            int row = ind % 4; int col = ind / 4;
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_temp[16 + 4*col]);
        }
        __syncthreads();
        //
        // Now extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            // xyz is easy
            int xyzInd = ind % 3; int eeInd = ind / 3; T *s_Xmat_hom = &s_temp[0 + 16*eeInd];
            s_eePos[6*eeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];
            // roll pitch yaw is a bit more difficult
            if(xyzInd > 0){continue;}
            s_eePos[6*eeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);
            s_eePos[6*eeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));
            s_eePos[6*eeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);
        }
        __syncthreads();
    }

    /**
     * Computes the End Effector Position
     *
     * @param s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_device(T *s_eePos, const T *s_q, const robotModel<T> *d_robotModel) {
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
    }

    /**
     * Computes the End Effector Position
     *
     * @param s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_temp_in is the pointer to the temporary shared memory
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_device(T *s_eePos, const T *s_q, T* s_temp_in, const robotModel<T> *d_robotModel) {
        T* s_XHomTemp = s_temp_in;
        T* s_XmatsHom = s_XHomTemp;
        T* s_temp = &s_XHomTemp[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
    }


    /**
     * Compute the End Effector Position
     *
     * @param d_eePos is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_kernel_single_timing(T *d_eePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_eePos[6];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            d_eePos[ind] = s_eePos[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the End Effector Position
     *
     * @param d_eePos is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_kernel(T *d_eePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_eePos[6];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_eePos_k = &d_eePos[k*6];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                d_eePos_k[ind] = s_eePos[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_eePos,hd_data->d_eePos,6*NUM_EES*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel_single_timing<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel_single_timing<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_eePos,hd_data->d_eePos,6*NUM_EES*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call EEPOS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the End Effector Pose
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_kernel<T><<<block_dimms,thread_dimms,EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * Notes:
     *   Assumes the Xhom and dXhom matricies have already been updated for the given q
     *
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_dXhom is the pointer to the gradient of the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 448
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_inner(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, T *s_temp) {
        //
        // For each branch/gradient in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        T *s_eeTemp = &s_temp[0]; T *s_deeTemp = &s_temp[224];
        // Serial chain manipulator so optimize as parent is jid-1
        // First set the leaf transforms for eePos and deePos
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int eeIndStart = 16*6;
            s_eeTemp[ind] = s_Xhom[eeIndStart + rc];
            s_deeTemp[ind] = (djid == 6) ? s_dXhom[eeIndStart + rc] : s_Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_eeTemp[0 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 5) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*5 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_eeTemp[112 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 4) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*4 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_eeTemp[0 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 3) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*3 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_eeTemp[112 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 2) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*2 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_eeTemp[0 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 1) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*1 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_eeTemp[112 + colInd]);
            const T *s_Xhom_dXhom = ((djid == 0) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*0 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        //
        // Now extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int outputInd = ind % 6; int deeInd = ind / 6;
            T *s_Xmat_hom = &s_eeTemp[0 + 16*deeInd]; T *s_dXmat_hom = &s_deeTemp[0 + 16*deeInd];
            // xyz is easy
            if (outputInd < 3){s_deePos[6*deeInd + outputInd] = s_dXmat_hom[12 + outputInd];}
            // roll pitch yaw is a bit more difficult
            // note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
            // Also note that d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
            else {
                // simpler to recompute
                T sqrtTerm = sqrt(s_Xmat_hom[10]*s_Xmat_hom[10] + s_Xmat_hom[6]*s_Xmat_hom[6]);
                T dsqrtTerm = (s_Xmat_hom[10]*s_dXmat_hom[10] + s_Xmat_hom[6]*s_dXmat_hom[6])/sqrtTerm;
                // branch to get pointer locations
                T y; T x; T y_prime; T x_prime;
                     if (outputInd == 3){ y = s_Xmat_hom[6]; x = s_Xmat_hom[10]; y_prime = s_dXmat_hom[6]; x_prime = s_dXmat_hom[10]; }
                else if (outputInd == 4){ y = -s_Xmat_hom[2]; x = sqrtTerm; y_prime = -s_dXmat_hom[2]; x_prime = dsqrtTerm; }
                else              { y = s_Xmat_hom[1]; x = s_Xmat_hom[0]; y_prime = s_dXmat_hom[1]; x_prime = s_dXmat_hom[0]; }
                s_deePos[6*deeInd + outputInd] = (-x_prime*y + x*y_prime)/(x*x + y*y);
            }
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_temp_in is the pointer to the temporary shared memory
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_device(T *s_deePos, const T *s_q, T* s_temp_in, const robotModel<T> *d_robotModel) {
        T* s_XHomTemp = s_temp_in; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_kernel_single_timing(T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            d_deePos[ind] = s_deePos[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_kernel(T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_deePos_k = &d_deePos[k*42];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_deePos_k[ind] = s_deePos[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call DEEPOS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Computes the Gradient of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * Notes:
     *   Assumes the Xhom and dXhom matricies have already been updated for the given q
     *
     * @param s_d2eePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_dXhom is the pointer to the 1st derivative of the homogenous transformation matricies 
     * @param s_d2Xhom is the pointer to the 2nd derivative of the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 448
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_hessian_inner(T *s_d2eePos, T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, const T *s_d2Xhom, T *s_temp) {
        //
        // For each branch/gradient in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        // Keep all gradients (and the value) as we need them for the hessian
        //
        T *s_eeTemp = &s_temp[0]; T *s_deeTemp = &s_temp[32];
        // Serial chain manipulator so optimize as parent is jid-1
        // First set the leaf transforms for eePos and deePos
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int eeIndStart = 16*6;
            if(djid == 0){s_eeTemp[ind] = s_Xhom[eeIndStart + rc];}
            s_deeTemp[ind] = (djid == 6) ? s_dXhom[eeIndStart + rc] : s_Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*5 + row], &s_eeTemp[0 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 5) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*5 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*4 + row], &s_eeTemp[16 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 4) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*4 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*3 + row], &s_eeTemp[0 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 3) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*3 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*2 + row], &s_eeTemp[16 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 2) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*2 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 16] = dot_prod<T,4,4,1>(&s_Xhom[16*1 + row], &s_eeTemp[0 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 1) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*1 + row], &s_deeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            if(djid == 0){s_eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom[16*0 + row], &s_eeTemp[16 + colInd]);}
            const T *s_Xhom_dXhom = ((djid == 0) ? s_dXhom : s_Xhom);
            s_deeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*0 + row], &s_deeTemp[112 + colInd]);
        }
        __syncthreads();
        int eeTemp_Offset = 0;
        int deeTemp_Offset = 0;
        //
        // For each hessian term in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        T *s_d2eeTemp = &s_deeTemp[224];
        // set eeTemp and deeTemp to the right offsets
        s_eeTemp = &s_eeTemp[eeTemp_Offset];
        s_deeTemp = &s_deeTemp[deeTemp_Offset];
        // Serial chain manipulator so optimize as parent is jid-1
        // First set the leaf transforms
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int eeIndStart = 16*6;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 6)) ? s_d2Xhom : (((djid_i == 6) || (djid_j == 6)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind] = s_Xhom_dXhom_d2Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 5)) ? s_d2Xhom : (((djid_i == 5) || (djid_j == 5)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 784] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*5 + row], &s_d2eeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 4)) ? s_d2Xhom : (((djid_i == 4) || (djid_j == 4)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*4 + row], &s_d2eeTemp[784 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 3)) ? s_d2Xhom : (((djid_i == 3) || (djid_j == 3)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 784] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*3 + row], &s_d2eeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 2)) ? s_d2Xhom : (((djid_i == 2) || (djid_j == 2)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*2 + row], &s_d2eeTemp[784 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 1)) ? s_d2Xhom : (((djid_i == 1) || (djid_j == 1)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 784] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*1 + row], &s_d2eeTemp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 784; ind += blockDim.x*blockDim.y){
            int djid_ij = ind / 16; int rc = ind % 16; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom_d2Xhom = ((djid_i == djid_j) && (djid_i == 0)) ? s_d2Xhom : (((djid_i == 0) || (djid_j == 0)) ? s_dXhom : s_Xhom);
            s_d2eeTemp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom_d2Xhom[16*0 + row], &s_d2eeTemp[784 + colInd]);
        }
        __syncthreads();
        int d2eeTemp_Offset = 0;
        //
        // Finally Extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        // set d2eeTemp to the right offsets
        s_d2eeTemp = &s_d2eeTemp[d2eeTemp_Offset];
        // For all n*n*num_ee in parallel
        for(int ind6 = threadIdx.x + threadIdx.y*blockDim.x; ind6 < 294; ind6 += blockDim.x*blockDim.y){
            int ind = ind6 / 6; int outputInd = ind6 % 6;
            int curr_ee = ind / 49; int djid_ij = ind % 49; int djid_i = djid_ij / 7; int djid_j = djid_ij % 7;
            int ee_offset = 16 * curr_ee; int dee_offset = ee_offset * 7; int d2ee_offset = dee_offset * 7;
            T *s_Xmat_hom = &s_eeTemp[ee_offset]; T *s_d2Xmat_hom_ij = &s_d2eeTemp[d2ee_offset + 16 * djid_i * 7 + 16 * djid_j];
            T *s_dXmat_hom_i = &s_deeTemp[dee_offset + 16 * djid_i]; T *s_dXmat_hom_j = &s_deeTemp[dee_offset + 16 * djid_j];
            T *s_deePos_i = &s_deePos[6*djid_i]; int d2eePosOffset_ij = djid_i * 7 + djid_j; int d2eePosOffset_ind = 49;
            // Note: djid_j == 0 computes gradient too
            // xyz is easy
            if (outputInd < 3){
                if (djid_j == 0){s_deePos[outputInd + 6*djid_i] = s_dXmat_hom_i[12 + outputInd];}
                s_d2eePos[outputInd*d2eePosOffset_ind + d2eePosOffset_ij] = s_d2Xmat_hom_ij[12 + outputInd];
            }
            // roll pitch yaw is a bit more difficult
            // note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
            //       d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
            //       d2/dz of arctan2(y(z),x(z)) is (bottom*dtop - top*dbottom) / (bottom*bottom) of:
            //          top = -x_prime_i*y + x*y_prime_i
            //          dtop = -x_prime_prime*y + x*y_prime_prime + (i != j)*(-x_prime_i*y_prime_j + x_prime_j*y_prime_i)
            //          bottom = x*x + y*y
            //          dbottom = 2*x*x_prime_j + 2*y*y_prime_j
            else {
                T pitchSqrtTerm = sqrt(s_Xmat_hom[10]*s_Xmat_hom[10] + s_Xmat_hom[6]*s_Xmat_hom[6]);
                T dpitchSqrtTerm_i = (s_Xmat_hom[10]*s_dXmat_hom_i[10] + s_Xmat_hom[6]*s_dXmat_hom_i[6])/pitchSqrtTerm;
                T dpitchSqrtTerm_j = (s_Xmat_hom[10]*s_dXmat_hom_j[10] + s_Xmat_hom[6]*s_dXmat_hom_j[6])/pitchSqrtTerm;
                T dpitchSqrtTerm_i_top_dj = s_dXmat_hom_j[10]*s_dXmat_hom_i[10] + s_Xmat_hom[10]*s_d2Xmat_hom_ij[10] + 
                                            s_dXmat_hom_j[6]*s_dXmat_hom_i[6] + s_Xmat_hom[6]*s_d2Xmat_hom_ij[6];
                T d2pitchSqrtTerm_ij = (pitchSqrtTerm*dpitchSqrtTerm_i_top_dj - dpitchSqrtTerm_i*dpitchSqrtTerm_j) / (pitchSqrtTerm*pitchSqrtTerm);
                // branch to get pointer locations
                T y; T x; T y_prime_i; T x_prime_i; T y_prime_j; T x_prime_j; T y_dprime_ij; T x_dprime_ij;
                     if (outputInd == 3){ y = s_Xmat_hom[6]; x = s_Xmat_hom[10]; y_prime_i = s_dXmat_hom_i[6]; x_prime_i = s_dXmat_hom_i[10]; y_prime_j = s_dXmat_hom_j[6]; x_prime_j = s_dXmat_hom_j[10]; y_dprime_ij = s_d2Xmat_hom_ij[6]; x_dprime_ij = s_d2Xmat_hom_ij[10]; }
                else if (outputInd == 4){ y = -s_Xmat_hom[2]; x = pitchSqrtTerm; y_prime_i = -s_dXmat_hom_i[2]; x_prime_i = dpitchSqrtTerm_i; y_prime_j = -s_dXmat_hom_j[2]; x_prime_j = dpitchSqrtTerm_j; y_dprime_ij = -s_d2Xmat_hom_ij[2]; x_dprime_ij = d2pitchSqrtTerm_ij; }
                else              { y = s_Xmat_hom[1]; x = s_Xmat_hom[0]; y_prime_i = s_dXmat_hom_i[1]; x_prime_i = s_dXmat_hom_i[0]; y_prime_j = s_dXmat_hom_j[1]; x_prime_j = s_dXmat_hom_j[0]; y_dprime_ij = s_d2Xmat_hom_ij[1]; x_dprime_ij = s_d2Xmat_hom_ij[0]; }
                T top = -x_prime_i*y + x*y_prime_i;     T bottom = x*x + y*y;
                T dtop = -x_dprime_ij*y + x*y_dprime_ij + (djid_i != djid_j)*(-x_prime_i*y_prime_j + x_prime_j*y_prime_i);
                T dbottom = 2*x*x_prime_j + 2*y*y_prime_j;
                if (djid_j == 0){s_deePos[outputInd + 6*djid_i] = top/bottom;}
                s_d2eePos[outputInd*d2eePosOffset_ind + d2eePosOffset_ij] = (bottom*dtop - top*dbottom) / (bottom*bottom);
            }
        }
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param s_d2eePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_pose_gradient_hessian_device(T *s_d2eePos, T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel) {
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_d2XmatsHom = &s_dXmatsHom[112]; T *s_temp = &s_d2XmatsHom[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_pose_gradient_hessian_inner<T>(s_d2eePos, s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_temp);
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param d_d2eePos is the vector of end effector positions gradients
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_hessian_kernel_single_timing(T *d_d2eePos, T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_d2eePos[294];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_d2XmatsHom = &s_dXmatsHom[112]; T *s_temp = &s_d2XmatsHom[112];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_hessian_inner<T>(s_d2eePos, s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 294; ind += blockDim.x*blockDim.y){
            d_d2eePos[ind] = s_d2eePos[ind];
        }
        __syncthreads();
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            d_deePos[ind] = s_deePos[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param d_d2eePos is the vector of end effector positions gradients
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_pose_gradient_hessian_kernel(T *d_d2eePos, T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_d2eePos[294];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_d2XmatsHom = &s_dXmatsHom[112]; T *s_temp = &s_d2XmatsHom[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_pose_gradient_hessian_inner<T>(s_d2eePos, s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_d2XmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_d2eePos_k = &d_d2eePos[k*294];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 294; ind += blockDim.x*blockDim.y){
                d_d2eePos_k[ind] = s_d2eePos[ind];
            }
            __syncthreads();
            // save down to global
            T *d_deePos_k = &d_deePos[k*42];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_deePos_k[ind] = s_deePos[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_hessian(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hd_data->h_d2eePos,hd_data->d_d2eePos,6*NUM_EES*NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_hessian_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_hessian_kernel_single_timing<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_hessian_kernel_single_timing<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(hd_data->h_d2eePos,hd_data->d_d2eePos,6*NUM_EES*NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call DEEPOS %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Computes the Gradient and Hessian of the End Effector Pose with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void end_effector_pose_gradient_hessian_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                             const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {end_effector_pose_gradient_hessian_kernel<T><<<block_dimms,thread_dimms,D2EE_POS_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_d2eePos,hd_data->d_deePos,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1] + !vFlag * s_qdd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2] + !vFlag * s_qdd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3] + !vFlag * s_qdd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4] + !vFlag * s_qdd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5] + !vFlag * s_qdd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6] + !vFlag * s_qdd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int dof_id = threadIdx.x + threadIdx.y*blockDim.x; dof_id < 7; dof_id += blockDim.x*blockDim.y){
            s_c[dof_id] = s_vaf[84 + 6*dof_id + 2];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int dof_id = threadIdx.x + threadIdx.y*blockDim.x; dof_id < 7; dof_id += blockDim.x*blockDim.y){
            s_c[dof_id] = s_vaf[84 + 6*dof_id + 2];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1] + !vFlag * s_qdd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2] + !vFlag * s_qdd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3] + !vFlag * s_qdd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4] + !vFlag * s_qdd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5] + !vFlag * s_qdd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6] + !vFlag * s_qdd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 126
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 42
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[72], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 42; int jid6 = 6 * 6;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[6]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*5]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[78], &s_vaf[36], s_qd[6]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 42 + jid6;
            T *dst = IaFlag ? &s_vaf[84] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[252 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[84 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[84 + 6*1]);
            int dstOffset = 84 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 667
     */
    template <typename T>
    __device__
    void direct_minv_inner(T *s_Minv, const T *s_q, T *s_XImats, T *s_temp) {
        // T *s_F = &s_temp[0]; T *s_IA = &s_temp[294]; T *s_U = &s_temp[546]; T *s_Dinv = &s_temp[588]; T *s_Ia = &s_temp[595]; T *s_IaTemp = &s_temp[631];
        // Initialize IA = I
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 252; ind += blockDim.x*blockDim.y){
            s_temp[294 + ind] = s_XImats[252 + ind];
        }
        // Zero Minv and F
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            if(ind < 294){s_temp[0 + ind] = static_cast<T>(0);}
            else{s_Minv[ind - 294] = static_cast<T>(0);}
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // backward pass updates where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 36 + row] = s_temp[294 + 6*36 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 6] = static_cast<T>(1)/s_temp[546 + 36 + 2];
                s_Minv[8 * 6] = s_temp[588 + 6];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            s_Minv[42 + 6] -= s_temp[588 + 6] * s_temp[0 + 42*6 + 36 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*6 + 36 + row] += s_temp[546 + 6*6 + row] * s_Minv[42 + 6];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[510 + ind] - (s_temp[582 + row] * s_temp[594] * s_temp[582 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*6 + 6*6]; T *dst = &s_temp[0 + 42*5 + 6*6];
            // adjust for temp comps
            if (col >= 1) {
                col -= 1; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[474 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[216 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 30 + row] = s_temp[294 + 6*30 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 5] = static_cast<T>(1)/s_temp[546 + 30 + 2];
                s_Minv[8 * 5] = s_temp[588 + 5];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(5 + ind); int jid_subtreeN = 7*(5 + ind);
            s_Minv[jid_subtreeN + 5] -= s_temp[588 + 5] * s_temp[0 + 42*5 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*5 + jid_subtree6 + row] += s_temp[546 + 6*5 + row] * s_Minv[jid_subtreeN + 5];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[474 + ind] - (s_temp[576 + row] * s_temp[593] * s_temp[576 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*5 + 6*(5 + col)]; T *dst = &s_temp[0 + 42*4 + 6*(5 + col)];
            // adjust for temp comps
            if (col >= 2) {
                col -= 2; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[438 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[180 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 24 + row] = s_temp[294 + 6*24 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 4] = static_cast<T>(1)/s_temp[546 + 24 + 2];
                s_Minv[8 * 4] = s_temp[588 + 4];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(4 + ind); int jid_subtreeN = 7*(4 + ind);
            s_Minv[jid_subtreeN + 4] -= s_temp[588 + 4] * s_temp[0 + 42*4 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*4 + jid_subtree6 + row] += s_temp[546 + 6*4 + row] * s_Minv[jid_subtreeN + 4];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[438 + ind] - (s_temp[570 + row] * s_temp[592] * s_temp[570 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 54; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*4 + 6*(4 + col)]; T *dst = &s_temp[0 + 42*3 + 6*(4 + col)];
            // adjust for temp comps
            if (col >= 3) {
                col -= 3; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[402 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[144 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 18 + row] = s_temp[294 + 6*18 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 3] = static_cast<T>(1)/s_temp[546 + 18 + 2];
                s_Minv[8 * 3] = s_temp[588 + 3];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(3 + ind); int jid_subtreeN = 7*(3 + ind);
            s_Minv[jid_subtreeN + 3] -= s_temp[588 + 3] * s_temp[0 + 42*3 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*3 + jid_subtree6 + row] += s_temp[546 + 6*3 + row] * s_Minv[jid_subtreeN + 3];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[402 + ind] - (s_temp[564 + row] * s_temp[591] * s_temp[564 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*3 + 6*(3 + col)]; T *dst = &s_temp[0 + 42*2 + 6*(3 + col)];
            // adjust for temp comps
            if (col >= 4) {
                col -= 4; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[366 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[108 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 12 + row] = s_temp[294 + 6*12 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 2] = static_cast<T>(1)/s_temp[546 + 12 + 2];
                s_Minv[8 * 2] = s_temp[588 + 2];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 5; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(2 + ind); int jid_subtreeN = 7*(2 + ind);
            s_Minv[jid_subtreeN + 2] -= s_temp[588 + 2] * s_temp[0 + 42*2 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*2 + jid_subtree6 + row] += s_temp[546 + 6*2 + row] * s_Minv[jid_subtreeN + 2];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[366 + ind] - (s_temp[558 + row] * s_temp[590] * s_temp[558 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 66; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*2 + 6*(2 + col)]; T *dst = &s_temp[0 + 42*1 + 6*(2 + col)];
            // adjust for temp comps
            if (col >= 5) {
                col -= 5; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[330 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[72 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 6 + row] = s_temp[294 + 6*6 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 1] = static_cast<T>(1)/s_temp[546 + 6 + 2];
                s_Minv[8 * 1] = s_temp[588 + 1];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(1 + ind); int jid_subtreeN = 7*(1 + ind);
            s_Minv[jid_subtreeN + 1] -= s_temp[588 + 1] * s_temp[0 + 42*1 + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 42*1 + jid_subtree6 + row] += s_temp[546 + 6*1 + row] * s_Minv[jid_subtreeN + 1];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[595 + ind] = s_temp[330 + ind] - (s_temp[552 + row] * s_temp[589] * s_temp[552 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            T *src = &s_temp[0 + 42*1 + 6*(1 + col)]; T *dst = &s_temp[0 + 42*0 + 6*(1 + col)];
            // adjust for temp comps
            if (col >= 6) {
                col -= 6; src = &s_temp[595 + 6*col]; dst = &s_temp[631 + 6*col];
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            s_temp[294 + 6*col + row] += dot_prod<T,6,6,1>(&s_temp[631 + row],&s_XImats[36 + 6*col]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 0
        //     joints are: A1
        //     links are: L1
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            s_temp[546 + 0 + row] = s_temp[294 + 6*0 + 6*2 + row];
            if(row == 2){
                s_temp[588 + 0] = static_cast<T>(1)/s_temp[546 + 0 + 2];
                s_Minv[8 * 0] = s_temp[588 + 0];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            int jid_subtree6 = 6*(0 + ind); int jid_subtreeN = 7*(0 + ind);
            s_Minv[jid_subtreeN + 0] -= s_temp[588 + 0] * s_temp[0 + 42*0 + jid_subtree6 + 2];
        }
        __syncthreads();
        //
        // Forward Pass
        //   Note that due to the i: operation we need to go serially over all n
        //
        // forward pass for jid: 0
        // F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[0 + ind] = (row == 2) * s_Minv[0 + 7 * col];
        }
        __syncthreads();
        // forward pass for jid: 1
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 6;
            s_temp[42 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[36 + row], &s_temp[0 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 1;
            T *s_Fcol = &s_temp[42 + 6*col_ind];
            s_Minv[7 * col_ind + 1] -= s_temp[589] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[552]);
            s_Fcol[2] += s_Minv[7 * col_ind + 1];
        }
        __syncthreads();
        // forward pass for jid: 2
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 30; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 12;
            s_temp[84 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[72 + row], &s_temp[42 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 5; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 2;
            T *s_Fcol = &s_temp[84 + 6*col_ind];
            s_Minv[7 * col_ind + 2] -= s_temp[590] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[558]);
            s_Fcol[2] += s_Minv[7 * col_ind + 2];
        }
        __syncthreads();
        // forward pass for jid: 3
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 18;
            s_temp[126 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[108 + row], &s_temp[84 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 3;
            T *s_Fcol = &s_temp[126 + 6*col_ind];
            s_Minv[7 * col_ind + 3] -= s_temp[591] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[564]);
            s_Fcol[2] += s_Minv[7 * col_ind + 3];
        }
        __syncthreads();
        // forward pass for jid: 4
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 18; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 24;
            s_temp[168 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[144 + row], &s_temp[126 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 4;
            T *s_Fcol = &s_temp[168 + 6*col_ind];
            s_Minv[7 * col_ind + 4] -= s_temp[592] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[570]);
            s_Fcol[2] += s_Minv[7 * col_ind + 4];
        }
        __syncthreads();
        // forward pass for jid: 5
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 30;
            s_temp[210 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[180 + row], &s_temp[168 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 5;
            T *s_Fcol = &s_temp[210 + 6*col_ind];
            s_Minv[7 * col_ind + 5] -= s_temp[593] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[576]);
            s_Fcol[2] += s_Minv[7 * col_ind + 5];
        }
        __syncthreads();
        // forward pass for jid: 6
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 36;
            s_temp[252 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[216 + row], &s_temp[210 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 6;
            T *s_Fcol = &s_temp[252 + 6*col_ind];
            s_Minv[7 * col_ind + 6] -= s_temp[594] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[582]);
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void direct_minv_device(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel){
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel_single_timing(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            d_Minv[ind] = s_Minv[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
            __syncthreads();
            // save down to global
            T *d_Minv_k = &d_Minv[k*49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_Minv_k[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                     const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {direct_minv_kernel_single_timing<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel_single_timing<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call Minv %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                  const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Finish the forward dynamics computation with qdd = Minv*(u-c)
     *
     * Notes:
     *   Assumes s_Minv and s_c are already computed
     *   Does not internally sync the thread group, so it should be called after all threads have finished computing their values
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_u is the vector of joint input torques
     * @param s_c is the bias vector
     * @param s_Minv is the inverse mass matrix
     */
    template <typename T>
    __device__
    void forward_dynamics_finish(T *s_qdd, const T *s_u, const T *s_c, const T *s_Minv) {
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 7; row += blockDim.x*blockDim.y){
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * (s_u[col] - s_c[col]);
            }
            s_qdd[row] = val;
        }
    }

    /**
     * Computes forward dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *   Does not internally sync the thread group, so it should be called after all threads have finished computing their values
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is the pointer to the shared memory needed of size: 891
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_XImats, T *s_temp, const T gravity) {
        direct_minv_inner<T>(s_temp, s_q, s_XImats, &s_temp[49]);
        inverse_dynamics_inner<T>(&s_temp[49], &s_temp[56], s_q, s_qd, s_XImats, &s_temp[182], gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, &s_temp[49], s_temp);
    }

    /**
     * Computes forward dynamics
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_device(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel_single_timing(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_qdd[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            s_q_qd_u[ind] = d_q_qd_u[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_qdd[ind] = s_qdd[ind];
        }
        __syncthreads();
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_qdd[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd_k[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        forward_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call FD %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // then call the kernel
        forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_vaf are the helper intermediate variables computed by inverse_dynamics
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 66*NUM_JOINTS + 6*sparse_dv,da,df_col_needs = 1722
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, T *s_temp, const T gravity) {
        //
        // dv and da need 28 cols per dq,dqd
        // df needs 49 cols per dq,dqd
        //    out of a possible 49 cols per dq,dqd
        // Gradients are stored compactly as dv_i/dq_[0...a], dv_i+1/dq_[0...b], etc
        //    where a and b are the needed number of columns
        //
        // Temp memory offsets are as follows:
        // T *s_dv_dq = &s_temp[0]; T *s_dv_dqd = &s_temp[168]; T *s_da_dq = &s_temp[336];
        // T *s_da_dqd = &s_temp[504]; T *s_df_dq = &s_temp[672]; T *s_df_dqd = &s_temp[966];
        // T *s_FxvI = &s_temp[1260]; T *s_MxXv = &s_temp[1512]; T *s_MxXa = &s_temp[1554];
        // T *s_Mxv = &s_temp[1596]; T *s_Mxf = &s_temp[1638]; T *s_Iv = &s_temp[1680];
        //
        // Initial Temp Comps
        //
        // First compute Imat*v and Xmat*v_parent, Xmat*a_parent (store in FxvI for now)
        // Note that if jid_parent == -1 then v_parent = 0 and a_parent = gravity
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 126; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int jid = col % 7; int jid6 = 6*jid;
            bool parentIsBase = (jid-1) == -1;
            bool comp1 = col < 7; bool comp3 = col >= 14;
            int XIOffset  =  comp1 * 252 + 6*jid6 + row; // rowCol of I (comp1) or X (comp 2 and 3)
            int vaOffset  = comp1 * jid6 + !comp1 * 6*(jid-1) + comp3 * 42; // v_i (comp1) or va_parent (comp 2 and 3)
            int dstOffset = comp1 * 1680 + !comp1 * 1260 + comp3 * 42 + jid6 + row; // rowCol of dst
            s_temp[dstOffset] = (parentIsBase && !comp1) ? comp3 * s_XImats[XIOffset + 30] * gravity : 
                                                           dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);
        }
        __syncthreads();
        // Then compute Mx(Xv), Mx(Xa), Mx(v), Mx(f)
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 28; col += blockDim.x*blockDim.y){
            int dof_id = col / 4; int selector = col % 4; int dof_id6 = 6*dof_id;
            int jid6 = dof_id6;
            // branch to get pointer locations
            int dstOffset; const T * src;
                 if (selector == 0){ dstOffset = 1512; src = &s_temp[1260]; }
            else if (selector == 1){ dstOffset = 1554; src = &s_temp[1302]; }
            else if (selector == 2){ dstOffset = 1596; src = &s_vaf[0]; }
            else              { dstOffset = 1638; src = &s_vaf[84]; }
            mx2<T>(&s_temp[dstOffset + dof_id6], &src[jid6]);
        }
        __syncthreads();
        //
        // Forward Pass
        //
        // We start with dv/du noting that we only have values
        //    for ancestors and for the current index else 0
        // dv/du where bfs_level is 0
        //     joints are: A1
        //     links are: L1
        // when parent is base dv_dq = 0, dv_dqd = S
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int dq_flag = (ind / 6) == 0;
            int du_offset = dq_flag ? 0 : 168;
            s_temp[du_offset + 6*0 + row] = (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 1; int col_jid = col_du % 1;
            int dq_flag = col < 1;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*1 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*1 + row],&s_temp[du_col_offset + 6*0]);
            // then add {Mx(Xv) or S for col ind}
            s_temp[du_col_offset + 6*1 + 6 + row] = 
                dq_flag * s_temp[1512 + 6*1 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2; int col_jid = col_du % 2;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*3 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*2 + row],&s_temp[du_col_offset + 6*1]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 1) {
                s_temp[du_col_offset + 6*3 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*2 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 3; int col_jid = col_du % 3;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*6 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*3 + row],&s_temp[du_col_offset + 6*3]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 2) {
                s_temp[du_col_offset + 6*6 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*3 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4; int col_jid = col_du % 4;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*10 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*4 + row],&s_temp[du_col_offset + 6*6]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 3) {
                s_temp[du_col_offset + 6*10 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*4 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 5; int col_jid = col_du % 5;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*15 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*5 + row],&s_temp[du_col_offset + 6*10]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 4) {
                s_temp[du_col_offset + 6*15 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*5 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6; int col_jid = col_du % 6;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 0 + !dq_flag * 168 + 6 * col_jid;
            s_temp[du_col_offset + 6*21 + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*6 + row],&s_temp[du_col_offset + 6*15]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 5) {
                s_temp[du_col_offset + 6*21 + 6 + row] = 
                    dq_flag * s_temp[1512 + 6*6 + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // start da/du by setting = MxS(dv/du)*qd + {MxXa, Mxv} for all n in parallel
        // start with da/du = MxS(dv/du)*qd
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 56; col += blockDim.x*blockDim.y){
            int col_du = col % 28;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            mx2_scaled<T>(&s_temp[336 + 6*col], &s_temp[0 + 6*col], s_qd[jid]);
            // then add {MxXa, Mxv} to the appropriate column
            int dq_flag = col == col_du; int src_offset = dq_flag * 1554 + !dq_flag * 1596 + 6*jid;
            if(col_du == ((jid+1)*(jid+2)/2 - 1)){
                for(int row = 0; row < 6; row++){
                    s_temp[336 + 6*col + row] += s_temp[src_offset + row];
                }
            }
        }
        __syncthreads();
        // Finish da/du with parent updates noting that we only have values
        //    for ancestors and for the current index and nothing for bfs 0
        // da/du where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 1;
            int dq_flag = col == col_du; int col_jid = col_du % 1;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*1 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*1 + row],&s_temp[du_col_offset + 6*0]);
        }
        __syncthreads();
        // da/du where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2;
            int dq_flag = col == col_du; int col_jid = col_du % 2;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*3 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*2 + row],&s_temp[du_col_offset + 6*1]);
        }
        __syncthreads();
        // da/du where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 3;
            int dq_flag = col == col_du; int col_jid = col_du % 3;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*6 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*3 + row],&s_temp[du_col_offset + 6*3]);
        }
        __syncthreads();
        // da/du where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4;
            int dq_flag = col == col_du; int col_jid = col_du % 4;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*10 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*4 + row],&s_temp[du_col_offset + 6*6]);
        }
        __syncthreads();
        // da/du where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 5;
            int dq_flag = col == col_du; int col_jid = col_du % 5;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*15 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*5 + row],&s_temp[du_col_offset + 6*10]);
        }
        __syncthreads();
        // da/du where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6;
            int dq_flag = col == col_du; int col_jid = col_du % 6;
            int du_col_offset = dq_flag * 336 + !dq_flag * 504 + 6 * col_jid;
            s_temp[du_col_offset + 6*21 + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*6 + row],&s_temp[du_col_offset + 6*15]);
        }
        __syncthreads();
        // Init df/du to 0
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 588; ind += blockDim.x*blockDim.y){
            s_temp[672 + ind] = static_cast<T>(0);
        }
        __syncthreads();
        // Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I 
        //    aka do all of the Fx comps in parallel
        // note that while df has more cols than dva the dva cols are the first few df cols
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 98; col += blockDim.x*blockDim.y){
            int col_du = col % 28;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            // Compute Offsets and Pointers
            int dq_flag = col == col_du; int dva_to_df_adjust = 7*jid - jid*(jid+1)/2;
            int Offset_col_du_src = dq_flag * 0 + !dq_flag * 168 + 6*col_du;
            int Offset_col_du_dst = dq_flag * 672 + !dq_flag * 966 + 6*(col_du + dva_to_df_adjust);
            T *dst = &s_temp[Offset_col_du_dst]; const T *fx_src = &s_temp[Offset_col_du_src]; const T *mult_src = &s_temp[1680 + 6*jid];
            // Adjust pointers for temp comps (if applicable)
            if (col >= 56) {
                int comp = col - 56; int comp_col = comp % 6; // int jid = comp / 6;
                int jid6 = comp - comp_col; int jid36_col6 = 6*jid6 + 6*comp_col;
                dst = &s_temp[1260 + jid36_col6]; fx_src = &s_vaf[jid6]; mult_src = &s_XImats[252 + jid36_col6];
            }
            fx_times_v<T>(dst, fx_src, mult_src);
        }
        __syncthreads();
        // Then in parallel finish df/du += I*da/du + (Fx(v)I)*dv/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 336; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col6 = ind - row; int col_du = (col % 28);
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du >= 21) * 6;
            // Compute Offsets and Pointers
            int dva_to_df_adjust = 7*jid - jid*(jid+1)/2;
            if (col >= 28){dva_to_df_adjust += 21;}
            T *df_row_col = &s_temp[672 + 6*dva_to_df_adjust + ind];
            const T *dv_col = &s_temp[0 + col6]; const T *da_col = &s_temp[336 + col6];
            int jid36 = 36*jid; const T *I_row = &s_XImats[252 + jid36 + row]; const T *FxvI_row = &s_temp[1260 + jid36 + row];
            // Compute the values
            *df_row_col += dot_prod<T,6,6,1>(I_row,da_col) + dot_prod<T,6,6,1>(FxvI_row,dv_col);
        }
        // At the same time compute the last temp var: -X^T * mx(f)
        // use Mx(Xv) temp memory as those values are no longer needed
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int XTcol = ind % 6; int jid6 = ind - XTcol;
            s_temp[1512 + ind] = -dot_prod<T,6,1,1>(&s_XImats[6*(jid6 + XTcol)], &s_temp[1638 + jid6]);
        }
        __syncthreads();
        //
        // BACKWARD Pass
        //
        // df/du update where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 6) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = &s_temp[du_col_offset + 6*35 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],&s_temp[du_col_offset + 6*42])
                          + dq_flag * (col_du == 6) * s_temp[1512 + 6*6 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 5) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = &s_temp[du_col_offset + 6*28 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],&s_temp[du_col_offset + 6*35])
                          + dq_flag * (col_du == 5) * s_temp[1512 + 6*5 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 4) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = &s_temp[du_col_offset + 6*21 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],&s_temp[du_col_offset + 6*28])
                          + dq_flag * (col_du == 4) * s_temp[1512 + 6*4 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 3) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = &s_temp[du_col_offset + 6*14 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],&s_temp[du_col_offset + 6*21])
                          + dq_flag * (col_du == 3) * s_temp[1512 + 6*3 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 2) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = &s_temp[du_col_offset + 6*7 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],&s_temp[du_col_offset + 6*14])
                          + dq_flag * (col_du == 2) * s_temp[1512 + 6*2 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            int dst_adjust = (col_du >= 1) * 6 * 0; // adjust for sparsity compression offsets
            T *dst = &s_temp[du_col_offset + 6*0 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row],&s_temp[du_col_offset + 6*7])
                          + dq_flag * (col_du == 1) * s_temp[1512 + 6*1 + row];
            *dst += update_val;
        }
        __syncthreads();
        // Finally dc[i]/du = S[i]^T*df[i]/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int jid = ind % 7; int jid_dq_qd = ind / 7; int jid_du = jid_dq_qd % 7; int dq_flag = jid_du == jid_dq_qd;
            int Offset_src = dq_flag * 672 + !dq_flag * 966 + 6 * 7 * jid + 6 * jid_du + 2;
            int Offset_dst = !dq_flag * 49 + 7 * jid_du + jid;
            s_dc_du[Offset_dst] = s_temp[Offset_src];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_dc_du[ind] = s_dc_du[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_dc_du[ind] = s_dc_du[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                 const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID_DU %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'
     *
     * @param s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ T s_dc_du[98];
        __shared__ T s_Minv[49];
        __shared__ T s_qdd[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
        inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, &s_temp[7], gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
        __syncthreads();
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                // Also save MIV as df_dtau
                if (col < 7){
                    s_df_du[ind + 49] = s_Minv[index];
                }
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'
     *
     * @param s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param s_Minv is the mass matrix
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[126];
        __shared__ T s_dc_du[98];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            int row = ind % 7; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 7; col++) {
                int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                // Also save MIV as df_dtau
                if (col < 7){
                    s_df_du[ind + 49] = s_Minv[index];
                }
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            s_Minv[ind] = d_Minv[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                    // Also save MIV as df_dtau
                    if (col < 7){
                        s_temp[ind + 49] = s_Minv[index];
                    }
                }
                s_temp[ind] = -val;
            }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_df_du[ind] = s_temp[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            const T *d_Minv_k = &d_Minv[k*49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                    // Also save MIV as df_dtau
                    if (col < 7){
                        s_temp[ind + 49] = s_Minv[index];
                    }
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            s_q_qd_u[ind] = d_q_qd_u[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
            inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, &s_temp[7], gravity);
            forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
            __syncthreads();
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                    // Also save MIV as df_dtau
                    if (col < 7){
                        s_temp[ind + 49] = s_Minv[index];
                    }
                }
                s_temp[ind] = -val;
            }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
            d_df_du[ind] = s_temp[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
            inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, &s_temp[7], gravity);
            forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
            __syncthreads();
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                    // Also save MIV as df_dtau
                    if (col < 7){
                        s_temp[ind + 49] = s_Minv[index];
                    }
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call FD_DU %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd= 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the Articulated Body Algorithm
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *
     * @param s_qdd is the vector of joint accelerations
     * @param s_va is a pointer to shared memory of size 2*6*NUM_JOINTS = 84
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_tau is the vector of joint torques
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is the pointer to the shared memory needed of size: 891
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void aba_inner(T *s_qdd, T *s_va, const T *s_q, const T *s_qd, const T *s_tau, T *s_XImats, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v where parent is base
        //     joints are: A1
        //     links are: L1
        // s_v[k] = S[k]*qd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid = 0;
            int jid6 = 6*jid;
            s_va[jid6 + row] = static_cast<T>(0);
            if (row == 2){s_va[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 1;
            int jid6 = 6 * jid;
            T qd_val = (row == 2) * (s_qd[1]);
            s_va[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_va[6*0]) + qd_val;
        }
        __syncthreads();
        // s_v where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 2;
            int jid6 = 6 * jid;
            T qd_val = (row == 2) * (s_qd[2]);
            s_va[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_va[6*1]) + qd_val;
        }
        __syncthreads();
        // s_v where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 3;
            int jid6 = 6 * jid;
            T qd_val = (row == 2) * (s_qd[3]);
            s_va[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_va[6*2]) + qd_val;
        }
        __syncthreads();
        // s_v where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 4;
            int jid6 = 6 * jid;
            T qd_val = (row == 2) * (s_qd[4]);
            s_va[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_va[6*3]) + qd_val;
        }
        __syncthreads();
        // s_v where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 5;
            int jid6 = 6 * jid;
            T qd_val = (row == 2) * (s_qd[5]);
            s_va[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_va[6*4]) + qd_val;
        }
        __syncthreads();
        // s_v where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 6;
            int jid6 = 6 * jid;
            T qd_val = (row == 2) * (s_qd[6]);
            s_va[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_va[6*5]) + qd_val;
        }
        __syncthreads();
        // c[k] = mxS(v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            int jid = ind;
            int jid6 = 6 * jid;
            mx2_scaled<T>(&s_temp[72 * 7+jid6], &s_va[jid6], s_qd[jid]);
        }
        // Initialize IA = I
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 252; ind += blockDim.x*blockDim.y){
            s_temp[ind] = s_XImats[252 + ind];
        }
        // Initialize vcross[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            int jid = ind;
            int jid6 = 6 * jid;
            vcross<T>(&s_temp[36*(7+jid)], &s_va[jid6]);
        }
        __syncthreads();
        // temp[k] = -vcross.T*I[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 252; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6; int jid = ind / 36;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + jid6*6 + row+col*6] = -1 * dot_prod<T,6,1,1>(&s_temp[36*(7+jid)+row*6], &s_XImats[36 * (7+jid) + col*6]);
        }
        __syncthreads();
        // pA[k] = temp[k]*v[k][0]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 7;
            int jid6 = 6 * jid;
            s_temp[78 * 7 + jid6 + row] = dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6*jid6+row], &s_va[jid6]);
        }
        //
        // Backward Pass
        //
        // Backward pass where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 6;
            int jid6 = 6 * 6;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 6;
            int jid6 = 6 * 6;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 6;
            int jid6 = 6 * 6;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 6;
            int jid6 = 6 * 6;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        // temp[k] = X[k].T*Ia[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 6;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * 7+jid6*6+6*col]);
        }
        __syncthreads();
        // IA[parent] += temp[k]*X[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 6;
            int jid6 = 6 * jid;
            T prodtemp = static_cast<T>(0);
            prodtemp =  dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);
            atomicAdd(&s_temp[36 * 5 + row + 6*col], prodtemp);
        }
        __syncthreads();
        // pA[parent] += X[k].T*pa[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 6;
            int jid6 = 6 * 6;
            s_temp[134 * 7 + jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*jid+6*row],&s_temp[90*7+jid6]);
            atomicAdd(&s_temp[78 * 7 + 6 * 5 + row], s_temp[134 * 7 + jid6 + row]);
        }
        __syncthreads();
        // Backward pass where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 5;
            int jid6 = 6 * 5;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 5;
            int jid6 = 6 * 5;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 5;
            int jid6 = 6 * 5;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 5;
            int jid6 = 6 * 5;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        // temp[k] = X[k].T*Ia[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 5;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * 7+jid6*6+6*col]);
        }
        __syncthreads();
        // IA[parent] += temp[k]*X[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 5;
            int jid6 = 6 * jid;
            T prodtemp = static_cast<T>(0);
            prodtemp =  dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);
            atomicAdd(&s_temp[36 * 4 + row + 6*col], prodtemp);
        }
        __syncthreads();
        // pA[parent] += X[k].T*pa[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 5;
            int jid6 = 6 * 5;
            s_temp[134 * 7 + jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*jid+6*row],&s_temp[90*7+jid6]);
            atomicAdd(&s_temp[78 * 7 + 6 * 4 + row], s_temp[134 * 7 + jid6 + row]);
        }
        __syncthreads();
        // Backward pass where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 4;
            int jid6 = 6 * 4;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 4;
            int jid6 = 6 * 4;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 4;
            int jid6 = 6 * 4;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 4;
            int jid6 = 6 * 4;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        // temp[k] = X[k].T*Ia[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 4;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * 7+jid6*6+6*col]);
        }
        __syncthreads();
        // IA[parent] += temp[k]*X[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 4;
            int jid6 = 6 * jid;
            T prodtemp = static_cast<T>(0);
            prodtemp =  dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);
            atomicAdd(&s_temp[36 * 3 + row + 6*col], prodtemp);
        }
        __syncthreads();
        // pA[parent] += X[k].T*pa[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 4;
            int jid6 = 6 * 4;
            s_temp[134 * 7 + jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*jid+6*row],&s_temp[90*7+jid6]);
            atomicAdd(&s_temp[78 * 7 + 6 * 3 + row], s_temp[134 * 7 + jid6 + row]);
        }
        __syncthreads();
        // Backward pass where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 3;
            int jid6 = 6 * 3;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 3;
            int jid6 = 6 * 3;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 3;
            int jid6 = 6 * 3;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 3;
            int jid6 = 6 * 3;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        // temp[k] = X[k].T*Ia[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 3;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * 7+jid6*6+6*col]);
        }
        __syncthreads();
        // IA[parent] += temp[k]*X[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 3;
            int jid6 = 6 * jid;
            T prodtemp = static_cast<T>(0);
            prodtemp =  dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);
            atomicAdd(&s_temp[36 * 2 + row + 6*col], prodtemp);
        }
        __syncthreads();
        // pA[parent] += X[k].T*pa[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 3;
            int jid6 = 6 * 3;
            s_temp[134 * 7 + jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*jid+6*row],&s_temp[90*7+jid6]);
            atomicAdd(&s_temp[78 * 7 + 6 * 2 + row], s_temp[134 * 7 + jid6 + row]);
        }
        __syncthreads();
        // Backward pass where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 2;
            int jid6 = 6 * 2;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 2;
            int jid6 = 6 * 2;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 2;
            int jid6 = 6 * 2;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 2;
            int jid6 = 6 * 2;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        // temp[k] = X[k].T*Ia[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 2;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * 7+jid6*6+6*col]);
        }
        __syncthreads();
        // IA[parent] += temp[k]*X[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 2;
            int jid6 = 6 * jid;
            T prodtemp = static_cast<T>(0);
            prodtemp =  dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);
            atomicAdd(&s_temp[36 * 1 + row + 6*col], prodtemp);
        }
        __syncthreads();
        // pA[parent] += X[k].T*pa[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 2;
            int jid6 = 6 * 2;
            s_temp[134 * 7 + jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*jid+6*row],&s_temp[90*7+jid6]);
            atomicAdd(&s_temp[78 * 7 + 6 * 1 + row], s_temp[134 * 7 + jid6 + row]);
        }
        __syncthreads();
        // Backward pass where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 1;
            int jid6 = 6 * 1;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 1;
            int jid6 = 6 * 1;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 1;
            int jid6 = 6 * 1;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 1;
            int jid6 = 6 * 1;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        // temp[k] = X[k].T*Ia[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 1;
            int jid6 = 6 * jid;
            s_temp[98 * 7 + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * 7+jid6*6+6*col]);
        }
        __syncthreads();
        // IA[parent] += temp[k]*X[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 1;
            int jid6 = 6 * jid;
            T prodtemp = static_cast<T>(0);
            prodtemp =  dot_prod<T,6,6,1>(&s_temp[98 * 7 + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);
            atomicAdd(&s_temp[36 * 0 + row + 6*col], prodtemp);
        }
        __syncthreads();
        // pA[parent] += X[k].T*pa[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 1;
            int jid6 = 6 * 1;
            s_temp[134 * 7 + jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*jid+6*row],&s_temp[90*7+jid6]);
            atomicAdd(&s_temp[78 * 7 + 6 * 0 + row], s_temp[134 * 7 + jid6 + row]);
        }
        __syncthreads();
        // Backward pass where bfs_level is 0
        //     joints are: A1
        //     links are: L1
        // U[k] = IA[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 0;
            int jid6 = 6 * 0;
            s_temp[84*7+jid6+row] = s_temp[36*jid+row+6*(2)];
        }
        __syncthreads();
        // d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 0;
            int jid6 = 6 * 0;
            s_temp[96 * 7 + jid] = s_temp[84 * 7 + jid6 + 2];
            T tempval = s_temp[78 * 7 + jid6 + 2];
            s_temp[97 * 7 + jid] = s_tau[jid] - tempval;
        }
        __syncthreads();
        // Ia[k] = IA[k] - U[k]*U[k].T/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = (ind / 6) %6;
            int jid = 0;
            int jid6 = 6 * 0;
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[84*7+jid6+row]*s_temp[84*7+jid6+col]/s_temp[96 *7+jid];
            s_temp[36 * 7+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * 7+6*jid6+row+6*col];
        }
        __syncthreads();
        // pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 0;
            int jid6 = 6 * 0;
            T Uval = s_temp[84 * 7+jid6+row]*s_temp[97*7+jid]/s_temp[96*7+jid];
            s_temp[90 * 7 + jid6 + row] = s_temp[78 * 7 + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*(7+jid)+row], &s_temp[72*7+jid6]) + Uval;
        }
        //
        // Second Forward Pass
        //
        // s_a, qdd where parent is base
        //     joints are: A1
        //     links are: L1
        // a[k] = X[k]*gravity_vec + c[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid = 0;
            int jid6 = 6*0;
            T gravity_vec[] = {0,0,0,0,0,gravity};
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &gravity_vec[0]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 0;
            int jid6 = 6 * 0;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 0;
            int jid6 = 6 * 0;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
        // s_a, s_qdd where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        // a[k] = X[k]*a[parent] + c[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 1;
            int jid6 = 6 * 1;
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &s_va[6*7+(6 * 0)]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 1;
            int jid6 = 6 * 1;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 1;
            int jid6 = 6 * 1;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
        // s_a, s_qdd where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        // a[k] = X[k]*a[parent] + c[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 2;
            int jid6 = 6 * 2;
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &s_va[6*7+(6 * 1)]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 2;
            int jid6 = 6 * 2;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 2;
            int jid6 = 6 * 2;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
        // s_a, s_qdd where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        // a[k] = X[k]*a[parent] + c[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 3;
            int jid6 = 6 * 3;
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &s_va[6*7+(6 * 2)]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 3;
            int jid6 = 6 * 3;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 3;
            int jid6 = 6 * 3;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
        // s_a, s_qdd where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        // a[k] = X[k]*a[parent] + c[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 4;
            int jid6 = 6 * 4;
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &s_va[6*7+(6 * 3)]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 4;
            int jid6 = 6 * 4;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 4;
            int jid6 = 6 * 4;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
        // s_a, s_qdd where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        // a[k] = X[k]*a[parent] + c[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 5;
            int jid6 = 6 * 5;
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &s_va[6*7+(6 * 4)]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 5;
            int jid6 = 6 * 5;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 5;
            int jid6 = 6 * 5;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
        // s_a, s_qdd where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        // a[k] = X[k]*a[parent] + c[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 6;
            int jid6 = 6 * 6;
            s_va[6*7+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &s_va[6*7+(6 * 5)]) + s_temp[72*7+jid6+row];
        }
        __syncthreads();
        // qdd[k] = (u[k] - U[k].T*a[k])/d[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int jid = 6;
            int jid6 = 6 * 6;
            T tempval = s_temp[97 * 7+jid] - dot_prod<T,6,1,1>(&s_temp[84*7+jid6], &s_va[6*7+jid6]);
            s_qdd[jid] = tempval / s_temp[96*7+jid];
        }
        __syncthreads();
        // a[k] += qdd[k]*S[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            int jid = 6;
            int jid6 = 6 * 6;
            T qdd_val = (row == 2) * (s_qdd[jid]);
            s_va[6*7+jid6+row] += qdd_val;
        }
        __syncthreads();
    }

    /**
     * Compute the ABA (Articulated Body Algorithm)
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_tau is the vector of joint torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void aba_device(const T *s_q, const T *s_qd, const T *s_tau, const robotModel<T> *d_robotModel, const T gravity) {
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        extern __shared__ T s_va[2*6*7];
        extern __shared__ T s_qdd[7];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        aba_inner<T>(s_qdd, s_va, s_q, s_qd, s_tau, s_XImats, s_temp, gravity);
    }

    /**
     * Compute the ABA (Articulated Body Algorithm)
     *
     * @param d_q_qd_tau is the vector of joint positions and velocities
     * @param stride_q_qd is the stride between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_tau is the vector of joint torques
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void aba_kernel_single_timing(T *d_qdd, const T *d_q_qd_tau, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_qdd[7];
        __shared__ T s_q_qd_tau[3*7]; T *s_q = s_q_qd_tau; T *s_qd = &s_q_qd_tau[7]; T *s_tau = &s_q_qd_tau[2 * 7];
        __shared__ T s_va[84];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            s_q_qd_tau[ind] = d_q_qd_tau[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            aba_inner<T>(s_qdd, s_va, s_q, s_qd, s_tau, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
            d_qdd[ind] = s_qdd[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the ABA (Articulated Body Algorithm)
     *
     * @param d_q_qd_tau is the vector of joint positions and velocities
     * @param stride_q_qd is the stride between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_tau is the vector of joint torques
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void aba_kernel(T *d_qdd, const T *d_q_qd_tau, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_qdd[7];
        __shared__ T s_q_qd_tau[3*7]; T *s_q = s_q_qd_tau; T *s_qd = &s_q_qd_tau[7]; T *s_tau = &s_q_qd_tau[2 * 7];
        __shared__ T s_va[84];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_tau_k = &d_q_qd_tau[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_tau[ind] = d_q_qd_tau_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            aba_inner<T>(s_qdd, s_va, s_q, s_qd, s_tau, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_qdd_k = &d_qdd[k*1];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd_k[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the ABA (Articulated Body Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void aba(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        aba_kernel<T><<<block_dimms,thread_dimms,ABA_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the ABA (Articulated Body Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void aba_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        aba_kernel_single_timing<T><<<block_dimms,thread_dimms,ABA_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ABA %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the ABA (Articulated Body Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void aba_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = 3*NUM_JOINTS;
        // then call the kernel
        aba_kernel<T><<<block_dimms,thread_dimms,ABA_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the Composite Rigid Body Algorithm
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_M is a pointer to the matrix of inertias_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 980
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void crba_inner(T *s_M, const T *s_q, const T *s_qd, T *s_XImats, T *s_temp, const T gravity) {
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 49; i += blockDim.x*blockDim.y){
            s_M[i] = static_cast<T>(0);
        }
        __syncthreads();
        T *alpha = &s_temp[0];
        T *beta = &s_temp[252];
        T *s_fh = &s_temp[504];
        T *s_jid_list = &s_temp[553];
        //
        // first loop (split into 2 parallel loops in bfs loop)
        // each bfs level runs in parallel
        //
        // pass updates where bfs_level is 6
        //     joints are: A7
        //     links are: L7
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 6 ;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 6 ;
            int parent_ind = 5;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);
            s_XImats[36*(parent_ind +0+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];
        }
        __syncthreads();
        // pass updates where bfs_level is 5
        //     joints are: A6
        //     links are: L6
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 5 ;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 5 ;
            int parent_ind = 4;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);
            s_XImats[36*(parent_ind +0+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];
        }
        __syncthreads();
        // pass updates where bfs_level is 4
        //     joints are: A5
        //     links are: L5
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 4 ;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 4 ;
            int parent_ind = 3;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);
            s_XImats[36*(parent_ind +0+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];
        }
        __syncthreads();
        // pass updates where bfs_level is 3
        //     joints are: A4
        //     links are: L4
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 3 ;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 3 ;
            int parent_ind = 2;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);
            s_XImats[36*(parent_ind +0+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];
        }
        __syncthreads();
        // pass updates where bfs_level is 2
        //     joints are: A3
        //     links are: L3
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 2 ;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 2 ;
            int parent_ind = 1;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);
            s_XImats[36*(parent_ind +0+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];
        }
        __syncthreads();
        // pass updates where bfs_level is 1
        //     joints are: A2
        //     links are: L2
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 1 ;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int jid = 1 ;
            int parent_ind = 0;
            int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;
            beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);
            s_XImats[36*(parent_ind +0+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];
        }
        __syncthreads();
        //
        // Calculation of M[ind, ind] 
        //
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            s_M[jid+jid*7] = s_XImats[252 + 36*jid + 6*2 + 2];
        }
        __syncthreads();
        //
        // Calculation of M[ind, parent]
        //
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 42; i += blockDim.x*blockDim.y){
            int jid = i / 6; int ind = i % 6;
            s_fh[i] = s_XImats[252 + 36*jid + 6*2 + ind];
        }
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            int jid_parents[] = {-1, -1, -1, -1, -1, -1};
            int num_parents = 0;
            switch (jid) {
                case 0:
                    num_parents += 0;
                    break;
                case 1:
                    jid_parents[0] = 0;
                    num_parents += 1;
                    break;
                case 2:
                    jid_parents[0] = 1;
                    jid_parents[1] = 0;
                    num_parents += 2;
                    break;
                case 3:
                    jid_parents[0] = 2;
                    jid_parents[1] = 1;
                    jid_parents[2] = 0;
                    num_parents += 3;
                    break;
                case 4:
                    jid_parents[0] = 3;
                    jid_parents[1] = 2;
                    jid_parents[2] = 1;
                    jid_parents[3] = 0;
                    num_parents += 4;
                    break;
                case 5:
                    jid_parents[0] = 4;
                    jid_parents[1] = 3;
                    jid_parents[2] = 2;
                    jid_parents[3] = 1;
                    jid_parents[4] = 0;
                    num_parents += 5;
                    break;
                case 6:
                    jid_parents[0] = 5;
                    jid_parents[1] = 4;
                    jid_parents[2] = 3;
                    jid_parents[3] = 2;
                    jid_parents[4] = 1;
                    jid_parents[5] = 0;
                    num_parents += 6;
                    break;
            }
            T s_alpha[6];
            for (int i = 0; i < num_parents; i++) {
                int X_ind = i==0 ? jid : jid_parents[i-1];
                for (int k = 0; k < 6; k++) s_alpha[k] = s_fh[jid*6+k];
                for (int k = 0; k < 6; k++) s_fh[jid*6 + k] = dot_prod<T,6,1,1>(&s_XImats[36*X_ind+k*6], &s_alpha[0]);
                int parent_ind = jid_parents[i];
                s_M[jid*7 + parent_ind] = s_fh[jid*6 + 2];
                s_M[parent_ind*7 + jid] = s_M[jid*7 + parent_ind];
            }
        }
        __syncthreads();
    }

    /**
     * Compute the CRBA (Composite Rigid Body Algorithm)
     *
     * @param s_M is a pointer to the matrix of inertia
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void crba_device(T *s_M, const T *s_q, const T *s_qd,const robotModel<T> *d_robotModel, const T gravity) {
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
        crba_inner<T>(s_M, s_q, s_qd, s_XImats, s_temp, gravity);
    }

    /**
     * Compute the CRBA (Composite Rigid Body Algorithm)
     *
     * @param d_M is the pointer to the matrix of inertia
     * @param d_q_qd is the vector of joint positions and velocities
     * @param stride_q_qd is the stride between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void crba_kernel_single_timing(T *d_M, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_M[49];
        __shared__ T s_q_qd[3*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            crba_inner<T>(s_M, s_q, s_qd, s_XImats, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            d_M[ind] = s_M[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the CRBA (Composite Rigid Body Algorithm)
     *
     * @param d_M is the pointer to the matrix of inertia
     * @param d_q_qd is the vector of joint positions and velocities
     * @param stride_q_qd is the stride between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void crba_kernel(T *d_M, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_M[49];
        __shared__ T s_q_qd[3*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            crba_inner<T>(s_M, s_q, s_qd, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_M_k = &d_M[k*1];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_M_k[ind] = s_M[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the CRBA (Composite Rigid Body Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void crba(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        // then call the kernel
        if (USE_COMPRESSED_MEM) {crba_kernel<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        else                    {crba_kernel<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_M,hd_data->d_M,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the CRBA (Composite Rigid Body Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void crba_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {crba_kernel_single_timing<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        else                    {crba_kernel_single_timing<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_M,hd_data->d_M,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the CRBA (Composite Rigid Body Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void crba_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {crba_kernel<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        else                    {crba_kernel<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the second order derivatives of inverse dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *
     * @param s_idsva_so is a pointer to memory for the final result of size 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS = 1372
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is a pointer to helper shared memory of size  = 3744
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void idsva_so_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, T *s_XImats, T *s_temp, const T gravity) {
        // Relevant Tensors in the order they appear
        T *I = s_XImats + XIMAT_SIZE*NUM_JOINTS;
        T *Xup = s_temp + 11*XIMAT_SIZE*NUM_JOINTS;
        T *IC = s_temp + XIMAT_SIZE*NUM_JOINTS;
        T *Xdown = s_temp;

        T *S = IC + XIMAT_SIZE*NUM_JOINTS;
        T *vJ = Xdown;
        T *v = vJ + 6*NUM_JOINTS;
        T *Sd = v + 6*NUM_JOINTS;
        T *aJ = Sd + 6*NUM_JOINTS;
        T *a = aJ + 6*NUM_JOINTS;
        T *psid = a + 6*NUM_JOINTS;
        T *psidd = S + 6*NUM_JOINTS;
        T *a_world = psidd + 6*NUM_JOINTS;
        T *BC = S + 30*NUM_JOINTS + 6;
        T *f = vJ;
        T *B_IC_S = BC + 36*NUM_JOINTS;
        


        // Temporary Variables for Computations
        T *I_Xup = S;
        T *crm_v = B_IC_S + 36*NUM_JOINTS;
        T *crf_v = crm_v + 36*NUM_JOINTS;
        T *IC_v = aJ;
        T *crm_S = crm_v;
        T *crf_S = crf_v;
        T *IC_S = IC_v;
        T *crm_psid = crf_v + 36*NUM_JOINTS;
        T *crf_psid = crm_psid + 36*NUM_JOINTS;
        T *IC_psid = a;
        T *icrf_f = crf_psid + 36*NUM_JOINTS;
        T *psid_Sd = v;
        T *ICT_S = f;
        


        // Main Temporary Tensors For Backward Pass
        T *T1 = IC_S;
        T *T2 = a_world + 6;
        T *T3 = T2 + 6*NUM_JOINTS;
        T *T4 = T3 + 6*NUM_JOINTS;
        T *D1 = icrf_f;
        T *D2 = D1 + 36*NUM_JOINTS;
        T *D3 = B_IC_S;
        T *D4 = crf_psid;
        T *t = D2 + 36*NUM_JOINTS;
        T *p1 = t;
        T *p2 = p1 + 6*28;
        T *p3 = p2 + 6*28;
        T *p4 = p3 + 6*28;
        T *p5 = p4 + 6*28;
        T *p6 = p5 + 6*28;
        T *crf_S_IC = crm_psid;
        


        // Final Tensors for Output
        T *d2tau_dq2 = s_idsva_so;
        T *d2tau_dqd2 = d2tau_dq2+ NUM_JOINTS*NUM_JOINTS*NUM_JOINTS;
        T *d2tau_dvdq = d2tau_dqd2 + NUM_JOINTS*NUM_JOINTS*NUM_JOINTS;
        T *dM_dq = d2tau_dvdq + NUM_JOINTS*NUM_JOINTS*NUM_JOINTS;
        

        // Compute Xup - parent to child transformation matrices
        #pragma unroll
        for (int jid = 0; jid < NUM_JOINTS; ++jid) {
            // Compute Xup[joint]
            int X_idx = jid*XIMAT_SIZE;
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < XIMAT_SIZE; i += blockDim.x*blockDim.y){
                if ((jid-1) == -1) Xup[X_idx + i] = s_XImats[X_idx + i]; // Parent is base
                else matmul<T>(i, &Xup[(jid-1) * XIMAT_SIZE], &s_XImats[X_idx], &Xup[X_idx], XIMAT_SIZE, 0);
            }
            __syncthreads();
        }
        


        // Compute IC - Centroidal Rigid Body Inertia
        // First I @ Xup
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < XIMAT_SIZE*NUM_JOINTS; i += blockDim.x*blockDim.y){
            // All involved matrices are 6x6
            matmul<T>(i, Xup, I, I_Xup, 36, false);
        }
        __syncthreads();
        // Next Xup.T @ I
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < XIMAT_SIZE*NUM_JOINTS; i += blockDim.x*blockDim.y){
            // All involved matrices are 6x6
            int mat_idx = (i / 36) * 36;
            matmul_trans<T>(i % 36, &Xup[mat_idx], &I_Xup[mat_idx], &IC[mat_idx], 'a');
        }
        __syncthreads();
        


        // Compute Xdown - child to parent transformation matrices
        for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < XIMAT_SIZE*NUM_JOINTS; i += blockDim.x*blockDim.y){
            size_t idx = i % XIMAT_SIZE;
            size_t sub_idx = idx % 18;
            if (idx % 18 == 1 || idx % 18 == 4 || idx % 18 == 8 || idx % 18 == 11) {
                Xdown[i] = Xup[i+5];
                Xdown[i+5] = Xup[i];
            }
            else if (idx % 18 == 2 || idx % 18 == 5) {
                Xdown[i] = Xup[i+10];
                Xdown[i+10] = Xup[i];
            }
            else if (sub_idx != 6 && sub_idx != 9 && sub_idx != 13 && sub_idx != 16 &&
                        sub_idx != 12 && sub_idx != 15)
                Xdown[i] = Xup[i];
            }
            __syncthreads();
            


            // Transform S
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int joint = i / 6;
                S[i] = Xdown[joint*XIMAT_SIZE + 2*6 + (i % 6)];
            }
            __syncthreads();
            


            // Compute vJ = S @ qd & aJ = S @ qdd
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 2*6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int joint = i / 6;
                if (joint < NUM_JOINTS) vJ[i] = S[i] * s_qd[joint];
                else aJ[i - 6*NUM_JOINTS] = S[i - 6*NUM_JOINTS] * s_qdd[joint - NUM_JOINTS];
            }
            __syncthreads();
            


            // Compute v = v[parent] + vJ
            #pragma unroll
            for (int jid = 0; jid < NUM_JOINTS; ++jid) {
                for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 6; i += blockDim.x*blockDim.y){
                    if ((jid-1) == -1) v[jid*6 + i] = vJ[jid*6 + i];
                    else v[jid*6 + i] = v[(jid-1)*6 + i] + vJ[jid*6 + i];
                }
                __syncthreads();
            }
            


            // Finish aJ += crm(v[parent])@vJ
            // For base, v[parent] = 0
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = i / 6;
                int index = i % 6;
                if ((jid-1) != -1) aJ[i] += crm_mul<T>(index, &v[(jid-1)*6], &vJ[jid*6]);
            }
            __syncthreads();
            


            // Compute Sd = crm(v) @ S & psid = crm(v[parent]) @ S
            // For base, v[parent] = 0
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 2*6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 6) % NUM_JOINTS;
                int index = i % 6;
                if (i < 6*NUM_JOINTS) Sd[i] = crm_mul<T>(index, &v[jid*6], &S[jid*6]);
                else {
                    if ((jid-1) == -1) psid[jid*6 + index] = 0;
                    else psid[i - 6 * NUM_JOINTS] = crm_mul<T>(index, &v[(jid-1)*6], &S[jid*6]);
                }
            }
            __syncthreads();
            


            // Compute a = a[parent] + aJ
            #pragma unroll
            for (int jid = 0; jid < NUM_JOINTS; ++jid) {
                for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 6; i += blockDim.x*blockDim.y){
                    if ((jid-1) == -1) a[jid*6+ i] = aJ[jid*6 + i] + gravity * (i == 5); // Base joint's parent is the world
                    else a[jid*6 + i] = a[(jid-1)*6 + i] + aJ[jid*6 + i];
                }
                __syncthreads();
            }
            


            // Initialize a_world
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 6; i += blockDim.x*blockDim.y){
                if (i < 5) a_world[i] = 0;
                else a_world[5] = gravity;
            }
            __syncthreads();
            


            // Compute psidd = crm(a[parent])@S + crm(v[:,i])@psid[:,i] & IC @ v (for BC) in parallel
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 2*6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 6) % NUM_JOINTS;
                int index = i % 6;
                if (i < 6*NUM_JOINTS) {
                    if ((jid-1) == -1) psidd[i] = crm_mul<T>(index, a_world, &S[jid*6]);
                    else psidd[i] = crm_mul<T>(index, &a[(jid-1)*6], &S[jid*6]) + crm_mul<T>(index, &v[(jid-1)*6], &psid[jid*6]);
                }
                else IC_v[i - 6*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&IC[index + jid*36], &v[jid*6]);
            }
            __syncthreads();
            


            // Need crm(v), crf(v) for BC computation
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 2*36*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 36) % NUM_JOINTS;
                int col = (i / 6) % 6;
                int row = i % 6;
                if (i < 36*NUM_JOINTS) crm_v[i] = crm<T>(i % 36, &v[jid*6]);
                else crf_v[(jid*36) + row*6 + col] = -crm<T>(i % 36, &v[jid*6]); // crf is negative tranpose of crm
            }
            __syncthreads();
            


            // Finish BC = crf(v) @ IC + icrf(IC @ v) - IC @ crm(v)
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 36*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = i / 36;
                int row = i % 6;
                int col_idx = (i / 6) * 6;
                BC[i] = dot_prod<T, 6, 6, 1>(&crf_v[jid*36 + row], &IC[col_idx]) +
                        icrf<T>(i % 36, &IC_v[jid*6]) -
                        dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_v[col_idx]);
            }
            __syncthreads();
            


            // Compute f = IC @ a + crf(v) @ IC @ v
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = i / 6;
                int row = i % 6;
                f[i] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &a[jid*6]) +
                        dot_prod<T, 6, 6, 1>(&crf_v[jid*36 + row], &IC_v[jid*6]);
            }
            __syncthreads();
            


            // Forward Pass Completed
            // Now compute the backward pass
            


            // Compute IC[parent] += IC[i], BC[parent] += BC[i], f[parent] += f[i]
            #pragma unroll
            for (int jid = NUM_JOINTS-1; jid > 0; --jid) {
                for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 36*2 + 6; i += blockDim.x*blockDim.y){
                    if ((jid-1) != -1) {
                        if (i < 36) IC[(jid-1)*36 + i] += IC[jid*36 + i];
                        else if (i < 36*2) BC[(jid-1)*36 + i - 36] += BC[jid*36 + i - 36];
                        else f[(jid-1)*6 + i - 36*2] += f[jid*6 + i - 36*2];
                    }
                }
                __syncthreads();
            }
            


            // Need crm(S), crf(S), IC@S, crm(psid), crf(psid), IC@psid for B computations & icrf(f), psid+Sd for T3,T4
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 5*36*NUM_JOINTS + 3*6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 36) % NUM_JOINTS;
                int jidMatmul = (i / 6) % NUM_JOINTS;
                int col = (i / 6) % 6;
                int row = i % 6;
                if (i < 36*NUM_JOINTS) crm_S[i] = crm<T>(i % 36, &S[jid*6]);
                else if (i < 2*36*NUM_JOINTS) crf_S[(jid*36) + row*6 + col] = -crm<T>(i % 36, &S[jid*6]); // crf is negative tranpose of crm
                else if (i < 3*36*NUM_JOINTS) crm_psid[jid*36 + col*6 + row] = crm<T>(i % 36, &psid[jid*6]);
                else if (i < 4*36*NUM_JOINTS) crf_psid[(jid*36) + row*6 + col] = -crm<T>(i % 36, &psid[jid*6]); // crf is negative tranpose of crm
                else if (i < 5*36*NUM_JOINTS) icrf_f[i - 4*36*NUM_JOINTS] = icrf<T>(i % 36, &f[jid*6]);
                else if (i < 5*36*NUM_JOINTS + 6*NUM_JOINTS) IC_S[i - 5*36*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&IC[row + jidMatmul*36], &S[jidMatmul*6]);
                else if (i < 5*36*NUM_JOINTS + 2*6*NUM_JOINTS) psid_Sd[i - 5*36*NUM_JOINTS - 6*NUM_JOINTS] = psid[i - 5*36*NUM_JOINTS - 6*NUM_JOINTS] + Sd[i - 5*36*NUM_JOINTS - 6*NUM_JOINTS];
                else IC_psid[i - 5*36*NUM_JOINTS - 2*6*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&IC[row + jidMatmul*36], &psid[jidMatmul*6]);
            }
            __syncthreads();
            


            // Finish B_IC_S, Start D2
            // B_IC_S = crf(S) @ IC + icrf(IC @ S) - IC @ crm(S)
            // D2 = crf(psid) @ IC + icrf(IC @ psid) - IC @ crm(psid)
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 2*36*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 36) % NUM_JOINTS;
                int row = i % 6;
                int col = (i / 6) % 6;
                if (i < 36*NUM_JOINTS) {
                    B_IC_S[i] = dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &IC[jid*36 + col*6]) + 
                                icrf<T>(i % 36, &IC_S[jid*6]) -
                                dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_S[jid*36 + col*6]);
                }
                else {
                    D2[i - 36*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&crf_psid[jid*36 + row], &IC[jid*36 + col*6]) + 
                                                    icrf<T>(i % 36, &IC_psid[jid*6]) -
                                                    dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_psid[jid*36 + col*6]);
                }
            }
            __syncthreads();
            


            // Compute T2 = -BC.T @ S
            // Compute T3 = BC @ psid + IC @ psidd + icrf(f) @ S
            // Compute T4 = BC @ S + IC @ (psid + Sd)
            // Compute IC.T @ S for D4
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 4*6*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 6) % NUM_JOINTS;
                int row = i % 6;
                if (i < 6*NUM_JOINTS) T2[i] = -dot_prod<T, 6, 1, 1>(&BC[jid*36 + row*6], &S[jid*6]);
                else if (i < 2*6*NUM_JOINTS) {
                    T3[i - 6*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &psid[jid*6]) +
                                        dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psidd[jid*6]) +
                                        dot_prod<T, 6, 6, 1>(&icrf_f[jid*36 + row], &S[jid*6]);
                }
                else if (i < 3*6*NUM_JOINTS) {
                    T4[i - 2*6*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &S[jid*6]) +
                                        dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psid_Sd[jid*6]);
                }
                else ICT_S[i - 3*6*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &S[jid*6]);
            }
            __syncthreads();
            


            // Compute D1, D2, D4, crf_S_IC
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 4*36*NUM_JOINTS; i += blockDim.x*blockDim.y){
                int jid = (i / 36) % NUM_JOINTS;
                int row = i % 6;
                int col = (i / 6) % 6;
                if (i < 36*NUM_JOINTS) {
                    D1[i] = dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &IC[jid*36 + col*6]) -
                            dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_S[jid*36 + col*6]);
                }
                else if (i < 2*36*NUM_JOINTS) {
                    D2[i - 36*NUM_JOINTS] += dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &BC[jid*36 + col*6]) -
                                            dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &crm_S[jid*36 + col*6]);
                }
                else if (i < 3*36*NUM_JOINTS) D4[i - 2*36*NUM_JOINTS] = icrf<T>(i % 36, &ICT_S[jid*6]);
                else crf_S_IC[i - 3*36*NUM_JOINTS] = dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &IC[jid*36 + col*6]);
            }
            __syncthreads();
            


            // Compute t1 = outer(S[j], psid[ancestor])
            // t1[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            static const int jids[] = { 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6 }; // Joints with ancestor at equivalent index of ancestors_j
            static const int ancestors_j[] = { 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0 }; // Joint or ancestor of joint at equivalent index of jids_a
            const int t_index_map[7][7] = {
                {  0, -1, -1, -1, -1, -1, -1 },
                {  2,  1, -1, -1, -1, -1, -1 },
                {  5,  4,  3, -1, -1, -1, -1 },
                {  9,  8,  7,  6, -1, -1, -1 },
                { 14, 13, 12, 11, 10, -1, -1 },
                { 20, 19, 18, 17, 16, 15, -1 },
                { 27, 26, 25, 24, 23, 22, 21 },
            };
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[jid*6], &psid[ancestor_j*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t1
            static const int jids_compute[] = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6 }; // Joints with ancestor at equivalent index of ancestors_j
            static const int ancestors_j_compute[] = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0 }; // Joint or ancestor of joint at equivalent index of jids
            static const int st[] = { 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6 }; // Subtree of joint at equivalent index of jids
            // d2tau_dvdq[child, joint, ancestor] = -np.dot(t1, D3[:, child])
            // d2tau_dq[joint, ancestor, child] = np.dot(t1, D2[:, child])
            // d2tau_dq[joint, child, ancestor] = -np.dot(t1, D2[:, child])
            // d2tau_dvdq[joint, child, ancestor] = np.dot(t1, D3[:, child])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 336; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (i < 84) d2tau_dvdq[st_j*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                else if (i < 168 && jid != st_j) d2tau_dq2[jid*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);
                else if (i < 252 && jid != st_j) d2tau_dq2[jid*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);
                else if (jid != st_j) d2tau_dvdq[jid*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
            }
            __syncthreads();
            


            // Compute t2 = outer(S[j], S[ancestor])
            // t2[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[jid*6], &S[ancestor_j*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t2
            // for ancestor d2tau_dqd[child, ancestor, joint] = -np.dot(t2, D3[child])
            // for joint d2tau_dqd[child, joint, joint] = -np.dot(t2, D1[child])
            // for child d2tau_dqd[joint, ancestor, child] = np.dot(t2, D3[child])
            // for ancestor d2tau_dqd[child, joint, ancestor] = -np.dot(t2, D3[child])
            // for child d2tau_dqd[joint, child, ancestor] = np.dot(t2, D3[child])
            // for child d2tau_dvdq[joint, ancestor, child] = np.dot(t2, D2[child])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 420; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (i < 84 && ancestor_j < jid) d2tau_dqd2[st_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + ancestor_j] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                else if (i < 84 && jid == ancestor_j) d2tau_dqd2[st_j*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
                else if (i < 168 && jid != st_j) d2tau_dqd2[jid*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                else if (i < 252 && ancestor_j < jid) d2tau_dqd2[st_j*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                else if (i < 336 && jid != st_j) d2tau_dqd2[jid*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                else if (i >= 336 && jid != st_j) d2tau_dvdq[jid*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);
            }
            __syncthreads();
            


            // Compute t3 = outer(psid[j], psid[ancestor])
            // t3[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&psid[jid*6], &psid[ancestor_j*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t3
            // for joint d2tau_dqd[child, joint, ancestor] = -np.dot(t3, D3[:, st_j])
            // for ancestor d2tau_dqd[child, ancestor, joint] = -np.dot(t3, D3[:, st_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 168; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (i < 84) d2tau_dq2[st_j*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                else if (ancestor_j < jid) d2tau_dq2[st_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + ancestor_j] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
            }
            __syncthreads();
            


            // Compute t4 = outer(S[j], psidd[ancestor])
            // t4[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[jid*6], &psidd[ancestor_j*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t4
            // for child d2tau_dq[dd, cc, succ_j] += np.dot(t4, D1[:, succ_j])
            // for child d2tau_dq[dd, succ_j, cc] += np.dot(t4, D1[:, succ_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 168; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (i < 84 && jid != st_j) d2tau_dq2[jid*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + ancestor_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
                else if (jid != st_j) d2tau_dq2[jid*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + st_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
            }
            __syncthreads();
            


            // Compute t5 = outer(S[j], (Sd+psid)[ancestor])
            // t5[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[jid*6], &psid_Sd[ancestor_j*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t5
            // for child d2tau_dvdq[dd, cc, succ_j] += np.dot(t5, D1[:, succ_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 84; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (st_j != jid) d2tau_dvdq[jid*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + ancestor_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
            }
            __syncthreads();
            


            // Compute t6 = outer(S[ancestor], psid[joint])
            // t6[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[ancestor_j*6], &psid[jid*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t6
            // for ancestor d2tau_dvdq[st_j, cc, dd] = -np.dot(t6, D3[:, st_j])
            // for ancestor d2tau_dq[cc, st_j, dd] = np.dot(t6, D2[:, st_j])
            // for ancestor d2tau_dvdq[cc, st_j, dd] = np.dot(t6, D3[:, st_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 252; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (ancestor_j < jid) {
                    if (i < 84) d2tau_dvdq[st_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + ancestor_j] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                    else if (i < 168) d2tau_dq2[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);
                    else d2tau_dvdq[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                }
            }
            __syncthreads();
            


            // Compute t7 = outer(S[ancestor], psidd[joint])
            // t7[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[ancestor_j*6], &psidd[jid*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t7
            // for ancestor d2tau_dq[cc, st_j, dd] += np.dot(t7, D1[:, st_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 84; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (ancestor_j < jid) d2tau_dq2[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
            }
            __syncthreads();
            


            // Compute t8 = outer(S[ancestor], S[joint])
            // t8[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[ancestor_j*6], &S[jid*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t8
            // for ancestor dM_dq[cc,st_j,dd] = t8.T @ D4[:, st_j]
            // for ancestor dM_dq[st_j,cc,dd] = t8.T @ D4[:, st_j]
            // for child dM_dq[cc, dd, succ_j] = np.dot(t8, D1[:, succ_j])
            // for child dM_dq[dd, cc, succ_j] = np.dot(t8, D1[:, succ_j])// for child & ancestor d2tau_dqd[cc, succ_j, dd] = np.dot(t8, D3[:, succ_j])
            // for child & ancestor d2tau_dqd[cc, dd, succ_j] = np.dot(t8, D3[:, succ_j])
            // for child & ancestor d2tau_dvdq[cc, dd, succ_j] = np.dot(t8, D2[:, succ_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 588; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (ancestor_j < jid) {
                    if (i < 84) dM_dq[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D4[st_j*36]);
                    else if (i < 168) dM_dq[st_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D4[st_j*36]);
                    if (st_j != jid) {
                        if (i < 252) d2tau_dqd2[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                        else if (i < 336) d2tau_dqd2[ancestor_j*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);
                        else if (i < 420) d2tau_dvdq[ancestor_j*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);
                    }
                }
                if (jid != st_j && i < 504) dM_dq[ancestor_j*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
                else if (jid != st_j) dM_dq[jid*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
            }
            __syncthreads();
            


            // Compute t9 = outer(S[ancestor], (Sd+psid)[joint])
            // t9[j][k] is stored at t[((j*(j+1)/2) + k)*36]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28*36; i += blockDim.x*blockDim.y){
                int jid = jids[i / 36];
                int ancestor_j = ancestors_j[i / 36];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                outerProduct<T>(&S[ancestor_j*6], &psid_Sd[jid*6], &t[t_idx], 6, 6, i%36);
            }
            __syncthreads();
            


            // Perform all computations with t9
            // for ancestor & child d2tau_dvdq[cc, dd, succ_j] += np.dot(t9, D1[:, succ_j])
            // for ancestor & child d2tau_dq[cc, dd, succ_j] = d2tau_dq[cc, succ_j, dd]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 168; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int t_idx = t_index_map[jid][ancestor_j]*36;
                if (i < 84 && ancestor_j < jid && st_j != jid) d2tau_dvdq[ancestor_j*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + jid] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);
                else if (ancestor_j < jid & st_j != jid) d2tau_dq2[ancestor_j*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + jid] = d2tau_dq2[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j];
            }
            __syncthreads();
            


            // Compute p1..p6 in parallel
            // p1 = self.crm(psid_c) @ S_d
            // p2 = self.crm(psidd[:, k]) @ S_d
            // p3 = self.crm(S_c) @ S_d
            // p4 = self.crm(Sd_c + psid_c) @ S_d - 2 * self.crm(psid_d) @ S_c
            // p5 = self.crm(S_d) @ S_c
            // p6 = IC_S[joint] @ crm(S[ancestor]) + S[ancestor] @ crf_S_IC[joint]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 1008; i += blockDim.x*blockDim.y){
                int index = i % 168;
                int jid = jids[index / 6];
                int ancestor_j = ancestors_j[index / 6];
                int p_idx = t_index_map[jid][ancestor_j]*6;
                if (i < 168) p1[p_idx + i % 6] = crm_mul<T>(i % 6, &psid[ancestor_j*6], &S[jid*6]);
                else if (i < 336) p2[p_idx + i % 6] = crm_mul<T>(i % 6, &psidd[ancestor_j*6], &S[jid*6]);
                else if (i < 504) p3[p_idx + i % 6] = crm_mul<T>(i % 6, &S[ancestor_j*6], &S[jid*6]);
                else if (i < 672) p4[p_idx + i % 6] = crm_mul<T>(i % 6, &psid_Sd[ancestor_j*6], &S[jid*6]) - 2 * crm_mul<T>(i % 6, &psid[jid*6], &S[ancestor_j*6]);
                else if (i < 840) p5[p_idx + i % 6] = crm_mul<T>(i % 6, &S[jid*6], &S[ancestor_j*6]);
                else p6[p_idx + i % 6] = dot_prod<T, 6, 1, 1>(&IC_S[jid*6], &crm_S[ancestor_j*36 + (i % 6)*6]) + dot_prod<T, 6, 1, 1>(&S[ancestor_j*6], &crf_S_IC[jid*36 + (i % 6)*6]);
            }
            __syncthreads();
            


            // Finish all computations with p1..p5
            // for joint d2tau_dq[st_j, dd, cc] += -np.dot(p1, T2[:, st_j]) + np.dot(p2, T1[:, st_j])
            // for ancestor d2tau_dq[st_j, cc, dd] += -np.dot(p1, T2[:, st_j]) + np.dot(p2, T1[:, st_j])
            // for ancestor d2tau_dvdq[st_j, cc, dd] += -np.dot(p3, T2[:, st_j]) + np.dot(p4, T1[:, st_j])
            // for ancestor d2tau_dq[cc, st_j, dd] -= np.dot(p5, T3[:, st_j])
            // for ancestor && child d2tau_dq[cc, dd, succ_j] -= np.dot(p5, T3[:, st_j])
            // for ancestor d2tau_dvdq[cc, st_j, dd] -= np.dot(p5, T4[:, st_j])
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 504; i += blockDim.x*blockDim.y){
                int index = i % 84;
                int jid = jids_compute[index];
                int ancestor_j = ancestors_j_compute[index];
                int st_j = st[index];
                int p_idx = t_index_map[jid][ancestor_j]*6;
                if (i < 84) d2tau_dq2[st_j*NUM_JOINTS*NUM_JOINTS + ancestor_j * NUM_JOINTS + jid] += -dot_prod<T, 6, 1, 1>(&p1[p_idx], &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(&p2[p_idx], &T1[st_j*6]);
                else if (ancestor_j < jid) {
                    if (i < 168) d2tau_dq2[st_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + ancestor_j] += -dot_prod<T, 6, 1, 1>(&p1[p_idx], &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(&p2[p_idx], &T1[st_j*6]);
                    else if (i < 252) d2tau_dvdq[st_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + ancestor_j] += -dot_prod<T, 6, 1, 1>(&p3[p_idx], &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(&p4[p_idx], &T1[st_j*6]);
                    else if (i < 336) d2tau_dq2[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] -= dot_prod<T, 6, 1, 1>(&p5[p_idx], &T3[st_j*6]);
                    else if (i < 420 && st_j != jid) d2tau_dq2[ancestor_j*NUM_JOINTS*NUM_JOINTS + st_j * NUM_JOINTS + jid] -= dot_prod<T, 6, 1, 1>(&p5[p_idx], &T3[st_j*6]);
                    else if (i >= 420) d2tau_dvdq[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + st_j] -= dot_prod<T, 6, 1, 1>(&p5[p_idx], &T4[st_j*6]);
                }
            }
            __syncthreads();
            


            // Finish computation with p6
            // d2tau_dqd[ancestor, joint, joint] = p6[joint][ancestor] @ S[joint]
            for(int i = threadIdx.x + threadIdx.y*blockDim.x; i < 28; i += blockDim.x*blockDim.y){
                int jid = jids[i];
                int ancestor_j = ancestors_j[i];
                int p_idx = t_index_map[jid][ancestor_j]*6;
                if (ancestor_j < jid) d2tau_dqd2[ancestor_j*NUM_JOINTS*NUM_JOINTS + jid * NUM_JOINTS + jid] = dot_prod<T, 6, 1, 1>(&p6[p_idx], &S[jid*6]);
            }
            __syncthreads();
        }

        /**
         * Computes the second order derivatives of inverse dynamics
         *
         * @param d_idsva_so is a pointer to memory for the final result of size 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS = 1372
         * @param d_q_dq_u is the vector of joint positions, velocities, and accelerations
         * @param stride_q_qd_u is the stide between each q, qd, u
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         */
        template <typename T>
        __global__
        void idsva_so_kernel_single_timing(T *d_idsva_so, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
            __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_qdd = &s_q_qd_u[14];
            __shared__ T s_idsva_so[1372];
            __shared__ T s_XImats[504];
            __shared__ T s_temp[3744];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                idsva_so_inner<T>(s_idsva_so, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            }
            __syncthreads();
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
                d_idsva_so[ind] = s_idsva_so[ind];
            }
            __syncthreads();
        }

        /**
         * Computes the second order derivatives of inverse dynamics
         *
         * @param d_idsva_so is a pointer to memory for the final result of size 4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS = 1372
         * @param d_q_dq_u is the vector of joint positions, velocities, and accelerations
         * @param stride_q_qd_u is the stide between each q, qd, u
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         */
        template <typename T>
        __global__
        void idsva_so_kernel(T *d_idsva_so, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
            __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_qdd = &s_q_qd_u[14];
            __shared__ T s_idsva_so[1372];
            __shared__ T s_XImats[504];
            __shared__ T s_temp[3744];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                idsva_so_inner<T>(s_idsva_so, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
                __syncthreads();
                // save down to global
                T *d_idsva_so_k = &d_idsva_so[k*1372];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
                    d_idsva_so_k[ind] = s_idsva_so[ind];
                }
                __syncthreads();
            }
        }

        /**
         * Compute IDSVA-SO (Inverse Dynamics - Spatial Vector Algebra - Second Order)
         *
         * @param hd_data is the packaged input and output pointers
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant,
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         * @param streams are pointers to CUDA streams for async memory transfers (if needed)
         */
        template <typename T>
        __host__
        void idsva_so_host(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
            int stride_q_qd = 3*NUM_JOINTS;
            // start code with memory transfer
            gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
            gpuErrchk(cudaDeviceSynchronize());
            // then call the kernel
            idsva_so_kernel<T><<<block_dimms,thread_dimms>>>(hd_data->d_idsva_so,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);
            // finally transfer the result back
            gpuErrchk(cudaMemcpy(hd_data->h_idsva_so,hd_data->d_idsva_so,4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
            gpuErrchk(cudaDeviceSynchronize());
        }

        /**
         * Compute IDSVA-SO (Inverse Dynamics - Spatial Vector Algebra - Second Order)
         *
         * @param hd_data is the packaged input and output pointers
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant,
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         * @param streams are pointers to CUDA streams for async memory transfers (if needed)
         */
        template <typename T>
        __host__
        void idsva_so_host_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
            int stride_q_qd = 3*NUM_JOINTS;
            // start code with memory transfer
            gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
            gpuErrchk(cudaDeviceSynchronize());
            // then call the kernel
            struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
            idsva_so_kernel_single_timing<T><<<block_dimms,thread_dimms>>>(hd_data->d_idsva_so,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);
            clock_gettime(CLOCK_MONOTONIC,&end);
            // finally transfer the result back
            gpuErrchk(cudaMemcpy(hd_data->h_idsva_so,hd_data->d_idsva_so,4*NUM_JOINTS*NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
            gpuErrchk(cudaDeviceSynchronize());
            printf("Single Call ID-SO %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
        }

        /**
         * Compute IDSVA-SO (Inverse Dynamics - Spatial Vector Algebra - Second Order)
         *
         * @param hd_data is the packaged input and output pointers
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant,
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         * @param streams are pointers to CUDA streams for async memory transfers (if needed)
         */
        template <typename T>
        __host__
        void idsva_so_host_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                           const dim3 block_dimms, const dim3 thread_dimms) {
            int stride_q_qd = 3*NUM_JOINTS;
            // then call the kernel
            idsva_so_kernel<T><<<block_dimms,thread_dimms>>>(hd_data->d_idsva_so,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);
        }

        /**
         * Second Order of Forward Dynamics with Spatial Vector Algebra
         *
         * Notes:
         *   Assumes works with IDSVA
         *
         * @param s_df2 are the second derivatives of forward dynamics WRT q,qd,tau
         * @param s_idsva_so are the second derivative tensors of inverse dynamics
         * @param s_Minv is the inverse mass matrix
         * @param s_df_du is the gradient of the forward dynamics
         * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
         * @param s_temp is the pointer to the shared memory needed of size: 3744
         * @param gravity is the gravity constant
         */
        template <typename T>
        __device__
        void fdsva_so_inner(T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_XImats, T *s_temp, const T gravity) {
            // Second Derivatives of Inverse Dynamics
            T *d2tau_dqdq = &s_idsva_so[0];
            T *d2tau_dvdv = &s_idsva_so[343];
            T *d2tau_dvdq = &s_idsva_so[686];
            T *dM_dq = &s_idsva_so[1029];
            


            // First Derivatives of Forward Dynamics
            T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[49];
            


            // Second Derivatives of Forward Dynamics
            T *d2a_dqdq = s_df2;
            T *d2a_dvdv = &s_df2[343];
            T *d2a_dvdq = &s_df2[686];
            T *d2a_dtdq = &s_df2[1029];
            


            // Temporary Variables
            T *inner_dq = s_temp; // Inner term for d2a_dqdq (d2tau_dqdq + dM_dq*da_dq + (dM_dq*da_dq)^R)
            T *inner_cross = inner_dq + 343; // Inner term for d2a_dvdq (dM_dq*Minv)
            T *inner_tau = inner_cross + 343; // Inner term for d2a_dtdq (d2tau_dvdq + dM_dq*da_dv)
            T *rot_dq = inner_tau + 343; // Rotated (dM_dq*da_dq)^R term used to compute inner_dq
            


            // Start inner term for d2a_dqdq & Fill out Minv
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                int i = ind / 49 % 7; int j = ind / 7 % 7; int k = ind % 7;
                if (ind < 343) {
                    inner_dq[ind] = dot_prod<T, 7, 1, 1>(&dM_dq[49*i + 7*k], &s_df_dq[7*j]);
                    rot_dq[i*49 + k*7 + j] = inner_dq[ind];
                }
                else if (k > j) s_Minv[j*7 + k] = s_Minv[k*7 + j];
            }
            __syncthreads();
            // Compute relevant inner subterms in parallel
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1029; ind += blockDim.x*blockDim.y){
                int i = ind / 49 % 7; int j = ind / 7 % 7; int k = ind % 7;
                if (ind < 343) inner_dq[ind] += rot_dq[ind] + d2tau_dqdq[ind]; // Started with dM_dq*da_dq
                else if (ind < 686) inner_cross[i*49 + k*7 + j] = dot_prod<T, 7, 1, 1>(&dM_dq[49*i + 7*k], &s_df_dqd[7*j]) + d2tau_dvdq[i*49 + k*7 + j];
                else inner_tau[i*49 + k*7 + j] = dot_prod<T, 7, 1, 1>(&dM_dq[49*i + 7*k], &s_Minv[7*j]);
            }
            __syncthreads();
            // Multiply by -Minv to finish algorithm
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
                int i = ind / 49 % 7; int j = ind / 7 % 7; int k = ind % 7;
                if (ind < 343) d2a_dqdq[i*49 + j + k*7] = -dot_prod<T, 7, 7, 49>(&s_Minv[i], &inner_dq[j + k*7]);
                else if (ind < 686) d2a_dvdq[i*49 + j + k*7] = -dot_prod<T, 7, 7, 49>(&s_Minv[i], &inner_cross[j + k*7]);
                else if (ind < 1029) d2a_dvdv[i*49 + j + k*7] = -dot_prod<T, 7, 7, 49>(&s_Minv[i], &d2tau_dvdv[j + k*7]);
                else d2a_dtdq[i*49 + j + k*7] = -dot_prod<T, 7, 7, 49>(&s_Minv[i], &inner_tau[j + k*7]);
            }
            __syncthreads();
        }

        /**
         * Compute the FDSVA_SO (Second Order of Forward Dyamics with Spacial Vector Algebra)
         *
         * @param s_df2 is the second derivatives of forward dynamics WRT q,qd,tau
         * @param s_df_du is a pointer to memory for the derivative of forward dynamics WRT q,qd of size 2*NUM_JOINTS*NUM_JOINTS = 98
         * @param s_q is the vector of joint positions
         * @param s_qd is the vector of joint velocities
         * @param s_u is the vector of joint control inputs
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant
         */
        template <typename T>
        __device__
        void fdsva_so_device(T *s_df2, T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
            __shared__ T s_Minv[49];
            __shared__ T s_qdd[7];
            __shared__ T s_idsva_so[1372];
            __shared__ T s_XImats[504];
            __shared__ T s_temp[3744];
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
            __syncthreads();
            forward_dynamics_gradient_device(s_df_du, s_q, s_qd, s_u, d_robotModel, gravity);
            idsva_so_inner<T>(s_idsva_so, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            fdsva_so_inner<T>(s_df2, s_idsva_so, s_Minv, s_df_du, s_XImats, s_temp, gravity);
        }

        /**
         * Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)
         *
         * @param d_df2 is the second derivatives of forward dynamics WRT q,qd,tau
         * @param d_q_qd_u is the vector of joint positions, velocities, torques
         * @param stride_q_qd_u is the stride between each q, qd, qdd
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         */
        template <typename T>
        __global__
        void fdsva_so_kernel_single_timing(T *d_df2, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
            __shared__ T s_q_qd_u[4*7]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[2 * 7];
            __shared__ T s_Minv[49];
            __shared__ T s_qdd[7];
            __shared__ T s_df_du[98];
            __shared__ T s_idsva_so[1372];
            __shared__ T s_df2[1372];
            __shared__ T s_XImats[504];
            __shared__ T s_temp[3744];
            // load to shared mem
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u[ind];
            }
            __syncthreads();
            // compute with NUM_TIMESTEPS as NUM_REPS for timing
            for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
                load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
                forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
                __syncthreads();
                forward_dynamics_gradient_device(s_df_du, s_q, s_qd, s_u, d_robotModel, gravity);
                idsva_so_inner<T>(s_idsva_so, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
                fdsva_so_inner<T>(s_df2, s_idsva_so, s_Minv, s_df_du, s_XImats, s_temp, gravity);
            }
            // save down to global
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
                d_df2[ind] = s_df2[ind];
            }
            __syncthreads();
        }

        /**
         * Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)
         *
         * @param d_df2 is the second derivatives of forward dynamics WRT q,qd,tau
         * @param d_q_qd_u is the vector of joint positions, velocities, torques
         * @param stride_q_qd_u is the stride between each q, qd, qdd
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         */
        template <typename T>
        __global__
        void fdsva_so_kernel(T *d_df2, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
            __shared__ T s_q_qd_u[4*7]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[2 * 7];
            __shared__ T s_Minv[49];
            __shared__ T s_qdd[7];
            __shared__ T s_df_du[98];
            __shared__ T s_idsva_so[1372];
            __shared__ T s_df2[1372];
            __shared__ T s_XImats[504];
            __shared__ T s_temp[3744];
            for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
                // load to shared mem
                const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                    s_q_qd_u[ind] = d_q_qd_u_k[ind];
                }
                __syncthreads();
                // compute
                load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
                direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
                forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
                __syncthreads();
                forward_dynamics_gradient_device(s_df_du, s_q, s_qd, s_u, d_robotModel, gravity);
                idsva_so_inner<T>(s_idsva_so, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
                fdsva_so_inner<T>(s_df2, s_idsva_so, s_Minv, s_df_du, s_XImats, s_temp, gravity);
                __syncthreads();
                // save down to global
                T *d_df2_k = &d_df2[k*1372];
                for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
                    d_df2_k[ind] = s_df2[ind];
                }
                __syncthreads();
            }
        }

        /**
         * Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)
         *
         * @param hd_data is the packaged input and output pointers
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant,
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         * @param streams are pointers to CUDA streams for async memory transfers (if needed)
         */
        template <typename T>
        __host__
        void fdsva_so(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                              const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
            int stride_q_qd_qdd = 3*NUM_JOINTS;
            // start code with memory transfer
            gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_qdd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
            gpuErrchk(cudaDeviceSynchronize());
            // call the kernel
            fdsva_so_kernel<T><<<block_dimms,thread_dimms,FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df2,hd_data->d_q_qd_u,stride_q_qd_qdd,d_robotModel,gravity,num_timesteps);
            gpuErrchk(cudaDeviceSynchronize());
            // finally transfer the result back
            gpuErrchk(cudaMemcpy(hd_data->h_df2,hd_data->d_df2,num_timesteps*1372*sizeof(T),cudaMemcpyDeviceToHost));
            gpuErrchk(cudaDeviceSynchronize());
        }

        /**
         * Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)
         *
         * @param hd_data is the packaged input and output pointers
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant,
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         * @param streams are pointers to CUDA streams for async memory transfers (if needed)
         */
        template <typename T>
        __host__
        void fdsva_so_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
            int stride_q_qd_qdd = 3*NUM_JOINTS;
            // start code with memory transfer
            gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_qdd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
            gpuErrchk(cudaDeviceSynchronize());
            // call the kernel
            struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
            fdsva_so_kernel_single_timing<T><<<block_dimms,thread_dimms,FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df2,hd_data->d_q_qd_u,stride_q_qd_qdd,d_robotModel,gravity,num_timesteps);
            gpuErrchk(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC,&end);
            // finally transfer the result back
            gpuErrchk(cudaMemcpy(hd_data->h_df2,hd_data->d_df2,1372*sizeof(T),cudaMemcpyDeviceToHost));
            gpuErrchk(cudaDeviceSynchronize());
            printf("Single Call FDSVA_SO %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
        }

        /**
         * Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)
         *
         * @param hd_data is the packaged input and output pointers
         * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
         * @param gravity is the gravity constant,
         * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
         * @param streams are pointers to CUDA streams for async memory transfers (if needed)
         */
        template <typename T>
        __host__
        void fdsva_so_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                           const dim3 block_dimms, const dim3 thread_dimms) {
            int stride_q_qd_qdd = 3*NUM_JOINTS;
            // call the kernel
            fdsva_so_kernel<T><<<block_dimms,thread_dimms,FDSVA_SO_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df2,hd_data->d_q_qd_u,stride_q_qd_qdd,d_robotModel,gravity,num_timesteps);
            gpuErrchk(cudaDeviceSynchronize());
        }

        /**
         * Sets shared mem needed for gradient kernels and initializes streams for host functions
         *
         * @return A pointer to the array of streams
         */
        template <typename T>
        __host__
        cudaStream_t *init_grid(){
            // set the max temp memory for the gradient kernels to account for large robots
            auto id_grad_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
            auto id_grad_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
            auto id_grad_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
            auto id_grad_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
            auto fd_grad_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
            auto fd_grad_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
            auto fd_grad_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
            auto fd_grad_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
            cudaFuncSetAttribute(id_grad_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(id_grad_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(id_grad_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(id_grad_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(fd_grad_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(fd_grad_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(fd_grad_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            cudaFuncSetAttribute(fd_grad_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
            gpuErrchk(cudaDeviceSynchronize());
            // allocate streams
            cudaStream_t *streams = (cudaStream_t *)malloc(3*sizeof(cudaStream_t));
            int priority, minPriority, maxPriority;
            gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
            for(int i=0; i<3; i++){
                int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;
                gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));
            }
            return streams;
        }

        /**
         * Frees the memory used by grid
         *
         * @param streams allocated by init_grid
         * @param robotModel allocated by init_robotModel
         * @param data allocated by init_gridData
         */
        template <typename T>
        __host__
        void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data){
            gpuErrchk(cudaFree(d_robotModel));
            gpuErrchk(cudaFree(hd_data->d_q_qd_u)); gpuErrchk(cudaFree(hd_data->d_q_qd)); gpuErrchk(cudaFree(hd_data->d_q));
            gpuErrchk(cudaFree(hd_data->d_c)); gpuErrchk(cudaFree(hd_data->d_Minv)); gpuErrchk(cudaFree(hd_data->d_qdd));
            gpuErrchk(cudaFree(hd_data->d_dc_du)); gpuErrchk(cudaFree(hd_data->d_df_du));
            gpuErrchk(cudaFree(hd_data->d_eePos)); gpuErrchk(cudaFree(hd_data->d_deePos));
            gpuErrchk(cudaFree(hd_data->d_idsva_so));
            gpuErrchk(cudaFree(hd_data->d_df2));
            free(hd_data->h_idsva_so); free(hd_data->h_df2);
            free(hd_data->h_q_qd_u); free(hd_data->h_q_qd); free(hd_data->h_q);
            free(hd_data->h_c); free(hd_data->h_Minv); free(hd_data->h_qdd);
            free(hd_data->h_dc_du); free(hd_data->h_df_du);
            free(hd_data->h_eePos); free(hd_data->h_deePos);
            for(int i=0; i<3; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);
        }

    }
