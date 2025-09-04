/**
 * This instance of grid.cuh is optimized for the urdf: iiwa14
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
 *       __device__ end_effector_positions_inner<T>(T *s_eePos, const T *s_q, const T *s_Xhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_positions_device<T>(T *s_eePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_positions_kernel<T>(T *d_eePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_positions<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ end_effector_positions_gradient_inner<T>(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, int *s_topology_helpers, T *s_temp)
 *       __device__ end_effector_positions_gradient_device<T>(T *s_deePos, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ end_effector_positions_gradient_kernel<T>(T *d_deePos, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   end_effector_positions_gradient<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *   Suggested Type T is float
 *   
 *   Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file
 *   
 *   By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [ID, MINV, FD, ID_DU, FD_DU]
 *
 */
#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "utils/cuda_utils.cuh"
// single kernel timing helper code

/**
 * All functions are kept in this namespace
 *
 */
namespace grid {
    const int NUM_JOINTS = 7;
    const int EE_POS_SIZE = 6;
    const int NUM_EES = 1;
    const int ID_DYNAMIC_SHARED_MEM_COUNT = 770;
    const int MINV_DYNAMIC_SHARED_MEM_COUNT = 1395;
    const int FD_DYNAMIC_SHARED_MEM_COUNT = 1444;
    const int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 2450;
    const int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 2450;
    const int ID_DU_MAX_SHARED_MEM_COUNT = 2695;
    const int FD_DU_MAX_SHARED_MEM_COUNT = 2849;
    const int EE_POS_SHARED_MEM_COUNT = 144;
    const int DEE_POS_SHARED_MEM_COUNT = 448;
    const int SUGGESTED_THREADS = 352;
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
        T *d_dc_du;
        T *d_df_du;
        T *d_eePos;
        T *d_deePos;
        // CPU OUTPUTS
        T *h_c;
        T *h_Minv;
        T *h_qdd;
        T *h_dc_du;
        T *h_df_du;
        T *h_eePos;
        T *h_deePos;
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
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_eePos, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_deePos, 6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_eePos = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_deePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
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
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_eePos, 6*NUM_EES*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_deePos, 6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_eePos = (T *)malloc(6*NUM_EES*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_deePos = (T *)malloc(6*NUM_EES*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
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
            s_XImats[36] = static_cast<T>(-s_temp[8]);
            s_XImats[37] = static_cast<T>(s_temp[1]);
            s_XImats[45] = static_cast<T>(-0.2025*s_temp[8]);
            s_XImats[46] = static_cast<T>(0.2025*s_temp[1]);
            s_XImats[48] = static_cast<T>(s_temp[1]);
            s_XImats[49] = static_cast<T>(s_temp[8]);
            // X[2]
            s_XImats[72] = static_cast<T>(-s_temp[9]);
            s_XImats[73] = static_cast<T>(s_temp[2]);
            s_XImats[75] = static_cast<T>(0.2045*s_temp[2]);
            s_XImats[76] = static_cast<T>(0.2045*s_temp[9]);
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
            s_XImats[144] = static_cast<T>(-s_temp[11]);
            s_XImats[145] = static_cast<T>(s_temp[4]);
            s_XImats[147] = static_cast<T>(0.1845*s_temp[4]);
            s_XImats[148] = static_cast<T>(0.1845*s_temp[11]);
            s_XImats[156] = static_cast<T>(s_temp[4]);
            s_XImats[157] = static_cast<T>(s_temp[11]);
            s_XImats[159] = static_cast<T>(0.1845*s_temp[11]);
            s_XImats[160] = static_cast<T>(-0.1845*s_temp[4]);
            // X[5]
            s_XImats[180] = static_cast<T>(s_temp[12]);
            s_XImats[181] = static_cast<T>(-s_temp[5]);
            s_XImats[189] = static_cast<T>(0.2155*s_temp[12]);
            s_XImats[190] = static_cast<T>(-0.2155*s_temp[5]);
            s_XImats[192] = static_cast<T>(s_temp[5]);
            s_XImats[193] = static_cast<T>(s_temp[12]);
            // X[6]
            s_XImats[216] = static_cast<T>(-s_temp[13]);
            s_XImats[217] = static_cast<T>(s_temp[6]);
            s_XImats[219] = static_cast<T>(0.081*s_temp[6]);
            s_XImats[220] = static_cast<T>(0.081*s_temp[13]);
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
            s_XmatsHom[16] = static_cast<T>(-s_temp[8]);
            s_XmatsHom[18] = static_cast<T>(s_temp[1]);
            s_XmatsHom[20] = static_cast<T>(s_temp[1]);
            s_XmatsHom[22] = static_cast<T>(s_temp[8]);
            // X_hom[2]
            s_XmatsHom[32] = static_cast<T>(-s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[36] = static_cast<T>(s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(-s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[12]);
            s_XmatsHom[82] = static_cast<T>(s_temp[5]);
            s_XmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_XmatsHom[86] = static_cast<T>(s_temp[12]);
            // X_hom[6]
            s_XmatsHom[96] = static_cast<T>(-s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[100] = static_cast<T>(s_temp[6]);
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
            s_XmatsHom[16] = static_cast<T>(-s_temp[8]);
            s_XmatsHom[18] = static_cast<T>(s_temp[1]);
            s_XmatsHom[20] = static_cast<T>(s_temp[1]);
            s_XmatsHom[22] = static_cast<T>(s_temp[8]);
            // X_hom[2]
            s_XmatsHom[32] = static_cast<T>(-s_temp[9]);
            s_XmatsHom[34] = static_cast<T>(s_temp[2]);
            s_XmatsHom[36] = static_cast<T>(s_temp[2]);
            s_XmatsHom[38] = static_cast<T>(s_temp[9]);
            // X_hom[3]
            s_XmatsHom[48] = static_cast<T>(s_temp[10]);
            s_XmatsHom[50] = static_cast<T>(s_temp[3]);
            s_XmatsHom[52] = static_cast<T>(-s_temp[3]);
            s_XmatsHom[54] = static_cast<T>(s_temp[10]);
            // X_hom[4]
            s_XmatsHom[64] = static_cast<T>(-s_temp[11]);
            s_XmatsHom[66] = static_cast<T>(s_temp[4]);
            s_XmatsHom[68] = static_cast<T>(s_temp[4]);
            s_XmatsHom[70] = static_cast<T>(s_temp[11]);
            // X_hom[5]
            s_XmatsHom[80] = static_cast<T>(s_temp[12]);
            s_XmatsHom[82] = static_cast<T>(s_temp[5]);
            s_XmatsHom[84] = static_cast<T>(-s_temp[5]);
            s_XmatsHom[86] = static_cast<T>(s_temp[12]);
            // X_hom[6]
            s_XmatsHom[96] = static_cast<T>(-s_temp[13]);
            s_XmatsHom[98] = static_cast<T>(s_temp[6]);
            s_XmatsHom[100] = static_cast<T>(s_temp[6]);
            s_XmatsHom[102] = static_cast<T>(s_temp[13]);
            // dX_hom[0]
            s_dXmatsHom[0] = static_cast<T>(-s_temp[0]);
            s_dXmatsHom[1] = static_cast<T>(s_temp[7]);
            s_dXmatsHom[4] = static_cast<T>(-s_temp[7]);
            s_dXmatsHom[5] = static_cast<T>(-s_temp[0]);
            // dX_hom[1]
            s_dXmatsHom[16] = static_cast<T>(s_temp[1]);
            s_dXmatsHom[18] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[20] = static_cast<T>(s_temp[8]);
            s_dXmatsHom[22] = static_cast<T>(-s_temp[1]);
            // dX_hom[2]
            s_dXmatsHom[32] = static_cast<T>(s_temp[2]);
            s_dXmatsHom[34] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[36] = static_cast<T>(s_temp[9]);
            s_dXmatsHom[38] = static_cast<T>(-s_temp[2]);
            // dX_hom[3]
            s_dXmatsHom[48] = static_cast<T>(-s_temp[3]);
            s_dXmatsHom[50] = static_cast<T>(s_temp[10]);
            s_dXmatsHom[52] = static_cast<T>(-s_temp[10]);
            s_dXmatsHom[54] = static_cast<T>(-s_temp[3]);
            // dX_hom[4]
            s_dXmatsHom[64] = static_cast<T>(s_temp[4]);
            s_dXmatsHom[66] = static_cast<T>(s_temp[11]);
            s_dXmatsHom[68] = static_cast<T>(s_temp[11]);
            s_dXmatsHom[70] = static_cast<T>(-s_temp[4]);
            // dX_hom[5]
            s_dXmatsHom[80] = static_cast<T>(-s_temp[5]);
            s_dXmatsHom[82] = static_cast<T>(s_temp[12]);
            s_dXmatsHom[84] = static_cast<T>(-s_temp[12]);
            s_dXmatsHom[86] = static_cast<T>(-s_temp[5]);
            // dX_hom[6]
            s_dXmatsHom[96] = static_cast<T>(s_temp[6]);
            s_dXmatsHom[98] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[100] = static_cast<T>(s_temp[13]);
            s_dXmatsHom[102] = static_cast<T>(-s_temp[6]);
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
    void end_effector_positions_inner(T *s_eePos, const T *s_q, const T *s_Xhom, T *s_temp) {
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
    void end_effector_positions_device(T *s_eePos, const T *s_q, T *s_temp_in, const robotModel<T> *d_robotModel) {
        T *s_XHomTemp = s_temp_in; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
        end_effector_positions_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
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

    /**
     * Compute the End Effector Position
     *
     * @param d_eePos is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */

    /**
     * Compute the End Effector Positions
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the End Effector Positions
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the End Effector Positions
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * Notes:
     *   Assumes the Xhom and dXhom matricies have already been updated for the given q
     *
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param s_Xhom is the pointer to the homogenous transformation matricies 
     * @param s_dXhom is the pointer to the gradient of the homogenous transformation matricies 
     * @param s_temp is a pointer to helper shared memory of size 224
     */
    template <typename T>
    __device__
    void end_effector_positions_gradient_inner(T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, T *s_temp) {
        //
        // For each branch/gradient in parallel chain up the transform
        // Keep chaining until reaching the root (starting from the leaves)
        //
        // Serial chain manipulator so optimize as parent is jid-1
        // First set to leaf transform
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int eeIndStart = 16*6;
            s_temp[ind] = (djid == 6) ? s_dXhom[eeIndStart + rc] : s_Xhom[eeIndStart + rc];
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 1/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom = ((djid == 5) ? s_dXhom : s_Xhom);
            s_temp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*5 + row], &s_temp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 2/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom = ((djid == 4) ? s_dXhom : s_Xhom);
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*4 + row], &s_temp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 3/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom = ((djid == 3) ? s_dXhom : s_Xhom);
            s_temp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*3 + row], &s_temp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 4/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom = ((djid == 2) ? s_dXhom : s_Xhom);
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*2 + row], &s_temp[112 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 5/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom = ((djid == 1) ? s_dXhom : s_Xhom);
            s_temp[ind + 112] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*1 + row], &s_temp[0 + colInd]);
        }
        __syncthreads();
        // Serial chain manipulator so optimize as parent is jid-1
        // Update with parent transform until you reach the base [level 6/6]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 112; ind += blockDim.x*blockDim.y){
            int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;
            const T *s_Xhom_dXhom = ((djid == 0) ? s_dXhom : s_Xhom);
            s_temp[ind + 0] = dot_prod<T,4,4,1>(&s_Xhom_dXhom[16*0 + row], &s_temp[112 + colInd]);
        }
        __syncthreads();
        //
        // Now extract the eePos from the Tansforms
        // TODO: ADD OFFSETS
        //
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
            // xyz is easy
            int xyzInd = ind % 3; int deeInd = ind / 3; T *s_Xmat_hom = &s_temp[0 + 16*deeInd];
            s_deePos[6*deeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];
            // roll pitch yaw is a bit more difficult
            //
            //
            // TODO THESE ARE WRONG BECUASE THERE IS CHAIN RULE HERE
            //
            //
            if(xyzInd > 0){continue;}
            s_deePos[6*deeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);
            s_deePos[6*deeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));
            s_deePos[6*deeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);
        }
        __syncthreads();
    }

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = 7 and NUM_EE = 1
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void end_effector_positions_gradient_device(T *s_deePos, const T *s_q, T *s_temp_in, const robotModel<T> *d_robotModel) {
        T *s_XHomTemp = s_temp_in; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
        end_effector_positions_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
    }

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

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
        //     joints are: iiwa_joint_1
        //     links are: iiwa_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[84 + 6*jid + 2];
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
        //     joints are: iiwa_joint_1
        //     links are: iiwa_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 7; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[84 + 6*jid + 2];
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
        //     joints are: iiwa_joint_1
        //     links are: iiwa_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[42 + jid6 + 2] += s_qdd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_1
        //     links are: iiwa_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[42 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row], &s_vaf[84 + 6*6]);
            int dstOffset = 84 + 6*5 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[84 + 6*5]);
            int dstOffset = 84 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[84 + 6*4]);
            int dstOffset = 84 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[84 + 6*3]);
            int dstOffset = 84 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[84 + 6*2]);
            int dstOffset = 84 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_1
        //     links are: iiwa_link_1
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

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Finish the forward dynamics computation with qdd = Minv*(u-c)
     *
     * Notes:
     *   Assumes s_Minv and s_c are already computed
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
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_temp is the pointer to the shared memory needed of size: 716
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

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

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
            int jid = col / 4; int selector = col % 4; int jid6 = 6*jid;
            // branch to get pointer locations
            int dstOffset; const T * src;
                 if (selector == 0){ dstOffset = 1512; src = &s_temp[1260]; }
            else if (selector == 1){ dstOffset = 1554; src = &s_temp[1302]; }
            else if (selector == 2){ dstOffset = 1596; src = &s_vaf[0]; }
            else              { dstOffset = 1638; src = &s_vaf[84]; }
            mx2<T>(&s_temp[dstOffset + jid6], &src[jid6]);
        }
        __syncthreads();
        //
        // Forward Pass
        //
        // We start with dv/du noting that we only have values
        //    for ancestors and for the current index else 0
        // dv/du where bfs_level is 0
        //     joints are: iiwa_joint_1
        //     links are: iiwa_link_1
        // when parent is base dv_dq = 0, dv_dqd = S
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int dq_flag = (ind / 6) == 0;
            int du_offset = dq_flag ? 0 : 168;
            s_temp[du_offset + 6*0 + row] = (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
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
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
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
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
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
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
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
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
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
        //     joints are: iiwa_joint_7
        //     links are: iiwa_link_7
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 6) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*35 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*6 + 6*row],&s_temp[du_col_offset + 6*42])
                          + dq_flag * (col_du == 6) * s_temp[1512 + 6*6 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 5
        //     joints are: iiwa_joint_6
        //     links are: iiwa_link_6
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 5) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*28 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row],&s_temp[du_col_offset + 6*35])
                          + dq_flag * (col_du == 5) * s_temp[1512 + 6*5 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 4
        //     joints are: iiwa_joint_5
        //     links are: iiwa_link_5
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 4) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*21 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row],&s_temp[du_col_offset + 6*28])
                          + dq_flag * (col_du == 4) * s_temp[1512 + 6*4 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 3
        //     joints are: iiwa_joint_4
        //     links are: iiwa_link_4
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 3) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*14 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row],&s_temp[du_col_offset + 6*21])
                          + dq_flag * (col_du == 3) * s_temp[1512 + 6*3 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 2
        //     joints are: iiwa_joint_3
        //     links are: iiwa_link_3
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 2) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*7 + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row],&s_temp[du_col_offset + 6*14])
                          + dq_flag * (col_du == 2) * s_temp[1512 + 6*2 + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 1
        //     joints are: iiwa_joint_2
        //     links are: iiwa_link_2
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 7;
            int dq_flag = col == col_du;
            int dst_adjust = (col_du >= 1) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 672 + !dq_flag * 966 + 6*col_du;
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

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

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

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */

    /**
     * Sets shared mem needed for gradient kernels and initializes streams for host functions
     *
     * @return A pointer to the array of streams
     */
    template <typename T>
    __host__
    cudaStream_t *init_grid(){
        // set the max temp memory for the gradient kernels to account for large robots
        auto id_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
        auto id_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
        auto id_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
        auto id_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
        auto fd_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
        auto fd_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
        auto fd_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
        auto fd_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
        cudaFuncSetAttribute(id_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
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
    void free_robotModel(robotModel<T> *d_robotModel){
        gpuErrchk(cudaFree(d_robotModel));
    }

}
