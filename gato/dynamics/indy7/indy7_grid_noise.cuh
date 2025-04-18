#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "utils/cuda_utils.cuh"
#include "settings.h"



namespace grid {

    __device__
    constexpr uint32_t KNOTS_F_EXT() {return static_cast<uint32_t>(sqp::F_EXT_KNOTS);}

    /**
    * Compute the RNEA (Recursive Newton-Euler Algorithm)
    *
    * Notes:
    *   Assumes the XI matricies have already been updated for the given q
    *   optimized for qdd = 0
    *
    * @param s_c is the vector of output torques
    * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 108
    * @param s_q is the vector of joint positions
    * @param s_qd is the vector of joint velocities
    * @param s_XI is the pointer to the transformation and inertia matricies 
    * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
    * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 36
    * @param gravity is the gravity constant
    * @param d_f_ext is the vector of external forces and torques
    */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, T *s_temp, const T gravity, T *d_f_ext) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: joint0
        //     links are: link1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
            int jid6 = 6*0;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[36 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[0];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: joint1
        //     links are: link2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 36; int jid6 = 6 * 1;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[1]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[42], &s_vaf[6], s_qd[1]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: joint2
        //     links are: link3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 36; int jid6 = 6 * 2;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[2]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[12], s_qd[2]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: joint3
        //     links are: link4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 36; int jid6 = 6 * 3;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[3]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[18], s_qd[3]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: joint4
        //     links are: link5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 36; int jid6 = 6 * 4;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[4]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[24], s_qd[4]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: joint5
        //     links are: link6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
            int vaOffset = !vFlag * 36; int jid6 = 6 * 5;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[5]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[30], s_qd[5]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 6;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 36 + jid6;
            T *dst = IaFlag ? &s_vaf[72] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[216 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 6; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[72 + jid6], &s_vaf[jid6], &s_temp[jid6]);

            if (jid == 5) {
                // Add external wrench
                //if (blockIdx.x < KNOTS_F_EXT()){
                s_vaf[72 + jid6] -= d_f_ext[0];
                s_vaf[72 + jid6 + 1] -= d_f_ext[1];
                s_vaf[72 + jid6 + 2] -= d_f_ext[2];
                s_vaf[72 + jid6 + 3] -= d_f_ext[3];
                s_vaf[72 + jid6 + 4] -= d_f_ext[4];
                s_vaf[72 + jid6 + 5] -= d_f_ext[5];
                //}
            }
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 5
        //     joints are: joint5
        //     links are: link6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[72 + 6*5]);
            int dstOffset = 72 + 6*4 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: joint4
        //     links are: link5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[72 + 6*4]);
            int dstOffset = 72 + 6*3 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: joint3
        //     links are: link4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[72 + 6*3]);
            int dstOffset = 72 + 6*2 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: joint2
        //     links are: link3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[72 + 6*2]);
            int dstOffset = 72 + 6*1 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: joint1
        //     links are: link2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[72 + 6*1]);
            int dstOffset = 72 + 6*0 + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 6; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[72 + 6*jid + 2];
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
        * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 108
        * @param s_q is the vector of joint positions
        * @param s_qd is the vector of joint velocities
        * @param s_qdd is (optional vector of joint accelerations
        * @param s_XI is the pointer to the transformation and inertia matricies 
        * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
        * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 36
        * @param gravity is the gravity constant
        */
        template <typename T>
        __device__
        void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, T *s_temp, const T gravity, T *d_f_ext) {
            //
            // Forward Pass
            //
            // s_v, s_a where parent is base
            //     joints are: joint0
            //     links are: link1
            // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
            for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 6; row += blockDim.x*blockDim.y){
                int jid6 = 6*0;
                s_vaf[jid6 + row] = static_cast<T>(0);
                s_vaf[36 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
                if (row == 2){s_vaf[jid6 + 2] += s_qd[0]; s_vaf[36 + jid6 + 2] += s_qdd[0];}
            }
            __syncthreads();
            // s_v and s_a where bfs_level is 1
            //     joints are: joint1
            //     links are: link2
            // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
                int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
                int vaOffset = !vFlag * 36; int jid6 = 6 * 1;
                T qd_qdd_val = (row == 2) * (vFlag * s_qd[1] + !vFlag * s_qdd[1]);
                // compute based on the branch and use bool multiply for no branch
                s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*0]) + qd_qdd_val;
            }
            // sync before a += MxS(v)*qd[S] 
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
                mx2_peq_scaled<T>(&s_vaf[42], &s_vaf[6], s_qd[1]);
            }
            __syncthreads();
            // s_v and s_a where bfs_level is 2
            //     joints are: joint2
            //     links are: link3
            // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
                int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
                int vaOffset = !vFlag * 36; int jid6 = 6 * 2;
                T qd_qdd_val = (row == 2) * (vFlag * s_qd[2] + !vFlag * s_qdd[2]);
                // compute based on the branch and use bool multiply for no branch
                s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*1]) + qd_qdd_val;
            }
            // sync before a += MxS(v)*qd[S] 
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
                mx2_peq_scaled<T>(&s_vaf[48], &s_vaf[12], s_qd[2]);
            }
            __syncthreads();
            // s_v and s_a where bfs_level is 3
            //     joints are: joint3
            //     links are: link4
            // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
                int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
                int vaOffset = !vFlag * 36; int jid6 = 6 * 3;
                T qd_qdd_val = (row == 2) * (vFlag * s_qd[3] + !vFlag * s_qdd[3]);
                // compute based on the branch and use bool multiply for no branch
                s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*2]) + qd_qdd_val;
            }
            // sync before a += MxS(v)*qd[S] 
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
                mx2_peq_scaled<T>(&s_vaf[54], &s_vaf[18], s_qd[3]);
            }
            __syncthreads();
            // s_v and s_a where bfs_level is 4
            //     joints are: joint4
            //     links are: link5
            // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
                int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
                int vaOffset = !vFlag * 36; int jid6 = 6 * 4;
                T qd_qdd_val = (row == 2) * (vFlag * s_qd[4] + !vFlag * s_qdd[4]);
                // compute based on the branch and use bool multiply for no branch
                s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*3]) + qd_qdd_val;
            }
            // sync before a += MxS(v)*qd[S] 
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
                mx2_peq_scaled<T>(&s_vaf[60], &s_vaf[24], s_qd[4]);
            }
            __syncthreads();
            // s_v and s_a where bfs_level is 5
            //     joints are: joint5
            //     links are: link6
            // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
                int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 1; int vFlag = comp == comp_mod;
                int vaOffset = !vFlag * 36; int jid6 = 6 * 5;
                T qd_qdd_val = (row == 2) * (vFlag * s_qd[5] + !vFlag * s_qdd[5]);
                // compute based on the branch and use bool multiply for no branch
                s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*4]) + qd_qdd_val;
            }
            // sync before a += MxS(v)*qd[S] 
            __syncthreads();
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
                mx2_peq_scaled<T>(&s_vaf[66], &s_vaf[30], s_qd[5]);
            }
            __syncthreads();
            //
            // s_f in parallel given all v, a
            //
            // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
            // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
                int row = ind % 6; int comp = ind / 6; int jid = comp % 6;
                bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 36 + jid6;
                T *dst = IaFlag ? &s_vaf[72] : s_temp;
                // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
                dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[216 + 6*jid6 + row], &s_vaf[vaOffset]);
            }
            __syncthreads();
            // finish with s_f[k] += fx(v[k])*Iv[k]
            for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 6; jid += blockDim.x*blockDim.y){
                int jid6 = 6*jid;
                fx_times_v_peq<T>(&s_vaf[72 + jid6], &s_vaf[jid6], &s_temp[jid6]);

                if (jid == 5) {
                    // Add external wrench
                    //if (blockIdx.x < KNOTS_F_EXT()){
                    s_vaf[72 + jid6] -= d_f_ext[0];
                    s_vaf[72 + jid6 + 1] -= d_f_ext[1];
                    s_vaf[72 + jid6 + 2] -= d_f_ext[2];
                    s_vaf[72 + jid6 + 3] -= d_f_ext[3];
                    s_vaf[72 + jid6 + 4] -= d_f_ext[4];
                    s_vaf[72 + jid6 + 5] -= d_f_ext[5];
                    //}
                }
            }
            __syncthreads();
            //
            // Backward Pass
            //
            // s_f update where bfs_level is 5
            //     joints are: joint5
            //     links are: link6
            // s_f[parent_k] += X[k]^T*f[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                int row = ind % 6;
                T val = dot_prod<T,6,1,1>(&s_XImats[36*5 + 6*row], &s_vaf[72 + 6*5]);
                int dstOffset = 72 + 6*4 + row;
                s_vaf[dstOffset] += val;
            }
            __syncthreads();
            // s_f update where bfs_level is 4
            //     joints are: joint4
            //     links are: link5
            // s_f[parent_k] += X[k]^T*f[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                int row = ind % 6;
                T val = dot_prod<T,6,1,1>(&s_XImats[36*4 + 6*row], &s_vaf[72 + 6*4]);
                int dstOffset = 72 + 6*3 + row;
                s_vaf[dstOffset] += val;
            }
            __syncthreads();
            // s_f update where bfs_level is 3
            //     joints are: joint3
            //     links are: link4
            // s_f[parent_k] += X[k]^T*f[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                int row = ind % 6;
                T val = dot_prod<T,6,1,1>(&s_XImats[36*3 + 6*row], &s_vaf[72 + 6*3]);
                int dstOffset = 72 + 6*2 + row;
                s_vaf[dstOffset] += val;
            }
            __syncthreads();
            // s_f update where bfs_level is 2
            //     joints are: joint2
            //     links are: link3
            // s_f[parent_k] += X[k]^T*f[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                int row = ind % 6;
                T val = dot_prod<T,6,1,1>(&s_XImats[36*2 + 6*row], &s_vaf[72 + 6*2]);
                int dstOffset = 72 + 6*1 + row;
                s_vaf[dstOffset] += val;
            }
            __syncthreads();
            // s_f update where bfs_level is 1
            //     joints are: joint1
            //     links are: link2
            // s_f[parent_k] += X[k]^T*f[k]
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                int row = ind % 6;
                T val = dot_prod<T,6,1,1>(&s_XImats[36*1 + 6*row], &s_vaf[72 + 6*1]);
                int dstOffset = 72 + 6*0 + row;
                s_vaf[dstOffset] += val;
            }
            __syncthreads();
        }



    // called by forwardDynamics()
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
    * @param s_temp is the pointer to the shared memory needed of size: 732
    * @param gravity is the gravity constant
    * @param d_f_ext is the vector of external forces and torques (6D)
    */
    template <typename T>
    __device__
    void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_XImats, T *s_temp, const T gravity, T *d_f_ext) {
        direct_minv_inner<T>(s_temp, s_q, s_XImats, &s_temp[36]);
        inverse_dynamics_inner<T>(&s_temp[36], &s_temp[42], s_q, s_qd, s_XImats, &s_temp[150], gravity, d_f_ext);
        forward_dynamics_finish<T>(s_qdd, s_u, &s_temp[36], s_temp);
    }

}



