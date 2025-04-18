#pragma once
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include "constants.h"

using namespace gato::constants;

namespace cg = cooperative_groups;
#include "settings.h"


template<typename T>
__host__ __device__ 
T angle_wrap(T input){
    const T pi = static_cast<T>(3.14159);
    if (input > pi) { input = -(input - pi); }
    if (input < -pi) { input = -(input + pi); }
    return input;
}

template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt){

    for (unsigned ind = threadIdx.x; ind < STATE_SIZE/2; ind +=  blockDim.x){
        if (INTEGRATOR_TYPE == 0){ // euler
            s_qkp1[ind] = s_q[ind] + dt*s_qd[ind];
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
        } else if (INTEGRATOR_TYPE == 1){ // semi-implicit euler
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt*s_qdkp1[ind];
        } else if (INTEGRATOR_TYPE == 2) { // trapezoidal
            const T dt_sq_half = 0.5 * dt * dt;
            s_qdkp1[ind] = s_qd[ind] + dt * s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt * s_qd[ind] + dt_sq_half * s_qdd[ind];
        } else {printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Trapezoidal]",INTEGRATOR_TYPE);}

        if(ANGLE_WRAP){ s_qkp1[ind] = angle_wrap(s_qkp1[ind]); }
    }
}

template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false, bool ABSVAL = false>
__device__ 
void exec_integrator_error(T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt){
    T new_qkp1; 
    T new_qdkp1;
    for (unsigned ind = threadIdx.x; ind < STATE_SIZE/2; ind += blockDim.x) {

        if (INTEGRATOR_TYPE == 0) { // euler 
            // q_next = q_curr + dt * qd_curr
            // qd_next = qd_curr + dt * qdd_curr
            new_qkp1 = s_q[ind] + dt*s_qd[ind];
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
        } else if (INTEGRATOR_TYPE == 1) { // semi-inplicit euler
            // qd_next = qd_curr + dt*qdd_curr
            // q_next = q_curr + dt*qd_next
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
            new_qkp1 = s_q[ind] + dt*new_qdkp1;
        } else if (INTEGRATOR_TYPE == 2) { // trapezoidal
            // v_next = v + a * dt
            // q_next = q + v * dt + 0.5 * a * dt**2
            const T dt_sq_half = 0.5 * dt * dt;
            new_qdkp1 = s_qd[ind] + dt * s_qdd[ind];
            new_qkp1 = s_q[ind] + dt * s_qd[ind] + dt_sq_half * s_qdd[ind];
        
        } else { 
            if (threadIdx.x == 0) { 
                printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Trapezoidal]",INTEGRATOR_TYPE);
            }
        }

        if(ANGLE_WRAP){ new_qkp1 = angle_wrap(new_qkp1); }

        // compute error
        s_err[ind] = ABSVAL ? abs(s_qkp1[ind] - new_qkp1) : s_qkp1[ind] - new_qkp1;
        s_err[ind + STATE_SIZE/2] = ABSVAL ? abs(s_qdkp1[ind] - new_qdkp1) : s_qdkp1[ind] - new_qdkp1;    
    }
}

// computes A and B matrices
template <typename T, unsigned INTEGRATOR_TYPE = 1>
__device__
void exec_integrator_gradient(T *s_Ak, T *s_Bk, T *s_dqdd, T dt){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t half_state = STATE_SIZE / 2;
    
    if (INTEGRATOR_TYPE == 0) { // euler (v_next = v + a*dt, q_next = q + v*dt)
        // Calculate Ak = dx_next/dx = [[I, dt*I], [dt*dqdd/dq, I + dt*dqdd/dv]]
        for (uint32_t i = thread_id; i < STATE_SIZE_SQ; i += block_dim){
            int c = i / STATE_SIZE; // Column index (dx)
            int r = i % STATE_SIZE; // Row index (dx_next)
            int rdqdd = r % half_state; // Index within qdd block

            // Calculate Ak[r, c]
            T val = (r == c) ? static_cast<T>(1.0) : static_cast<T>(0.0); // Identity term

            // Top right dq_next/dv = dt*I
            if (r < half_state && r == (c - half_state)) {
                val += dt;
            }
            // Contribution from acceleration gradient: dt*dqdd/dx (only affects dv_next rows)
            if (r >= half_state) {
                // Index into s_dqdd for dqdd_{r'} / dx_c where r' = r - half_state
                int dqdd_idx = c * half_state + rdqdd;
                val += dt * s_dqdd[dqdd_idx];
            }
            s_Ak[i] = val;
        }
        // Calculate Bk = dx_next/du = [[0], [dt*dqdd/du]]
        for (uint32_t i = thread_id; i < STATE_P_CONTROL; i += block_dim) {
            int c = i / STATE_SIZE; // Control index (du)
            int r = i % STATE_SIZE; // Row index (dx_next)
            int rdqdd = r % half_state; // Index within qdd block

            if (r >= half_state) { // dv_next/du
                // Index into s_dqdd for dqdd_{r'} / du_c where r' = r - half_state
                int dqdd_du_idx = STATE_SIZE * half_state + c * half_state + rdqdd;
                s_Bk[i] = dt * s_dqdd[dqdd_du_idx];
            } else { // dq_next/du is zero
                s_Bk[i] = static_cast<T>(0.0);
            }
        }
    } else if (INTEGRATOR_TYPE == 1) { // semi-implicit euler (v_next = v + a*dt, q_next = q + v_next*dt)
        // Ak = [[I + dt^2*(dqdd/dq), dt*I + dt^2*(dqdd/dv)], [dt*(dqdd/dq), I + dt*(dqdd/dv)]]
        for (uint32_t i = thread_id; i < STATE_SIZE * STATE_SIZE; i += block_dim){
            int c = i / STATE_SIZE; // Column index (dx)
            int r = i % STATE_SIZE; // Row index (dx_next)
            int rdqdd = r % half_state; // Index within qdd block

            // dt^2 factor for dq_next rows, dt factor for dv_next rows
            T dt_factor = (r < half_state) ? dt : static_cast<T>(1.0);

            // Base term: I + [[0, dt*I], [0, 0]]
            T val = (r == c) ? static_cast<T>(1.0) : static_cast<T>(0.0);
            if (r < half_state && r == (c - half_state)) {
                 val += dt;
            }

            // Add contribution from acceleration gradient: dt * dqdd/dx * dt_factor
            // Index into s_dqdd for dqdd_{r'} / dx_c where r' = rdqdd
            int dqdd_idx = c * half_state + rdqdd;
            val += dt * s_dqdd[dqdd_idx] * dt_factor;
            s_Ak[i] = val;
        }
        // Calculate Bk = dx_next/du = [[dt^2*dqdd/du], [dt*dqdd/du]]
        for (uint32_t i = thread_id; i < STATE_P_CONTROL; i += block_dim) {
            int c = i / STATE_SIZE; // Control index (du)
            int r = i % STATE_SIZE; // Row index (dx_next)
            int rdqdd = r % half_state; // Index within qdd block

            // dt^2 factor for dq_next rows, dt factor for dv_next rows
            T dt_factor = (r < half_state) ? dt : static_cast<T>(1.0);

            // Index into s_dqdd for dqdd_{r'} / du_c where r' = rdqdd
            int dqdd_du_idx = STATE_SIZE * half_state + c * half_state + rdqdd;
            s_Bk[i] = dt * s_dqdd[dqdd_du_idx] * dt_factor;
        }
    } else if (INTEGRATOR_TYPE == 2) { // trapezoidal (v_next = v + a*dt, q_next = q + v*dt + 0.5*a*dt^2)
        // Ak = [[I + 0.5*dt^2*(dqdd/dq), dt*I + 0.5*dt^2*(dqdd/dv)], [dt*(dqdd/dq), I + dt*(dqdd/dv)]]
        const T dt_sq_half = 0.5 * dt * dt;
        for (uint32_t i = thread_id; i < STATE_SIZE_SQ; i += block_dim){
            int c = i / STATE_SIZE; // Column index (dx)
            int r = i % STATE_SIZE; // Row index (dx_next)
            int rdqdd = r % half_state; // Index within qdd block

            // Index into s_dqdd for dqdd_{rdqdd} / dx_c
            int dqdd_idx = c * half_state + rdqdd;
            T dqdd_rc = s_dqdd[dqdd_idx];

            // Calculate Ak[r, c]
            T val = (r == c) ? static_cast<T>(1.0) : static_cast<T>(0.0); // Diagonal term for I

            if (r < half_state) { // Top rows: dq_next/dx
                // dq_next/dv term: I*dt
                if (c >= half_state && r == (c - half_state)) {
                    val += dt;
                }
                // dq_next contribution from acceleration: 0.5 * dt^2 * dqdd/dx
                val += dt_sq_half * dqdd_rc;
            } else { // Bottom rows: dv_next/dx
                // dv_next contribution from acceleration: dt * dqdd/dx
                val += dt * dqdd_rc;
            }
            s_Ak[i] = val;
        }
         // Calculate Bk = dx_next/du = [[0.5*dt^2*dqdd/du], [dt*dqdd/du]]
        for (uint32_t i = thread_id; i < STATE_P_CONTROL; i += block_dim) {
            int c = i / STATE_SIZE; // Control index (du)
            int r = i % STATE_SIZE; // Row index (dx_next)
            int rdqdd = r % half_state; // Index within qdd block

            // Index into s_dqdd for dqdd_{rdqdd} / du_c
            int dqdd_du_idx = STATE_SIZE * half_state + c * half_state + rdqdd;
            T dqdd_du_rc = s_dqdd[dqdd_du_idx];

            if (r < half_state) { // dq_next/du
                s_Bk[i] = dt_sq_half * dqdd_du_rc;
            } else { // dv_next/du
                s_Bk[i] = dt * dqdd_du_rc;
            }
        }
    } else {
        if (thread_id == 0) { 
            printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Trapezoidal]", INTEGRATOR_TYPE);
        }
    }
}

// s_temp of size: ---> state_size/2 + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
void integrator(T *s_xkp1, T *s_xk, T *s_uk, T *s_temp, void *d_dynMem_const, T dt, T *d_f_ext = nullptr){

    T *s_q = s_xk; T *s_qd = s_q + STATE_SIZE/2; T *s_u = s_uk;
    T *s_qkp1 = s_xkp1; T *s_qdkp1 = s_qkp1 + STATE_SIZE/2;
    T *s_qdd = s_temp; T *s_extra_temp = s_temp + STATE_SIZE/2;

    if (d_f_ext == nullptr) {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    } else {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, d_f_ext);
    }
    __syncthreads();
    exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt);
}

// s_temp of size: ---> state_size + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
T integratorError(T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, T *d_f_ext = nullptr){
    T *s_q = s_xuk; T *s_qd = s_q + STATE_SIZE/2; T *s_u = s_qd + STATE_SIZE/2;
    T *s_qkp1 = s_xkp1; T *s_qdkp1 = s_qkp1 + STATE_SIZE/2;
    T *s_qdd = s_temp; T *s_err = s_qdd + STATE_SIZE/2; T *s_extra_temp = s_err + STATE_SIZE/2;

    if (d_f_ext == nullptr) {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    } else {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, d_f_ext);
    }
    __syncthreads();
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP,true>(s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt); 
    __syncthreads();
    block::reduce<T>(STATE_SIZE, s_err);
    __syncthreads();
    return s_err[0];
}

// s_temp of size: ---> state_size/2*(state_size + control_size + 1) + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, T *d_f_ext = nullptr){
    T *s_q = s_xux; T *s_qd = s_q + STATE_SIZE/2; T *s_u = s_qd + STATE_SIZE/2;
    T *s_qdd = s_temp; T *s_dqdd = s_qdd + STATE_SIZE/2;	
    T *s_extra_temp = s_dqdd + STATE_SIZE/2*(STATE_SIZE+CONTROL_SIZE);

    if (d_f_ext == nullptr) { // no external wrench
        gato::plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    } else {
        gato::plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, d_f_ext);
    }
    __syncthreads();

    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(s_xnew_err, &s_xux[STATE_SIZE+CONTROL_SIZE], &s_xux[STATE_SIZE+CONTROL_SIZE+STATE_SIZE/2], s_q, s_qd, s_qdd, dt);
    } else {
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(s_xnew_err, &s_xnew_err[STATE_SIZE/2], s_q, s_qd, s_qdd, dt);
    }
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(s_Ak, s_Bk, s_dqdd, dt);
}







    // const uint32_t thread_id = threadIdx.x;
    // const uint32_t block_dim = blockDim.x;

    // // and finally A and B
    // if (INTEGRATOR_TYPE == 0){
    //     // then apply the euler rule -- xkp1 = xk + dt*dxk thus AB = [I_{state},0_{control}] + dt*dxd
    //     // where dxd = [ 0, I, 0; dqdd/dq, dqdd/dqd, dqdd/du]
    //     for (unsigned ind = thread_id; ind < state_size*(state_size + control_size); ind += block_dim){
    //         int c = ind / state_size; int r = ind % state_size;
    //         T *dst = (c < state_size)? &s_Ak[ind] : &s_Bk[ind - state_size*state_size]; // dst
    //         T val = (r == c) * static_cast<T>(1); // first term (non-branching)
    //         val += (r < state_size/2 && r == c - state_size/2) * dt; // first dxd term (non-branching)
    //         if(r >= state_size/2) { val += dt * s_dqdd[c*state_size/2 + r - state_size/2]; }
    //         ///TODO: EMRE why didn't this error before?
    //         // val += (r >= state_size/2) * dt * s_dqdd[c*state_size/2 + r - state_size/2]; // second dxd term (non-branching)
    //         *dst = val;
    //     }
    // }
    // else if (INTEGRATOR_TYPE == 1){
    //     // semi-inplicit euler
    //     // qdkp1 = qdk + dt*qddk
    //     // qkp1 = qk  + dt*qdkp1 = qk + dt*qdk + dt^2*qddk
    //     // dxkp1 = [Ix | 0u ] + dt*[[0q, Iqd, 0u] + dt*dqdd
    //     //                                             dqdd]
    //     // Ak = I + dt * [[0,I] + dt*dqdd/dx; dqdd/dx]
    //     // Bk = [dt*dqdd/du; dqdd/du]
    //     for (unsigned ind = thread_id; ind < state_size*state_size; ind += block_dim){
    //         int c = ind / state_size; int r = ind % state_size; int rdqdd = r % (state_size/2);
    //         T dtVal = static_cast<T>((r == rdqdd)*dt + (r != rdqdd));
    //         s_Ak[ind] = static_cast<T>((r == c) + dt*(r == c - state_size/2)) +
    //                     dt * s_dqdd[c*state_size/2 + rdqdd] * dtVal;
    //         if(c < control_size){
    //             s_Bk[ind] = dt * s_dqdd[state_size*state_size/2 + c*state_size/2 + rdqdd] * dtVal;
    //         }
    //     }
    // }




    // //// Midpoint (Explicit)
    //     // qd_mid = qd + (dt/2)*qdd
    //     // q_mid = q + (dt/2)*qd
    //     // qdd_mid = dynamics(q_mid, qd_mid, u) <-- Requires re-evaluation
    //     // qdkp1 = qd + dt*qdd_mid
    //     // qkp1 = qk + dt*qd_mid = qk + dt*(qd + (dt/2)*qdd) = qk + dt*qd + (dt^2/2)*qdd
    //     //
    //     // NOTE: The gradient calculation below assumes we use dqdd evaluated at the *initial* state (xk, uk).
    //     // A more accurate gradient would require dqdd_mid (gradient of dynamics at the midpoint state),
    //     // and careful application of the chain rule through the midpoint calculation.
    //     // This implementation mirrors the structure of Euler/SIE but is likely not the exact gradient of the midpoint method.
        
    //     // Ak = d(xkp1)/d(xk) = d[qkp1; qdkp1]/d[qk; qdk]
    //     // Bk = d(xkp1)/d(uk) = d[qkp1; qdkp1]/d[uk]

    //     // qkp1 = qk + dt*qdk + (dt^2/2)*qddk
    //     // qdkp1 = qdk + dt*qdd_mid  <- This dependency on qdd_mid complicates the gradient.
    //     // Using qddk as approximation for qdd_mid for gradient calculation:
    //     // qdkp1 approx qdk + dt*qddk

    //     const uint32_t half_state = state_size / 2;
    //     const T dt_half = dt / 2.0;
    //     const T dt_sq_half = dt * dt_half;
        
    //     // Parallel computation of Ak and Bk
    //     if (thread_id < state_size * state_size) {
    //         // Calculate Ak
    //         int c = thread_id / state_size;
    //         int r = thread_id % state_size;
    //         int rdqdd = r % half_state;
            
    //         T dqdd_val = s_dqdd[c * half_state + rdqdd];
            
    //         // Calculate Ak entry
    //         T ak_val = static_cast<T>(r == c); // Identity matrix base
    //         if (r < half_state) { // d(qkp1)/d(xk)
    //             ak_val += static_cast<T>(r == c - half_state) * dt; // d(qkp1)/d(qdk) = dt*I
    //             ak_val += dt_sq_half * dqdd_val;                   // d(qkp1)/d(xk) = (dt^2/2)*dqdd/dxk
    //         } else { // d(qdkp1)/d(xk)
    //             ak_val += dt * dqdd_val;                           // d(qdkp1)/d(xk) approx dt*dqdd/dxk
    //         }
    //         s_Ak[thread_id] = ak_val;
    //     }
        
    //     // Direct mapping for Bk calculation
    //     if (thread_id < state_size * control_size) {
    //         int c = thread_id / state_size;
    //         int r = thread_id % state_size;
    //         int rdqdd = r % half_state;
            
    //         T dqdd_du_val = s_dqdd[state_size * half_state + c * half_state + rdqdd];
            
    //         T bk_val = 0.0;
    //         if (r < half_state) { // d(qkp1)/d(uk)
    //             bk_val = dt_sq_half * dqdd_du_val; // (dt^2/2) * dqdd/duk
    //         } else { // d(qdkp1)/d(uk)
    //             bk_val = dt * dqdd_du_val;         // dt * dqdd/duk
    //         }
    //         s_Bk[thread_id] = bk_val;