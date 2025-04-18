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
void exec_integrator_error(uint32_t state_size, T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cg::thread_block b, bool absval = false){
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x) {

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
        } else if (INTEGRATOR_TYPE == 2) { // midpoint
            // Estimate state at midpoint t + dt/2
            T dt_half = dt / 2.0;
            T q_mid = s_q[ind] + dt_half * s_qd[ind];
            T qd_mid = s_qd[ind] + dt_half * s_qdd[ind];

            // *** Need to compute acceleration at midpoint: qdd_mid = dynamics(q_mid, qd_mid, u) ***
            // *** This requires modifying the calling function to re-evaluate dynamics ***
            T qdd_mid = s_qdd[ind]; // Placeholder: Using initial qdd - THIS IS INCORRECT FOR STANDARD MIDPOINT

            // Full step using midpoint velocity and acceleration
            new_qdkp1 = s_qd[ind] + dt * qdd_mid; // Velocity update uses midpoint acceleration
            new_qkp1 = s_q[ind] + dt * qd_mid;   // Position update uses midpoint velocity
        
        } else { printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Midpoint]",INTEGRATOR_TYPE); }

        // wrap angles if needed
        if(ANGLE_WRAP){
            new_qkp1 = angle_wrap(new_qkp1);
        }

        // then computre error
        if(absval){
            s_err[ind] = abs(s_qkp1[ind] - new_qkp1);
            s_err[ind + state_size/2] = abs(s_qdkp1[ind] - new_qdkp1);    
        }
        else{
            s_err[ind] = s_qkp1[ind] - new_qkp1;
            s_err[ind + state_size/2] = s_qdkp1[ind] - new_qdkp1;
        }
        // printf("err[%f] with new qkp1[%f] vs orig[%f] and new qdkp1[%f] vs orig[%f] with qk[%f] qdk[%f] qddk[%f] and dt[%f]\n",s_err[ind],new_qkp1,s_qkp1[ind],new_qdkp1,s_qdkp1[ind],s_q[ind],s_qd[ind],s_qdd[ind],dt);
    }
}

template <typename T, unsigned INTEGRATOR_TYPE = 1>
__device__
void exec_integrator_gradient(uint32_t state_size, uint32_t control_size, T *s_Ak, T *s_Bk, T *s_dqdd, T dt, cg::thread_block b){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t half_state = state_size / 2;
    
    // Constants for midpoint integrator
    const T dt_half = dt / 2.0;
    const T dt_sq_half = dt * dt_half;
    
    if (INTEGRATOR_TYPE == 0) { // euler
        // Direct thread assignment for A matrix
        for (uint32_t i = thread_id; i < STATE_SIZE; i += block_dim){
            int c = i / STATE_SIZE;
            int r = i % STATE_SIZE;
            s_Ak[i] = (r == c) ? static_cast<T>(1) : // 1s on diagonal
                             ((r < half_state && r == c - half_state) ? dt : // dt on off-diagonal for + qd*dt
                             ((r >= half_state) ? s_dqdd[c * half_state + r - half_state] * dt : 0)); // qdd*dt on off-diagonal for + q*dt^2
        }
        
        // Direct thread assignment for B matrix
        if (thread_id < state_size * control_size) {
            int c = thread_id / state_size;
            int r = thread_id % state_size;
            s_Bk[thread_id] = (r >= half_state) ? dt * s_dqdd[state_size * half_state + c * half_state + r - half_state] : 0;
        }
    }
    else if (INTEGRATOR_TYPE == 1) {
        // Semi-Implicit Euler
        // Direct thread assignment for A matrix
        if (thread_id < state_size * state_size) {
            int c = thread_id / state_size;
            int r = thread_id % state_size;
            int rdqdd = r % half_state;
            T dtVal = (r == rdqdd) ? dt : static_cast<T>(1);
            
            s_Ak[thread_id] = (r == c) ? static_cast<T>(1) : 
                             ((r == c - half_state) ? dt : 0);
            s_Ak[thread_id] += dt * s_dqdd[c * half_state + rdqdd] * dtVal;
        }
        
        // Direct thread assignment for B matrix
        if (thread_id < state_size * control_size) {
            int c = thread_id / state_size;
            int r = thread_id % state_size;
            int rdqdd = r % half_state;
            T dtVal = (r == rdqdd) ? dt : static_cast<T>(1);
            
            s_Bk[thread_id] = dt * s_dqdd[state_size * half_state + c * half_state + rdqdd] * dtVal;
        }
    }



    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    // and finally A and B
    if (INTEGRATOR_TYPE == 0){
        // then apply the euler rule -- xkp1 = xk + dt*dxk thus AB = [I_{state},0_{control}] + dt*dxd
        // where dxd = [ 0, I, 0; dqdd/dq, dqdd/dqd, dqdd/du]
        for (unsigned ind = thread_id; ind < state_size*(state_size + control_size); ind += block_dim){
            int c = ind / state_size; int r = ind % state_size;
            T *dst = (c < state_size)? &s_Ak[ind] : &s_Bk[ind - state_size*state_size]; // dst
            T val = (r == c) * static_cast<T>(1); // first term (non-branching)
            val += (r < state_size/2 && r == c - state_size/2) * dt; // first dxd term (non-branching)
            if(r >= state_size/2) { val += dt * s_dqdd[c*state_size/2 + r - state_size/2]; }
            ///TODO: EMRE why didn't this error before?
            // val += (r >= state_size/2) * dt * s_dqdd[c*state_size/2 + r - state_size/2]; // second dxd term (non-branching)
            *dst = val;
        }
    }
    else if (INTEGRATOR_TYPE == 1){
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1 = qk + dt*qdk + dt^2*qddk
        // dxkp1 = [Ix | 0u ] + dt*[[0q, Iqd, 0u] + dt*dqdd
        //                                             dqdd]
        // Ak = I + dt * [[0,I] + dt*dqdd/dx; dqdd/dx]
        // Bk = [dt*dqdd/du; dqdd/du]
        for (unsigned ind = thread_id; ind < state_size*state_size; ind += block_dim){
            int c = ind / state_size; int r = ind % state_size; int rdqdd = r % (state_size/2);
            T dtVal = static_cast<T>((r == rdqdd)*dt + (r != rdqdd));
            s_Ak[ind] = static_cast<T>((r == c) + dt*(r == c - state_size/2)) +
                        dt * s_dqdd[c*state_size/2 + rdqdd] * dtVal;
            if(c < control_size){
                s_Bk[ind] = dt * s_dqdd[state_size*state_size/2 + c*state_size/2 + rdqdd] * dtVal;
            }
        }
    }
    else if (INTEGRATOR_TYPE == 2){
        // Midpoint (Explicit)
        // qd_mid = qd + (dt/2)*qdd
        // q_mid = q + (dt/2)*qd
        // qdd_mid = dynamics(q_mid, qd_mid, u) <-- Requires re-evaluation
        // qdkp1 = qd + dt*qdd_mid
        // qkp1 = qk + dt*qd_mid = qk + dt*(qd + (dt/2)*qdd) = qk + dt*qd + (dt^2/2)*qdd
        //
        // NOTE: The gradient calculation below assumes we use dqdd evaluated at the *initial* state (xk, uk).
        // A more accurate gradient would require dqdd_mid (gradient of dynamics at the midpoint state),
        // and careful application of the chain rule through the midpoint calculation.
        // This implementation mirrors the structure of Euler/SIE but is likely not the exact gradient of the midpoint method.
        
        // Ak = d(xkp1)/d(xk) = d[qkp1; qdkp1]/d[qk; qdk]
        // Bk = d(xkp1)/d(uk) = d[qkp1; qdkp1]/d[uk]

        // qkp1 = qk + dt*qdk + (dt^2/2)*qddk
        // qdkp1 = qdk + dt*qdd_mid  <- This dependency on qdd_mid complicates the gradient.
        // Using qddk as approximation for qdd_mid for gradient calculation:
        // qdkp1 approx qdk + dt*qddk

        const uint32_t half_state = state_size / 2;
        const T dt_half = dt / 2.0;
        const T dt_sq_half = dt * dt_half;
        
        // Parallel computation of Ak and Bk
        if (thread_id < state_size * state_size) {
            // Calculate Ak
            int c = thread_id / state_size;
            int r = thread_id % state_size;
            int rdqdd = r % half_state;
            
            T dqdd_val = s_dqdd[c * half_state + rdqdd];
            
            // Calculate Ak entry
            T ak_val = static_cast<T>(r == c); // Identity matrix base
            if (r < half_state) { // d(qkp1)/d(xk)
                ak_val += static_cast<T>(r == c - half_state) * dt; // d(qkp1)/d(qdk) = dt*I
                ak_val += dt_sq_half * dqdd_val;                   // d(qkp1)/d(xk) = (dt^2/2)*dqdd/dxk
            } else { // d(qdkp1)/d(xk)
                ak_val += dt * dqdd_val;                           // d(qdkp1)/d(xk) approx dt*dqdd/dxk
            }
            s_Ak[thread_id] = ak_val;
        }
        
        // Direct mapping for Bk calculation
        if (thread_id < state_size * control_size) {
            int c = thread_id / state_size;
            int r = thread_id % state_size;
            int rdqdd = r % half_state;
            
            T dqdd_du_val = s_dqdd[state_size * half_state + c * half_state + rdqdd];
            
            T bk_val = 0.0;
            if (r < half_state) { // d(qkp1)/d(uk)
                bk_val = dt_sq_half * dqdd_du_val; // (dt^2/2) * dqdd/duk
            } else { // d(qdkp1)/d(uk)
                bk_val = dt * dqdd_du_val;         // dt * dqdd/duk
            }
            s_Bk[thread_id] = bk_val;
        }

        // ----

        T dt_half = dt / 2.0;
        T dt_sq_half = dt * dt_half; // dt^2 / 2

        for (unsigned ind = thread_id; ind < state_size*state_size; ind += block_dim){
            int c = ind / state_size; // column index in Ak (relates to xk)
            int r = ind % state_size; // row index in Ak (relates to xkp1)
            int rdqdd = r % (state_size/2); // index within q or qd part for dqdd access

            T dqdd_val = s_dqdd[c*state_size/2 + rdqdd]; // dqdd/dxk or dqdd/duk depending on c

            // Calculate Ak entry
            T ak_val = static_cast<T>(r == c); // Identity matrix base
            if (r < state_size/2) { // d(qkp1)/d(xk)
                ak_val += static_cast<T>(r == c - state_size/2) * dt; // d(qkp1)/d(qdk) = dt*I
                ak_val += dt_sq_half * dqdd_val;                     // d(qkp1)/d(xk) = (dt^2/2)*dqdd/dxk
            } else { // d(qdkp1)/d(xk)  (Approximation using qddk instead of qdd_mid)
                ak_val += dt * dqdd_val;                             // d(qdkp1)/d(xk) approx dt*dqdd/dxk
            }
            s_Ak[ind] = ak_val;
        }
         // Calculate Bk entry (Approximation using qddk)
        for (unsigned ind = thread_id; ind < state_size*control_size; ind += block_dim){
             int c = ind / state_size; // column index in Bk (relates to uk)
             int r = ind % state_size; // row index in Bk (relates to xkp1)
             int rdqdd = r % (state_size/2); // index within q or qd part for dqdd access

             // dqdd/duk term, offset in s_dqdd array
             T dqdd_du_val = s_dqdd[state_size*state_size/2 + c*state_size/2 + rdqdd];

             T bk_val = 0.0;
             if (r < state_size/2) { // d(qkp1)/d(uk)
                 bk_val = dt_sq_half * dqdd_du_val; // (dt^2/2) * dqdd/duk
             } else { // d(qdkp1)/d(uk)
                 bk_val = dt * dqdd_du_val;         // dt * dqdd/duk
             }
             s_Bk[ind] = bk_val;
        }
    }
    else{printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Midpoint]",INTEGRATOR_TYPE);}
}


template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(uint32_t state_size, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cg::thread_block b){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    for (unsigned ind = thread_id; ind < state_size/2; ind += block_dim){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            s_qkp1[ind] = s_q[ind] + dt*s_qd[ind];
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt*s_qdkp1[ind];
        }
        // midpoint
        else if (INTEGRATOR_TYPE == 2) {
            // Estimate state at midpoint t + dt/2
            T dt_half = dt / 2.0;
            T q_mid = s_q[ind] + dt_half * s_qd[ind];
            T qd_mid = s_qd[ind] + dt_half * s_qdd[ind];

            // *** Need to compute acceleration at midpoint: qdd_mid = dynamics(q_mid, qd_mid, u) ***
            // *** This requires modifying the calling function to re-evaluate dynamics ***
            T qdd_mid = s_qdd[ind]; // Placeholder: Using initial qdd - THIS IS INCORRECT FOR STANDARD MIDPOINT

            // Full step using midpoint velocity and acceleration
            s_qdkp1[ind] = s_qd[ind] + dt * qdd_mid; // Velocity update uses midpoint acceleration
            s_qkp1[ind] = s_q[ind] + dt * qd_mid;   // Position update uses midpoint velocity
        }
        else{printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Midpoint]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){
            s_qkp1[ind] = angle_wrap(s_qkp1[ind]);
        }
    }
}

// s_temp of size state_size/2*(state_size + control_size + 1) + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(uint32_t state_size, uint32_t control_size, T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cg::thread_block b){

    
    // first compute qdd and dqdd
    T *s_qdd = s_temp; 	
    T *s_dqdd = s_qdd + state_size/2;	
    T *s_extra_temp = s_dqdd + state_size/2*(state_size+control_size);
    T *s_q = s_xux; 	
    T *s_qd = s_q + state_size/2; 		
    T *s_u = s_qd + state_size/2;
    gato::plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    b.sync();
    // first compute xnew or error
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xux[state_size+control_size], &s_xux[state_size+control_size+state_size/2], s_q, s_qd, s_qdd, dt, b);
    }
    else{
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, b);
    }
    
    // then compute gradient
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(state_size, control_size, s_Ak, s_Bk, s_dqdd, dt, b);
}

// add external wrench
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(uint32_t state_size, uint32_t control_size, T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cg::thread_block b, T *d_f_ext){

    
    // first compute qdd and dqdd
    T *s_qdd = s_temp; 	
    T *s_dqdd = s_qdd + state_size/2;	
    T *s_extra_temp = s_dqdd + (state_size/2)*(state_size+control_size);
    T *s_q = s_xux; 	
    T *s_qd = s_q + state_size/2; 		
    T *s_u = s_qd + state_size/2;
    gato::plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, d_f_ext);
    b.sync();
    // first compute xnew or error
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xux[state_size+control_size], &s_xux[state_size+control_size+state_size/2], s_q, s_qd, s_qdd, dt, b);
    }
    else{
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, b);
    }
    
    // then compute gradient
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(state_size, control_size, s_Ak, s_Bk, s_dqdd, dt, b);
}


// s_temp of size 3*state_size/2 + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
T integratorError(uint32_t state_size, T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, cg::thread_block b){

    // first compute qdd
    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;
    T *s_qdd = s_temp; 					
    T *s_err = s_qdd + state_size/2;
    T *s_extra_temp = s_err + state_size/2;
    gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, b);
    b.sync();
    // if(blockIdx.x == 0 && threadIdx.x==0){
    //     printf("\n");
    //     for(int i = 0; i < state_size/2; i++){
    //         printf("%f ", s_qdd[i]);
    //     }
    //     printf("\n");
    // }
    // b.sync();
    // then apply the integrator and compute error
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, b, true);
    b.sync();

    // finish off forming the error
    block::reduce<T>(state_size, s_err);
    b.sync();
    // if(GATO_LEAD_THREAD){printf("in integratorError with reduced error of [%f]\n",s_err[0]);}
    return s_err[0];
}


// add external wrench
template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
T integratorError(uint32_t state_size, T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, cg::thread_block b, T *d_f_ext){

    // first compute qdd
    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;
    T *s_qdd = s_temp; 					
    T *s_err = s_qdd + state_size/2;
    T *s_extra_temp = s_err + state_size/2;
    gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, b, d_f_ext);
    b.sync();
    // if(blockIdx.x == 0 && threadIdx.x==0){
    //     printf("\n");
    //     for(int i = 0; i < state_size/2; i++){
    //         printf("%f ", s_qdd[i]);
    //     }
    //     printf("\n");
    // }
    // b.sync();
    // then apply the integrator and compute error
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, b, true);
    b.sync();

    // finish off forming the error
    block::reduce<T>(state_size, s_err);
    b.sync();
    // if(GATO_LEAD_THREAD){printf("in integratorError with reduced error of [%f]\n",s_err[0]);}
    return s_err[0];
}

template <typename T, unsigned INTEGRATOR_TYPE = 1, bool ANGLE_WRAP = false>
__device__ 
void integrator(uint32_t state_size, T *s_xkp1, T *s_xk, T *s_uk, T *s_temp, void *d_dynMem_const, T dt, cg::thread_block b, T *d_f_ext){

    T *s_q = s_xk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_uk;

    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;

    T *s_qdd = s_temp; 					

    T *s_extra_temp = s_temp + state_size/2;

    gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, b, d_f_ext);
    b.sync();
    exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, b);
}
