#pragma once
#include <algorithm>
#include <cmath>
#include "constants.h"
#include "settings.h"
using namespace gato::constants;

namespace gato::plant {

template<typename T>
__host__ __device__ 
T angle_wrap(T input){
    const T pi = static_cast<T>(3.14159);
    if (input > pi) { input = -(input - pi); }
    if (input < -pi) { input = -(input + pi); }
    return input;
}

template <typename T, unsigned INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false>
__device__ 
void integrate_forward(T *s_q_next, T *s_qd_next, T *s_q, T *s_qd, T *s_qdd, T dt){
    for (unsigned i = threadIdx.x; i < STATE_SIZE/2; i += blockDim.x) {
        if (INTEGRATOR_TYPE == 0) { // euler 
            // q_next = q_curr + dt * qd_curr  |  qd_next = qd_curr + dt * qdd_curr
            s_q_next[i] = s_q[i] + dt*s_qd[i];
            s_qd_next[i] = s_qd[i] + dt*s_qdd[i];

        } else if (INTEGRATOR_TYPE == 1) { // semi-inplicit euler
            // qd_next = qd_curr + dt*qdd_curr  |  q_next = q_curr + dt*qd_next
            s_qd_next[i] = s_qd[i] + dt*s_qdd[i];
            s_q_next[i] = s_q[i] + dt*s_qd_next[i];

        } else if (INTEGRATOR_TYPE == 2) { // trapezoidal
            // v_next = v + a * dt  |  q_next = q + v * dt + 0.5 * a * dt**2
            s_qd_next[i] = s_qd[i] + dt * s_qdd[i];
            s_q_next[i] = s_q[i] + dt * s_qd[i] + 0.5 * s_qdd[i] * dt * dt;
        
        } else { if (threadIdx.x == 0) { printf("Integrator [%d] not defined. Currently support [0: Euler, 1: Semi-Implicit Euler, 2: Trapezoidal]",INTEGRATOR_TYPE); } }
        
        if(ANGLE_WRAP){ s_q_next[i] = angle_wrap(s_q_next[i]); }
    }
}

// shared mem size: STATE_SIZE + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false, bool ABSVAL = false>
__device__ 
void compute_integrator_error(T *s_err, T *s_q_next, T *s_qd_next, T *s_q, T *s_qd, T *s_qdd, T dt, T *s_temp){

    T *s_q_new = s_temp;  T *s_qd_new = s_q_new + STATE_SIZE/2;
    integrate_forward(s_q_new, s_qd_new, s_q, s_qd, s_qdd, dt);

    // compute error
    for (uint32_t i = threadIdx.x; i < STATE_SIZE/2; i += blockDim.x) {
        s_err[i] = ABSVAL ? abs(s_q_next[i] - s_q_new[i]) : s_q_next[i] - s_q_new[i];
        s_err[i + STATE_SIZE/2] = ABSVAL ? abs(s_qd_next[i] - s_qd_new[i]) : s_qd_next[i] - s_qd_new[i];    
    }
}

// computes A and B matrices
template <typename T, unsigned INTEGRATOR_TYPE = 2>
__device__
void compute_integrator_gradient(T *s_Ak, T *s_Bk, T *s_dqdd, T dt){
    // s_Ak: [STATE_SIZE x STATE_SIZE]
    // s_Bk: [STATE_SIZE x CONTROL_SIZE]
    // s_dqdd: [STATE_SIZE x STATE_SIZE]

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
template <typename T, unsigned INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false>
__device__ 
void sim_step(T *s_xkp1, T *s_xk, T *s_uk, T *s_temp, void *d_dynMem_const, T dt, T *d_f_ext = nullptr){

    T *s_q = s_xk; T *s_qd = s_q + STATE_SIZE/2; T *s_u = s_uk;
    T *s_qkp1 = s_xkp1; T *s_qdkp1 = s_qkp1 + STATE_SIZE/2;
    T *s_qdd = s_temp; T *s_extra_temp = s_temp + STATE_SIZE/2;

    if (d_f_ext == nullptr) {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    } else {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, d_f_ext);
    }
    __syncthreads();
    integrate_forward(s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt);
}

// s_temp of size: ---> 2 * state_size + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false>
__device__ 
T integrator_error(T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, T *d_f_ext = nullptr){
    T *s_q = s_xuk; T *s_qd = s_q + STATE_SIZE/2; T *s_u = s_qd + STATE_SIZE/2;
    T *s_qkp1 = s_xkp1; T *s_qdkp1 = s_qkp1 + STATE_SIZE/2;
    T *s_qdd = s_temp; T *s_err = s_qdd + STATE_SIZE/2; T *s_extra_temp = s_err + STATE_SIZE/2;

    if (d_f_ext == nullptr) {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    } else {
        gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, d_f_ext);
    }
    __syncthreads();
    compute_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP,true>(s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, s_extra_temp); 
    __syncthreads();
    block::reduce<T>(STATE_SIZE, s_err);
    __syncthreads();
    return s_err[0];
}

// s_temp of size: ---> state_size/2*(state_size + control_size + 1) + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 2, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void linearize_dynamics(T *s_xux, T *s_Ak, T *s_Bk, T *s_out, T *s_temp, void *d_dynMem_const, T dt, T *d_f_ext = nullptr){
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
        compute_integrator_error(s_out, &s_xux[STATE_SIZE+CONTROL_SIZE], &s_xux[STATE_SIZE+CONTROL_SIZE+STATE_SIZE/2], s_q, s_qd, s_qdd, dt, s_extra_temp);
    } else {
        integrate_forward(s_out, &s_out[STATE_SIZE/2], s_q, s_qd, s_qdd, dt);
    }
    compute_integrator_gradient(s_Ak, s_Bk, s_dqdd, dt);
}

} // namespace gato::plant