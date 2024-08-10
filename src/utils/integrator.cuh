#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gato.cuh"

/**
 * @brief Wraps an angle to range [-pi, pi].
 * @tparam T Data type of the angle
 * @param input Input angle
 * @return T Wrapped angle
 */
 template<typename T>
 __host__ __device__ 
 T angleWrap(T input){
     const T pi = static_cast<T>(M_PI);
     if(input > pi) {input -= 2 * pi;}
     else if (input < -pi) {input += 2 * pi;}
     return input;
 }


/**
 * @brief Execute integration to find next state and velocity
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator (0: Euler, 1: Semi-Implicit Euler)
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of state vector
 * @param s_qkp1, s_qdkp1 Next state, next velocity
 * @param s_q, s_qd, s_qdd Linearized state, velocity, acceleration
 * @param dt Time step
 * @param block Thread block
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(uint32_t state_size, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block){

    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
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
        else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){
            s_qkp1[ind] = angleWrap(s_qkp1[ind]);
        }
    }
}

// ---------- Integrator for simulation ----------

/**
 * @brief Compute forward dynamics and integrate to find next state.
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of the state vector
 * @param s_xkp1 Output next state
 * @param s_xuk Input state and control
 * @param s_temp Temporary storage
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 * @param block Thread block
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void integrator(uint32_t state_size, T *s_xkp1, T *s_xuk, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;

    T *s_qdd = s_temp; 					
    T *s_extra_temp = s_qdd + state_size/2;

    //first compute qdd
    gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block);
}
