#pragma once

#include "gato.cuh"


/**
* @brief Execute integration and finds gradients of next state wrt current state and control
* @tparam T Data type
* @tparam INTEGRATOR_TYPE Type of integrator (0: Euler, 1: Semi-Implicit Euler)
* @param state_size Size of the state vector
* @param control_size Size of the control vector
* @param s_Ak, s_Bk df
* @param s_dqdd Acceleration gradient
* @param dt Time step
* @param block Thread block
*/
template <typename T, unsigned INTEGRATOR_TYPE = 0>
__device__
void exec_integrator_gradient(uint32_t state_size, uint32_t control_size, T *s_Ak, T *s_Bk, T *s_dqdd, T dt, cgrps::thread_block block){
        
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
    else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}
}

// ---------- Integrator and gradients for KKT system ----------

/**
 * @brief Integrates to find next state and computes the gradient of next state wrt current state and control
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @tparam COMPUTE_INTEGRATOR_ERROR Whether to compute integrator error
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param s_xux Input state and control
 * @param s_Ak, s_Bk Output matrices
 * @param s_xnew_err Output new state or error
 * @param s_temp Temporary storage, size: (state_size/2*(state_size + control_size + 1) + DYNAMICS_TEMP)
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 * @param block Thread block
 */
 template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
 __device__ __forceinline__
 void integratorAndGradient(uint32_t state_size, uint32_t control_size, T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
     
     T *s_q = s_xux; 	
     T *s_qd = s_q + state_size/2; 		
     T *s_u = s_qd + state_size/2;
     
     T *s_qdd = s_temp; // linearized acceleration
     T *s_dqdd = s_qdd + state_size/2;
     T *s_extra_temp = s_dqdd + (state_size/2)*(state_size+control_size);
 
     // first compute qdd and dqdd
     gato::plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
     block.sync();
 
     // then compute xnew or error
     if (COMPUTE_INTEGRATOR_ERROR){
         exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xux[state_size+control_size], &s_xux[state_size+control_size+state_size/2], s_q, s_qd, s_qdd, dt, block);
     } else {
         exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, block);
     }
     
     // then compute gradients to form Ak and Bk
     exec_integrator_gradient<T,INTEGRATOR_TYPE>(state_size, control_size, s_Ak, s_Bk, s_dqdd, dt, block);
 }