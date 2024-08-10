#pragma once

#include "gato.cuh"
#include "integrator.cuh"


/**
* @brief Execute integration to get next state and compute error.
* @tparam T Data type
* @tparam INTEGRATOR_TYPE Type of integrator (0: Euler, 1: Semi-Implicit Euler)
* @tparam ANGLE_WRAP Whether to wrap angles
* @param state_size Size of state vector
* @param s_err Output error
* @param s_qkp1, s_qdkp1 Next state input
* @param s_q, s_qd, s_qdd Current state and acceleration
* @param dt Time step
* @param block Thread block
* @param absval Whether to compute absolute value of error
*/
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator_error(uint32_t state_size, T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block, bool absval = false){
    
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            new_qkp1 = s_q[ind] + dt*s_qd[ind];
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
            new_qkp1 = s_q[ind] + dt*new_qdkp1;
        } else {printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){ printf("ANGLE_WRAP!\n");
            new_qkp1 = angleWrap(new_qkp1);
        }

        // then computre error
        if(absval){
            s_err[ind] = abs(s_qkp1[ind] - new_qkp1);
            s_err[ind + state_size/2] = abs(s_qdkp1[ind] - new_qdkp1);    
        } else {
            s_err[ind] = s_qkp1[ind] - new_qkp1;
            s_err[ind + state_size/2] = s_qdkp1[ind] - new_qdkp1;
        }
        // printf("err[%f] with new qkp1[%f] vs orig[%f] and new qdkp1[%f] vs orig[%f] with qk[%f] qdk[%f] qddk[%f] and dt[%f]\n",s_err[ind],new_qkp1,s_qkp1[ind],new_qdkp1,s_qdkp1[ind],s_q[ind],s_qd[ind],s_qdd[ind],dt);
    }
}

// ---------- Error for Merit Function ----------

/**
 * @brief Integrate to get next state and compute error.
 * 
 * s_temp of size: (3*state_size/2 + DYNAMICS_TEMP)
 *
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of the state vector
 * @param s_xuk Input state and control
 * @param s_xkp1 Next state
 * @param s_temp Temporary storage
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 * @param block Thread block
 * @return T Computed error
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
T integratorError(uint32_t state_size, T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){

    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;

    T *s_qdd = s_temp; 					
    T *s_err = s_qdd + state_size/2;
    T *s_extra_temp = s_err + state_size/2;

    // first compute qdd
    gato::plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();

    // then apply the integrator and compute error
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block, true);
    block.sync();

    // finish off forming the error
    glass::reduce<T>(state_size, s_err);
    block.sync();
    
    return s_err[0];
}
