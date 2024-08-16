#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "glass.cuh" //GPU linear algebra simple subroutines
#include "dynamics/rbd_plant.cuh" //this is where the gato namespace is originally defined
#include "solver_settings.h" //solver settings
#include "sim_settings.h" //simulation settings

namespace cgrps = cooperative_groups;

/**
 * @brief GPU-Accelerated Trajectory Optimization
 * 
**/
namespace gato {
    
    // define constants
    const uint32_t STATE_SIZE = grid::NUM_JOINTS * 2; // joints and velocities
    const uint32_t CONTROL_SIZE = grid::NUM_JOINTS;
    const uint32_t KNOT_POINTS = 32; //move this to a settings file in config/
    
    const float TIMESTEP = 0.015625; // 1/64 TODO: add to settings.h

    const float SIM_STEP_TIME = 2e-4; //TODO: this was arbitrarily defined

    // other constants for memory offsets
    const uint32_t STATES_SQ = STATE_SIZE * STATE_SIZE;
    const uint32_t CONTROLS_SQ = CONTROL_SIZE * CONTROL_SIZE;
    const uint32_t STATES_P_CONTROLS = STATE_SIZE * CONTROL_SIZE;
    const uint32_t STATES_S_CONTROLS = STATE_SIZE + CONTROL_SIZE;
    const uint32_t TRAJ_LEN = KNOT_POINTS * STATES_S_CONTROLS - CONTROL_SIZE;

    //const unsigned GRiD_SUGGESTED_THREADS; //defined in _plant.cuh

    namespace plant {

        void *d_dynMem_const; // GRiD constant memory (to be initialized in main)

        // ---------- Everything below here is defined in _plant.cuh ----------
        // --------------------------------------------------------------------

        template <typename T>
		void *initializeDynamicsConstMem();

        template <typename T>
		void freeDynamicsConstMem(void *d_dynMem_const);

        template<class T>
		__host__ __device__
		constexpr T PI();
	
		// TODO: turn on gravity?
		template<class T>
		__host__ __device__
		constexpr T GRAVITY();

        // get QD cost for tracking cost (error in derivative of state)
        template<class T>
        __host__ __device__
        constexpr T COST_QD();

        // get R cost for tracking cost (control effort)
        template<class T>
        __host__ __device__
        constexpr T COST_R();

        __host__ __device__
		constexpr unsigned forwardDynamics_TempMemSize_Shared();

		__host__ __device__
		constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared();

        //shared memory size for tracking cost kernel
        __host__
		unsigned trackingCost_TempMemCt_Shared(uint32_t state_size, 
                                            uint32_t control_size, 
                                            uint32_t knot_points);

        template <typename T>
		__device__
		void forwardDynamics(T *s_qdd, 
                            T *s_q, 
                            T *s_qd, 
                            T *s_u, 
                            T *s_XITemp, 
                            void *d_dynMem_const, 
                            cgrps::thread_block block);
		
        template <typename T>
		__device__
		T trackingCost(uint32_t state_size, 
                    uint32_t control_size, 
                    uint32_t knot_points, 
                    T *s_xu, 
                    T *s_eePos_traj, 
                    T *s_temp, 
                    const grid::robotModel<T> *d_robotModel);

        template <typename T, bool computeR>
		__device__
		void trackingCostGradientAndHessian(uint32_t state_size, 
											uint32_t control_size, 
											T *s_xu, 
											T *s_eePos_traj, 
											T *s_Qk, 
											T *s_qk, 
											T *s_Rk, 
											T *s_rk,
											T *s_temp,
											void *d_robotModel);

        template <typename T>
		__device__
		void trackingCostGradientAndHessian_lastblock(uint32_t state_size, 
													uint32_t control_size, 
													T *s_xux, 
													T *s_eePos_traj, 
													T *s_Qk, 
													T *s_qk, 
													T *s_Rk, 
													T *s_rk, 
													T *s_Qkp1, 
													T *s_qkp1,
													T *s_temp,
													void *d_dynMem_const);

    }

}