#pragma once

#define Q_COST 10.0
#define R_COST 0.1
#define QD_COST 1.0

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "GBD-PCG/include/gpuassert.cuh"

namespace grid {
    const int NUM_JOINTS = 1;
	const int EE_POS_SIZE = 2;
	const int EE_POS_SIZE_COST = 2;
    const int EE_POS_SHARED_MEM_COUNT = 0;
    const int DEE_POS_SHARED_MEM_COUNT = 0;
    template <typename T>
    struct robotModel {
        T *d_XImats;
        int *d_topology_helpers;
    };
    template <typename T>
    __global__
    void end_effector_positions_kernel(T *d_eePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        if (threadIdx.x == 0){
        	d_eePos[0] = d_q[0];
        	d_eePos[1] = d_q[1];
        }
    }
}

namespace gato {
	const unsigned GRiD_SUGGESTED_THREADS = 128;

	namespace plant {
		template<class T>
		__host__ __device__
		constexpr T PI() {return static_cast<T>(3.14159);}

		template<class T>
		__host__ __device__
		constexpr T COST_QD() {return static_cast<T>(1.0);}

		template<class T>
		__host__ __device__
		constexpr T COST_R() {return static_cast<T>(0.1);}

		template<class T>
		__host__ __device__
		constexpr T GRAVITY() {return static_cast<T>(-9.81);}

		
		template <typename T>
		void *initializeDynamicsConstMem(){
			return 0;
		}
		template <typename T>
		void freeDynamicsConstMem(void *d_dynMem_const){
			return;
		}

		// Start at x = [0,0]
		template <typename T>
		__host__
		void loadInitialState(T *x){
			x[0] = static_cast<T>(0); x[1] = static_cast<T>(0);
		}

		template <typename T>
		__host__
		void loadInitialControl(T *u){u[0] = static_cast<T>(0);}

		// goal at X = [PI,0]
		template <typename T>
		__host__
		void loadGoalState(T *xg){
			xg[0] = static_cast<T>(PI); xg[1] = static_cast<T>(0);
		}

		template <typename T>
		__device__
		void forwardDynamics(T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
			if (threadIdx.x == 0){
				s_qdd[0] = s_u[0] + GRAVITY<T>()*sin(s_q[0]);
			}
			__syncthreads();
		}

		__host__ __device__
		constexpr unsigned forwardDynamics_TempMemSize_Shared(){return 0;}

		template <typename T>
		__device__
		void forwardDynamicsGradient( T *s_dqdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const){
			if (threadIdx.x == 0){
				s_dqdd[0] = GRAVITY<T>()*cos(s_q[0]); //dq
				s_dqdd[1] = 0.0; //dqd
				s_dqdd[2] = 1;   //du
			}
			__syncthreads();
		}

		__host__ __device__
		constexpr unsigned forwardDynamicsGradient_TempMemSize_Shared(){return 0;}

		template <typename T>
		__device__
		void forwardDynamicsAndGradient(T *s_dqdd, T *s_qdd, T *s_q, T *s_qd, T *s_u,  T *s_temp, void *d_dynMem_const){
			const T gravity = GRAVITY<T>();

			if (threadIdx.x == 0){
				s_qdd[0] = s_u[0] + gravity*sin(s_q[0]);
				s_dqdd[0] = gravity*cos(s_q[0]); //dq
				s_dqdd[1] = 0.0; //dqd
				s_dqdd[2] = 1;   //du
			}
			__syncthreads();
		}


		__host__ __device__
		constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared(){return 0;}


		__host__
		unsigned trackingCost_TempMemCt_Shared(uint32_t state_size, uint32_t control_size, uint32_t knot_points){
			return 0;
		}

		// Note that there is 
		template <typename T>
		__device__
		T trackingCost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *s_xu, T *s_eePos_traj, T *s_temp, const grid::robotModel<T> *d_robotModel){
			T *s_xg = s_eePos_traj; // abuse this and pass in xg somehow
			T *s_cost = s_temp;
			if (threadIdx.x == 0){
				T err = (s_xu[0] - s_xg[0]);
				*s_cost = Q_COST * err * err;
				*s_cost += COST_QD<T>() * s_xu[1] * s_xu[1];
				*s_cost += COST_R<T>() * s_xu[2] * s_xu[2];
				*s_cost *= 0.5;
			}
			__syncthreads();
			
			return s_cost[0];
		}	


		template <typename T, bool computeR=true>
		__device__
		void trackingCostGradientAndHessian(uint32_t state_size, 
											uint32_t control_size, 
											T *s_xu, 
											T *s_eePos_traj, // abuse this and pass in xg somehow
											T *s_Qk, 
											T *s_qk, 
											T *s_Rk, 
											T *s_rk,
											T *s_temp,
											void *d_robotModel)
		{
			T *s_xg = s_eePos_traj; // abuse this and pass in xg somehow
			if (threadIdx.x == 0){
				s_Qk[0] = Q_COST;
				s_Qk[1] = 0;
				s_Qk[2] = 0;
				s_Qk[3] = COST_QD<T>();
				s_qk[0] = Q_COST * (s_xu[0] - s_xg[0]);
				s_qk[1] = COST_QD<T>() * s_xu[1];
				if (computeR){
					s_Rk[0] = COST_R<T>();
					s_rk[0] = COST_R<T>() * s_xu[2];
				}
			}
		}

		// last block
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
													void *d_dynMem_const
													)
		{
			trackingCostGradientAndHessian<T>(state_size, control_size, s_xux, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp, d_dynMem_const);
			__syncthreads();
			trackingCostGradientAndHessian<T, false>(state_size, control_size, s_xux, &s_eePos_traj[grid::EE_POS_SIZE], s_Qkp1, s_qkp1, nullptr, nullptr, s_temp, d_dynMem_const);
			__syncthreads();
		}
	}
}
