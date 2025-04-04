#pragma once
// // values assumed coming from an instance of grid
// namespace grid{
// 	//
// 	// TODO do I need all of these?
// 	//

// 	const int NUM_JOINTS = 30;
//     const int ID_DYNAMIC_SHARED_MEM_COUNT = 2340;
//     const int MINV_DYNAMIC_SHARED_MEM_COUNT = 9210;
//     const int FD_DYNAMIC_SHARED_MEM_COUNT = 10110;
//     const int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 10980;
//     const int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 10980;
//     const int ID_DU_MAX_SHARED_MEM_COUNT = 13410;
//     const int FD_DU_MAX_SHARED_MEM_COUNT = 16140;
//     const int SUGGESTED_THREADS = 512;

// 	template <typename T>
//     struct robotModel {
//         T *d_XImats;
//         int *d_topology_helpers;
//     };
// }

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "indy7_grid.cuh"
#include "GLASS/glass.cuh"
#include "settings.h"

// #include <random>
// #define RANDOM_MEAN 0
// #define RANDOM_STDEV 0.001
// std::default_random_engine randEng(time(0)); //seed
// std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

using namespace sqp;

namespace gato{

	const unsigned GRiD_SUGGESTED_THREADS = grid::SUGGESTED_THREADS;

	namespace plant{
		const unsigned SUGGESTED_THREADS = grid::SUGGESTED_THREADS;

		template<class T>
		__host__ __device__
		constexpr T PI() {return static_cast<T>(3.14159);}
		template<class T>
		__host__ __device__
		constexpr T GRAVITY() {return static_cast<T>(9.81);}
		
		template<class T>
		__host__ __device__
		constexpr T COST_QD() {return static_cast<T>(VELOCITY_COST);}

		template<class T>
		__host__ __device__
		constexpr T COST_R() {return static_cast<T>(CONTROL_COST);}

        template<class T>
		__host__ __device__
		constexpr T COST_EE_POS() {return static_cast<T>(EE_POS_COST);}

		template<class T>
		__host__ __device__
		constexpr T COST_EE_POS_TERMINAL() {return static_cast<T>(EE_POS_TERMINAL_COST);}

        template<class T>
		__host__ __device__
		constexpr T COST_BARRIER() {return static_cast<T>(BARRIER_COST);}

		template<class T>
		__host__ __device__
		constexpr T BARRIER_MARGIN() {return static_cast<T>(JOINT_BARRIER_MARGIN);}

		__device__
		constexpr float POS_LIMITS_DATA[6][2] = { // from indy7.urdf
			{-3.0543f, 3.0543f}, // joint 0
			{-3.0543f, 3.0543f}, // joint 1
			{-3.0543f, 3.0543f}, // joint 2
			{-3.0543f, 3.0543f}, // joint 3
			{-3.0543f, 3.0543f}, // joint 4
			{-3.7520f, 3.7520f}, // joint 5
		};

        __device__
        constexpr float VEL_LIMITS_DATA_FLOAT[6][2] = { // from indy7.urdf
            {-2.61799f, 2.61799f}, // joint 0
            {-2.61799f, 2.61799f}, // joint 1
            {-2.61799f, 2.61799f}, // joint 2
            {-3.14159f, 3.14159f}, // joint 3
            {-3.14159f, 3.14159f}, // joint 4
            {-3.14159f, 3.14159f}, // joint 5
        };

		template<class T>
		__device__
		constexpr const float (&POS_LIMITS())[6][2] {
			return POS_LIMITS_DATA;
		}

		template<class T>
		__device__
		constexpr const float (&VEL_LIMITS())[6][2] {
			return VEL_LIMITS_DATA_FLOAT;
		}

		template <typename T>
		void *initializeDynamicsConstMem(){
			grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
			return (void *)d_robotModel;
		}
		template <typename T>
		void freeDynamicsConstMem(void *d_dynMem_const){
			grid::free_robotModel((grid::robotModel<T>*) d_dynMem_const);
		}

		template<class T>
		__device__
		T jointBarrier(T q, T q_min, T q_max) {
			T dist_min = q - (q_min + BARRIER_MARGIN<T>());
			T dist_max = (q_max - BARRIER_MARGIN<T>()) - q;
			return -log(dist_min) - log(dist_max);
		}

		template<class T>
		__device__
		T jointBarrierGradient(T q, T q_min, T q_max) {
			T dist_min = q - (q_min + BARRIER_MARGIN<T>());
			T dist_max = (q_max - BARRIER_MARGIN<T>()) - q;
			return -1/dist_min + 1/dist_max;
		}

        template<class T>
        __device__
        T jointBarrierHessian(T q, T q_min, T q_max) {
            T dist_min = q - (q_min + BARRIER_MARGIN<T>());
            T dist_max = (q_max - BARRIER_MARGIN<T>()) - q;
            return 1/(dist_min*dist_min) + 1/(dist_max*dist_max);
        }

		// template <typename T>
		// __host__
		// void loadInitialState(T *x){
		// 	T q[6] = {PI<T>(),0.25*PI<T>(),0.167*PI<T>(),-0.167*PI<T>(),PI<T>(),0.167*PI<T>()};
		// 	for (int i = 0; i < 6; i++){
		// 		x[i] = q[i]; x[i + 6] = 0;
		// 	}
		// }

		// template <typename T>
		// __host__
		// void loadInitialControl(T *u){for (int i = 0; i < 7; i++){u[i] = 0;}}


        // ********** DYNAMICS ADAPTED FROM indy7_grid.cuh **********

		__host__ __device__
		constexpr unsigned forwardDynamicsSMemSize(){return grid::FD_DYNAMIC_SHARED_MEM_COUNT;}
        
		template <typename T>
		__device__
		void forwardDynamics(T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_XITemp, void *d_dynMem_const){
			T *s_XImats = s_XITemp; 
            T *s_temp = &s_XITemp[864];
			grid::load_update_XImats_helpers<T>(s_XImats, s_q, (grid::robotModel<float> *) d_dynMem_const, s_temp);
			grid::forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gato::plant::GRAVITY<T>());
		}

		__host__ __device__
		constexpr unsigned forwardDynamicsAndGradientSMemSize(){return grid::FD_DU_MAX_SHARED_MEM_COUNT;}

		template <typename T, bool INCLUDE_DU = true>
		__device__
		void forwardDynamicsAndGradient(T *s_df_du, T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_temp_in, void *d_dynMem_const){
			grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *) d_dynMem_const;
			T *s_XITemp = s_temp_in;
			T *s_XImats = s_XITemp; 
            T *s_vaf = &s_XITemp[432]; 
            T *s_dc_du = &s_vaf[108]; 
            T *s_Minv = &s_dc_du[72]; 
            T *s_temp = &s_Minv[36];

			grid::load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
			grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
			grid::inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, &s_temp[6], GRAVITY<T>()); 
			grid::forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv); 
			grid::inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, GRAVITY<T>()); 
			grid::inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, GRAVITY<T>()); 
			
			for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
				int row = ind % 6; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
				T val = static_cast<T>(0);
				for(int col = 0; col < 6; col++) {
					int index = (row <= col) * (col * 6 + row) + (row > col) * (row * 6 + col);
					val += s_Minv[index] * s_dc_du[dc_col_offset + col];
				}
				s_df_du[ind] = -val;
				if (INCLUDE_DU && ind < 36){
					int col = ind / 6; int index = (row <= col) * (col * 6 + row) + (row > col) * (row * 6 + col);
					s_df_du[ind + 72] = s_Minv[index];
				}
			}
            __syncthreads();
		}

		__host__
		unsigned trackingCostSMemSize(){
            //TODO
        }

		template <typename T>
		__device__
		T trackingCost(uint32_t state_size, uint32_t control_size, T *s_xu, T *s_eePos_traj, T *s_temp, const grid::robotModel<T> *d_robotModel){
			
            // 3 ee_pos + positions + velocities + controls (unless last block)
			const uint32_t num_threads = 3 + state_size + control_size * (blockIdx.x < gridDim.x - 1);
            const T ee_pos_cost = (blockIdx.x < gridDim.x - 1) ? COST_EE_POS<T>() : COST_EE_POS_TERMINAL<T>();
			
			T *s_cost_vec = s_temp;
			T *s_eePos_cost = s_cost_vec + num_threads;
			T *s_extra_temp = s_eePos_cost + 6;

            T err, x;

            grid::end_effector_positions_device<T>(s_eePos_cost, s_xu, s_extra_temp, d_robotModel);

			for(int i = threadIdx.x; i < num_threads; i += blockDim.x){
                err = s_eePos_cost[i] - s_eePos_traj[i];
                x = s_xu[i-3]; // offset by 3 to account for ee_pos
                if (i < 3){ 
                    // ee_pos cost
                    s_cost_vec[i] = ee_pos_cost * err * err;
                } else if (i < 3 + state_size/2){                  
                    // joint barrier cost
                    s_cost_vec[i] = COST_BARRIER<T>() * jointBarrier(x, POS_LIMITS<T>()[i][0], POS_LIMITS<T>()[i][1]);
				} else if (i < state_size){ 
                    // joint velocity cost
                    s_cost_vec[i] = static_cast<T>(0.5) * COST_QD<T>() * x * x;
                    // joint velocity barrier cost
                    //s_cost_vec[i] += COST_BARRIER<T>() * jointBarrier(x, VEL_LIMITS<T>()[i][0], VEL_LIMITS<T>()[i][1]);
				} else {
                    // control cost
					s_cost_vec[i] = static_cast<T>(0.5) * COST_R<T>() * x * x;
				}
			}

			__syncthreads();
			glass::reduce<T>(num_threads, s_cost_vec); // TODO: use https://nvidia.github.io/cccl/cub/api/enum_namespacecub_1add0251c713859b8974806079e498d10a.html
			__syncthreads();
			
			return s_cost_vec[0];
		}

		template <typename T, bool computeR=true>
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
											void *d_robotModel)
		{	
            const uint32_t num_threads = state_size + control_size * computeR;
			const T ee_pos_cost = (blockIdx.x < gridDim.x - 1) ? COST_EE_POS<T>() : COST_EE_POS_TERMINAL<T>();
            
			T *s_eePos = s_temp;
			T *s_eePos_grad = s_eePos + 6;
			T *s_scratch = s_eePos_grad + 6 * state_size/2;

			T x_err, y_err, z_err, x;

			grid::end_effector_positions_device<T>(s_eePos, s_xu, s_scratch, (grid::robotModel<T> *)d_robotModel);
			grid::end_effector_positions_gradient_device<T>(s_eePos_grad, s_xu, s_scratch, (grid::robotModel<T> *)d_robotModel);

            // gradient
			for (int i = threadIdx.x; i < num_threads; i += blockDim.x){
				if(i < state_size/2){ // joint position gradient
                    x_err = (s_eePos[0] - s_eePos_traj[0]);
                    y_err = (s_eePos[1] - s_eePos_traj[1]);
                    z_err = (s_eePos[2] - s_eePos_traj[2]);

                    s_qk[i] = s_eePos_grad[6 * i + 0] * x_err + 
                                s_eePos_grad[6 * i + 1] * y_err + 
                                s_eePos_grad[6 * i + 2] * z_err;
                    s_qk[i] *= ee_pos_cost;

                    s_qk[i] += COST_BARRIER<T>() * jointBarrierGradient(s_xu[i], POS_LIMITS<T>()[i][0], POS_LIMITS<T>()[i][1]);
				} else if (i < state_size){ // joint velocity gradient
                    x = s_xu[i];
					s_qk[i] = COST_QD<T>() * x;
                    //s_qk[i] += COST_BARRIER<T>() * jointBarrierGradient(x, VEL_LIMITS<T>()[i][0], VEL_LIMITS<T>()[i][1]);
				} else { // control gradient
					s_rk[i - state_size] = COST_R<T>() * s_xu[i];
				}
			}

            // hessian
			for (int i = threadIdx.x; i < num_threads; i += blockDim.x){
                if (i < state_size){
                    for(int j = 0; j < state_size; j++){
                        // joint position hessian
                        if(j < state_size / 2 && i < state_size / 2){
                            s_Qk[i*state_size + j] = s_eePos_grad[6 * i + 0] * s_eePos_grad[6 * j + 0] + 
                                                        s_eePos_grad[6 * i + 1] * s_eePos_grad[6 * j + 1] + 
                                                        s_eePos_grad[6 * i + 2] * s_eePos_grad[6 * j + 2];

                            s_Qk[i*state_size + j] *= ee_pos_cost;

                            s_Qk[i*state_size + j] += (i == j) ?
                                COST_BARRIER<T>() * jointBarrierHessian(s_xu[i], POS_LIMITS<T>()[i][0], POS_LIMITS<T>()[i][1]) 
                                : static_cast<T>(0);
                        } else { 
                            // joint velocity hessian   
                            s_Qk[i*state_size + j] = (i == j) ? COST_QD<T>() : static_cast<T>(0); 
                                // + COST_BARRIER<T>() * jointBarrierHessian(s_xu[i], VEL_LIMITS<T>()[i][0], VEL_LIMITS<T>()[i][1]) 
                                //: static_cast<T>(0);
                        }
                    }
                } else {
                    // control hessian
                    uint32_t offset = i - state_size;
                    for(int j = 0; j < control_size; j++){
						s_Rk[(offset)*control_size+j] = (offset == j) ? COST_R<T>() : static_cast<T>(0);
					}
				}
			}
		}

		// last block doesn't compute control gradient and hessian
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
			trackingCostGradientAndHessian<T, false>(state_size, control_size, s_xux, &s_eePos_traj[6], s_Qkp1, s_qkp1, nullptr, nullptr, s_temp, d_dynMem_const);
		}
	}
}