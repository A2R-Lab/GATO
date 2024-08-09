#pragma once

/*******************************************************************************
 *                           SQP Settings                               *
 *******************************************************************************/

/* time_linsys = 1 to record linear system solve times. 
time_linsys = 0 to record number of sqp iterations. 
In both cases, the tracking error will also be recorded. */
#ifndef TIME_LINSYS
#define TIME_LINSYS 1
#endif

#if TIME_LINSYS == 1
    #define SQP_MAX_ITER   20
    typedef double toplevel_return_type;
#else
    #define SQP_MAX_ITER    40
    typedef uint32_t toplevel_return_type;
#endif

#ifndef LINSYS_SOLVE //use pcg or qdldl
#define LINSYS_SOLVE 1 
#endif

#ifndef USE_DOUBLES //change type of linsys_t (T)
#define USE_DOUBLES 0
#endif

#if USE_DOUBLES
typedef double linsys_t;
#else
typedef float linsys_t;
#endif

#ifndef CONST_UPDATE_FREQ //exits sqp solver early to reach a constant update frequency
#define CONST_UPDATE_FREQ 1
#endif

#ifndef SQP_MAX_TIME_US //max time to run sqp solver, if CONST_UPDATE_FREQ == 1 (us). think about matching with SIMULATION_PERIOD in sim_settings.h
#define SQP_MAX_TIME_US 2500 
#endif

/*******************************************************************************
 *                           Thread Settings                               *
 *******************************************************************************/

#ifndef KKT_THREADS
#define KKT_THREADS 128
#endif

#ifndef SCHUR_THREADS
#define SCHUR_THREADS 128
#endif

#ifndef PCG_NUM_THREADS
#define PCG_NUM_THREADS	32
#endif

#ifndef DZ_THREADS
#define DZ_THREADS 128
#endif 

#ifndef MERIT_THREADS
#define MERIT_THREADS 64
#endif 

/*******************************************************************************
 *                           PCG Settings                               *
 *******************************************************************************/

// Values found using experiments
#ifndef PCG_MAX_ITER
	#if LINSYS_SOLVE
// 		#if KNOT_POINTS == 32
// #define PCG_MAX_ITER 173 
// 		#elif KNOT_POINTS == 64
// #define PCG_MAX_ITER 167
// 		#elif KNOT_POINTS == 128
// #define PCG_MAX_ITER 167
// 		#elif KNOT_POINTS == 256
// #define PCG_MAX_ITER 118
// 		#elif KNOT_POINTS == 512
// #define PCG_MAX_ITER 67
// 		#else
// #define PCG_MAX_ITER 200	
// 		#endif	
		#define PCG_MAX_ITER 173 // TODO: knot points are now defined in gato.cuh
	#else 
#define PCG_MAX_ITER -1
#define PCG_EXIT_TOL -1 
	#endif
#endif

/*******************************************************************************
 *                           Rho Settings                               *
 *******************************************************************************/

#ifndef RHO_MIN
#define RHO_MIN 1e-3
#endif

//TODO: get rid of rho in defines
#ifndef RHO_FACTOR
#define RHO_FACTOR 1.2 
#endif

#ifndef RHO_MAX
#define RHO_MAX 10 
#endif