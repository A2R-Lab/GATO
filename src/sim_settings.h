#pragma once

/*******************************************************************************
 *                           Test Settings                               *
 *******************************************************************************/

#ifndef SAVE_DATA //save stats from mpcsim to file
#define SAVE_DATA   0
#endif 

#ifndef TEST_ITERS //number of times to test each trajectory for track_iiwa_pcg.cu
#define TEST_ITERS 1
#endif

/*******************************************************************************
 *                           Print Settings                      *
 *******************************************************************************/

#ifndef LIVE_PRINT_PATH
#define LIVE_PRINT_PATH 0
#endif 

#ifndef LIVE_PRINT_STATS
#define LIVE_PRINT_STATS 0
#endif

/*******************************************************************************
 *                           MPC Settings                               *
 *******************************************************************************/


#ifndef SIMULATION_PERIOD // how long to simulate the system during mpcsim loop (us) if CONST_UPDATE_FREQ == 1, otherwise uses last sqp solve time
#define SIMULATION_PERIOD 2500
#endif

#ifndef REMOVE_JITTERS // run sqp solver a bunch of times before starting to track
#define REMOVE_JITTERS 0
#endif

#ifndef SHIFT_THRESHOLD // this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#define SHIFT_THRESHOLD (1 * timestep)
#endif


