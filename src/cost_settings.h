#pragma once

//TODO: Select between different cost functions

// control effort penalty
#ifndef R_COST
	#if KNOT_POINTS == 64
#define R_COST .001 
	#else 
#define R_COST .0001 
	#endif
#endif

// penalty for error in derivative of state
#ifndef QD_COST
#define QD_COST .0001 
#endif