/* 
* sepintgr.h - This file is a part of the sep-library 
*
* Copyright (C) 2008 Jesper Schmidt Hansen 
* 
* License: GPL - see COPYING for copying conditions.
* There is ABSOLUTELY NO WARRANTY, not even for MERCHANTIBILITY or
* FITNESS FOR A PARTICULAR PURPOSE.
*
* Contact: schmidt@zigzak.net
*/


#ifndef __SEPINTGR_H__
#define __SEPINTGR_H__

#include "sepmisc.h"
#include "sepstrct.h"


/**
 * Shifts/translates the particle positions (periodic bc) when  
 * particle is located outside box. 
 * @param atoms Pointer to seplib particle structure
 * @param n Particle index
 * @return True displacement 
 */
double sep_periodic(sepatom *atoms, unsigned n, sepsys *sys);

/**
 * Updates the positions and velocities using the leap-frog algorithm
 * @param ptr Pointer to the seplib particle structure
 * @param sys Pointer to the seplib system structure
 * @param retval Pointer to the seplib return structure
 */
void sep_leapfrog(seppart *ptr, sepsys *sys, sepret *retval);

/**
 * Updates the Nose-Hoover thermostat friction coefficient and adds the
 * corresponding force to particles of specified type. Used before and in
 * connection with the leap-frog algorithm 
 * @param ptr Pointer to the seplib particle structure
 * @param type Particle types to thermostat
 * @param Td Desired temperature
 * @param alpha Thermostat state; double array of length 3. Must be initialized eg alpha[3]={0.1}
 * @param Q Thermostat 'mass'
 * @param sys Pointer to seplib system structure
 */
void sep_nosehoover_type(seppart *ptr, char type, double Td, 
			 double *alpha, const double Q, sepsys *sys);

/**
 * Updates the Nose-Hoover thermostat friction coefficient and adds the
 * corresponding force to all system particles. Used before and in
 * connection with the leap-frog algorithm 
 * @param ptr Pointer to the seplib particle structure
 * @param Td Desired temperature
 * @param alpha Thermostat state; double array of length 3. Must be initialized eg alpha[3]={0.1}
 * @param Q Thermostat 'mass'
 * @param sys Pointer to seplib system structure
 */
void sep_nosehoover(seppart *ptr, double Td, double *alpha, const double Q, 
		    sepsys *sys);
 
/**
 * Updates the positions and velocties with Langevin dynamics - Fokker-Planck level. 
 * Currently the integrator applies to all system particles.
 * @param ptr Pointer to the seplib particle structure
 * @param temp_desired The desired system temperature
 * @param sys Pointer to seplib system structure
 * @param retval Pointer to seplib return structure 
 */
void sep_fp(seppart *ptr, double temp_desired, sepsys *sys, sepret *retval);

/**
 * Dissipative particle dynamics integrator ( Groot and Warren )
 * @param ptr Pointer to the seplib particle structure 
 * @param lambda Integrator parameter (recommendation is 0.5)
 * @param stepnow Iteration number
 * @param sys Pointer to seplib system structure
 * @param retval Pointer to seplib return structure
 */
void sep_verlet_dpd(seppart *ptr, double lambda, int stepnow,
		    sepsys *sys, sepret *retval);


#ifndef DOXYGEN_SKIP

void sep_set_shake(sepmol *mols, unsigned nuau, 
		   int nb, double *blength, sepsys sys);

void sep_shake(sepatom *atoms, sepmol *mols, sepsys *sys, 
	       double tol, sepret *retval);

void sep_set_leapfrog(sepsys *sys, double dt);

#endif

#endif
