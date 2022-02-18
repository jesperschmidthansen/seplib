

/** @example prg6.c
 *
 * DPD simulation
 *
 * Tests Groot and Warren integrator
 */ 

#include "sep.h"

int main( void ){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  double t, dt, tend,
    dens, lbox,
    etot, sump, 
    temp, sigma,
    a_AA, cf;
  int natoms, n;

  // Setting parameter values 
  dens  = 3.0;
  temp  = 1.0;
  sigma = 3.0;
  tend  = 100.0;

  natoms = 1024;
  dt     = 0.02;
  lbox   = pow(natoms/dens, 1./3.);
  cf     = 1.0;
  a_AA   = 25.0;
  
  // Setting the system
  atoms = sep_init(natoms, SEP_NUM_NEIGHB);
  sys   = sep_sys_setup(lbox, lbox, lbox, cf, dt, natoms, SEP_LLIST_NEIGHBLIST);

  sep_set_lattice(atoms, sys);
  sep_set_vel_seed(atoms, temp, 42, sys);

  sep_set_omp(2, &sys);
  //  sep_set_skin(&sys, 2.0);
  
  // Main loop 
  t=0.0; n = 0;
  while ( t <= tend ){
	 
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);

    // Evaluatforces acting on between part.
    sep_force_dpd(atoms, "AA", cf, a_AA, temp, sigma, &sys, &ret, SEP_ALL);

    // Integrate forward
    sep_verlet_dpd(atoms, 0.5, n, &sys, &ret);
    
    // Printing stuff 
    if ( n%100 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      etot  = (ret.epot + ret.ekin)/natoms;

      sep_pressure_tensor(&ret, &sys);
     
      printf("%d %.2f %f %f %f %f %1.4e %.3f\n", n, t, 
	     ret.epot/natoms, ret.ekin/natoms, ret.ekin*2/(3*natoms), 
	     etot, sump, ret.p);
      SEP_FLUSH;
    }

    t += dt; n++;
  }
  
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&sys);

  return 0;
} 


