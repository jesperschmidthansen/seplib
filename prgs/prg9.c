
/** @example prg9.c
 *
 * Brownian dynamics - Fokker-Planck level (small system) 
 *
 * Tests the temperature
 *
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys system;
  sepret returns;
  sepsampler sampler;
  double t, dt, cutoff,
    dens, lbox,
    etot, sump, tstart;
  int natoms, n, nloops,
    lvacf, tvacf;
 
  // Setting parameter values 
  natoms = 216;             // Number of atoms
  dens = 0.75;               // Density
  tstart = 1.12;             // temperature

  cutoff = SEP_WCACF;         // Potential cut-off radius
  dt = 0.001;               // Integrator time step
  nloops = 10000;           // Number of loops
  lbox = pow(natoms/dens, 1.0/3.0);  // The box lengths
  
  // Allocating memory 
  atoms = sep_init(natoms, SEP_NO_NEIGHB);

  // Setting up the system
  system = sep_sys_setup(lbox, lbox, lbox, cutoff, dt, natoms, SEP_BRUTE);
  
  // Initializing the positions and momenta
  sep_set_lattice(atoms, system);
  sep_set_vel_seed(atoms, tstart, 42, system);

  // Main loop 
  t=0.0; n = 0;
  while ( n<nloops ){
  
    // Reset return values 
    sep_reset_retval(&returns);

    // Reset force 
    sep_reset_force(atoms, &system);

    // Evaluate forces acting on between part.
    sep_force_pairs(atoms, "AA", cutoff, sep_lj_shift, &system, &returns, SEP_ALL);
    
    // Integrate particles forward in time
    sep_fp(atoms, tstart, &system, &returns);
        		      
    // Printing stuff 
    if ( n%100 == 0 ){
      printf("%d %.2f %f %f %f \n", n, returns.ekin*2/(3*natoms),
	     atoms[10].x[0],atoms[10].x[1],atoms[10].x[2]);
      SEP_FLUSH;
    }

    t += dt; n++;
  }
   
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&system);
  
  return 0;
} 
