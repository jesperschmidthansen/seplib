
/** @example prg4.c
 *
 * Simple NVE Lennard-Jones system
 * 
 * Tests openMP optimization model I
 */

#include "sep.h"

int main( int argc, char **argv ){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  double dt, 
    dens, temp, lbox,
    etot, sump;
  int natoms, n, nthreads;

  if ( argc != 2 )
    sep_error("Incorrect number of argument");
  
  // Setting parameter values
  natoms = 10000;
  dens = 0.80;
  dt   = 0.005;
  lbox = pow(natoms/dens, 1.0/3.0);
  temp = 1.2;
  nthreads = atoi(argv[1]);
  
  // Allocating memory 
  atoms = sep_init(natoms, SEP_NEIGHB);
  
  // Setting up the system
  sys = sep_sys_setup(lbox, lbox, lbox, 2.5, dt, natoms, SEP_LLIST_NEIGHBLIST);
  
  // Initializing the positions and momenta
  sep_set_lattice(atoms, sys);
  sep_set_vel_seed(atoms, temp, 42, sys);

  // Openmp support
  sep_set_omp(nthreads, &sys);

  // Increase the skin to reduce the neighbour list update
  sep_set_skin(&sys, 1.0);
  
  // Main loop 
  n = 0;
  while ( n<1000 ){
  
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);

    // Evaluate forces acting on between part.
    sep_force_pairs(atoms, "AA", 2.5, sep_lj_shift, &sys, &ret, SEP_ALL);

    // Integrate particles forward in time
    sep_leapfrog(atoms, &sys, &ret);
        
    // Printing stuff 
    if ( n%100 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      etot  = (ret.epot + ret.ekin)/natoms;

      printf("%d %.3f %.3f %.10f %1.4e %.1f\n", n, ret.epot/natoms,
	     ret.ekin/natoms, etot, sump, (double)n/sys.nupdate_neighb);
      SEP_FLUSH;
    }
    
    n++;
  }
   
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&sys);
  
  return 0;
} 
