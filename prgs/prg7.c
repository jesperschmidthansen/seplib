
/** @example prg7.c
 *
 * NPT of Lennard-Jones system
 * 
 * Tests the Berendsen barostat
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  double t, dt, 
    dens, lbox,
    etot, sump, alpha;
  int natoms, n;
   
  // Setting parameter values 
  dens = 0.844; 
  natoms = 216;
  dt = 0.005;
  lbox = pow(natoms/dens, 1.0/3.0);
  alpha = 0.1;
  
  // Allocating memory 
  atoms = sep_init(natoms, SEP_NO_NEIGHB);
   
  // Setting up the system
  sys = sep_sys_setup(lbox, lbox, lbox, 2.5, dt, natoms, SEP_BRUTE);

  // Initializing the positions and momenta
  sep_set_lattice(atoms, sys);
  sep_set_vel(atoms, 0.728, sys);
 
  // Main loop 
  t=0.0; n = 0;
  while ( t <= 50.0 ){
	 
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);

    // Evaluate forces acting on between part. 
    sep_force_pairs(atoms, "AA", 2.5, sep_lj_shift, &sys, &ret, SEP_ALL);

    // Integrate particles forward in time 
    sep_nosehoover(atoms, 0.5, &alpha, 0.1, &sys);
    sep_leapfrog(atoms, &sys, &ret);
   
    // The Berensen barostat
    sep_berendsen(atoms, 5.91, 0.1, &ret, &sys);
		
    // Printing stuff 
    if ( n%100 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      etot  = (ret.epot + ret.ekin)/natoms;
   
      printf("%d %.2f %f %f %f %f %1.4e %.3f %.3f\n", n, t, 
	     ret.epot/natoms, ret.ekin/natoms, ret.ekin*2/(3*natoms), 
	     etot, sump, ret.p, sys.volume);
      SEP_FLUSH;
    }
	 
    t += dt; n++;
  }
   
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&sys);

  return 0;
} 
