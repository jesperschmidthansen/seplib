
/** @example prg8.c
 *
 * Slit-pore simulations
 *
 * Test for virtuel lattice and prof sampler
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  double t, dt, rcut, rcutWW, lbox[3], temp;
  int natoms, n, nloops;
    
  // Setting parameter values 
  temp = 1.4;
  rcut = 2.5; rcutWW=pow(2.0, 1./6.);
  nloops = 10000;
  dt = 0.005;
 
  // Allocating memory 
  atoms = sep_init_xyz(lbox, &natoms, "prg8.xyz", 'q');
      
  // Setting up the system
  sys = sep_sys_setup(lbox[0], lbox[1], lbox[2], rcut, dt,
		      natoms, SEP_LLIST_NEIGHBLIST);

  sep_set_x0(atoms, sys.npart);

  sepsampler sampler = sep_init_sampler();
  sep_add_sampler(&sampler, "profs", sys, 100, 'F', 10);
  
  
  // Main loop 
  t=0.0; n = 0;
  while ( n<nloops ){
  
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);

    // Evaluate forces acting on between part. 
    sep_force_pairs(atoms, "FF", rcut, sep_lj_shift, &sys, &ret, SEP_ALL);
    sep_force_pairs(atoms, "WF", rcut, sep_lj_shift, &sys, &ret, SEP_ALL);
    sep_force_pairs(atoms, "WW", rcutWW, sep_wca, &sys, &ret, SEP_ALL);

    sep_force_x0(atoms, 'W',  sep_spring_x0, &sys);
    
    // Integrate particles forward in time
    sep_leapfrog(atoms, &sys, &ret);
    sep_relax_temp(atoms, 'W', temp, 0.01, &sys);

    sep_sample(atoms, &sampler, &ret, sys, n);

    if ( n%100==0 ){
      printf("%d %f\n", n, 2.0/3.0*ret.ekin/sys.npart);
      fflush(stdout);
    }
    
    t += dt; n++;
  }

  sep_save_xyz(atoms, "WF", "slitpore.xyz", "w", &sys);
  
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&sys);

  return 0;
} 
