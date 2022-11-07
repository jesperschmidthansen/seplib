
/** @example prg1.c
 *
 * Standard NVT Lennard-Jones simulation (large system)
 *
 * Tests for Nose-Hoover thermostat and linked list + neighbour list.
 * Tests for velocity and stress autocorrelations and mean square displacement
 * samplers.  
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  sepsampler sampler;
  double t, dt, rcut, 
    dens, lbox, temp,
    etot, sump, tau, alpha=0.1;
  int natoms, n, nloops;

  // Setting parameter values 
  dens = 0.7;
  temp = 1.0;
  natoms = 1000;
  rcut = pow(2.0, 1./6.); 
  tau = 0.01;
  nloops = 50000;
  dt = 0.005;

  lbox = pow(natoms/dens, 1.0/3.0);

  // Allocating memory 
  atoms = sep_init(natoms, SEP_NEIGHB);
   
  // Setting up the system
  sys = sep_sys_setup(lbox, lbox, lbox, rcut, dt, natoms, SEP_LLIST_NEIGHBLIST);

  // Initializing the positions and momenta
  sep_set_lattice(atoms, sys);
  sep_set_vel_seed(atoms, temp, 42, sys);

  // Set samplers
  sampler = sep_init_sampler();
  sep_add_sampler(&sampler, "vacf", sys, 100, 5.0);
  sep_add_sampler(&sampler, "sacf", sys, 50, 2.0);
  sep_add_sampler(&sampler, "msd", sys, 125, 10.0, 5, 'A');

  // Main loop 
  t=0.0; n = 0;
  while ( n<nloops ){
  
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);

    // Evaluate forces acting on between part. Particle lable is 'A' as default 
    sep_force_pairs(atoms, "AA", rcut, sep_lj_shift, &sys, &ret, SEP_ALL);
      
    // Integrate particles forward in time
    sep_nosehoover(atoms, temp, &alpha, tau, &sys);
    sep_leapfrog(atoms, &sys, &ret);
    
    // Sample
    sep_sample(atoms, &sampler, &ret, sys, n);
		
    // Printing stuff 
    if ( n%1000 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      etot  = (ret.epot + ret.ekin)/natoms;
      sep_pressure_tensor(&ret, &sys);

      printf("%d %.2f %f %f %f %.10f %1.4e %.3f %f\n", n, t, 
	     ret.epot/natoms, ret.ekin/natoms, ret.ekin*2/(3*natoms), 
	     etot, sump, ret.p, (double)n/sys.nupdate_neighb);
      SEP_FLUSH;
    }
    
    t += dt; n++;
  }
   
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&sys);

  // Closing sampler
  sep_close_sampler(&sampler);

  return 0;
} 
