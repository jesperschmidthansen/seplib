
/** @example prg0.c
 *
 * Standard NVE Lennard-Jones simulation (small system) 
 *
 * Tests for simple energy and momentum conservation using brute-force
 * and leap-frog, On fail - stop everything!! 
 *
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys system;
  sepret returns;
  sepsampler sampler;
  double t, dt, rcut,
    dens, lbox,
    etot, sump, tstart;
  int natoms, n, nloops,
    lvacf, tvacf;
 
  // Setting parameter values 
  natoms = 216;             // Number of atoms
  dens = 0.8;               // Density
  tstart = 1.0;             // Start temperature

  rcut = 2.5;               // Potential cut-off radius
  dt = 0.005;               // Integrator time step
  nloops = 10000;           // Number of loops
  lbox = pow(natoms/dens, 1.0/3.0);  // The box lengths
  
  // Allocating memory 
  atoms = sep_init(natoms, SEP_NO_NEIGHB);

  // Setting up the system
  system = sep_sys_setup(lbox, lbox, lbox, rcut, dt, natoms, SEP_BRUTE);
  
  // Initializing the positions and momenta
  sep_set_lattice(atoms, system);
  sep_set_vel_seed(atoms, tstart, 42, system);
  double param[]={2.5, 1.0,1.0,1.0};
  
  // Main loop 
  t=0.0; n = 0;
  while ( n<nloops ){
  
    // Reset return values 
    sep_reset_retval(&returns);

    // Reset force 
    sep_reset_force(atoms, &system);

    // Evaluate forces acting on between part.
    sep_force_lj(atoms, "AA", param, &system, &returns, SEP_ALL);
      
    // Integrate particles forward in time
    sep_leapfrog(atoms, &system, &returns);
        		      
    // Printing stuff 
    if ( n%1000 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      etot  = (returns.epot + returns.ekin)/natoms;
      sep_pressure_tensor(&returns, &system);

      printf("%d %.2f %f %f %f %.10f %1.4e %.5f %f\n", n, t, 
	     returns.epot/natoms, returns.ekin/natoms,
	     returns.ekin*2/(3*natoms), etot, sump, returns.p, atoms[0].f[0]);
      SEP_FLUSH;
    }

    t += dt; n++;
  }
   
  // Freeing memory 
  sep_close(atoms, natoms);
  sep_free_sys(&system);
  
  return 0;
} 
