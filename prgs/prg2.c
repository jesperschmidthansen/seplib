
/** @example prg2.c
 * 
 * Butane with flexible bonds 
 * epsilon/kB = 72.1 K, m = 14.5 a.m.u , sigma = 3.9 Å 
 *
 * Tests bond, angle and torsion forces. 
 * Tests the molecular hydrodynamic correlation function sampler, 
 * molecular stress and molecular vel. acf.
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  sepmol *mols;
  double t, lbox[3],
    etot, sump;
  int natoms,  n;
 
  // Setting parameter values 
  double dt = 0.001;
  double alpha= 0.1;
  double temp = 4.0;
  double rbcoef[] = {15.5000,  20.3050, -21.9170, -5.1150,  43.8340, -52.6070};

  // Load atom info and allocate memory
  atoms = sep_init_xyz(lbox, &natoms, "prg1.xyz", 'v');

  // Setting system 
  sys = sep_sys_setup(lbox[0], lbox[1], lbox[2], 2.5, dt, natoms, 
		      SEP_LLIST_NEIGHBLIST);

  // Reading the molecular topologies 
  sep_read_topology_file(atoms, "prg1.top", &sys, 'v');

  // Initialize the mol structure
  mols = sep_init_mol(atoms, &sys);

  // Sampler
  sepsampler sampler = sep_init_sampler();
  sep_add_mol_sampler(&sampler, mols);
  sep_add_sampler(&sampler, "msacf", sys, 100, 5.0);
  sep_add_sampler(&sampler, "mvacf", sys, 100, 5.0);
  sep_add_sampler(&sampler, "mgh", sys, 100, 5.0, 10, true);
  
  // Main loop 
  t=0.0; n = 0; 
  while ( n < 10000 ){
	 
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);
    sep_reset_force_mol(&sys);
    
    // Evaluate forces acting on between part. 
    sep_force_pairs(atoms, "CC", 2.5, sep_lj_shift, &sys, &ret, SEP_EXCL_SAME_MOL);

    // Bond forces
    sep_stretch_harmonic(atoms, 0, 0.407, 2074.0, &sys, &ret);

    // Angles forces (here harmonic)
    sep_angle_harmonic(atoms, 0, 1.90, 400.0, &sys, &ret);

    // RB-torsion potential
    sep_torsion_Ryckaert(atoms, 0, rbcoef, &sys, &ret);

    // Integrate particles forward in time 
    sep_nosehoover(atoms, 'C', temp, &alpha, 0.1, &sys);
    sep_leapfrog(atoms, &sys, &ret);

    // Sample
    sep_sample(atoms, &sampler, &ret, sys, n);

    // Printing stuff 
    if ( n%100 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      etot  = (ret.epot + ret.ekin)/natoms;

      sep_mol_pressure_tensor(atoms, mols, &ret, &sys); 
      sep_pressure_tensor(&ret, &sys);

      printf("%d %.2f %f %f %f %.3f %1.4e %.2f %.2f\n", n, t, 
	     ret.epot/natoms, ret.ekin/natoms, etot, 
	     ret.ekin*2.0/(3.0*natoms), sump, ret.p, ret.p_mol);
      
      SEP_FLUSH;
    }	
	
    t += dt; n++;
  }

  sep_save_xyz(atoms, "C", "final.xyz", "w",  sys);

  /* Freeing memory */
  sep_free_mol(mols, &sys);
  sep_close(atoms, natoms);

  sep_free_sys(&sys);

  return 0;
} 

