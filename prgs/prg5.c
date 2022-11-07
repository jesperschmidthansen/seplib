
/** @example prg5.c
 *
 * Butane simulation.
 *
 * Tests openMP optimization model II 
 */

#include "sep.h"

int main(void){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  sepmol *mols;
  double t, lbox[3], rho=0.0;
  int natoms,  n;
 
  // Setting parameter values 
  double dt = 0.001;
  double alpha  = 0.1;
  double desired_temp = 4.0;
  double desired_rho = 1.46;
  double rbcoef[] = {15.5000,  20.3050, -21.9170, -5.1150,  43.8340, -52.6070};

  // Load atom info and allocate memory
  atoms = sep_init_xyz(lbox, &natoms, "prg1.xyz", 'v');

  // Setting system 
  sys = sep_sys_setup(lbox[0], lbox[1], lbox[2], 2.5, dt, natoms, 
		      SEP_LLIST_NEIGHBLIST);

  // Reading positions 
  sep_read_topology_file(atoms, "prg1.top", &sys,  'v');

  // Initialize the mol structure
  mols = sep_init_mol(atoms, &sys);

  // Used to calculate the different force contributions
  double **f_1 = sep_matrix(natoms, 3);
  double **f_2 = sep_matrix(natoms, 3);

  sep_set_omp(2, &sys);
      
  sep_set_skin(&sys, 1.0);

  // Main loop 
  t=0.0; n = 0; 
  while ( n < 1000 ){
	 
    // Reset return values 
    sep_reset_retval(&ret);
    sep_reset_force(atoms, &sys);
	
    // Evaluate forces acting on between part.
    sep_force_pairs(atoms, "CC", 2.5, sep_lj_shift, &sys, &ret, SEP_EXCL_SAME_MOL);
  
#pragma omp parallel sections
    {
#pragma omp section
      {
	sep_matrix_set(f_1, natoms, 3, 0.0);
	sep_omp_bond(f_1, atoms, 0, 0.407, 2074.0, &sys);
	sep_omp_angle(f_1, atoms, 0, 1.90, 425.0, &sys);
      }
#pragma omp section
      {      
	sep_matrix_set(f_2, natoms, 3, 0.0);
	sep_omp_torsion(f_2, atoms, 0, rbcoef, &sys);
      }
    }

    
    // Collecting the forces
    for ( int i=0; i<natoms; i++ ){
      for ( int k=0; k<3; k++ ){
	atoms[i].f[k] += f_1[i][k] + f_2[i][k];
      }
    }
      
    // Integrate particles forward in time 
    sep_nosehoover(atoms, desired_temp, &alpha, 0.1, &sys);
    sep_leapfrog(atoms, &sys, &ret);
     
    // Compress - equilibrate
    if ( n%10 == 0 && rho < desired_rho ) {
      sep_compress_box(atoms, desired_rho, 0.998, &sys);
      sep_reset_momentum(atoms, 'C', &sys);
    }

    // Printing stuff 
    if ( n%100 == 0 ){
      printf("%d %.2f %.3f\n", n, t, ret.ekin*2.0/(3.0*natoms));
      SEP_FLUSH;
    }	
	
    t += dt; n++;
  }

  sep_free_matrix(f_1, natoms);
  sep_free_matrix(f_2, natoms);
  
  sep_save_xyz(atoms, "C", "final.xyz", "w",  sys);

  /* Freeing memory */
  sep_free_mol(mols, &sys);
  sep_close(atoms, natoms);
  sep_free_sys(&sys);

  return 0;
} 
