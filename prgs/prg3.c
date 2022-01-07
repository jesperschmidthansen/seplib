
/** @example prg3.c
 * 
 * SPC/Fw-SF water (see Wu et al. JCP (2007))
 * espilon/kB = 78.2, m=16 a.m.u, sigma = 3.16 Å
 *
 * Tests for compression and SF Coulomb
 *  
 */

#include "sep.h"

#define CE 26.30 
#define SIGMA 3.16

int main(void){
  sepatom *atoms;
  sepsys sys;
  sepret ret;
  sepmol *mols;
  double t, lbox[3], sump, rho=0.0;
  int natoms,  n;

  // Setting parameter values 
  double dt = 5.0e-4;
  double alpha[3] = {0.1};
  double temp = 4.0;
  double drho = 3.15;
 
  double lbond = 0.316;
  double angle = 1.97;
  
  double kspring = 68421;
  double kangle = 490;

  double cf = 2.9;

  // Load atom info and allocate memory
  atoms = sep_init_xyz(lbox, &natoms, "prg2.xyz", 'q');
  
  // Setting system 
  sys = sep_sys_setup(lbox[0], lbox[1], lbox[2], cf, dt, natoms, SEP_BRUTE);
  
  // Reading positions 
  sep_read_topology_file(atoms, "prg2.top",  &sys, 'q');

  // Setting initial velocity 
  sep_set_vel(atoms, temp, sys);
  
  // Initialize the mol structure
  mols = sep_init_mol(atoms, &sys);

  // Main loop 
  t=0.0; n = 0; 
  while ( n < 5000 ){
    
    // Reset return values 
    sep_reset_retval(&ret);

    // Reset force 
    sep_reset_force(atoms, &sys);
    sep_reset_force_mol(&sys);
    
    // Evaluate forces acting on between part. 
    sep_force_pairs(atoms, "OO", 2.5, sep_lj_shift, &sys, &ret, SEP_EXCL_SAME_MOL);
    
    sep_stretch_harmonic(atoms, 0, lbond, kspring, &sys, &ret);
    sep_angle_cossq(atoms, 0, angle, kangle, &sys, &ret);
    
    sep_coulomb_sf(atoms, cf, &sys, &ret, SEP_EXCL_SAME_MOL);

    // Integrate particles forward in time 
    sep_nosehoover(atoms, temp, alpha, 10.0, &sys);
    sep_leapfrog(atoms, &sys, &ret);

    // Isotropic compression 
    if ( n%10 == 0 && rho < drho ) {
      sep_compress_box(atoms, drho, 0.995, &sys);
      sep_reset_momentum(atoms, 'O', &sys);
      sep_reset_momentum(atoms, 'H', &sys);
    }

    // Printing stuff 
    if ( n%100 == 0 ){
      sump  = sep_eval_mom(atoms, natoms);
      rho = sys.npart/sys.volume;

      sep_pressure_tensor(&ret, &sys);
      sep_mol_pressure_tensor(atoms, mols, &ret, &sys);
      
      printf("%d %.2f %.3f %1.4e %.3f %.3f %.3f %.3f\n", n, t, 
	     ret.ekin*2.0/(3.0*natoms), sump, rho, 
	     ret.epot/sys.molptr->num_mols, ret.p, ret.p_mol);
      SEP_FLUSH;
    }
	
    t += dt; n++;
  }

  sep_save_xyz(atoms, "OH", "final.xyz", "w", sys);

  /* Freeing memory */
  sep_free_mol(mols, &sys);
  sep_close(atoms, natoms);
  sep_free_sys(&sys);

  return 0;
} 


void radialdistr(sepatom *atom, double lbox, int npart, int opt){
  static int count = 0;
  static double *histOO, *histOH, *histHH; 
  double r, rv[3], r2, dg, rr;
  int i, j, k, index, nhist;
  FILE *fout;

  nhist = 0.5*lbox/0.01;

  if ( count== 0 ){
    histOO = sep_vector(nhist);
    histHH = sep_vector(nhist);
    histOH = sep_vector(nhist);
  }
  
  switch (opt){
    
  case 0: 

    dg = 0.5*lbox/nhist;
 
    for ( i=0; i<npart-1; i++ ){
      for ( j=i+1; j<npart; j++ ){
       
	if ( atom[i].molindex == atom[j].molindex ) continue;

	r2 = 0.0;
	for ( k=0; k<3; k++){
	  rv[k]  = atom[i].x[k]-atom[j].x[k];
	  sep_Wrap( rv[k], lbox );
	  r2   += rv[k]*rv[k];
	}
	
	r = sqrt(r2);
	index = (int)(r/dg);
	if ( index < nhist ){
	  if ( atom[j].type == 'O' && atom[i].type == 'O' )
	    histOO[index] += 2.0;
	  else if ( atom[j].type == 'H' && atom[i].type == 'H' )
	    histHH[index] += 2.0;
	  else 
	    histOH[index] += 2.0;
	}

      }
    }

    count++;
   
    fout = fopen("hist.dat", "w");
    if ( fout == NULL ) sep_error("Couldn't open file");

    for ( i=0; i<nhist; i++ ){
     
      double vi = pow(i*dg, 3.0);
      double vii = pow((i+1)*dg, 3.0);

      double dv = vii-vi;
      double fac = 1.0/(count*npart*dv);

      rr = (i+0.5)*dg*SIGMA;
      fprintf(fout, "%f %f %f %f\n", rr, histOO[i]*fac, 
	      histHH[i]*fac, histOH[i]*fac); 
    }
   
    fclose(fout);


    break;
    
  case 1:
    
    free(histHH); free(histOO); free(histOH);
    
    break;
  }

}
