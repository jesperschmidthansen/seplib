// Molecules + electro-statics
// Check against Wu et al. SPC/Fw water

#include "sepcuda.h"

int main(void){

	sepcupart *aptr = sep_cuda_load_xyz("start_water.xyz");
	sepcusys *sptr = sep_cuda_sys_setup(aptr);
	
	sepcumol *mptr = sep_cuda_init_mol();
	sep_cuda_read_bonds(aptr, mptr, "start_water.top", 'q');
	sep_cuda_read_angles(aptr, mptr, "start_water.top", 'q');
	
	sep_cuda_set_exclusion(aptr, "molecule");
	
	float ljparam[3]={1.0,1.0,2.5};
	
	sptr->dt = 0.0005;

	int sintdip = 1000;	int sintstrs = 10;
	sep_cuda_set_molforcecalc_on(sptr, sintstrs);

	sepcusampler_stress* stresscorr = sep_cuda_sample_stress_init(sptr, 100, 5, sptr->dt*sintstrs);
	sepcusampler_dipole* polcorr = sep_cuda_sample_dipole_init(sptr, 100, 10, sptr->dt*sintdip);
	
	FILE *fout = fopen("test.dat", "w");

	int nloops = 10000000; int counter = 0; char filestr[100];
	for ( int n=0; n<nloops; n++ ){
	
		sep_cuda_reset_iteration(aptr, sptr);
		
		if ( n%10==0 ){
			sep_cuda_update_neighblist(aptr, sptr, 2.9);
		}
		
		sep_cuda_force_lj(aptr, "OO", ljparam);
		sep_cuda_force_sf(aptr, 2.9);
		
		sep_cuda_force_harmonic(aptr, mptr, 0, 68000, 0.316);
		sep_cuda_force_angle(aptr, mptr, 0, 490 , 1.97);
		
		sep_cuda_thermostat_nh(aptr, sptr, 3.86, 0.1);
		sep_cuda_integrate_leapfrog(aptr, sptr);

		if ( n%sintdip==0 )
		 	sep_cuda_sample_dipole(polcorr, aptr, sptr, mptr);
		if ( n%sintstrs==0 )
			sep_cuda_sample_stress(stresscorr, aptr, sptr, mptr);
			
	 	//double  P[9];
		//sep_cuda_mol_calc_cmprop(aptr, mptr);
		//sep_cuda_mol_calc_dipoles(aptr, mptr); 
		//sep_cuda_mol_calc_molpress(P, aptr, mptr);
		//double mu = sep_cuda_mol_calc_avdipole(mptr);

		//for ( int k=0; k<9; k++ ) printf("%f ", P[k]);
		//for ( int k=0; k<3; k++ ) printf("%f ", p[k]); 
		//printf("%f\n", mu);

		if 	( n%10000==0 ){
			sprintf(filestr, "molsim-%05d.xyz", counter);
			sep_cuda_save_xyz(aptr, filestr);
			counter ++;
		}

		if ( n%1000==0 ){printf("\r%d  ", n); fflush(stdout);}

	}


	sep_cuda_save_xyz(aptr, "test.xyz");
		
	sep_cuda_sample_stress_free(stresscorr); 
	sep_cuda_sample_dipole_free(polcorr);
	
	sep_cuda_free_memory(aptr, sptr);
	
	sep_cuda_free_bonds(mptr);
	sep_cuda_free_angles(mptr);

	return 0;
}
