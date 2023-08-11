// Molecules + electro-statics
// Check against Wu et al. SPC/Fw water

#include "sepcuda.h"


int main(void){
	
	sepcupart *aptr = sep_cuda_load_xyz("tmp.xyz");
	sepcusys *sptr = sep_cuda_sys_setup(aptr);
	
	sepcumol *mptr = sep_cuda_init_mol();
	sep_cuda_read_bonds(aptr, mptr, "tmp.top");
	sep_cuda_read_angles(aptr, mptr, "tmp.top");
	
	sep_cuda_set_exclusion(aptr, "molecule");
	
	float ljparam[3]={1.0,1.0,2.5};
	
	sptr->dt = 0.0005;

	sepcumgh *sampler = sep_cuda_sample_mgh_init(sptr, 200, 5, 10*sptr->dt);
	sep_cuda_set_molprop_on(sptr);

	int nloops = 1000000; int counter = 0; char filestr[100];
	for ( int n=0; n<nloops; n++ ){
	
		sep_cuda_reset_iteration(aptr, sptr);
		
		if ( n%10==0 ){
			sep_cuda_update_neighblist(aptr, sptr, 3.0);
		}
		
		sep_cuda_force_lj(aptr, "OO", ljparam);
		sep_cuda_force_sf(aptr, 3.0);
		
		sep_cuda_force_harmonic(aptr, mptr, 0, 68000, 0.316);
		sep_cuda_force_angle(aptr, mptr, 0, 490 , 1.97);
		
		sep_cuda_thermostat_nh(aptr, sptr, 3.86, 0.1);
		sep_cuda_integrate_leapfrog(aptr, sptr);
	
		if ( n%10==0 )
			sep_cuda_sample_mgh(sampler, aptr, sptr, mptr);
	
		if ( n%1000==0 ){
			sprintf(filestr, "molsim-%05d.xyz", counter);
			sep_cuda_save_xyz(aptr, filestr);
			
			counter ++;
		}
		
	}

	sep_cuda_save_xyz(aptr, "test.xyz");

	sep_cuda_sample_mgh_free(sampler);

	sep_cuda_free_memory(aptr, sptr);
	
	sep_cuda_free_bonds(mptr);
	sep_cuda_free_angles(mptr);

	return 0;
}
