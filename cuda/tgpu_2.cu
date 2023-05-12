// Molecules + electro-statics
// Check against Wu et al. SPC/Fw water

#include "sepcuda.h"
#include "sepcudamol.h"


int main(void){
	
	sepcupart *aptr = sep_cuda_load_xyz("start_water.xyz");
	sepcusys *sptr = sep_cuda_sys_setup(aptr);
	
	sepcumol *mptr = sep_cuda_init_mol();
	sep_cuda_read_bonds(aptr, mptr, "start_water.top");
	sep_cuda_read_angles(aptr, mptr, "start_water.top");
	
	float ljparam[3]={1.0,1.0,2.5};
	
	sptr->dt = 0.0005;
	
	int nloops = 10000; int counter = 0; char filestr[100];
	for ( int n=0; n<nloops; n++ ){

		sep_cuda_reset_iteration(aptr, sptr);
		
		if ( n%10==0 ){
			sep_cuda_update_neighblist(aptr, sptr, 2.5);
		}
		
		sep_cuda_force_lj(aptr, "OO", ljparam);
		sep_cuda_force_sf(aptr, 2.5);
		
		sep_cuda_force_harmonic(aptr, mptr, 0, 68000, .316);
		sep_cuda_force_angle(aptr, mptr, 0, 490 , 1.97);
		
		sep_cuda_thermostat_nh(aptr, sptr, 3.86, 0.1);
		sep_cuda_integrate_leapfrog(aptr, sptr);
		
		if ( n%100==0 ){
			sprintf(filestr, "molsim-%05d.xyz", counter);
			sep_cuda_save_xyz(aptr, filestr);
			counter ++;
		}
		
	}

	sep_cuda_save_xyz(aptr, "test.xyz");
	
	sep_cuda_free_memory(aptr, sptr);
	sep_cuda_free_mols(mptr);
	
	return 0;
}
