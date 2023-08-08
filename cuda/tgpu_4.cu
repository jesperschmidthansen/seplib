// Molecules with dihedrals - propanol
// Check against rdf_prop_OO.dat and rdf_prop_OH.dat 
// Akiyama et al., Journal of Solution Chemistry, Vol. 33, Nos. 6/7, 797

#include "sepcuda.h"

__global__ void printlist(int index, int *dlist){
	
	int offset = index*(SEP_MAX_NUMB_EXCLUSION+1);
	printf("Device %d -> ", dlist[offset]);
	for ( int n=1; n<=SEP_MAX_NUMB_EXCLUSION; n++ )
		printf("%d ", dlist[offset+n]);
	printf("\n");
}

void printlistHost(int index, int *hlist){
	
	
	int offset = index*(SEP_MAX_NUMB_EXCLUSION+1);
	printf("Host %d -> ", hlist[offset]);
	for ( int n=1; n<=SEP_MAX_NUMB_EXCLUSION; n++ )
		printf("%d ", hlist[offset+n]);
	printf("\n");
	
}

int main(void){
	
	sepcupart *aptr = sep_cuda_load_xyz("start_propanol.xyz");

	sepcusys *sptr = sep_cuda_sys_setup(aptr);
	
	sepcumol *mptr = sep_cuda_init_mol();
	
	sep_cuda_read_bonds(aptr, mptr, "start_propanol.top");
	sep_cuda_read_angles(aptr, mptr, "start_propanol.top");
	sep_cuda_read_dihedrals(aptr, mptr, "start_propanol.top");

	sep_cuda_set_exclusion(aptr, "molecule");
	
	float ljparamCC[3]={1.24,1.05,2.5}; 
	float ljparamcc[3]={1.31,0.49,2.5};
	float ljparamOO[3]={1.00,1.00,2.5};
	
	float ljparamCc[3]={1.28, 0.72, 2.5};
	float ljparamCO[3]={1.12, 1.02, 2.5};
	
	float ljparamcO[3]={1.16, 0.70, 2.5};
	
	float rbparam0[6] = {9.0,   22.9,   1.1,  -33.1, 0.0,  0.0};
	float rbparam1[6] = {3.6,   3.8,    0.62, -8.1,  0.0,  0.0};
	
	sptr->dt = 0.0005;
	sepcugh *ghptr = sep_cuda_sample_gh_init(sptr, 50, 20, 10*sptr->dt);
	
	int nloops = 1000; int counter = 0; char filestr[100];
	for ( int n=0; n<nloops; n++ ){

	
		sep_cuda_reset_iteration(aptr, sptr);
		
		if ( n%10==0 ){
			sep_cuda_update_neighblist(aptr, sptr, 3.5);
		}
		
		sep_cuda_force_lj(aptr, "CC", ljparamCC);
		sep_cuda_force_lj(aptr, "cc", ljparamcc);
		sep_cuda_force_lj(aptr, "OO", ljparamOO);
		sep_cuda_force_lj(aptr, "Cc", ljparamCc);
		sep_cuda_force_lj(aptr, "CO", ljparamCO);
		sep_cuda_force_lj(aptr, "cO", ljparamcO);
		
		sep_cuda_force_sf(aptr, 3.5);
		
		sep_cuda_force_harmonic(aptr, mptr, 0, 31288, 0.51);
		sep_cuda_force_harmonic(aptr, mptr, 1, 31288, 0.47);
		sep_cuda_force_harmonic(aptr, mptr, 2, 52307, 0.31);

		sep_cuda_force_angle(aptr, mptr, 0, 672 , 1.99);
		sep_cuda_force_angle(aptr, mptr, 1, 541 , 1.91);
		sep_cuda_force_angle(aptr, mptr, 2, 592 , 1.89);
		
		sep_cuda_force_dihedral(aptr, mptr, 0, rbparam0);
		sep_cuda_force_dihedral(aptr, mptr, 1, rbparam1);

		sep_cuda_thermostat_nh(aptr, sptr, 3.86, 0.1);
		sep_cuda_integrate_leapfrog(aptr, sptr);
	
	
		if ( n%10 ==0 ){
			sep_cuda_sample_gh(ghptr, aptr, sptr);
		}
		
					
		if ( n%10000==0 ){
			sprintf(filestr, "molsim-%05d.xyz", counter);
			sep_cuda_save_xyz(aptr, filestr);
			counter ++;
		}
	
	}

	sep_cuda_save_xyz(aptr, "test.xyz");
	
	sep_cuda_free_memory(aptr, sptr);
	sep_cuda_sample_gh_free(ghptr);

	sep_cuda_free_bonds(mptr);
	sep_cuda_free_angles(mptr);
	sep_cuda_free_dihedrals(mptr);
	
	return 0;
}
