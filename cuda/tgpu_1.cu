// Sanity check prg
/*
 *  Ref. seplib. Dens 0.7513, T=2.0, <etot> = -0.82, <press>=4.31 eta_0 = 1.6
 *               Radial distribution function found in rdf_single.dat
 * 
 *  Tested 23 April            
 */
#include "sepcuda.h"
#include "sepcudasampler.h"


float sep_cuda_eval_momentum(sepcupart *aptr){
	
	sep_cuda_copy(aptr, 'v', 'h');
	
	float sumv = 0.0;
	
	for ( int n=0; n<aptr->npart; n++ ) 
		sumv += aptr->hv[n].x;
	
	return sumv/aptr->npart;
}


bool sep_cuda_logrem(unsigned n, int base){
	static unsigned counter = 0;
	bool retval=false;
	
	if ( n%(int)pow(base, counter)==0 ){
		retval = true;
		counter++;
	}
	
	return retval;
}

int main(int argc, char **argv){
	
	if ( argc != 2 ) {
		fprintf(stderr, "Provide ensemble option\n");
		exit(EXIT_FAILURE);
	}
	
	char ensemble[10]="nve";
	if ( atoi(argv[1])==1 ) ensemble[2]='t';
	
	printf("Ensemble is %s\n", ensemble);
	
	sepcupart *ptr = sep_cuda_load_xyz("start_singleAN8000.xyz");
	sepcusys *sptr = sep_cuda_sys_setup(ptr);

	sepcugh *ghptr = sep_cuda_sample_gh_init(sptr, 50, 20, 10*sptr->dt);
	
	float ljparam[3] = {1.0, 1.0, 2.5};
	
	float temp0 = 2.0; 	char filestr[100];
	int n = 0; int nloops = 1000; bool update = true; int counter = 0;
	while ( n<nloops ){

		sep_cuda_reset_iteration(ptr, sptr);

		if ( update ) sep_cuda_update_neighblist(ptr, sptr, 2.5);
	
		sep_cuda_force_lj(ptr, ljparam);
			
		if ( atoi(argv[1])==1 )	sep_cuda_thermostat_nh(ptr, sptr, temp0, 0.1);	

		sep_cuda_integrate_leapfrog(ptr, sptr);
		
		update = sep_cuda_check_neighblist(ptr, sptr->skin);
	
		if ( n%10 ==0 ){
			sep_cuda_sample_gh(ghptr, ptr, sptr);
		}
				
		//if ( sep_cuda_logrem(n, 2) ){
		if ( n%5==0 ){
			sprintf(filestr, "molsim-%05d.xyz", counter);
			sep_cuda_save_xyz(ptr, filestr);
			
			sprintf(filestr, "crossings-%05d.dat", counter);
			sep_cuda_save_crossings(ptr, filestr, n*sptr->dt);
			
			counter ++;
		}
		
		if ( n%1000 == 0 ){
			double normalpress, shearpress[3];
			sep_cuda_get_pressure(&normalpress, shearpress, ptr);
			sep_cuda_get_energies(ptr, sptr, ensemble);
		
			printf("%f %f %f %f %f %f %f %f %f\n", 
				   sptr->ekin, sptr->epot, sptr->etot, sptr->temp, normalpress, 
				   shearpress[0], shearpress[1], shearpress[2], sep_cuda_eval_momentum(ptr));
		}
		
		n++;
	}

	sep_cuda_save_xyz(ptr, "test.xyz");
	
	sep_cuda_sample_gh_free(ghptr);
	sep_cuda_free_memory(ptr, sptr);
	
	return 0;
}
