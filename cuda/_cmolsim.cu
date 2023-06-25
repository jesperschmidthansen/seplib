
/********************************
 * 
 * A wrapper for sepcuda
 * 
 * CUDA headers and libraries are stored different places
 * depending on platform. This wrapper will hopefully make 
 * the installation procedure less combersom
*
 ********************************/
 
#include "cmolsim.h"
#include "sepcuda.h"
#include "sepcuda.cu"

#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

sepcupart *pptr; 
sepcusys *sptr;

float maxcutoff = 2.5;

int iterationnumber = 0;
int neighbupdatefreq = 10;

bool init = false;


void load_xyz(const char file[]){
	
	pptr = sep_cuda_load_xyz(file);
	sptr = sep_cuda_sys_setup(pptr);
	
	init = true;
}

void free_memory(void){
	
	
	if ( init ) {
		sep_cuda_free_memory(pptr, sptr);
		init = false;
	}
}

void get_positions(void){
	
	sep_cuda_copy(pptr, 'x', 'h');
	
	printf("%f %f %f\n", pptr->hx[276].x, pptr->hx[341].y, pptr->hx[311].z);
}

void reset_iteration(void){
	
	sep_cuda_reset_iteration(pptr, sptr);
	
}

void update_neighblist(void){
	
	sep_cuda_update_neighblist(pptr, sptr, maxcutoff);
	
}

void force_lj(char *types, float *ljparams){
	
	if ( iterationnumber%neighbupdatefreq == 0 )
		sep_cuda_update_neighblist(pptr, sptr, maxcutoff);
	
	sep_cuda_force_lj(pptr, types, ljparams);
	
}

void integrate_leapfrog(void){
	
	sep_cuda_integrate_leapfrog(pptr, sptr);
	iterationnumber ++;
}

void save_xyz(char *filename){
	
	sep_cuda_save_xyz(pptr, filename);
	
}
