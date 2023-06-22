
/********************************
 * 
 * A wrapper for sepcuda
 * 
 * CUDA headers and libraries are stored different places
 * depending on platform; therefore this wrapper
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
bool init = false;

void load_xyz(const char file[]){
	
	pptr = sep_cuda_load_xyz(file);
	sptr = sep_cuda_sys_setup(pptr);
	
	init = true;
}

void free_memory(void){
	
	
	sep_cuda_free_memory(pptr, sptr);

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
	
	sep_cuda_force_lj(pptr, types, ljparams);
	
}
