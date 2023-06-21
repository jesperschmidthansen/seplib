
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

sepcupart *pptr; 
sepcusys *sptr;

void load_xyz(const char file[]){
	
	pptr = sep_cuda_load_xyz(file);
	sptr = sep_cuda_sys_setup(pptr);
	
}

void free_memory(void){
	
	sep_cuda_free_memory(pptr, sptr);

}

void get_positions(void){
	
	sep_cuda_copy(pptr, 'x', 'h');
	
	printf("%f %f %f\n", pptr->hx[276].x, pptr->hx[341].y, pptr->hx[311].z);
}
