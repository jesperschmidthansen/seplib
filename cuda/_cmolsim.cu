
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
int resetmomentum = -1;

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
	
	if ( resetmomentum >=0 && resetmomentum%iterationnumber==0 )
		sep_cuda_reset_momentum(pptr);
	
}

void reset_momentum(int freq){
	resetmomentum = freq;
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

void thermostat_nh(float temp0, float mass){
	
	sep_cuda_thermostat_nh(pptr, sptr, temp0, mass);
	
}

void get_pressure(double *presspointer){
	
	double normalpress, shearpress[3];
	sep_cuda_get_pressure(&normalpress, shearpress, pptr);
	
	presspointer[1] = normalpress;
	for ( int k=1; k<4; k++ ) presspointer[k]=shearpress[k-1];
	
}

void get_energies(double *energypointer){
	
	sep_cuda_get_energies(pptr, sptr, "nve");
	
	energypointer[0] = sptr->ekin;
	energypointer[1] = sptr->epot;
}
