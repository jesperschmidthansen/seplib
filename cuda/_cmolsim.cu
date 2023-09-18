
/********************************
 * 
 * A wrapper for sepcuda
 * 
 * CUDA headers and libraries are stored different places
 * depending on platform. This wrapper will hopefully make 
 * the installation procedure less cumbersom
*
 ********************************/
 
#include "cmolsim.h"

#include "sepcuda.h"

#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

sepcupart *pptr; 
sepcusys *sptr;
sepcumol *mptr;

float maxcutoff = 2.5;

int iterationnumber = 0;
int neighbupdatefreq = 10;
int resetmomentumfreq = -1;

int ensemble = 0; // 0: nve, 1: nvt 

bool init = false, initmol = false;
bool needneighbupdate = true;


void load_xyz(const char file[]){
	
	pptr = sep_cuda_load_xyz(file);
	sptr = sep_cuda_sys_setup(pptr);
	
	init = true;
}

void load_top(const char file[]){
	
	mptr = sep_cuda_init_mol();
	
	sep_cuda_read_bonds(pptr, mptr, file, 'v');
	sep_cuda_read_angles(pptr, mptr, file, 'v');
	sep_cuda_read_dihedrals(pptr, mptr, file, 'v');
	
	initmol = true;
}

void free_memory(void){
	
	if ( initmol ){
		sep_cuda_free_bonds(mptr);
		sep_cuda_free_angles(mptr);
		sep_cuda_free_dihedrals(mptr);
		
		initmol = false;
	}
	
	if ( init ) {
		sep_cuda_free_memory(pptr, sptr);
		init = false;
	}
}

void reset_iteration(void){
	
	sep_cuda_reset_iteration(pptr, sptr);

	if ( iterationnumber%neighbupdatefreq == 0 ) 	
		sep_cuda_update_neighblist(pptr, sptr, maxcutoff);

}

void reset_momentum(int freq){
	resetmomentumfreq = freq;
}


void force_lj(const char *types, float *ljparams){
	
	if ( needneighbupdate && iterationnumber%neighbupdatefreq == 0 )
		sep_cuda_update_neighblist(pptr, sptr, maxcutoff);
	
	sep_cuda_force_lj(pptr, types, ljparams);
	
	needneighbupdate = false;
}

void force_coulomb(float cf){
	
	sep_cuda_force_sf(pptr, cf);
	
}

void integrate_leapfrog(void){
	
	sep_cuda_integrate_leapfrog(pptr, sptr);
	
	iterationnumber ++;
	needneighbupdate = true;
		
	if ( resetmomentumfreq >= 0 && resetmomentumfreq%iterationnumber==0 )
		sep_cuda_reset_momentum(pptr);
	
}

void save_xyz(const char filename[]){
	
	sep_cuda_save_xyz(pptr, filename);
	
}

void thermostat_nh(float temp0, float mass){
	
	ensemble = 1;
	sep_cuda_thermostat_nh(pptr, sptr, temp0, mass);
	
}

void get_pressure(double *presspointer){
	
	double normalpress, shearpress[3];
	sep_cuda_get_pressure(&normalpress, shearpress, pptr);
	
	presspointer[0] = normalpress;
	for ( int k=1; k<4; k++ ) presspointer[k]=shearpress[k-1];
	
}

void get_energies(double *energypointer){
	
	if ( ensemble==1 )
		sep_cuda_get_energies(pptr, sptr, "nvt");
	else 
		sep_cuda_get_energies(pptr, sptr, "nve");
	
	energypointer[0] = sptr->ekin;
	energypointer[1] = sptr->epot;
}
