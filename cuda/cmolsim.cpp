/*******************************************
 * 
 * Basically just a wrapper
 * 
 *******************************************/

#include <mex.h>
#include <string.h>
#include "cmolsim.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	if ( nrhs == 0 ){
		mexPrintf("molsim - a wrapper for seplib. Check documentation. \n");
		return;
	}

	char *action = mxArrayToString(prhs[0]);
	
	if ( strcmp(action, "load")==0 ){
		char *specifier = mxArrayToString(prhs[1]);
		
		if ( strcmp(specifier, "xyz")==0 ){
			char *file = mxArrayToString(prhs[2]);
			load_xyz(file);
		
			free(file);
		}
		
		free(specifier);
	}
	else if ( strcmp(action, "get")==0 ) {
		char *specifier = mxArrayToString(prhs[1]);
		if ( strcmp(specifier, "positions")==0 ){
			get_positions();
		}
		free(specifier);
	}
	else if ( strcmp(action, "clear")==0 ) {
		free_memory();
	}
	
	free(action);
}
