#include <mex.h>
#include "cmolsim.h"

#include <string.h>

sepcupart *pptr; 
sepcusys *sptr;

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	if ( nrhs == 0 ){
		mexPrintf("molsim - a wrapper for seplib. Check documentation. \n");
		return;
	}
	
	
	// ...and ACTION!!
	char *action = mxArrayToString(prhs[0]);
		
	if ( strcmp(action, "load")==0 ){
		char *specifier = mxArrayToString(prhs[1]);
 
		if ( strcmp(specifier, "xyz")==0 ){
			char *file = mxArrayToString(prhs[2]);
			pptr = sep_cuda_load_xyz(file);
			sptr = sep_cuda_sys_setup(pptr);
		}
		
	} else if ( strcmp(action, "clear")==0 ) {
		sep_cuda_free_memory(pptr, sptr);
	}
	
	
}
