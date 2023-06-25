/*******************************************
 * 
 * Basically just a wrapper
 * 
 *******************************************/

#include <mex.h>
#include <string.h>
#include "cmolsim.h"


// Hard-coded hash values for switch - *I* cannot "optimize" further
// Hash value is simply the string (lower case) character sum
enum {
  RESET=547, CALCFORCE=930, INTEGRATE=963,
  THERMOSTAT=1099, SAMPLE=642, ADD=297,
  GET=320, PRINT=557, SAVE=431,
  TASK=435, COMPRESS=876, CLEAR=519,
  SET=332, HELLO=532, LOAD=416,
  HASHVALUE=961, BAROSTAT=864, CONVERT=769, NUPDATE=753
};

unsigned hashfun(const char *key);

void action_load(int nrhs, const mxArray *prhs[]);
void action_clear(void);
void action_reset(void);
void action_nupdate();
void action_calcforce(int nrhs, const mxArray *prhs[]);
void action_integrate(const mxArray *prhs[]);
void action_save(const mxArray *prhs[]);
void action_thermostat(const mxArray *prhs[]);
void action_set(const mxArray *prhs[]);
void action_get(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);


void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	if ( nrhs == 0 ){
		mexPrintf("cmolsim - a wrapper for CUDA seplib. Check documentation. \n");
		return;
	}

	char *action = mxArrayToString(prhs[0]);
	
	switch ( hashfun(action) ){

		case LOAD: action_load(nrhs, prhs); break; 

		case RESET: action_reset(); break;

		case NUPDATE: action_nupdate(); break;
			
		case CALCFORCE: action_calcforce(nrhs, prhs); break;
		
		case INTEGRATE: action_integrate(prhs); break;
		
		case THERMOSTAT: action_thermostat(prhs); break;
		
		case GET: action_get(nlhs, plhs, nrhs, prhs); break;

		case SET: action_set(prhs); break;
		
		case SAVE: action_save(prhs); break;

		case CLEAR: action_clear(); break;

		default:
			mexPrintf("Action %s given -> ", action);
			mexErrMsgTxt("Not a valid action\n");

			break;
	}

}



unsigned hashfun(const char *key){

  const size_t len_key = strlen(key);

  unsigned sum_char = 0;
  for ( size_t n=0; n<len_key; n++ ) sum_char += (unsigned)key[n];
  
  return sum_char;

}

void action_load(int nrhs, const mxArray *prhs[]){
	char *specifier = mxArrayToString(prhs[1]);
	
	if ( strcmp(specifier, "xyz")==0 ){
		char *file = mxArrayToString(prhs[2]);
		load_xyz(file);
		free(file);
	}
	
	free(specifier);

}

void action_get(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	char *specifier = mxArrayToString(prhs[1]);
	
	if ( strcmp(specifier, "positions")==0 ){
		get_positions();
	}
	else if ( strcmp(specifier, "pressure")==0 ){
		plhs[0] = mxCreateDoubleMatrix(4, 1, mxREAL);

		double *pressptr = mxGetPr(plhs[0]);
    
		get_pressure(pressptr);
	}
	else if ( strcmp(specifier, "energies")==0 ){
		plhs[0] = mxCreateDoubleMatrix(2, 1, mxREAL);
		double *energypointer = mxGetPr(plhs[0]);
		
		get_energies(energypointer);
	}
	
	free(specifier);
	
}

void action_clear(void){
	free_memory();
}

void action_reset(void){
	reset_iteration();
}

void action_nupdate(void){
	update_neighblist();
}

void action_calcforce(int nrhs, const mxArray *prhs[]){
	
	char *specifier = mxArrayToString(prhs[1]);
    
    // van der Waal
    if ( strcmp(specifier, "lj")==0 ){
      char *types =  mxArrayToString(prhs[2]);
      float cf = mxGetScalar(prhs[3]);
      float sigma = mxGetScalar(prhs[4]);
      float epsilon = mxGetScalar(prhs[5]);
      
      float ljparam[3]={sigma, epsilon, cf};

      force_lj(types, ljparam);
      free(types);
	}
	
	free(specifier);
}

void action_integrate(const mxArray *prhs[]){
	
	char *specifier = mxArrayToString(prhs[1]);
    
    if ( strcmp(specifier, "leapfrog") == 0 ){
		integrate_leapfrog();
    }
    
    free(specifier);
}


void action_save(const mxArray *prhs[]){
	
	char *filename = mxArrayToString(prhs[1]);
    
	save_xyz(filename);
	
	free(filename);
}

void action_thermostat(const mxArray *prhs[]){
	
	char *specifier = mxArrayToString(prhs[1]);
	
	if ( strcmp(specifier, "nosehoover") == 0 ){
		int type = (int)mxGetScalar(prhs[2]); // Still not implemented
		float temperature = mxGetScalar(prhs[3]);
		float thermostatMass = mxGetScalar(prhs[4]);
      
		thermostat_nh(temperature, thermostatMass);
	}
	
	free(specifier);
	
}

void action_set(const mxArray *prhs[]){
	
	char *specifier = mxArrayToString(prhs[1]);
	
	if ( strcmp(specifier, "resetmomentum")==0 ){
		int resetfreq = (int)mxGetScalar(prhs[2]);
		reset_momentum(resetfreq);
	}
	
	free(specifier);
}
