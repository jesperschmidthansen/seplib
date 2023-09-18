#include <octave/oct.h>
#include "cmolsim.h"

#define HELP "usage: cmolsim(<action>, <specifier>, <specifier args>) - see cmolsim documentation for more help"

enum {
  RESET=547, CALCFORCE=930, INTEGRATE=963,
  THERMOSTAT=1099, SAMPLE=642, ADD=297,
  GET=320, PRINT=557, SAVE=431,
  TASK=435, COMPRESS=876, CLEAR=519,
  SET=332, HELLO=532, LOAD=416,
  HASHVALUE=961, BAROSTAT=864, CONVERT=769 
};

unsigned hashfun(const std::string key);

void action_load(const octave_value_list& args);
void action_reset(void);
void action_calcforce(const octave_value_list& args);
void action_integrate(const octave_value_list& args);
void action_save(const octave_value_list& args);
void action_clear(void);
void action_get(octave_value_list& retval, const octave_value_list& args);


DEFUN_DLD(cmolsim, args, , HELP){
	octave_value_list retval;
 
	const std::string action = args(0).string_value();

	switch ( hashfun(action) ){
		case LOAD:
			action_load(args);	break;
		case RESET:
			action_reset();	break;
		case CALCFORCE:
			action_calcforce(args);	break;
		case INTEGRATE:
			action_integrate(args); break;
		case GET:
			action_get(retval, args); break;	
		case SAVE:
			action_save(args); break;
		case CLEAR:
			action_clear();	break;
		default:
			octave_stdout << "Not a valid action\n";
	}

	
	return retval;	
}


unsigned hashfun(const std::string key){

  unsigned sum_char = 0;
  for ( size_t n=0; n<key.length(); n++ ) sum_char += (unsigned)key[n];
  
  return sum_char;

}

void action_load(const octave_value_list& args){
	
	const std::string specifier = args(1).string_value();
	const std::string filename = args(2).string_value();

	if ( strcmp(specifier.c_str(), "xyz")==0 ){
		load_xyz(filename.c_str());
	}
	else if ( strcmp(specifier.c_str(), "top")==0 ){
		load_top(filename.c_str());
	}

}

void action_reset(void){
	reset_iteration();
}

void action_calcforce(const octave_value_list& args){

	const std::string specifier = args(1).string_value();

	if ( strcmp(specifier.c_str(), "lj")==0 && args.length()==7 ){
      const std::string types  =  args(2).string_value();
	  
      float cf = args(3).scalar_value();
      float sigma = args(4).scalar_value();
      float epsilon = args(5).scalar_value();
	  float aw = args(6).scalar_value();      
      
	  float ljparam[4]={sigma, epsilon, cf, aw};

	  force_lj(types.c_str(), ljparam);
	}
	else if ( strcmp(specifier.c_str(), "coulomb")==0 && (args.length()==4 || args.length()==5) ) {
		float cf = args(3).scalar_value();
		force_coulomb(cf);
	}
}

void action_integrate(const octave_value_list& args){

	const std::string specifier = args(1).string_value();

	if ( strcmp(specifier.c_str(), "leapfrog")==0 ){
		integrate_leapfrog();		
	}

}

void action_get(octave_value_list& retval, const octave_value_list& args){

	const std::string specifier = args(1).string_value();

	if ( strcmp(specifier.c_str(), "energies")==0 )	{
		double energies[2];
		get_energies(energies);
		RowVector en(2); en(0)=energies[0]; en(1)=energies[1];
		retval.append(en);
	}	
	else if ( strcmp(specifier.c_str(), "pressure")==0 ){
		double press[4];
		get_pressure(press);
		RowVector pr(4);
		for ( int k=0; k<4; k++ ) pr(k)=press[k];
		retval.append(pr);	
	}
}

void action_save(const octave_value_list& args){

	const std::string specifier = args(1).string_value();

	save_xyz(specifier.c_str());		
}

void action_clear(void){ 

	free_memory();

}



