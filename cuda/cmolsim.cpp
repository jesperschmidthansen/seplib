#include <octave/oct.h>
#include "cmolsim.h"

#define HELP "usage: cmolsim(<action>, <specifier>, <specifier args>) - see cmolsim documentation for more help"

enum {
  RESET=547, CALCFORCE=930, INTEGRATE=963,
  THERMOSTAT=1099, SAMPLE=642, ADD=297,
  GET=320, PRINT=557, SAVE=431,
  TASK=435, COMPRESS=876, CLEAR=519,
  SET=332, HELLO=532, LOAD=416,
  HASHVALUE=961, BAROSTAT=864, CONVERT=769, NUPDATE=753
};

unsigned hashfun(const std::string key);
void action_load(const octave_value_list& args);
void action_get(const octave_value_list& args);
void action_clear(void);


DEFUN_DLD(cmolsim, args, , HELP){
	octave_value_list retval;
 
   	const unsigned nargs = args.length();	
	const std::string action = args(0).string_value();

	switch ( hashfun(action) ){
		case LOAD:
			action_load(args);	
			break;
		case GET:
			action_get(args);
			break;
		case CLEAR:
			action_clear();
			break;
		default:
			octave_stdout << "Not a valid action\n";
	}

	retval.append(1);

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


void action_get(const octave_value_list& args){
	
	const std::string specifier = args(1).string_value();
	
	if ( strcmp(specifier.c_str(), "positions")==0 ){
		get_positions();
	}
}


void action_clear(void){ 

	free_memory();

}


