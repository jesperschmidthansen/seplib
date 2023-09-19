// Minimal and benchmark prg
// At the moment manual neighbour list update 

#include "sepcuda.h"

int main(int argc, char **argv){
	
	if ( argc != 2 ) {
		fprintf(stderr, "Please provide filename\n");
		exit(EXIT_FAILURE);
	}
	
	sepcupart *ptr = sep_cuda_load_xyz(argv[1]);
	sepcusys *sptr = sep_cuda_sys_setup(ptr);

	float ljparam[3]={1.0,1.0,2.5};
		
	int n=0; int nloops = 100000; 
	while ( n<nloops ){
		
		sep_cuda_reset_iteration(ptr, sptr);

		if ( n%10==0 )	sep_cuda_update_neighblist(ptr, sptr, 2.5);
		
		sep_cuda_force_lj(ptr, ljparam);

		sep_cuda_integrate_leapfrog(ptr, sptr);
		
		n++;
	}
	
	sep_cuda_save_xyz(ptr, "test.xyz");
	
	sep_cuda_free_memory(ptr, sptr);
	
	return 0;
}
