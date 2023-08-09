#include "sepcudamisc.h"

void sep_cuda_mem_error(void){
	
	fprintf(stderr, "Memory allocation error");
	
	exit(EXIT_FAILURE);
}

void sep_cuda_file_error(void){
	
	fprintf(stderr, "Couldn't open or read file");
	
	exit(EXIT_FAILURE);
}


void sep_cuda_copy(sepcupart *ptr, char opt_quantity, char opt_direction){
	
	size_t nbytes = ptr->npart_padding*sizeof(float4);
	
	switch (opt_direction){
		
		case 'd':
			if ( opt_quantity == 'x' )  // Position
				cudaMemcpy(ptr->dx, ptr->hx, nbytes, cudaMemcpyHostToDevice); 
			else if ( opt_quantity == 'v' ) // Velocity
				cudaMemcpy(ptr->dv, ptr->hv, nbytes, cudaMemcpyHostToDevice);
			else if ( opt_quantity == 'f' ){ // Force + type
				for ( unsigned n=0; n<ptr->npart_padding; n++ ) ptr->hf[n].w = (float)(ptr->ht[n]);
				cudaMemcpy(ptr->df, ptr->hf, nbytes, cudaMemcpyHostToDevice);	
			} 
			else if ( opt_quantity == 'c' ){  // Particle crossings array
				nbytes = ptr->npart_padding*sizeof(int3);
				cudaMemcpy(ptr->dcrossings, ptr->hcrossings, nbytes, cudaMemcpyHostToDevice);
			}
			else if ( opt_quantity == 'l' )  // Lattice sites
				cudaMemcpy(ptr->dx0, ptr->hx0, nbytes, cudaMemcpyHostToDevice);
			else {
				fprintf(stderr, "Invalid opt_quantity");
				exit(EXIT_FAILURE);
			}
			break;
		
		case 'h':
			if ( opt_quantity == 'x' )
				cudaMemcpy(ptr->hx, ptr->dx, nbytes, cudaMemcpyDeviceToHost);
			else if ( opt_quantity == 'v' )
				cudaMemcpy(ptr->hv, ptr->dv, nbytes, cudaMemcpyDeviceToHost);
			else if ( opt_quantity == 'f' )
				cudaMemcpy(ptr->hf, ptr->df, nbytes, cudaMemcpyDeviceToHost);
			else if ( opt_quantity == 'c' ){
				nbytes = ptr->npart_padding*sizeof(int3);
				cudaMemcpy(ptr->hcrossings, ptr->dcrossings, nbytes, cudaMemcpyDeviceToHost);
			}
			else if ( opt_quantity == 'l' )
				cudaMemcpy(ptr->hx0, ptr->dx0, nbytes, cudaMemcpyDeviceToHost);
			else {
				fprintf(stderr, "Invalid opt_quantity");
				exit(EXIT_FAILURE);
			}
			break;
		
		default:
			fprintf(stderr, "Invalid opt_direction");
			exit(EXIT_FAILURE);
	}
	
	// The device not synchronized copying to host
	cudaDeviceSynchronize();
	
}

void sep_cuda_set_molprop_on(sepcusys *sptr) { sptr->molprop = true; }

void sep_cuda_copy_energies(sepcusys *sptr){
	
	cudaMemcpy(sptr->henergies, sptr->denergies, sizeof(float3), cudaMemcpyDeviceToHost);
	
}

void sep_cuda_load_lattice_positions(sepcupart *ptr, const char *xyzfile){
	char cdum; float vdum[3], mdum, qdum; unsigned npart;

	FILE *fin = fopen(xyzfile, "r");
	if ( fin == NULL ) sep_cuda_file_error();
 
	fscanf(fin, "%d\n", &npart);
	fscanf(fin, "%f %f %f\n", &(vdum[0]), &(vdum[1]), &(vdum[2]));
	
	for ( unsigned n=0; n<npart; n++ ) {
		fscanf(fin, "%c %f %f %f %f %f %f %f %f\n", 
			   &cdum, &(ptr->hx0[n].x),&(ptr->hx0[n].y),&(ptr->hx0[n].z),
			   &(vdum[0]), &(vdum[1]), &(vdum[2]), &mdum, &qdum);
	}
	
	fclose(fin);

	for ( unsigned n=npart; n<ptr->npart_padding; n++ ) 
		ptr->hx0[n].x = ptr->hx0[n].y = ptr->hx0[n].z = 0.0f;
	
	sep_cuda_copy(ptr, 'l', 'd');
}

void sep_cuda_save_crossings(sepcupart *ptr, const char *filestr, float time){

	FILE *fout = fopen(filestr, "w");
	if ( fout == NULL ) sep_cuda_file_error();
		
	sep_cuda_copy(ptr, 'c','h');
	
	fprintf(fout, "%f %f %f\n", time, 0.0f, 0.0f); // To make it easier to load
	
	
	for ( unsigned n=0; n<ptr->npart; n++ )
		fprintf(fout, "%d %d %d \n", 
				ptr->hcrossings[n].x, ptr->hcrossings[n].y, ptr->hcrossings[n].z);
	
	fclose(fout);
	
}

void sep_cuda_save_xyz(sepcupart *ptr, const char *filestr){

	FILE *fout = fopen(filestr, "w");
	if ( fout == NULL ) sep_cuda_file_error();
		
	sep_cuda_copy(ptr, 'x','h');
	sep_cuda_copy(ptr, 'v','h');
	sep_cuda_copy(ptr, 'f','h');
	
	fprintf(fout, "%d\n",ptr->npart);
	
	fprintf(fout, "%f %f %f\n", ptr->lbox.x, ptr->lbox.y, ptr->lbox.z);
	for ( unsigned n=0; n<ptr->npart; n++ ){
		fprintf(fout, "%c %f %f %f ", (int)ptr->hf[n].w, ptr->hx[n].x, ptr->hx[n].y, ptr->hx[n].z);
		fprintf(fout, "%f %f %f ", ptr->hv[n].x, ptr->hv[n].y, ptr->hv[n].z);
		fprintf(fout, "%f %f\n", ptr->hx[n].w, ptr->hv[n].w);
	}
	
	fclose(fout);
	
}


void sep_cuda_compressbox(sepcupart *aptr, float rho0, float compressfactor[3]){
	
	float volume = (aptr->lbox.x)*(aptr->lbox.y)*(aptr->lbox.z); 
	float rho = aptr->npart/volume;

	// This shoulb done nicer - with a compression prop to fraction
	if ( rho < rho0 ){
		aptr->lbox.x = (aptr->lbox.x)*compressfactor[0];
		aptr->lbox.y = (aptr->lbox.y)*compressfactor[1];
		aptr->lbox.z = (aptr->lbox.z)*compressfactor[2];
	}
	
}

void sep_cuda_get_pressure(double *npress, double *shearpress, sepcupart *aptr){


	float4 press; press.x = press.y = press.z = press.w = 0.0f;
	cudaMemcpy(aptr->sumpress, &press, sizeof(float4), cudaMemcpyHostToDevice);
	
	sep_cuda_getpress<<<aptr->nblocks, aptr->nthreads>>>
		(aptr->sumpress, aptr->dx, aptr->dv, aptr->press, aptr->npart);
	cudaDeviceSynchronize();
	
	cudaMemcpy(&press, aptr->sumpress, sizeof(float4), cudaMemcpyDeviceToHost);

	double volume = aptr->lbox.x*aptr->lbox.y*aptr->lbox.z;
	
	*npress = press.x/volume;
	shearpress[0] = press.y/volume; shearpress[1] = press.z/volume; shearpress[2] = press.w/volume;
	
}


float sep_cuda_eval_momentum(float *momentum, sepcupart *aptr){
	
	sep_cuda_copy(aptr, 'v', 'h');
	sep_cuda_copy(aptr, 'x', 'h');
	
	for ( int k=0; k<3; k++ ) momentum[k] = 0.0f;
	
	for ( unsigned n=0; n<aptr->npart; n++ ){
		float mass = aptr->hx[n].w;
		momentum[0] += aptr->hv[n].x*mass;
		momentum[1] += aptr->hv[n].y*mass;
		momentum[2] += aptr->hv[n].z*mass;
	}
	
	float retval = (momentum[0]+momentum[1]+momentum[2])/(3.*aptr->npart);
	return retval;
}


void sep_cuda_reset_momentum(sepcupart *aptr){
	float momentum[3];

	// Note; hv, hx updated 	
	sep_cuda_eval_momentum(momentum, aptr);
	
	float totalmass = 0.0f;
	for ( unsigned n=0; n<aptr->npart; n++ ) totalmass += aptr->hx[n].w;
	
	for ( unsigned n=0; n<aptr->npart; n++ ){
		aptr->hv[n].x -=  momentum[0]/totalmass;
		aptr->hv[n].y -=  momentum[1]/totalmass;
		aptr->hv[n].z -=  momentum[2]/totalmass;
	}
	
	sep_cuda_copy(aptr, 'v', 'd');
	
}

bool sep_cuda_logrem(unsigned n, int base){
	static unsigned counter = 0;
	bool retval=false;
	
	if ( n%(int)pow(base, counter)==0 ){
		retval = true;
		counter++;
	}
	
	return retval;
}

float sep_cuda_wrap_host(float x, float lbox){
		
	if ( x > 0.5*lbox ) 
		x -= lbox;
	else if  ( x < -0.5*lbox ) 
		x += lbox;
	
	return x;
}



__global__ void sep_cuda_lattice_force(const char type, float springConstant, float4 *pos, float4 *pos0, float4 *force,
									   float3 lbox, const unsigned npart){

	
	unsigned pidx = blockDim.x * blockIdx.x + threadIdx.x;
	int itype = __float2int_rd(force[pidx].w);
		
	if ( pidx < npart && itype == (int)type ){
		
		float dx = pos[pidx].x - pos0[pidx].x; dx = sep_cuda_wrap(dx, lbox.x);
		float dy = pos[pidx].y - pos0[pidx].y; dy = sep_cuda_wrap(dy, lbox.y);
		float dz = pos[pidx].z - pos0[pidx].z; dz = sep_cuda_wrap(dz, lbox.z);

		force[pidx].x = - springConstant*dx;
		force[pidx].y = - springConstant*dy;
		force[pidx].z = - springConstant*dz;
		
	}
	
}


__global__ void sep_cuda_sumdistance(float *totalsum, float *dist, float maxdist, unsigned npart){
	
	__shared__ float sum;
	if (threadIdx.x==0) sum=.0f;
	__syncthreads();
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if ( id < npart ) atomicAdd(&sum, dist[id]);
	__syncthreads();
	
	if ( threadIdx.x == 0 ) {
		atomicAdd(totalsum, sum);
	}
}

__global__ void sep_cuda_setvalue(float *variable, float value){
	
	*variable = value;

	
}

__global__ void sep_cuda_printvalue(float *value){
	
	printf("Device value %f with address %p\n", *value, value);

}

__global__ void sep_cuda_setvalue(int *variable, int value){
	
	*variable = value;
}

__global__ void sep_cuda_sumenergies(float3 *totalsum, float4* dx, float4 *dv, float4 *df, 
									 float dt, float *epot, unsigned npart){
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float3 sums;
	
	if ( threadIdx.x==0 ) {
		sums.x = sums.y = sums.z = 0.0f;
	}
	__syncthreads();
	
	if ( id < npart ){
		
		float4 vel; 
		vel.x =  dv[id].x - 0.5*dt*df[id].x/dx[id].w;
		vel.y =  dv[id].y - 0.5*dt*df[id].y/dx[id].w;
		vel.z =  dv[id].z - 0.5*dt*df[id].z/dx[id].w;
		
		float mykin = 0.5*sep_cuda_dot(vel)*dx[id].w;
		float mymom = (dv[id].x + dv[id].y + dv[id].z)*dx[id].w;
		
		atomicAdd(&sums.x, mykin);
		atomicAdd(&sums.y, epot[id]);
		atomicAdd(&sums.z, mymom);
	}

	__syncthreads();
	
	if ( id < npart && threadIdx.x == 0 ) {
		atomicAdd(&(totalsum->x), sums.x);
		atomicAdd(&(totalsum->y), sums.y);
		atomicAdd(&(totalsum->z), sums.z);
	}
	
}

__global__ void sep_cuda_getpress(float4 *press, float4 *pos, float4 *vel, float4 *ppress, int npart){
	
	__shared__ float4 sums;
	if ( threadIdx.x==0) {
		sums.x = sums.y = sums.z = sums.w = 0.0f;
	}
	__syncthreads();
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if ( id < npart ){
		float kinetics = pos[id].w*(vel[id].x*vel[id].x + vel[id].y*vel[id].y + vel[id].z*vel[id].z);
		atomicAdd(&sums.x, (kinetics + 0.5*ppress[id].x)/3.0);

		kinetics = pos[id].w*vel[id].x*vel[id].y;
		atomicAdd(&sums.y, kinetics + 0.5*ppress[id].y);
		
		kinetics = pos[id].w*vel[id].x*vel[id].z;
		atomicAdd(&sums.z, kinetics + 0.5*ppress[id].z);
		
		kinetics = pos[id].w*vel[id].y*vel[id].z;
		atomicAdd(&sums.w, kinetics + 0.5*ppress[id].w);
	}
	__syncthreads();
	
	if ( threadIdx.x == 0 ) {
		atomicAdd(&(press->x), sums.x);
		atomicAdd(&(press->y), sums.y);
		atomicAdd(&(press->z), sums.z);
		atomicAdd(&(press->w), sums.w);
	}
	
}


__global__ void sep_cuda_reset(float4 *force, float *epot, float4 *press, float4 *sumpress, float3 *energies, unsigned npart){

	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( i < npart ) {
		force[i].x = force[i].y = force[i].z = 0.0f;
		press[i].x = press[i].y = press[i].z = press[i].w = 0.0f;
		epot[i] = 0.0f;
	
		if ( i==0 )	{ 
			energies->x = 0.0f; energies->y = 0.0f; energies->z = 0.0f; 
			sumpress->x = sumpress->y = sumpress->z = sumpress->w = 0.0f;
		}
	}

}


__global__ void sep_cuda_reset_mol(float3 *force, unsigned nmol){
	

	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

	if ( i < nmol )
		force[i].x = force[i].y = force[i].z = 0.0f;
}

// Wrapper/interface functions
void sep_cuda_force_lattice(sepcupart *pptr, const char type, float springConstant){
	const int nb = pptr->nblocks; 
	const int nt = pptr->nthreads;
	
	sep_cuda_lattice_force<<<nb, nt>>>
		(type, springConstant, pptr->dx, pptr->dx0, pptr->df, pptr->lbox, pptr->npart);
		
	cudaDeviceSynchronize();

}

void sep_cuda_reset_iteration(sepcupart *pptr, sepcusys *sptr){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;

	sep_cuda_reset<<<nb,nt>>>
			(pptr->df, pptr->epot, pptr->press, pptr->sumpress, sptr->denergies, pptr->npart);

	if ( sptr->molprop )	
		sep_cuda_reset_mol<<<nb,nt>>>(sptr->mptr->df, sptr->mptr->nmols);

	cudaDeviceSynchronize();

}



void sep_cuda_get_energies(sepcupart *ptr, sepcusys *sptr, const char ensemble[]){

	// This summation has been done for the nh-thermostat
	if ( strcmp("nve", ensemble)==0 ){
		sep_cuda_sumenergies<<<sptr->nblocks,sptr->nthreads>>>
							(sptr->denergies, ptr->dx, ptr->dv, ptr->df, sptr->dt, ptr->epot, sptr->npart);
		cudaDeviceSynchronize();
	}
	
	sep_cuda_copy_energies(sptr);
			
	sptr->ekin = (sptr->henergies->x)/sptr->npart;
	sptr->epot = (sptr->henergies->y)/sptr->npart;
	
	sptr->etot = sptr->ekin + sptr->epot;
	sptr->temp = 2./3*sptr->ekin;
}



__device__ float sep_cuda_dot(float4 a){
	
	return (a.x*a.x + a.y*a.y + a.z*a.z);
	
}

__device__ float sep_cuda_dot(float3 a, float3 b){
	
	return (a.x*b.x + a.y*b.y + a.z*b.z);
	
}

__device__ float sep_cuda_wrap(float x, float lbox){
	
	if ( x > 0.5*lbox ) 
		x -= lbox;
	else if  ( x < -0.5*lbox ) 
		x += lbox;
	
	return x;
}

__device__ float sep_cuda_periodic(float x, float lbox, int *crossing){
	
	if ( x > lbox ) {
		x -= lbox;  
		*crossing = *crossing + 1;
	}
	else if  ( x < 0 ) {
		x += lbox;
		*crossing = *crossing - 1;
	}
	
	return x;
}

__device__ bool sep_cuda_exclude_pair(int *exclusionlist, int numbexclude, int offset, int idxj){
	
	int retval = false;
	
	for ( int n=1; n<=numbexclude; n++ ){
		if ( exclusionlist[n+offset] == idxj ) {
			retval = true;  break;
		}
	}

	return retval;
}

