
#include "sepcuda.h"


void sep_cuda_mem_error(void){
	
	fprintf(stderr, "Memory allocation error");
	
	exit(EXIT_FAILURE);
}

void sep_cuda_file_error(void){
	
	fprintf(stderr, "Couldn't open or read file");
	
	exit(EXIT_FAILURE);
}

sepcupart* sep_cuda_allocate_memory(unsigned npartPadding){
	sepcupart* ptr;
	
	if ( cudaMallocHost((void **)&ptr, sizeof(sepcupart))== cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	size_t nbytes = npartPadding*sizeof(float4);
	
	if ( cudaMallocHost((void **)&(ptr->hx), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hv), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hf), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->ht), npartPadding*sizeof(char)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hexclusion), npartPadding*sizeof(int4)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hcrossings), npartPadding*sizeof(int3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	
	if ( cudaMalloc((void **)&(ptr->dx), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->dv), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->df), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->ddist), npartPadding*sizeof(float)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->epot), npartPadding*sizeof(float)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->press), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->sumpress), sizeof(float4)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->dexclusion), npartPadding*sizeof(int4)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	ptr->maxneighb = SEP_CUDA_MAXNEIGHBS;
	if ( cudaMalloc(&(ptr->neighblist), sizeof(int)*npartPadding*(ptr->maxneighb)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();

	if ( cudaMalloc((void **)&(ptr->dcrossings), npartPadding*sizeof(int3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	return ptr;
}

void sep_cuda_free_memory(sepcupart *ptr, sepcusys *sptr){
	
	// Particle structure
	cudaFreeHost(ptr->hx); 	cudaFreeHost(ptr->hv); 
	cudaFreeHost(ptr->hf); 	cudaFreeHost(ptr->ht);
	
	cudaFreeHost(ptr->hexclusion); cudaFreeHost(ptr->hcrossings);
	
	cudaFree(ptr->dx); cudaFree(ptr->dv); cudaFree(ptr->df);
	cudaFree(ptr->ddist); cudaFree(ptr->neighblist);
	cudaFree(ptr->epot); cudaFree(ptr->press); cudaFree(ptr->sumpress); 
	
	cudaFree(ptr->dexclusion); cudaFree(ptr->dcrossings);
	cudaFreeHost(ptr);
	
	// System structure
	cudaFree(sptr->denergies); cudaFreeHost(sptr->henergies);
	cudaFree(sptr->dalpha); cudaFree(sptr->dupdate); 
	
	cudaFreeHost(sptr);
}


void sep_cuda_copy(sepcupart *ptr, char opt_quantity, char opt_direction){
	
	size_t nbytes = ptr->npart_padding*sizeof(float4);
	
	switch (opt_direction){
		
		case 'd':
			if ( opt_quantity == 'x' )
				cudaMemcpy(ptr->dx, ptr->hx, nbytes, cudaMemcpyHostToDevice); 
			else if ( opt_quantity == 'v' )
				cudaMemcpy(ptr->dv, ptr->hv, nbytes, cudaMemcpyHostToDevice);
			else if ( opt_quantity == 'f' ){
				for ( int n=0; n<ptr->npart_padding; n++ ) ptr->hf[n].w = (float)(ptr->ht[n]);
				cudaMemcpy(ptr->df, ptr->hf, nbytes, cudaMemcpyHostToDevice);	
			} 
			else if ( opt_quantity == 'c' ){
				nbytes = ptr->npart_padding*sizeof(int3);
				cudaMemcpy(ptr->dcrossings, ptr->hcrossings, nbytes, cudaMemcpyHostToDevice);
			}
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
			else {
				fprintf(stderr, "Invalid opt_quantity");
				exit(EXIT_FAILURE);
			}
			break;
		
		default:
			fprintf(stderr, "Invalid opt_direction");
			exit(EXIT_FAILURE);
	}
	
	// Aparently, the device not synchronized copying to hist ...
	cudaDeviceSynchronize();
	
}

void sep_cuda_copy_energies(sepcusys *sptr){
	
	cudaMemcpy(sptr->henergies, sptr->denergies, sizeof(float3), cudaMemcpyDeviceToHost);
	
}

sepcupart* sep_cuda_load_xyz(const char *xyzfile){
	int npart;
	int nthreads = SEP_CUDA_NTHREADS;
	
	FILE *fin = fopen(xyzfile, "r");
	if ( fin == NULL )
		sep_cuda_file_error();
	
	fscanf(fin, "%d\n", &npart);
	
	unsigned nblocks = (npart + nthreads - 1) / nthreads;
	unsigned npartwithPadding = nblocks*nthreads;
	
	sepcupart *ptr = sep_cuda_allocate_memory(npartwithPadding);

	ptr->nblocks = nblocks; 
	ptr->nthreads = nthreads;
	ptr->npart = npart; 
	ptr->npart_padding = npartwithPadding;
	
	fscanf(fin, "%f %f %f\n", &(ptr->lbox.x), &(ptr->lbox.y), &(ptr->lbox.z));
	
	for ( unsigned n=0; n<npart; n++ ) {
		fscanf(fin, "%c %f %f %f %f %f %f %f %f\n", 
			   &(ptr->ht[n]), &(ptr->hx[n].x),&(ptr->hx[n].y),&(ptr->hx[n].z), 
			   &(ptr->hv[n].x),&(ptr->hv[n].y),&(ptr->hv[n].z), &(ptr->hx[n].w), &(ptr->hv[n].w));
		ptr->hcrossings[n].x = ptr->hcrossings[n].y = ptr->hcrossings[n].z = 0;
	}
	
	fclose(fin);

	for ( unsigned n=npart; n<npartwithPadding; n++ ){
		ptr->hx[n].x = ptr->hx[n].y = ptr->hx[n].z = 0.0f;
		ptr->hv[n].x = ptr->hv[n].y = ptr->hv[n].z = 0.0f;
		ptr->hv[n].w = 1.0;	ptr->ht[n] = 'A';
	}
	
	sep_cuda_copy(ptr, 'x', 'd'); 
	sep_cuda_copy(ptr, 'v', 'd');
	sep_cuda_copy(ptr, 'f', 'd');
	sep_cuda_copy(ptr, 'c', 'd');
	
	sep_cuda_reset_exclusion(ptr);
	
	return ptr;
}


sepcusys *sep_cuda_sys_setup(sepcupart *pptr){
	
	sepcusys *sptr;
	if ( cudaMallocHost((void **)&sptr, sizeof(sepcusys)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	sptr->npart = pptr->npart;
	sptr->npart_padding = pptr->npart_padding;
	sptr->nblocks = pptr->nblocks;
	sptr->nthreads = pptr->nthreads;
	sptr->dt = 0.005;
	sptr->skin = 0.3;
	sptr->lbox = pptr->lbox;
	
	if ( cudaMallocHost((void **)&(sptr->henergies), sizeof(float3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(sptr->denergies), sizeof(float3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(sptr->dalpha), sizeof(float)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	sep_cuda_setvalue<<<1,1>>>(sptr->dalpha, 0.2);
	
	
	if ( cudaMalloc((void **)&(sptr->dupdate), sizeof(int)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	sep_cuda_setvalue<<<1,1>>>(sptr->dupdate, 1);
	
	return sptr;
}

bool sep_cuda_check_neighblist(sepcupart *ptr, float maxdist){
		
	sep_cuda_sumdistance<<<ptr->nblocks,ptr->nthreads>>>(&(ptr->dsumdist), ptr->ddist, maxdist, ptr->npart);
	cudaDeviceSynchronize();
	
	float sumdr=0.0f;
	cudaMemcpy(&sumdr, &(ptr->dsumdist), sizeof(float), cudaMemcpyDeviceToHost);
	
	float avsumdr = sumdr/ptr->npart;
		
	if ( avsumdr > maxdist ){
		sep_cuda_setvalue<<<1,1>>>(&(ptr->dsumdist), 0);
		cudaDeviceSynchronize();
		return true;
	}	
	else 
		return false;
	
}

void sep_cuda_save_xyz(sepcupart *ptr, const char *filestr){

	FILE *fout = fopen(filestr, "w");
	if ( fout == NULL ) sep_cuda_file_error();
		
	sep_cuda_copy(ptr, 'x','h');
	sep_cuda_copy(ptr, 'v','h');
	sep_cuda_copy(ptr, 'f','h');
	
	fprintf(fout, "%d\n",ptr->npart);
	
	fprintf(fout, "%f %f %f\n", ptr->lbox.x, ptr->lbox.y, ptr->lbox.z);
	for ( int n=0; n<ptr->npart; n++ ){
		fprintf(fout, "%c %f %f %f ", (int)ptr->hf[n].w, ptr->hx[n].x, ptr->hx[n].y, ptr->hx[n].z);
		fprintf(fout, "%f %f %f ", ptr->hv[n].x, ptr->hv[n].y, ptr->hv[n].z);
		fprintf(fout, "%f %f\n", ptr->hx[n].w, ptr->hv[n].w);
	}
	
	fclose(fout);
	
}

void sep_cuda_save_crossings(sepcupart *ptr, const char *filestr, float time){

	FILE *fout = fopen(filestr, "w");
	if ( fout == NULL ) sep_cuda_file_error();
		
	sep_cuda_copy(ptr, 'c','h');
	
	fprintf(fout, "%f %f %f\n", time, 0.0f, 0.0f); // To make it easier to load
	
	
	for ( int n=0; n<ptr->npart; n++ )
		fprintf(fout, "%d %d %d \n", 
				ptr->hcrossings[n].x, ptr->hcrossings[n].y, ptr->hcrossings[n].z);
	
	
	fclose(fout);
	
}


void sep_cuda_reset_exclusion(sepcupart *pptr){
	
	for ( int n=0; n<pptr->npart_padding; n++ ){
		pptr->hexclusion[n].x = 0;
		pptr->hexclusion[n].y = -1;
		pptr->hexclusion[n].z = -1;
		pptr->hexclusion[n].w = -1;
	}
	
	sep_cuda_copy_exclusion(pptr);	
}

void sep_cuda_copy_exclusion(sepcupart *pptr){
		
	size_t nbytes = pptr->npart_padding*sizeof(int4);
	cudaError_t __err = cudaMemcpy(pptr->dexclusion, pptr->hexclusion, nbytes, cudaMemcpyHostToDevice);
	if ( __err != cudaSuccess ) fprintf(stderr, "Error copying\n");	
	
}

void sep_cuda_set_hexclusion(sepcupart *pptr, int a, int b){
	
	switch (pptr->hexclusion[a].x){
		case 0: 
			pptr->hexclusion[a].y = b;
			break;
		case 1:
			pptr->hexclusion[a].z = b;
			break;
		case 2: 
			pptr->hexclusion[a].w = b;
			break;
		default:
			fprintf(stderr, "Error in exclusion list");
			break;
	}
	
	pptr->hexclusion[a].x = pptr->hexclusion[a].x + 1;
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


// Kernels

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


__global__ void sep_cuda_build_neighblist(float *dalpha, int *neighlist, int4 *exclusion, float4 *p, float *dist, float cf, 
										  float3 lbox, unsigned nneighmax, unsigned npart) {

	int pidx = blockDim.x * blockIdx.x + threadIdx.x;
		
	if ( pidx < npart ){
		float cfsqr = cf*cf; 
		int arrayOffset = pidx*nneighmax;

		float mpx = __ldg(&p[pidx].x); float mpy = __ldg(&p[pidx].y); float mpz = __ldg(&p[pidx].z);

		#pragma unroll	
		for ( int n=0; n<nneighmax; n++ ) neighlist[arrayOffset + n] = -1; //<- this should be optimized 
		
		dist[pidx] = 0.0f;
		
		int exclx = __ldg(&exclusion[pidx].x); int excly = __ldg(&exclusion[pidx].y); 
		int exclz = __ldg(&exclusion[pidx].z); int exclw = __ldg(&exclusion[pidx].w); 
		
		
		int shift = 0;
		for ( int tile = 0; tile < gridDim.x; tile++ ) {

			/*
			__shared__ float4 spos[SEP_CUDA_NTHREADS];
			spos[threadIdx.x] = p[tile * blockDim.x + threadIdx.x];
			__syncthreads();
			*/
			
			for ( int j = 0; j < SEP_CUDA_NTHREADS; j++ ) {
				int idxj = tile*blockDim.x + j;
				
				if ( idxj >= npart )  break;

				if ( sep_cuda_check_exclude(exclx, excly, exclz, exclw, idxj) ) continue;
				
				/*
				float dx = mpx - spos[j].x; dx = sep_cuda_wrap(dx, lbox.x);
				float dy = mpy - spos[j].y; dy = sep_cuda_wrap(dy, lbox.y);
				float dz = mpz - spos[j].z; dz = sep_cuda_wrap(dz, lbox.z);
				*/
				
				float dx = mpx - p[idxj].x; dx = sep_cuda_wrap(dx, lbox.x);
				float dy = mpy - p[idxj].y; dy = sep_cuda_wrap(dy, lbox.y);
				float dz = mpz - p[idxj].z; dz = sep_cuda_wrap(dz, lbox.z);
				
				
				float distSqr = dx*dx + dy*dy + dz*dz;

				if ( distSqr < 2.0*FLT_EPSILON ) continue; // Self contribution
				
				if ( distSqr < cfsqr ) {
						
					if ( shift < nneighmax )
						neighlist[arrayOffset + shift] = idxj;
					else if ( shift >= nneighmax ) {
						printf("Neighbour list generation failed\n");
						return;
					}	
					
					shift++;
				}
			}

			__syncthreads();
			
		}
	}
}
	


__global__ void sep_cuda_lj(const char type1, const char type2, float3 params, int *neighblist, float4 *pos, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart){

	
	int pidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( pidx < npart ) {
		
		int itype = __float2int_rd(force[pidx].w);
		int atype = (int)type1; int btype = (int)type2; //cast stupid
		
		if ( itype != atype && itype != btype ) return;
		
		float sigma = params.x; 
		float epsilon = params.y; 
		float cf = params.z; //__ldg does not work..?
		float cfsqr = cf*cf;
		float Epot_shift = 4.0*epsilon*(powf(sigma/cf, 12.) - powf(sigma/cf,6.));
		
		int offset = pidx*maxneighb;
			
		float mpx = __ldg(&pos[pidx].x); float mpy = __ldg(&pos[pidx].y); float mpz = __ldg(&pos[pidx].z);
				
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f; 
		float Epot = 0.0f; 
		float4 mpress; mpress.x = mpress.y = mpress.z = mpress.w = 0.0f;
		int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
			int jtype = __float2int_rd(force[pjdx].w);
			
			if ( (itype == atype && jtype == btype) || (itype == btype && jtype == atype) ){
				
				float dx = mpx - pos[pjdx].x; dx = sep_cuda_wrap(dx, lbox.x);
				float dy = mpy - pos[pjdx].y; dy = sep_cuda_wrap(dy, lbox.y);
				float dz = mpz - pos[pjdx].z; dz = sep_cuda_wrap(dz, lbox.z);

				float distSqr = dx*dx + dy*dy + dz*dz;

				if ( distSqr < cfsqr ) {
					float rri = sigma/distSqr; 
					float rri3 = rri*rri*rri;
					float ft = 48.0*epsilon*rri3*(rri3 - 0.5)*rri;
				
					Fx += ft*dx; Fy += ft*dy; Fz += ft*dz;
					Epot += 0.5*(4.0*epsilon*rri3*(rri3 - 1.0) - Epot_shift);
					mpress.x += dx*ft*dx + dy*ft*dy + dz*ft*dz; 
					mpress.y += dx*ft*dy; mpress.z += dx*ft*dz; mpress.w += dy*ft*dz;
				}
			}
			
			n++;
		}
		
		force[pidx].x += Fx; force[pidx].y += Fy; force[pidx].z += Fz;
		epot[pidx] += Epot; 
		press[pidx].x += mpress.x;
		press[pidx].y += mpress.y; press[pidx].z += mpress.z; press[pidx].w += mpress.w; 
	}
		
}



__global__ void sep_cuda_lj(float3 params, int *neighblist, float4 *pos, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart){

	
	int pidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( pidx < npart ) {
		
		float sigma = params.x; 
		float epsilon = params.y; 
		float cf = params.z; //__ldg does not work..?
		float cfsqr = cf*cf;
		float Epot_shift = 4.0*epsilon*(powf(sigma/cf, 12.) - powf(sigma/cf,6.));
		
		int offset = pidx*maxneighb;
			
		float mpx = __ldg(&pos[pidx].x); float mpy = __ldg(&pos[pidx].y); float mpz = __ldg(&pos[pidx].z);
				
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f; 
		float Epot = 0.0f; 
		float4 mpress; mpress.x = mpress.y = mpress.z = mpress.w = 0.0f;
		int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
				
			float dx = mpx - pos[pjdx].x; dx = sep_cuda_wrap(dx, lbox.x);
			float dy = mpy - pos[pjdx].y; dy = sep_cuda_wrap(dy, lbox.y);
			float dz = mpz - pos[pjdx].z; dz = sep_cuda_wrap(dz, lbox.z);

			float distSqr = dx*dx + dy*dy + dz*dz;

			if ( distSqr < cfsqr ) {
				float rri = sigma/distSqr; 
				float rri3 = rri*rri*rri;
				float ft =  48.0*epsilon*rri3*(rri3 - 0.5)*rri; //pow( sqrtf(1.0/distSqr), 11.0 ); //
				
				Fx += ft*dx; Fy += ft*dy; Fz += ft*dz;
				Epot += 0.5*(4.0*epsilon*rri3*(rri3 - 1.0) - Epot_shift);
				mpress.x += dx*ft*dx + dy*ft*dy + dz*ft*dz; 
				mpress.y += dx*ft*dy; mpress.z += dx*ft*dz; mpress.w += dy*ft*dz;
			}
			
			n++;
		}
			
		
			
		force[pidx].x += Fx; force[pidx].y += Fy; force[pidx].z += Fz;
		epot[pidx] += Epot; 
		press[pidx].x += mpress.x;
		press[pidx].y += mpress.y; press[pidx].z += mpress.z; press[pidx].w += mpress.w; 
	}
}



__global__ void sep_cuda_lj_sf(const char type1, const char type2, float3 params, int *neighblist, float4 *pos, float4 *force,
								float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart){

	
	int pidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( pidx < npart ) {
		
		int itype = __float2int_rd(force[pidx].w);
		int atype = (int)type1; int btype = (int)type2; //cast stupid
		
		if ( itype != atype && itype != btype ) return;
		
		float sigma = params.x; 
		float epsilon = params.y; 
		float cf = params.z; //__ldg does not work..?
		float cfsqr = cf*cf; 
		float Epot_shift = 4.0*epsilon*(powf(sigma/cf, 12.) - powf(sigma/cf,6.));
		float force_shift = 48.0*epsilon*powf(sigma/cf,6.0)*(powf(sigma/cf,3.0) - 0.5)*pow(sigma/cf, 2.0);
		
		int offset = pidx*maxneighb;
			
		float mpx = __ldg(&pos[pidx].x); float mpy = __ldg(&pos[pidx].y); float mpz = __ldg(&pos[pidx].z);
				
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f; 
		float Epot = 0.0f; 
		float4 mpress; mpress.x = mpress.y = mpress.z = mpress.w = 0.0f;
		int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
			int jtype = __float2int_rd(force[pjdx].w);
			
			if ( (itype == atype && jtype == btype) || (itype == btype && jtype == atype) ){
				
				float dx = mpx - pos[pjdx].x; dx = sep_cuda_wrap(dx, lbox.x);
				float dy = mpy - pos[pjdx].y; dy = sep_cuda_wrap(dy, lbox.y);
				float dz = mpz - pos[pjdx].z; dz = sep_cuda_wrap(dz, lbox.z);

				float distSqr = dx*dx + dy*dy + dz*dz;

				if ( distSqr < cfsqr ) {
					float rri = sigma/distSqr; 
					float rri3 = rri*rri*rri;
					float ft = 48.0*epsilon*rri3*(rri3 - 0.5)*rri + force_shift;
				
					Fx += ft*dx; Fy += ft*dy; Fz += ft*dz;
					Epot += 0.5*(4.0*epsilon*rri3*(rri3 - 1.0) - Epot_shift);
					mpress.x += dx*ft*dx + dy*ft*dy + dz*ft*dz; 
					mpress.y += dx*ft*dy; mpress.z += dx*ft*dz; mpress.w += dy*ft*dz;
				}
			}
			
			n++;
		}
		
		force[pidx].x += Fx; force[pidx].y += Fy; force[pidx].z += Fz;
		epot[pidx] += Epot; 
		press[pidx].x += mpress.x;
		press[pidx].y += mpress.y; press[pidx].z += mpress.z; press[pidx].w += mpress.w; 
	}
		
}



__global__ void sep_cuda_sf(float cf, int *neighblist, float4 *pos, float4 *vel, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart){
	
	__const__ int pidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( pidx < npart ) {
		
		float cfsqr = cf*cf;
		float icf = 1.0/cf;
		float icf2 = 1.0/cfsqr;
		
		int offset = pidx*maxneighb;
			
		float mpx = __ldg(&pos[pidx].x); 
		float mpy = __ldg(&pos[pidx].y); 
		float mpz = __ldg(&pos[pidx].z);
				
		float Fx = 0.0; float Fy = 0.0; float Fz = 0.0; float Epot = 0.0;		
		float4 mpress; mpress.x = mpress.y = mpress.z = mpress.w = 0.0f;
		
		int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
				
			float dx = mpx - pos[pjdx].x; dx = sep_cuda_wrap(dx, lbox.x);
			float dy = mpy - pos[pjdx].y; dy = sep_cuda_wrap(dy, lbox.y);
			float dz = mpz - pos[pjdx].z; dz = sep_cuda_wrap(dz, lbox.z);

			float distSqr = dx*dx + dy*dy + dz*dz;

			if ( distSqr < cfsqr ) {
				float zizj = vel[pidx].w*vel[pjdx].w;
				float dist = sqrtf(distSqr); 
				float ft = zizj*(1.0/distSqr - icf2)/dist; 
				
				Fx += ft*dx; Fy += ft*dy; Fz += ft*dz;
				
				Epot += 0.5*zizj*(1.0/dist + (dist-cf)*icf2 - icf);
				mpress.x += dx*ft*dx + dy*ft*dy + dz*ft*dz; 
				mpress.y += dx*ft*dy; mpress.z += dx*ft*dz; mpress.w += dy*ft*dz;
			}

			n++;
		}
		
		force[pidx].x += Fx; force[pidx].y += Fy; force[pidx].z += Fz;
		epot[pidx] += Epot;	
		press[pidx].x += mpress.x;
		press[pidx].y += mpress.y; press[pidx].z += mpress.z; press[pidx].w += mpress.w;
	}	
		
}


__global__ void sep_cuda_leapfrog(float4 *pos, float4 *vel, float4 *force, float *dist, int3 *crossing, float dt, float3 lbox, unsigned npart){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	float4 oldpos = make_float4(pos[i].x, pos[i].y, pos[i].z, 0.0f);
	
	if ( i < npart ) {
		float imass = 1.0/pos[i].w;
		vel[i].x += force[i].x*imass*dt;
		vel[i].y += force[i].y*imass*dt;
		vel[i].z += force[i].z*imass*dt;
		
		pos[i].x += vel[i].x*dt; 
		pos[i].x = sep_cuda_periodic(pos[i].x, lbox.x, &(crossing[i].x));
		
		pos[i].y += vel[i].y*dt;
		pos[i].y = sep_cuda_periodic(pos[i].y, lbox.y, &(crossing[i].y));
		
		pos[i].z += vel[i].z*dt; 
		pos[i].z = sep_cuda_periodic(pos[i].z, lbox.z, &(crossing[i].z));
					
		float dx = oldpos.x - pos[i].x; dx = sep_cuda_wrap(dx, lbox.x);
		float dy = oldpos.y - pos[i].y; dy = sep_cuda_wrap(dy, lbox.y);
		float dz = oldpos.z - pos[i].z; dz = sep_cuda_wrap(dz, lbox.z);

		dist[i] += sqrtf(dx*dx + dy*dy + dz*dz);
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

__global__ void sep_cuda_update_nosehoover(float *alpha, float3 *denergies, float temp0, 
										   float tau, float dt, unsigned int npart){

	float temp = (2.0/3.0)*denergies->x/npart; 

	*alpha = *alpha + dt/(tau*tau)*(temp/temp0 - 1.0);
	
}


__global__ void sep_cuda_nosehoover(float *alpha, float4 *pos, float4 *vel, float4 *force, unsigned npart){
	
	unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( id < npart ){	
		float fac = (*alpha)*pos[id].w;
		force[id].x -= fac*vel[id].x; 
		force[id].y -= fac*vel[id].y; 
		force[id].z -= fac*vel[id].z;		
	}
}

// Device functions

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

__device__ bool sep_cuda_check_exclude(int x, int y, int z, int w, int idxj){
	
	int retval = false;
		
	if ( x > 0 ){
		if ( y == idxj ||  z == idxj ||  w == idxj ) return true;
	}
	
	return retval;
}

// Wrappers

void sep_cuda_force_lj(sepcupart *pptr, const char types[], float params[3]){
	const int nb = pptr->nblocks; 
	const int nt = pptr->nthreads;
	
	float3 ljparams = make_float3(params[0],params[1],params[2]);
	
	sep_cuda_lj<<<nb, nt>>>
		(types[0], types[1], ljparams, pptr->neighblist, pptr->dx, pptr->df, pptr->epot, 
											pptr->press, pptr->maxneighb, pptr->lbox, pptr->npart);
	cudaDeviceSynchronize();

}

void sep_cuda_force_lj(sepcupart *pptr, float params[3]){
	const int nb = pptr->nblocks; 
	const int nt = pptr->nthreads;
	
	float3 ljparams = make_float3(params[0],params[1],params[2]);
	
	sep_cuda_lj<<<nb, nt>>>
		(ljparams, pptr->neighblist, pptr->dx, pptr->df, pptr->epot, pptr->press, pptr->maxneighb, pptr->lbox, pptr->npart);
		
	cudaDeviceSynchronize();

}


void sep_cuda_force_lj_sf(sepcupart *pptr, const char types[], float params[3]){
	const int nb = pptr->nblocks; 
	const int nt = pptr->nthreads;
	
	float3 ljparams = make_float3(params[0],params[1],params[2]);
	
	sep_cuda_lj_sf<<<nb, nt>>>
		(types[0], types[1], ljparams, pptr->neighblist, pptr->dx, pptr->df, pptr->epot, 
											pptr->press, pptr->maxneighb, pptr->lbox, pptr->npart);
	cudaDeviceSynchronize();

}



void sep_cuda_force_sf(sepcupart *pptr, const float cf){
	const int nb = pptr->nblocks; 
	const int nt = pptr->nthreads;
	
	sep_cuda_sf<<<nb,nt>>>
		(cf, pptr->neighblist, pptr->dx, pptr->dv, pptr->df, pptr->epot, 
											pptr->press, pptr->maxneighb, pptr->lbox, pptr->npart);
	cudaDeviceSynchronize();

}

void sep_cuda_thermostat_nh(sepcupart *pptr, sepcusys *sptr, float temp0, float tau){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;
	
	// Get current system kinetic energy
	sep_cuda_sumenergies<<<nb,nt>>>
		(sptr->denergies, pptr->dx, pptr->dv, pptr->df, sptr->dt, pptr->epot, sptr->npart);
	cudaDeviceSynchronize();
	
	
	// Update nh-alpha dynamics (single thread)
	sep_cuda_update_nosehoover<<<1,1>>>
		(sptr->dalpha, sptr->denergies, temp0, tau, sptr->dt, sptr->npart);
	cudaDeviceSynchronize();
	
	// Add thermostat force
	sep_cuda_nosehoover<<<nb, nt>>>
		(sptr->dalpha, pptr->dx, pptr->dv, pptr->df, sptr->npart);
	cudaDeviceSynchronize();		
	
}


void sep_cuda_reset_iteration(sepcupart *pptr, sepcusys *sptr){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;

	sep_cuda_reset<<<nb,nt>>>
			(pptr->df, pptr->epot, pptr->press, pptr->sumpress, sptr->denergies, pptr->npart);
	cudaDeviceSynchronize();

}

void sep_cuda_update_neighblist(sepcupart *pptr, sepcusys *sptr, float maxcutoff){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;

	sep_cuda_build_neighblist<<<nb, nt>>>
		(sptr->dalpha, pptr->neighblist, pptr->dexclusion, pptr->dx, pptr->ddist, sptr->skin+maxcutoff, 
		 pptr->lbox, pptr->maxneighb,pptr->npart);
	
	cudaDeviceSynchronize();

}

void sep_cuda_integrate_leapfrog(sepcupart *pptr, sepcusys *sptr){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;

	sep_cuda_leapfrog<<<nb, nt>>>
		(pptr->dx, pptr->dv, pptr->df, pptr->ddist, pptr->dcrossings, sptr->dt, pptr->lbox, pptr->npart);
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

