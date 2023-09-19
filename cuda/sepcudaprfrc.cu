
#include "sepcudaprfrc.h"

#ifdef OCTAVE

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

#endif


// Host functions

bool sep_cuda_check_neighblist(sepcupart *ptr, float maxdist){
		
	sep_cuda_sumdistance<<<ptr->nblocks,ptr->nthreads>>>
		(&(ptr->dsumdist), ptr->ddist, maxdist, ptr->npart);
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

void sep_cuda_reset_exclusion(sepcupart *pptr){
	
	for ( unsigned n=0; n<pptr->npart_padding; n++ ){
		int offset = n*(SEP_MAX_NUMB_EXCLUSION+1);

		pptr->hexclusion[offset] = 0;
		for ( int m=1; m<=SEP_MAX_NUMB_EXCLUSION; m++ )
			pptr->hexclusion[offset+m] = -1;
		
	}
	
	sep_cuda_copy_exclusion(pptr);	
}

void sep_cuda_copy_exclusion(sepcupart *pptr){

	size_t nbytes_excludelist = (SEP_MAX_NUMB_EXCLUSION+1)*pptr->npart_padding*sizeof(int);
		
	cudaError_t __err = cudaMemcpy(pptr->dexclusion, pptr->hexclusion, nbytes_excludelist, cudaMemcpyHostToDevice);
	if ( __err != cudaSuccess ) fprintf(stderr, "Error copying\n");	

}

void sep_cuda_set_hexclusion(sepcupart *pptr, int a, int b){
	
	int offset_a = a*(SEP_MAX_NUMB_EXCLUSION+1); 
	int offset_lst = pptr->hexclusion[offset_a];
	
	pptr->hexclusion[offset_a + offset_lst + 1] = b;
	pptr->hexclusion[offset_a] = pptr->hexclusion[offset_a] + 1;
	
}



void sep_cuda_set_exclusion(sepcupart *aptr, const char rule[]){
	
	if ( strcmp(rule, "bonds")==0 ){
		aptr->hexclusion_rule = SEP_CUDA_EXCL_BONDS;
	}
	else if (strcmp(rule, "molecule")==0 ){
		aptr->hexclusion_rule = SEP_CUDA_EXCL_MOLECULE;
	}
	else {
		fprintf(stderr, "Not valid exclusion rule\n");
	}
	
	size_t nbytes = sizeof(unsigned);
	cudaMemcpy(&(aptr->dexclusion_rule), &(aptr->hexclusion_rule), nbytes, cudaMemcpyHostToDevice);
	
}


// Kernel functions

/* Neighbourlist for particles - no exclusion */
__global__ void sep_cuda_build_neighblist(int *neighlist, float4 *p, float *dist, float cf, 
										  float3 lbox, unsigned nneighmax, unsigned npart) {

	int pidx = blockDim.x * blockIdx.x + threadIdx.x;
		
	if ( pidx < npart ){
		float cfsqr = cf*cf; 
		int arrayOffset = pidx*nneighmax;
	
		float mpx = __ldg(&p[pidx].x); float mpy = __ldg(&p[pidx].y); float mpz = __ldg(&p[pidx].z);

		#pragma unroll	
		for ( int n=0; n<nneighmax; n++ ) neighlist[arrayOffset + n] = -1; //<- this should be optimized 
		
		dist[pidx] = 0.0f;
		
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
		
				float dx = mpx - p[idxj].x;
				if ( dx > 0.5*lbox.x ) dx -= lbox.x;
				else if  ( dx < -0.5*lbox.x ) dx += lbox.x;
	
				float dy = mpy - p[idxj].y; 
				if ( dy > 0.5*lbox.y ) dy -= lbox.y;
				else if  ( dy < -0.5*lbox.y ) dy += lbox.y;
	
				float dz = mpz - p[idxj].z; 
				if ( dz > 0.5*lbox.z ) dz -= lbox.z;
				else if  ( dz < -0.5*lbox.z ) dz += lbox.z;
	
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
	
/* Neighbourlist for particles excluding particles in same molecule */
__global__ void sep_cuda_build_neighblist(int *neighlist, float *dist, float4 *p, int *molindex, 
										  float cf, float3 lbox, unsigned nneighmax, unsigned npart) {

	int pidx = blockDim.x * blockIdx.x + threadIdx.x;
		
	if ( pidx < npart ){
		float cfsqr = cf*cf; 
		int arrayOffset = pidx*nneighmax;
		int moli = molindex[pidx];
	//	float mpx = __ldg(&p[pidx].x); float mpy = __ldg(&p[pidx].y); float mpz = __ldg(&p[pidx].z);
		float mpx =p[pidx].x; float mpy = p[pidx].y; float mpz = p[pidx].z;
	
		#pragma unroll	
		for ( int n=0; n<nneighmax; n++ ) neighlist[arrayOffset + n] = -1; //<- this should be optimized 
		
		// Reset the distance travelled since last update - in danger zone
		dist[pidx] = 0.0f;
		
		int shift = 0;
		for ( int tile = 0; tile < gridDim.x; tile++ ) {

			for ( int j = 0; j < SEP_CUDA_NTHREADS; j++ ) {
				int idxj = tile*blockDim.x + j;
				
				if ( idxj >= npart )  break;
				
				if ( moli == molindex[idxj] ) continue;
	
				float dx = mpx - p[idxj].x;
				if ( dx > 0.5*lbox.x ) dx -= lbox.x;
				else if  ( dx < -0.5*lbox.x ) dx += lbox.x;
	
				float dy = mpy - p[idxj].y; 
				if ( dy > 0.5*lbox.y ) dy -= lbox.y;
				else if  ( dy < -0.5*lbox.y ) dy += lbox.y;
	
				float dz = mpz - p[idxj].z; 
				if ( dz > 0.5*lbox.z ) dz -= lbox.z;
				else if  ( dz < -0.5*lbox.z ) dz += lbox.z;
			
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
			// __syncthreads();
		}
	}

}

/* Pair interactions - types specified */
__global__ void sep_cuda_lj(const char type1, const char type2, float4 params, int *neighblist, float4 *pos, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart){


	int pidx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( pidx < npart ) {
		
		int itype = __float2int_rd(force[pidx].w);
		int atype = (int)type1; int btype = (int)type2; //cast is stupid!
		
		if ( itype != atype && itype != btype ) return;
		
		float sigma = params.x; 
		float epsilon = params.y; 
		float cf = params.z; //__ldg does not work..?
		float aw = params.w; 

		float cfsqr = cf*cf;
		float awh = 0.5*aw;
		float Epot_shift = 4.0*epsilon*(powf(sigma/cf, 12.) - aw*powf(sigma/cf,6.));
		
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
			
				float dx = mpx - pos[pjdx].x; //dx = sep_cuda_wrap(dx, lbox.x);
				if ( dx > 0.5*lbox.x ) dx -= lbox.x;
				else if  ( dx < -0.5*lbox.x ) dx += lbox.x;
		
				float dy = mpy - pos[pjdx].y; //dy = sep_cuda_wrap(dy, lbox.y);
				if ( dy > 0.5*lbox.y ) dy -= lbox.y;
				else if  ( dy < -0.5*lbox.y ) dy += lbox.y;
		
				float dz = mpz - pos[pjdx].z; //dz = sep_cuda_wrap(dz, lbox.z);
				if ( dz > 0.5*lbox.z ) dz -= lbox.z;
				else if  ( dz < -0.5*lbox.z ) dz += lbox.z;
		
				float distSqr = dx*dx + dy*dy + dz*dz;

				if ( distSqr < cfsqr ) {
					float rri = sigma*sigma/distSqr; 
					float rri3 = rri*rri*rri;
					float ft = 48.0*epsilon*rri3*(rri3 - awh)*rri;
				
					Fx += ft*dx; Fy += ft*dy; Fz += ft*dz;
					Epot += 0.5*(4.0*epsilon*rri3*(rri3 - aw) - Epot_shift);
					
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


/* Pair interactions - all types have same interactions (faster) */
__global__ void sep_cuda_lj(float3 params, int *neighblist, float4 *pos, float4 *force,
						float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart){


	int pidx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( pidx < npart ) {

		float sigma = params.x; 
		float epsilon = params.y; 
		float cf = params.z; 
		float cfsqr = cf*cf;
		float Epot_shift = 4.0*epsilon*(powf(sigma/cf, 12.) - powf(sigma/cf,6.));
		
		int offset = pidx*maxneighb;
			
		//float mpx = __ldg(&pos[pidx].x); float mpy = __ldg(&pos[pidx].y);float mpz = __ldg(&pos[pidx].z);
		float mpx =pos[pidx].x; float mpy = pos[pidx].y; float mpz = pos[pidx].z;
	
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f; 
		float Epot = 0.0f; 
		float4 mpress; mpress.x = mpress.y = mpress.z = mpress.w = 0.0f;
		
		register int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
				
			float dx = mpx - pos[pjdx].x; //dx = sep_cuda_wrap(dx, lbox.x);
			if ( dx > 0.5*lbox.x ) dx -= lbox.x;
			else if  ( dx < -0.5*lbox.x ) dx += lbox.x;
	
			float dy = mpy - pos[pjdx].y; //dy = sep_cuda_wrap(dy, lbox.y);
			if ( dy > 0.5*lbox.y ) dy -= lbox.y;
			else if  ( dy < -0.5*lbox.y ) dy += lbox.y;
	
			float dz = mpz - pos[pjdx].z; //dz = sep_cuda_wrap(dz, lbox.z);
			if ( dz > 0.5*lbox.z ) dz -= lbox.z;
			else if  ( dz < -0.5*lbox.z ) dz += lbox.z;
	
			float distSqr = dx*dx + dy*dy + dz*dz;

			if ( distSqr < cfsqr ) {
				float rri = sigma*sigma/distSqr; 
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
					float rri = sigma*sigma/distSqr; 
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

// Molecular force calculations (for pressure etc)
__global__ void sep_cuda_calc_molforce(float3 *mforce,  const char type1, const char type2, float4 params, float4 *pos, 
										int *neighblist,  unsigned maxneighb, float4 *force, 
										float3 lbox, int *molindex, unsigned nmol, const unsigned npart) {


	int pidx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( pidx < npart ) {
		
		int itype = __float2int_rd(force[pidx].w);
		int atype = (int)type1; int btype = (int)type2; //cast is stupid!
		
		if ( itype != atype && itype != btype ) return;
		
		float sigma = params.x;	float epsilon = params.y; float cf = params.z; float aw = params.w;
		float cfsqr = cf*cf; float awh = 0.5*aw;
		int molidx = molindex[pidx];

		int offset = pidx*maxneighb;
			
		float mpx = __ldg(&pos[pidx].x); float mpy = __ldg(&pos[pidx].y); float mpz = __ldg(&pos[pidx].z);
				
	//	float Fx=0.0f; float Fy = 0.0f; float Fz = 0.0f; 

		int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
			int jmolidx = molindex[pjdx]; 

			if ( jmolidx == molidx ) { n++; continue; }

			int jtype = __float2int_rd(force[pjdx].w);
				
			if ( (itype == atype && jtype == btype) || (itype == btype && jtype == atype) ){
				
				float dx = mpx - pos[pjdx].x; dx = sep_cuda_wrap(dx, lbox.x);
				float dy = mpy - pos[pjdx].y; dy = sep_cuda_wrap(dy, lbox.y);
				float dz = mpz - pos[pjdx].z; dz = sep_cuda_wrap(dz, lbox.z);

				float distSqr = dx*dx + dy*dy + dz*dz;

				if ( distSqr < cfsqr ) {
					float rri = sigma*sigma/distSqr; 
					float rri3 = rri*rri*rri;
					float ft = 48.0*epsilon*rri3*(rri3 - awh)*rri;
			
					unsigned offset = molidx*nmol + jmolidx;
					float Fx = ft*dx; float Fy = ft*dy; float Fz = ft*dz;
					
					atomicAdd(&mforce[offset].x, Fx);
					atomicAdd(&mforce[offset].y, Fy);
					atomicAdd(&mforce[offset].z, Fz);

					//Fx += ft*dx; Fy += ft*dy; Fz += ft*dz;
				}
			}
			
			n++;
		}
//		atomicAdd(&mforce[molidx].x, Fx);
//		atomicAdd(&mforce[molidx].y, Fy);
//		atomicAdd(&mforce[molidx].z, Fz);
	}
		
}

__global__ void sep_cuda_calc_molforce(float3 *mforce, float cf, int *neighblist, float4 *pos, float4 *vel, 
										 unsigned maxneighb, int *molindex, unsigned nmols, 
										 float3 lbox, const unsigned npart){

	__const__ int pidx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( pidx < npart ) {
		
		float cfsqr = cf*cf;
		float icf2 = 1.0/cfsqr;
		
		int offset = pidx*maxneighb;
			
		int molidx = molindex[pidx];

		float mpx = __ldg(&pos[pidx].x); 
		float mpy = __ldg(&pos[pidx].y); 
		float mpz = __ldg(&pos[pidx].z);
		
		//Force contribution to mol from particule			
	//		float Fx = 0.0; float Fy = 0.0; float Fz = 0.0; 		
		
		int n = 0;
		while ( neighblist[n+offset] != -1 ){
			int pjdx = neighblist[n+offset];
			int jmolidx = molindex[pjdx]; 
	
			if ( jmolidx == molidx ) { n++; continue; }

			float dx = mpx - pos[pjdx].x; dx = sep_cuda_wrap(dx, lbox.x);
			float dy = mpy - pos[pjdx].y; dy = sep_cuda_wrap(dy, lbox.y);
			float dz = mpz - pos[pjdx].z; dz = sep_cuda_wrap(dz, lbox.z);

			float distSqr = dx*dx + dy*dy + dz*dz;

			if ( distSqr < cfsqr ) {
				float zizj = vel[pidx].w*vel[pjdx].w;
				float dist = sqrtf(distSqr); 
				float ft = zizj*(1.0/distSqr - icf2)/dist; 
				
				unsigned offset = nmols*molidx + jmolidx;		
				float Fx = ft*dx; float Fy = ft*dy; float Fz = ft*dz;

				atomicAdd(&mforce[offset].x, Fx);
				atomicAdd(&mforce[offset].y, Fy);
				atomicAdd(&mforce[offset].z, Fz);
			}

			n++;
		}

		/*
		atomicAdd(&mforce[molidx].x, Fx);
		atomicAdd(&mforce[molidx].y, Fy);
		atomicAdd(&mforce[molidx].z, Fz);
		*/
	}	
		
}


// Wrapper section

void sep_cuda_force_lj(sepcupart *pptr, const char types[], float params[4]){
	const int nb = pptr->nblocks; 
	const int nt = pptr->nthreads;

	float4 ljparams = make_float4(params[0],params[1],params[2],params[3]);

	sep_cuda_lj<<<nb, nt>>>
		(types[0], types[1], ljparams, pptr->neighblist, pptr->dx, pptr->df,  
				pptr->epot, pptr->press, pptr->maxneighb, pptr->lbox, pptr->npart);

	// Calculates the molecular forces for molecular stress tensor
	if ( pptr->sptr->molprop && pptr->sptr->iteration%pptr->sptr->molpropinterval==0 ){
		sep_cuda_calc_molforce<<<nb,nt>>>(pptr->sptr->mptr->dfij, types[0], types[1], ljparams, 
					pptr->dx, pptr->neighblist, pptr->maxneighb, pptr->df, 
					pptr->sptr->lbox, pptr->dmolindex, pptr->sptr->mptr->nmols, pptr->sptr->npart);
	}

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

	if ( pptr->sptr->molprop && pptr->sptr->iteration%pptr->sptr->molpropinterval==0 ){
		sep_cuda_calc_molforce<<<nb, nt>>>(pptr->sptr->mptr->dfij, cf, pptr->neighblist, pptr->dx, pptr->dv, 
										 pptr->maxneighb, pptr->dmolindex, pptr->sptr->mptr->nmols, 
										 pptr->lbox, pptr->npart);
	}

	cudaDeviceSynchronize();

}

void sep_cuda_update_neighblist(sepcupart *pptr, sepcusys *sptr, float maxcutoff){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;

	if ( pptr->hexclusion_rule == SEP_CUDA_EXCL_NONE ) {
		sep_cuda_build_neighblist<<<nb, nt>>>
			(pptr->neighblist, pptr->dx, pptr->ddist, sptr->skin+maxcutoff, pptr->lbox, pptr->maxneighb,pptr->npart);
	}
	else if ( pptr->hexclusion_rule == SEP_CUDA_EXCL_MOLECULE ) {
		sep_cuda_build_neighblist<<<nb, nt>>>
			(pptr->neighblist, pptr->ddist, pptr->dx, pptr->dmolindex, sptr->skin+maxcutoff, pptr->lbox, pptr->maxneighb,pptr->npart);
	}
	else {
		fprintf(stderr, "Exclusion rule invalid");
	}
		
	cudaDeviceSynchronize();

}

