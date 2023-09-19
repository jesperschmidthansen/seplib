
#include "sepcudaintgr.h"

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

__global__ void sep_cuda_leapfrog(float4 *pos, float4 *vel, 
		  float4 *force, float *dist, int3 *crossing, float dt, float3 lbox, unsigned npart){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	float4 oldpos = make_float4(pos[i].x, pos[i].y, pos[i].z, 0.0f);
	float4 mypos = make_float4(pos[i].x, pos[i].y, pos[i].z, pos[i].w);

	if ( i < npart ) {
		
		float imass = 1.0/mypos.w;
		
		vel[i].x += force[i].x*imass*dt; 
		vel[i].y += force[i].y*imass*dt;
		vel[i].z += force[i].z*imass*dt;
		
		mypos.x += vel[i].x*dt;
		mypos.x = sep_cuda_periodic(mypos.x, lbox.x, &(crossing[i].x));
		
		mypos.y += vel[i].y*dt;
		mypos.y = sep_cuda_periodic(mypos.y, lbox.y, &(crossing[i].y));
	
		mypos.z += vel[i].z*dt;
		mypos.z = sep_cuda_periodic(mypos.z, lbox.z, &(crossing[i].z));
					
		float dx = oldpos.x - mypos.x; dx = sep_cuda_wrap(dx, lbox.x);
		float dy = oldpos.y - mypos.y; dy = sep_cuda_wrap(dy, lbox.y);
		float dz = oldpos.z - mypos.z; dz = sep_cuda_wrap(dz, lbox.z);
	
		dist[i] += sqrtf(dx*dx + dy*dy + dz*dz);

		pos[i].x = mypos.x; pos[i].y = mypos.y; pos[i].z = mypos.z;
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

   
void sep_cuda_integrate_leapfrog(sepcupart *pptr, sepcusys *sptr){
	const int nb = sptr->nblocks; 
	const int nt = sptr->nthreads;

	sep_cuda_leapfrog<<<nb, nt>>>
		(pptr->dx, pptr->dv, pptr->df, pptr->ddist, pptr->dcrossings, sptr->dt, pptr->lbox, pptr->npart);
	cudaDeviceSynchronize();
	
}


