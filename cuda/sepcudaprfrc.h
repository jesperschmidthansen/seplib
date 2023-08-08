
#ifndef __SEPCUDAPRFRC_H__
#define  __SEPCUDAPRFRC_H__

#include "sepcudadefs.h"
#include "sepcudamisc.h"


bool sep_cuda_check_neighblist(sepcupart *ptr, float maxdist);
void sep_cuda_reset_exclusion(sepcupart *pptr);
void sep_cuda_copy_exclusion(sepcupart *pptr);
void sep_cuda_set_hexclusion(sepcupart *pptr, int a, int b);
void sep_cuda_set_exclusion(sepcupart *aptr, const char rule[]);

__global__ 
void sep_cuda_reset(float4 *force, float *epot, float4 *press, float4 *sumpress, float3 *energies, unsigned npart);

__global__ 
void sep_cuda_build_neighblist(int *neighlist, float4 *p, float *dist, float cf, float3 lbox, unsigned nneighmax, unsigned npart);

__global__ 
void sep_cuda_build_neighblist(int *neighlist, float *dist, float4 *p, int *molindex, float cf, float3 lbox, unsigned nneighmax, unsigned npart); 

/* Pair interactions - types specified */
__global__ 
void sep_cuda_lj(const char type1, const char type2, float3 params, int *neighblist, float4 *pos, float4 *force,
							int *molindex, float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart);

/* Pair interactions - all types have same interactions (faster) */
__global__ 
void sep_cuda_lj(float3 params, int *neighblist, float4 *pos, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart);

__global__ 
void sep_cuda_lj_sf(const char type1, const char type2, float3 params, int *neighblist, float4 *pos, float4 *force,
								float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart);

__global__
void sep_cuda_sf(float cf, int *neighblist, float4 *pos, float4 *vel, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart);


/* Wrapper interfaces*/

void sep_cuda_force_lj(sepcupart *pptr, const char types[], float params[3]);
void sep_cuda_force_lj(sepcupart *pptr, float params[3]);
void sep_cuda_force_lj_sf(sepcupart *pptr, const char types[], float params[3]);
void sep_cuda_force_sf(sepcupart *pptr, const float cf);
void sep_cuda_update_neighblist(sepcupart *pptr, sepcusys *sptr, float maxcutoff);

#endif
