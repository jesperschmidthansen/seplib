
#ifndef ___SEPCUDA_H__
#define ___SEPCUDA_H__


#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define SEP_CUDA_NTHREADS 32
#define SEP_CUDA_MAXNEIGHBS 600

#define SEP_CUDA_PI 3.14159265

#define SEP_MAX_NUMB_EXCLUSION 20
#define SEP_CUDA_EXCL_NONE 0
#define SEP_CUDA_EXCL_BONDS 1
#define SEP_CUDA_EXCL_MOLECULE 2


typedef struct{
	
	float4 *hx, *dx; //x,y,z,mass
	float4 *hv, *dv; //vx,vy,vz,charge 
	float4 *hf, *df; //fx,f,fz,type 
	float4 *hx0, *dx0; //Virtual lattice sites, x0, y0, z0 
	
	unsigned maxneighb; 
	int *neighblist; // neighb indicies + trailing -1s
	int3 *hcrossings, *dcrossings; // Simulation box crossing
	int *hexclusion, *dexclusion; // Exclusions (atom index) 

	float *epot;  // Potential energy on particle - on device
	float4 *press; //sumdiag,xy,xz,yz pressures - on device 
	float4 *sumpress; // sum of diag, xy, xz, yz
	
	char *ht;   // Type on host only
	
	float *ddist;  // distance travelled by atom - on device
	float dsumdist; // total distance travelled by all atoms - on device
	
	// A few additional members in order to reduce functions API argument list
	unsigned nthreads, nblocks;
	float3 lbox;
	unsigned npart, npart_padding;
	
	// Molecule index 
	int *hmolindex, *dmolindex; // 
	
	// Pair exlusion rules 0 - no exclusion rule, 1 - exclude bonds, 2 - exclude mol.
	unsigned hexclusion_rule, dexclusion_rule; 
	
} sepcupart;


typedef struct{

	unsigned nthreads, nblocks;
	unsigned npart, npart_padding;
	float3 lbox; 

	float skin;
	
	float *dalpha; // On device <- what is this....? 
	int *dupdate;  // Neighbourlist update? On device 
	float dt;
	
	float3 *henergies, *denergies;  // ekin, epot, momentum
	float ekin, epot, etot;
	float temp;
	
	bool molprop;	
} sepcusys;



#endif

