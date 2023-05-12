

#ifndef __SEPCUDA_H__
#define __SEPCUDA_H__


#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define SEP_CUDA_NTHREADS 32
#define SEP_CUDA_MAXNEIGHBS 400

#define SEP_CUDA_PI 3.14159265

typedef struct{
	
	float4 *hx, *dx; //x,y,z,mass
	float4 *hv, *dv; //vx,vy,vz,charge 
	float4 *hf, *df; //fx,f,fz,type 
	float4 *hx0, *dx0; //Virtual lattice sites, x0, y0, z0 
	
	unsigned maxneighb; 
	int *neighblist; // neighb indicies + trailing -1s
	int3 *hcrossings, *dcrossings; // Simulation box crossing
	int4 *hexclusion, *dexclusion; // Exclusions (atom index) 

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
	
} sepcupart;


typedef struct{

	unsigned nthreads, nblocks;
	unsigned npart, npart_padding;
	float3 lbox; 

	float skin;
	
	float *dalpha; // On device
	int *dupdate;  // On device
	float dt;
	
	float3 *henergies, *denergies;  // ekin, epot, momentum
	float ekin, epot, etot;
	float temp;
} sepcusys;



void sep_cuda_mem_error(void);
void sep_cuda_file_error(void);

sepcupart* sep_cuda_allocate_memory(unsigned npart);
sepcupart* sep_cuda_load_xyz(const char *xyzfile);
sepcusys *sep_cuda_sys_setup(sepcupart *pptr);

void sep_cuda_load_lattice_positions(sepcupart *pptr, const char *xyzfile);

void sep_cuda_free_memory(sepcupart *ptr, sepcusys *sptr);

void sep_cuda_copy(sepcupart *ptr, char opt_quantity, char opt_direction);

bool sep_cuda_check_neighblist(sepcupart *ptr, float maxdist);

void sep_cuda_copy_energies(sepcusys *sptr);
void sep_cuda_save_xyz(sepcupart *ptr, const char *xyzfile);
void sep_cuda_save_crossings(sepcupart *ptr, const char *filestr, float time);

void sep_cuda_copy_exclusion(sepcupart *pptr);
void sep_cuda_reset_exclusion(sepcupart *pptr);
void sep_cuda_set_hexclusion(sepcupart *pptr, int a, int b);

void sep_cuda_compressbox(sepcupart *aptr, float rho0, float compressfactor[3]);

void sep_cuda_get_pressure(double *npress, double *shearpress, sepcupart *aptr);
float sep_cuda_eval_momentum(sepcupart *aptr);

bool sep_cuda_logrem(unsigned n, int base);


// Kernels
__global__ void sep_cuda_reset(float4 *force, float *epot, float4 *press, float4 *sumpress, float3 *energies, unsigned npart);
__global__ void sep_cuda_build_neighblist(float *alpha, int *neighlist, int4 *exclusion, float4 *p, float *dist, float cf, 
										  float3 lbox, unsigned nneighmax, unsigned npart);

__global__ void sep_cuda_lj(const char type1, const char type2, float3 ljparams, int *neighblist, float4 *pos, float4 *force, 	
							float *epot, float4 *press, unsigned nneighbmax, float3 lbox, const unsigned npart);
__global__ void sep_cuda_lj(float3 params, int *neighblist, float4 *pos, float4 *force,
							float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart);
__global__ void sep_cuda_lj_sf(const char type1, const char type2, float3 params, int *neighblist, float4 *pos, float4 *force,
								float *epot, float4 *press, unsigned maxneighb, float3 lbox, const unsigned npart);

__global__ void sep_cuda_sf(float cf, int *neighblist, float4 *pos, float4 *vel, float4 *force,
							float *epot, float4 *press, unsigned nneighbmax, float3 lbox, const unsigned npart);

__global__ void sep_cuda_leapfrog(float4 *pos, float4 *vel, float4 *force, float *dist, 
								  float dt, float3 lbox, unsigned npart);

__global__ void sep_cuda_setvalue(float *, float);
__global__ void sep_cuda_setvalue(int *, int);
__global__ void sep_cuda_printvalue(float *value);

__global__ void sep_cuda_getpress(float4 *press, float4 *pos, float4 *vel, float4 *ppress, int npart);
__global__ void sep_cuda_sumdistance(float *totalsum, float *dist, float maxdist, unsigned npart);
__global__ void sep_cuda_sumenergies(float3 *totalsum, float4* dx, float4 *dv, float4 *df, 
									 float dt, float *epot, unsigned npart);


__global__ void sep_cuda_update_nosehoover(float *alpha, float3 *denergies, float temp0, 
										   float tau, float dt, unsigned int npart);
__global__ void sep_cuda_nosehoover(float *alpha, float4 *pos, float4 *vel, float4 *force, unsigned npart);

__global__ void sep_cuda_lattice_force(const char type, float springConstant, float4 *pos, float4 *pos0, float4 *force,
									   float3 lbox, const unsigned npart);

// Device functions 
__device__ float sep_cuda_wrap(float x, float lbox);

__device__ float sep_cuda_periodic(float x, float lbox, int *crossings);

__device__ float sep_cuda_dot(float4 a);
__device__ float sep_cuda_dot(float3 a, float3 b);

__device__ bool sep_cuda_check_exclude(int x, int y, int z, int w, int idxj);

// Wrappers
void sep_cuda_force_lj(sepcupart *pptr, const char types[], float params[3]);
void sep_cuda_force_lj(sepcupart *pptr, float params[3]);
void sep_cuda_force_lj_sf(sepcupart *pptr, const char types[], float params[3]);
void sep_cuda_force_sf(sepcupart *pptr, const float cf);
void sep_cuda_force_lattice(sepcupart *pptr, const char type, float springConstant);
void sep_cuda_thermostat_nh(sepcupart *pptr, sepcusys *sptr, float temp0, float tau);
void sep_cuda_reset_iteration(sepcupart *pptr, sepcusys *sptr);
void sep_cuda_update_neighblist(sepcupart *pptr, sepcusys *sptr, float maxcutoff);
void sep_cuda_integrate_leapfrog(sepcupart *pptr, sepcusys *sptr);
void sep_cuda_get_energies(sepcupart *ptr, sepcusys *sptr, const char ensemble[]);


#endif
