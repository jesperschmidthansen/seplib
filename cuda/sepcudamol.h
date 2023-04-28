
#ifndef __SEPCUDAMOL_H__
#define __SEPCUDAMOL_H__

#include "sepcuda.h"

typedef struct {
	unsigned nmols; 
	unsigned *hnuau, *dnuau;
	
	unsigned nbonds;    /**< Total number of bonds */
	unsigned *hblist, *dblist;    /**< Bond list: (the two bonded part. indicies + bond type)*num_bonds */
	unsigned nbondblocks;
	
	unsigned nangles;    /**< Total number of bonds */
	unsigned *halist, *dalist;    /**< Bond list: (the two bonded part. indicies + bond type)*num_bonds */
	unsigned nangleblocks;
	
} sepcumol;


sepcumol * sep_cuda_init_mol(void);

FILE *sep_cuda_set_file_pointer(FILE *fptr, const char *section);

void sep_cuda_read_bonds(sepcupart *pptr, sepcumol *mptr, const char *file);
void sep_cuda_read_angles(sepcupart *pptr, sepcumol *mptr, const char *file);

void sep_cuda_free_bonds(sepcumol *mptr);
void sep_cuda_free_angles(sepcumol *mptr);
void sep_cuda_free_mols(sepcumol *mptr);

// Kernels
__global__ void sep_cuda_bond_harmonic(unsigned int *blist, unsigned nbonds, float3 bondspec, 
								  float4 *pos, float4 *force, float3 lbox);
__global__ void sep_cuda_angle(unsigned *alist, unsigned nangles, float3 anglespec, 
								  float4 *pos, float4 *force, float3 lbox);


// Wrappers
void sep_cuda_force_harmonic(sepcupart *pptr, sepcumol *mptr, int type, float ks, float lbond);
void sep_cuda_force_angle(sepcupart *pptr, sepcumol *mptr, int type, float ktheta, float angle0);

#endif