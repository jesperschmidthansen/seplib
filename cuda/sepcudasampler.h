
#ifndef __SEPCUDASAMPLER_H__
#define __SEPCUDASAMPLER_H__

#include "sepcudadefs.h"
#include "sepcudamisc.h"

typedef struct {
	
	double **dacf;
	double **tmacf;
	double **stress;
	
	double **mcoskrArray;
	double **msinkrArray;

	double **vcoskrArray;
	double **vsinkrArray;
	
	double **stressa, **stressb;
	
	double *wavevector;
	
	unsigned int lvec, nwaves;
	unsigned int index, nsample;
	
	double dtsample;
} sepcugh;

typedef struct {

	double **stress;
	double **stressa, **stressb;

	double *wavevector;
	
	unsigned int lvec, nwaves;
	unsigned int index, nsample;
	
	double dtsample;

} sepcumgh;

/*
typedef struct {

	double *corr;
	double *stress;

	unsigned index, nsample, lvec;
	double dtsample;

} sepcumsacf;
*/

// Aux
double** sep_cuda_matrix(size_t nrow, size_t ncol);
void sep_cuda_free_matrix(double **ptr, size_t nrow);

// gh samlper
sepcugh* sep_cuda_sample_gh_init(sepcusys *sysptr, int lvec, unsigned nk, double dtsample);
void sep_cuda_sample_gh(sepcugh *sampleptr, sepcupart *pptr, sepcusys *sptr);
void sep_cuda_sample_gh_free(sepcugh *ptr);

// mgh sampler
sepcumgh* sep_cuda_sample_mgh_init(sepcusys *sysptr, int lvec, unsigned nk, double dtsample);
void sep_cuda_sample_mgh_free(sepcumgh *ptr);
void sep_cuda_sample_mgh(sepcumgh *sampleptr, sepcupart *pptr, sepcusys *sptr, sepcumol *mptr);

#endif
