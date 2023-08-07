#include "sepcudasampler.h"

double** sep_cuda_matrix(size_t nrow, size_t ncol){
  double **ptr;
  size_t n, m;

  ptr = (double **)malloc(nrow*sizeof(double *));

  for (n=0; n<nrow; n++)
    ptr[n] = (double *)malloc(ncol*sizeof(double));

  for (n=0; n<nrow; n++)
    for (m=0; m<ncol; m++)
      ptr[n][m] = 0.0;

  return ptr;
}


void sep_cuda_free_matrix(double **ptr, size_t nrow){
  size_t n;

  for (n=0; n<nrow; n++)
    free(ptr[n]);

  free(ptr);
}


sepcugh* sep_cuda_sample_gh_init(sepcusys *sysptr, int lvec, unsigned nk, double dtsample){
	
	sepcugh *sptr = (sepcugh *)malloc(sizeof(sepcugh));
	
	sptr->dacf = (double **)sep_cuda_matrix(lvec, nk);
	sptr->tmacf = (double **)sep_cuda_matrix(lvec, nk);
	sptr->stress = (double **)sep_cuda_matrix(lvec, nk);
	
	sptr->wavevector = (double *)malloc(nk*sizeof(double));
	
	sptr->mcoskrArray = (double **)sep_cuda_matrix(lvec, nk);
	sptr->msinkrArray = (double **)sep_cuda_matrix(lvec, nk);

	sptr->vcoskrArray = (double **)sep_cuda_matrix(lvec, nk);
	sptr->vsinkrArray = (double **)sep_cuda_matrix(lvec, nk);
	
	sptr->stressa = (double **)sep_cuda_matrix(lvec, nk);
	sptr->stressb = (double **)sep_cuda_matrix(lvec, nk);
	
	sptr->nwaves = nk; sptr->lvec=lvec; sptr->dtsample = dtsample;
	
	sptr->index = 0; sptr->nsample = 0;
	
	FILE *fout = fopen("gh-wavevectors.dat", "w");
	if ( fout == NULL ) sep_cuda_file_error();
	
	for ( unsigned n=0; n<nk; n++ ){
		sptr->wavevector[n] = 2*SEP_CUDA_PI*(n+1)/sysptr->lbox.y;
		fprintf(fout, "%f\n", sptr->wavevector[n]);
	}
	fclose(fout);
	
	return sptr;
}	


void sep_cuda_sample_gh_free(sepcugh *ptr){
	
	sep_cuda_free_matrix(ptr->dacf, ptr->lvec);
	sep_cuda_free_matrix(ptr->tmacf, ptr->lvec);
	sep_cuda_free_matrix(ptr->stress, ptr->lvec);

	free(ptr->wavevector);
	
	sep_cuda_free_matrix(ptr->mcoskrArray, ptr->lvec);
	sep_cuda_free_matrix(ptr->msinkrArray, ptr->lvec);
	
	sep_cuda_free_matrix(ptr->vcoskrArray, ptr->lvec);
	sep_cuda_free_matrix(ptr->vsinkrArray, ptr->lvec);
	
	sep_cuda_free_matrix(ptr->stressa, ptr->lvec);
	sep_cuda_free_matrix(ptr->stressb, ptr->lvec);
	
	free(ptr);
}


void sep_cuda_sample_gh(sepcugh *sampleptr, sepcupart *pptr, sepcusys *sptr){
	
	sep_cuda_copy(pptr, 'x', 'h');
	sep_cuda_copy(pptr, 'v', 'h');
	sep_cuda_copy(pptr, 'f', 'h');
	
	unsigned index = sampleptr->index;
	
	for ( unsigned k=0; k<sampleptr->nwaves; k++ ){
	  
		double mcoskr = 0.0;	double msinkr = 0.0;
		double vcoskr = 0.0;	double vsinkr = 0.0;

		double stressa = 0.0; double stressb = 0.0;
		
		for ( unsigned n=0; n<sptr->npart; n++ ){
			double kr = sampleptr->wavevector[k]*pptr->hx[n].y;
			double mass = pptr->hx[n].w; double fx = pptr->hf[n].x;
			double velx = pptr->hv[n].x; double vely = pptr->hv[n].y;
			
			double ckr = cos(kr); double skr = sin(kr);
			mcoskr += mass*ckr; msinkr += mass*skr;
			vcoskr += mass*velx*ckr; vsinkr += mass*velx*skr;
			
			stressa += fx/sampleptr->wavevector[k]*ckr - mass*velx*vely*skr;
			stressb += fx/sampleptr->wavevector[k]*skr + mass*velx*vely*ckr;			
		}
		
		sampleptr->mcoskrArray[index][k] = mcoskr;
		sampleptr->msinkrArray[index][k] = msinkr;
		
		sampleptr->vcoskrArray[index][k] = vcoskr;
		sampleptr->vsinkrArray[index][k] = vsinkr;
		
		sampleptr->stressa[index][k] = stressa;
		sampleptr->stressb[index][k] = stressb;
	
	}
	
	(sampleptr->index)++;
	if ( sampleptr->index == sampleptr->lvec){
	
	   for ( unsigned k=0; k<sampleptr->nwaves; k++ ){
			
			for ( unsigned n=0; n<sampleptr->lvec; n++ ){
				for ( unsigned nn=0; nn<sampleptr->lvec-n; nn++ ){
					
					double costerm = (sampleptr->mcoskrArray[nn][k])*(sampleptr->mcoskrArray[nn+n][k]);
					double sinterm = (sampleptr->msinkrArray[nn][k])*(sampleptr->msinkrArray[nn+n][k]);
					
					sampleptr->dacf[n][k]  += costerm + sinterm;
					
					costerm = (sampleptr->vcoskrArray[nn][k])*(sampleptr->vcoskrArray[nn+n][k]);
					sinterm = (sampleptr->vsinkrArray[nn][k])*(sampleptr->vsinkrArray[nn+n][k]);
					
					sampleptr->tmacf[n][k] += costerm + sinterm;
					
					double asqr = (sampleptr->stressa[nn][k])*(sampleptr->stressa[nn+n][k]);
					double bsqr = (sampleptr->stressb[nn][k])*(sampleptr->stressb[nn+n][k]);
					
					sampleptr->stress[n][k] += asqr + bsqr;
				}
			}
		}
		(sampleptr->nsample)++;

		FILE *fout_tmacf = fopen("gh-tmacf.dat", "w");
		FILE *fout_dacf = fopen("gh-dacf.dat", "w");
		FILE *fout_stress = fopen("gh-stress.dat", "w");

		if ( fout_dacf == NULL || fout_tmacf == NULL || fout_stress == NULL ){
			fprintf(stderr, "Couldn't open file(s)\n");
		}

				double volume = sptr->lbox.x*sptr->lbox.y*sptr->lbox.z;

		for ( unsigned n=0; n<sampleptr->lvec; n++ ){
			double fac = 1.0/(sampleptr->nsample*volume*(sampleptr->lvec-n));
			double t   = n*sampleptr->dtsample;
      
			fprintf(fout_dacf, "%f ", t); fprintf(fout_tmacf, "%f ", t); fprintf(fout_stress, "%f ", t); 
		 
			 for ( unsigned k=0; k<sampleptr->nwaves; k++ ) {
				 fprintf(fout_dacf, "%f ", sampleptr->dacf[n][k]*fac);
				 fprintf(fout_tmacf, "%f ", sampleptr->tmacf[n][k]*fac);
				 fprintf(fout_stress, "%f ", sampleptr->stress[n][k]*fac);
				 
			 }
			 fprintf(fout_dacf, "\n"); fprintf(fout_tmacf, "\n");fprintf(fout_stress, "\n");
		}
	
		fclose(fout_dacf); 
		fclose(fout_tmacf);	
		fclose(fout_stress);
	
		sampleptr->index = 0;
	}
			
}
