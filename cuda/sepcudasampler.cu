#include "sepcudasampler.h"

// General purpose functions
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

// The gen. hydrodynamic sampler - atomic
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
					// God I miss C99!				
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


// The gen. hydrodynamic sampler - molecular UNDER CONSTRUCTION 
sepcumgh* sep_cuda_sample_mgh_init(sepcusys *sysptr, int lvec[2], unsigned nk, double dtsample){
	
	sepcumgh *sptr = (sepcumgh *)malloc(sizeof(sepcumgh));
	
	sptr->wavevector = (double *)malloc(nk*sizeof(double));

	sptr->stress = (double **)sep_cuda_matrix(lvec[0], nk);
	sptr->stressa = (double **)sep_cuda_matrix(lvec[0], nk);
	sptr->stressb = (double **)sep_cuda_matrix(lvec[0], nk);
	sptr->stresslvec = lvec[0]; 
	sptr->stressindex = 0; sptr->stressnsample = 0; 

	sptr->dipole = (double **)sep_cuda_matrix(lvec[1], nk);
	sptr->dipolea = (double **)sep_cuda_matrix(lvec[1], nk);
	sptr->dipoleb = (double **)sep_cuda_matrix(lvec[1], nk);
	sptr->dipolelvec = lvec[1];
	sptr->dipoleindex = 0; sptr->dipolensample = 0;

	sptr->nwaves = nk; sptr->dtsample = dtsample;
	
	FILE *fout = fopen("mgh-wavevectors.dat", "w");
	if ( fout == NULL ) sep_cuda_file_error();
	
	for ( unsigned n=0; n<nk; n++ ){
		sptr->wavevector[n] = 2*SEP_CUDA_PI*(n+1)/sysptr->lbox.y;
		fprintf(fout, "%f\n", sptr->wavevector[n]);
	}
	fclose(fout);
	
	return sptr;
}	


void sep_cuda_sample_mgh_free(sepcumgh *ptr){
	
	free(ptr->wavevector);
	
	sep_cuda_free_matrix(ptr->stress, ptr->stresslvec);
	sep_cuda_free_matrix(ptr->stressa, ptr->stresslvec);
	sep_cuda_free_matrix(ptr->stressb, ptr->stresslvec);
	
	sep_cuda_free_matrix(ptr->dipole, ptr->dipolelvec);
	sep_cuda_free_matrix(ptr->dipolea, ptr->dipolelvec);
	sep_cuda_free_matrix(ptr->dipoleb, ptr->dipolelvec);
	
	free(ptr);
}


void sep_cuda_print_current_corr(sepcusys *sptr, sepcumgh *sampler,  
		double **corr, double **a, double **b, unsigned lvec, int nsample, const char *filename){
		
			
	for ( unsigned k=0; k<sampler->nwaves; k++ ){
		
		for ( unsigned n=0; n<lvec; n++ ){
			for ( unsigned nn=0; nn<lvec-n; nn++ ){
				double asqr = a[nn][k]*a[nn+n][k];
				double bsqr = b[nn][k]*b[nn+n][k];
					
				corr[n][k] += asqr + bsqr;
			}	
		}
	}

	FILE *fout = fopen(filename, "w");
	if ( fout == NULL ) fprintf(stderr, "Couldn't open file(s)\n");
		
	double volume = sptr->lbox.x*sptr->lbox.y*sptr->lbox.z;
		
	for ( unsigned n=0; n<lvec; n++ ){
		double fac = 1.0/(nsample*volume*(lvec-n));
		double t   = n*sampler->dtsample;
	
		fprintf(fout, "%f ", t); 
 
		for ( unsigned k=0; k<sampler->nwaves; k++ ) {
			fprintf(fout, "%f ", corr[n][k]*fac); 
		}
		fprintf(fout, "\n") ;
	}
	
	fclose(fout); 
	
}

void sep_cuda_sample_mgh(sepcumgh *sampleptr, sepcupart *pptr, sepcusys *sptr, sepcumol *mptr){

	if ( !pptr->sptr->molprop ) {
		fprintf(stderr, "Mol. properties flag not set to 'on' - stress correlator not calculated\n");
		return;
	}

	if ( !sptr->cmflag ) 	
		sep_cuda_mol_calc_cmprop(pptr, mptr); // Calculations done and saved on host
	
	// Forces on molecular  - wavevector depedent stress and mech. properties
	sep_cuda_mol_calc_forceonmol(pptr, mptr);
	sep_cuda_copy(pptr, 'M', 'h');

	// Dipole moments - dielectric properties
	sep_cuda_mol_calc_dipoles(pptr, mptr);

	unsigned idxS = sampleptr->stressindex;
	unsigned idxD = sampleptr->dipoleindex;
	for ( unsigned k=0; k<sampleptr->nwaves; k++ ){
	  
		double stressa = 0.0; double stressb = 0.0;
		double dipolea = 0.0; double dipoleb = 0.0;

		for ( unsigned m=0; m<mptr->nmols; m++ ){
			double kr = sampleptr->wavevector[k]*mptr->hx[m].y;
			double ckr = cos(kr); double skr = sin(kr);

			double mass = mptr->masses[m]; double fx = mptr->hf[m].x;	
			double velx = mptr->hv[m].x; double vely = mptr->hv[m].y;
			
			stressa += fx/sampleptr->wavevector[k]*ckr - mass*velx*vely*skr;
			stressb += fx/sampleptr->wavevector[k]*skr + mass*velx*vely*ckr;			
			
			if ( k==0 )	kr = 0.0;
			else kr = sampleptr->wavevector[k-1]*mptr->hx[m].y;

			ckr = cos(kr); skr = sin(kr);

			double mui = mptr->hpel[m].y; 
			dipolea += mui*ckr; dipoleb += mui*skr; 
		}
				
		sampleptr->stressa[idxS][k] = stressa;	sampleptr->stressb[idxS][k] = stressb;
		sampleptr->dipolea[idxD][k] = dipolea;	sampleptr->dipoleb[idxD][k] = dipoleb;
	}
	
	(sampleptr->stressindex)++; (sampleptr->dipoleindex)++;

	if ( sampleptr->stressindex == sampleptr->stresslvec){
		(sampleptr->stressnsample)++;
		sampleptr->stressindex = 0;
		sep_cuda_print_current_corr(sptr, sampleptr, sampleptr->stress, sampleptr->stressa, sampleptr->stressb, 
										sampleptr->stresslvec, sampleptr->stressnsample, "mgh-stress.dat");
	}	
	
	if ( sampleptr->dipoleindex == sampleptr->dipolelvec ){
		(sampleptr->dipolensample)++;
		sampleptr->dipoleindex = 0;
		sep_cuda_print_current_corr(sptr, sampleptr, sampleptr->dipole, sampleptr->dipolea, sampleptr->dipoleb, 
										sampleptr->dipolelvec, sampleptr->dipolensample, "mgh-dipole.dat");
	}	


}

