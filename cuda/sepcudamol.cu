#include "sepcudamol.h"


sepcumol * sep_cuda_init_mol(void){
	
	sepcumol *mptr = (sepcumol *)malloc(sizeof(sepcumol));
	if ( mptr==NULL ) sep_cuda_mem_error();
	
	mptr->nmols = 0; 
	
	return mptr;
}



FILE *sep_cuda_set_file_pointer(FILE *fptr, const char *section){
	char line[256];

	do {

		if ( fgets(line, 256, fptr) == NULL ) break;

		if ( strcmp(line, section) == 0 ){
			if ( fgets(line, 256, fptr)==NULL )
				sep_cuda_file_error();
			return fptr;
		}

	}  while ( !feof(fptr) );

	return NULL;
}


void sep_cuda_read_bonds(sepcupart *pptr, sepcumol *mptr, const char *file){
	const char section[] = {'[', ' ', 'b', 'o', 'n', 'd', 's', ' ', ']','\n', '\0'};
	char line[256];
	fpos_t pos_file;
	unsigned moli, a, b, type;

	mptr->nbonds = 0; 

	// Broken
	//sep_cuda_reset_exclusion(pptr);
	
	FILE *fptr = fopen(file, "r");
	if ( fptr == NULL ) sep_cuda_file_error();
	
	// We *must* init the pointer since it will be free no matter if the read is sucessful or not
	mptr->hblist = (unsigned *)malloc(0);
	if ( mptr->hblist == NULL ) sep_cuda_mem_error();
 
	// Find the 'bonds' section 
	fptr = sep_cuda_set_file_pointer(fptr, section);
	if ( fptr==NULL ) sep_cuda_file_error();
	
	do {

		fgetpos(fptr, &pos_file); 
		if ( fgets(line, 256, fptr) == NULL ) sep_cuda_file_error();
		
		if ( line[0] == '[' ) {
			break;
		}
		else {
      
			fsetpos(fptr, &pos_file); 
      
			int sc = fscanf(fptr, "%u%u%u%u\n", &moli, &a, &b, &type);
			if ( sc != 4 ) sep_cuda_file_error();
     
			(mptr->nbonds) ++;
    
			mptr->hblist = (unsigned *)realloc((mptr->hblist), sizeof(unsigned)*3*mptr->nbonds);
			if ( mptr->hblist == NULL ) sep_cuda_mem_error();
      
			int index0 = (mptr->nbonds-1)*3;
      
			pptr->hmolindex[a]=moli;
			pptr->hmolindex[b]=moli;
			
			mptr->hblist[index0] = a;
			mptr->hblist[index0+1] = b;
			mptr->hblist[index0+2] = type;
			
			/* Broken
			sep_cuda_set_hexclusion(pptr, a, b); 
			sep_cuda_set_hexclusion(pptr, b, a);
			*/
			
			if ( moli > mptr->nmols ) mptr->nmols = moli;
		}
	} while ( !feof(fptr) ); 
	
	fclose(fptr);

	// Since number of mols is one bigger than the index
	(mptr->nmols)++; mptr->nbondblocks = mptr->nbonds/SEP_CUDA_NTHREADS + 1 ;
	
	fprintf(stdout, "Succesfully read 'bond' section in file %s -> ", file);
    fprintf(stdout, "Found %d molecule(s) and %d bond(s)\n", mptr->nmols, mptr->nbonds);
	fprintf(stdout, "Copying to device\n");
	
	// Broken
	//sep_cuda_copy_exclusion(pptr);
	
	size_t nbytes =  3*(mptr->nbonds)*sizeof(unsigned int);
	if ( cudaMalloc((void **)&(mptr->dblist),nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	cudaMemcpy(mptr->dblist, mptr->hblist, nbytes, cudaMemcpyHostToDevice);

	nbytes = pptr->npart_padding*sizeof(int);
	cudaMemcpy(pptr->dmolindex, pptr->hmolindex, nbytes, cudaMemcpyHostToDevice);
	
}


void sep_cuda_read_angles(sepcupart *pptr, sepcumol *mptr, const char *file){
	const char section[] = {'[', ' ', 'a', 'n', 'g', 'l', 'e', 's', ' ', ']','\n', '\0'};
	char line[256];
	fpos_t pos_file;
	unsigned moli, a, b, c, type;

	if ( mptr->nmols == 0 ) {
		fprintf(stderr, "Bond section must be read before angle section");
		exit(EXIT_FAILURE);
	}
	
	mptr->nangles = 0; 

	// Broke
	//sep_cuda_reset_exclusion(pptr);
	
	FILE *fptr = fopen(file, "r");
	if ( fptr == NULL ) sep_cuda_file_error();
	
	// We *must* init the pointer since it will be free no matter if the read is sucessful or not
	mptr->halist = (unsigned *)malloc(0);
	if ( mptr->halist == NULL ) sep_cuda_mem_error();
 
	// Find the 'angles' section 
	fptr = sep_cuda_set_file_pointer(fptr, section);
	if ( fptr==NULL ) sep_cuda_file_error();
	
	do {

		fgetpos(fptr, &pos_file); 
		if ( fgets(line, 256, fptr) == NULL ) sep_cuda_file_error();
		
		if ( line[0] == '[' ) {
			break;
		}
		else {
      
			fsetpos(fptr, &pos_file); 
      
			int sc = fscanf(fptr, "%u%u%u%u%u\n", &moli, &a, &b, &c, &type);
			if ( sc != 5 ) sep_cuda_file_error();
     
			(mptr->nangles) ++;
    
			mptr->halist = (unsigned *)realloc((mptr->halist), sizeof(unsigned)*4*mptr->nangles);
			if ( mptr->halist == NULL ) sep_cuda_mem_error();
      
			int index0 = (mptr->nangles-1)*4;
      
			mptr->halist[index0] = a;
			mptr->halist[index0+1] = b;
			mptr->halist[index0+2] = c;
			mptr->halist[index0+3] = type;
			
			/* Broken
			sep_cuda_set_hexclusion(pptr, a, b); sep_cuda_set_hexclusion(pptr, a, c); 
			sep_cuda_set_hexclusion(pptr, b, a); sep_cuda_set_hexclusion(pptr, b, c); 
			sep_cuda_set_hexclusion(pptr, c, a); sep_cuda_set_hexclusion(pptr, c, b);			
			*/
		}
	} while ( !feof(fptr) ); 
	
	fclose(fptr);

	// Since number of mols is one bigger than the index
	mptr->nangleblocks = mptr->nangles/SEP_CUDA_NTHREADS + 1 ;
	
	fprintf(stdout, "Succesfully read 'angle' section in file %s -> ", file);
    fprintf(stdout, "Found %d angles(s)\n", mptr->nangles);
	
	if ( mptr->nangles > 0 ){
		fprintf(stdout, "Copying to device\n");
	
		size_t nbytes =  4*(mptr->nangles)*sizeof(unsigned int);
		if ( cudaMalloc((void **)&(mptr->dalist),nbytes) == cudaErrorMemoryAllocation )
			sep_cuda_mem_error();
	
		cudaMemcpy(mptr->dalist, mptr->halist, nbytes, cudaMemcpyHostToDevice);
		
		// Broken
		//sep_cuda_copy_exclusion(pptr);
	}
	
}


void sep_cuda_read_dihedrals(sepcupart *pptr, sepcumol *mptr, const char *file){
	const char section[] = {'[', ' ', 'd', 'i', 'h', 'e', 'd', 'r', 'a', 'l', 's', ' ', ']','\n', '\0'};
	char line[256];
	fpos_t pos_file;
	unsigned moli, a, b, c, d, type;

	if ( mptr->nmols == 0 ) {
		fprintf(stderr, "Bond section must be read before dihedral section");
		exit(EXIT_FAILURE);
	}
	
	mptr->ndihedrals = 0; 

	// Broken
	//sep_cuda_reset_exclusion(pptr);
	
	FILE *fptr = fopen(file, "r");
	if ( fptr == NULL ) sep_cuda_file_error();
	
	// We *must* init the pointer since it will be free no matter if the read is sucessful or not
	mptr->hdlist = (unsigned *)malloc(0);
	if ( mptr->hdlist == NULL ) sep_cuda_mem_error();
 
	// Find the 'angles' section 
	fptr = sep_cuda_set_file_pointer(fptr, section);
	if ( fptr==NULL ) sep_cuda_file_error();
	
	do {

		fgetpos(fptr, &pos_file); 
		if ( fgets(line, 256, fptr) == NULL ) sep_cuda_file_error();
		
		if ( line[0] == '[' ) {
			break;
		}
		else {
      
			fsetpos(fptr, &pos_file); 
      
			int sc = fscanf(fptr, "%u%u%u%u%u%u\n", &moli, &a, &b, &c, &d, &type);
			if ( sc != 6 ) sep_cuda_file_error();
     
			(mptr->ndihedrals) ++;
    
			mptr->hdlist = (unsigned *)realloc((mptr->hdlist), sizeof(unsigned)*5*mptr->ndihedrals);
			if ( mptr->hdlist == NULL ) sep_cuda_mem_error();
      
			int index0 = (mptr->ndihedrals-1)*5;
      
			mptr->hdlist[index0] = a;
			mptr->hdlist[index0+1] = b;
			mptr->hdlist[index0+2] = c;
			mptr->hdlist[index0+3] = d;
			mptr->hdlist[index0+4] = type;
			
			/* Broken
			sep_cuda_set_hexclusion(pptr, a, b); sep_cuda_set_hexclusion(pptr, a, c); sep_cuda_set_hexclusion(pptr, a, d);  
			sep_cuda_set_hexclusion(pptr, b, a); sep_cuda_set_hexclusion(pptr, b, c); sep_cuda_set_hexclusion(pptr, b, d); 
			sep_cuda_set_hexclusion(pptr, c, a); sep_cuda_set_hexclusion(pptr, c, b); sep_cuda_set_hexclusion(pptr, c, d);
			sep_cuda_set_hexclusion(pptr, d, a); sep_cuda_set_hexclusion(pptr, d, b); sep_cuda_set_hexclusion(pptr, d, c);
			*/
		}
	} while ( !feof(fptr) ); 
	
	fclose(fptr);

	// Since number of mols is one bigger than the index
	mptr->ndihedralblocks = mptr->ndihedrals/SEP_CUDA_NTHREADS + 1 ;
	
	fprintf(stdout, "Succesfully read 'dihedral' section in file %s -> ", file);
    fprintf(stdout, "Found %d angles(s)\n", mptr->ndihedrals);
	
	if ( mptr->ndihedrals > 0 ){
		fprintf(stdout, "Copying to device\n");
	
		size_t nbytes =  5*(mptr->nangles)*sizeof(unsigned int);
		if ( cudaMalloc((void **)&(mptr->ddlist),nbytes) == cudaErrorMemoryAllocation )
			sep_cuda_mem_error();
	
		cudaMemcpy(mptr->ddlist, mptr->hdlist, nbytes, cudaMemcpyHostToDevice);
		
		// Broken
		//sep_cuda_copy_exclusion(pptr);
		
		cudaDeviceSynchronize();
	}
	
}


void sep_cuda_free_bonds(sepcumol *mptr){

	free(mptr->hblist);
	cudaFree(mptr->dblist);
  
}

void sep_cuda_free_angles(sepcumol *mptr){

	free(mptr->halist);
	cudaFree(mptr->dalist);
}

void sep_cuda_free_dihedrals(sepcumol *mptr){

	free(mptr->hdlist);
	cudaFree(mptr->ddlist);
}

/*
void sep_cuda_free_mols(sepcumol *mptr){
	
	sep_cuda_free_bonds(mptr);
	sep_cuda_free_angles(mptr);
	sep_cuda_free_dihedrals(mptr);
}
*/

__global__ void sep_cuda_bond_harmonic(unsigned *blist, unsigned nbonds, float3 bondspec, 
								  float4 *pos, float4 *force, float3 lbox){
	
	unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
	
	if ( i<nbonds ){
		
		int type =  __float2int_rd(bondspec.z);
		unsigned offset = i*3;
		
		if ( blist[offset+2] == type ) {
			
			unsigned a = blist[offset]; unsigned b = blist[offset+1]; 
			
			float dx = pos[a].x - pos[b].x; dx = sep_cuda_wrap(dx, lbox.x);
			float dy = pos[a].y - pos[b].y; dy = sep_cuda_wrap(dy, lbox.y);
			float dz = pos[a].z - pos[b].z; dz = sep_cuda_wrap(dz, lbox.z);

			float dist = sqrtf(dx*dx + dy*dy + dz*dz);
						
			float ft = -bondspec.x*(dist - bondspec.y)/dist;
			
			//ACHTUNG slow perhaps
			atomicAdd(&(force[a].x), ft*dx); 
			atomicAdd(&(force[a].y), ft*dy); 
			atomicAdd(&(force[a].z), ft*dz);
			
			atomicAdd(&(force[b].x), -ft*dx); 
			atomicAdd(&(force[b].y), -ft*dy); 
			atomicAdd(&(force[b].z), -ft*dz);
			
		}
	}
	
}

__global__ void sep_cuda_angle(unsigned *alist, unsigned nangles, float3 anglespec, 
								  float4 *pos, float4 *force, float3 lbox){
	
	unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
	
	if ( i<nangles ){
		
		int type =  __float2int_rd(anglespec.z);
		unsigned offset = i*4;
		float cCon = cos(SEP_CUDA_PI - anglespec.y);
		
		if ( alist[offset+3] == type ) {
			
			unsigned a = alist[offset]; 
			unsigned b = alist[offset+1];
			unsigned c = alist[offset+2];
			 
			float3 dr1, dr2;
			
			dr1.x = pos[b].x - pos[a].x; dr1.x = sep_cuda_wrap(dr1.x, lbox.x);
			dr1.y = pos[b].y - pos[a].y; dr1.y = sep_cuda_wrap(dr1.y, lbox.y);
			dr1.z = pos[b].z - pos[a].z; dr1.z = sep_cuda_wrap(dr1.z, lbox.z);

			dr2.x = pos[c].x - pos[b].x; dr2.x = sep_cuda_wrap(dr2.x, lbox.x);
			dr2.y = pos[c].y - pos[b].y; dr2.y = sep_cuda_wrap(dr2.y, lbox.y);
			dr2.z = pos[c].z - pos[b].z; dr2.z = sep_cuda_wrap(dr2.z, lbox.z);
	
			float c11 = sep_cuda_dot(dr1, dr1);
			float c12 = sep_cuda_dot(dr1, dr2);
			float c22 = sep_cuda_dot(dr2, dr2);
      
			float cD = sqrtf(c11*c22); float cc = c12/cD; 

			float f = -anglespec.x*(cc - cCon);
      
			float3 f1, f2;
			
			f1.x = f*((c12/c11)*dr1.x - dr2.x)/cD; 
			f1.y = f*((c12/c11)*dr1.y - dr2.y)/cD;
			f1.z = f*((c12/c11)*dr1.z - dr2.z)/cD;
			
			f2.x = f*(dr1.x - (c12/c22)*dr2.x)/cD;
			f2.y = f*(dr1.y - (c12/c22)*dr2.y)/cD;
			f2.z = f*(dr1.z - (c12/c22)*dr2.z)/cD;
				
			//ACHTUNG slow perhaps
			atomicAdd(&(force[a].x), f1.x); 
			atomicAdd(&(force[a].y), f1.y); 
			atomicAdd(&(force[a].z), f1.z);
			
			atomicAdd(&(force[b].x), -f1.x-f2.x); 
			atomicAdd(&(force[b].y), -f1.y-f2.y); 
			atomicAdd(&(force[b].z), -f1.z-f2.z); 
			
			atomicAdd(&(force[c].x), f2.x); 
			atomicAdd(&(force[c].y), f2.y); 
			atomicAdd(&(force[c].z), f2.z);
			
		}
	}
	
}



__global__ void sep_cuda_ryckertbellemann(unsigned *dlist, unsigned ndihedrals, int type, 
										  float params1, float params2, float params3, float params4, float params5,
											float4 *pos, float4 *force, float3 lbox){
	
	unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
	
	if ( i<ndihedrals ){
		
		unsigned offset = i*5;
		
		if ( dlist[offset+4] == type ) {
			
			unsigned a = dlist[offset]; 
			unsigned b = dlist[offset+1];
			unsigned c = dlist[offset+2];
			unsigned d = dlist[offset+3];
			
			float3 dr1, dr2, dr3;

			dr1.x = pos[b].x - pos[a].x; dr1.x = sep_cuda_wrap(dr1.x, lbox.x);
			dr1.y = pos[b].y - pos[a].y; dr1.y = sep_cuda_wrap(dr1.y, lbox.y);
			dr1.z = pos[b].z - pos[a].z; dr1.z = sep_cuda_wrap(dr1.z, lbox.z);

			dr2.x = pos[c].x - pos[b].x; dr2.x = sep_cuda_wrap(dr2.x, lbox.x);
			dr2.y = pos[c].y - pos[b].y; dr2.y = sep_cuda_wrap(dr2.y, lbox.y);
			dr2.z = pos[c].z - pos[b].z; dr2.z = sep_cuda_wrap(dr2.z, lbox.z);
	
			dr3.x = pos[d].x - pos[c].x; dr3.x = sep_cuda_wrap(dr3.x, lbox.x);
			dr3.y = pos[d].y - pos[c].y; dr3.y = sep_cuda_wrap(dr3.y, lbox.y);
			dr3.z = pos[d].z - pos[c].z; dr3.z = sep_cuda_wrap(dr3.z, lbox.z);
	
			float c11 = sep_cuda_dot(dr1, dr1); 
			float c12 = sep_cuda_dot(dr1, dr2);
			float c13 = sep_cuda_dot(dr1, dr3);
			float c22 = sep_cuda_dot(dr2, dr2);
			float c23 = sep_cuda_dot(dr2, dr3);
			float c33 = sep_cuda_dot(dr3, dr3);
			 
			float cA = c13*c22 - c12*c23;
			float cB1 = c11*c22 - c12*c12;
			float cB2 = c22*c33 - c23*c23;
			float cD = sqrt(cB1*cB2); 
			float cc = cA/cD;

			float f = -(params1+(2.*params2+(3.*params3+(4.*params4+5.*params5*cc)*cc)*cc)*cc);
			
			float t1 = cA; 
			float t2 = c11*c23 - c12*c13;
			float t3 = -cB1; 
			float t4 = cB2;
			float t5 = c13*c23 - c12*c33; 
			float t6 = -cA;
			float cR1 = c12/c22; 
			float cR2 = c23/c22;
			
			float3 f1;
			f1.x = f*c22*(t1*dr1.x + t2*dr2.x + t3*dr3.x)/(cD*cB1);
			f1.y = f*c22*(t1*dr1.y + t2*dr2.y + t3*dr3.y)/(cD*cB1);
			f1.z = f*c22*(t1*dr1.z + t2*dr2.z + t3*dr3.z)/(cD*cB1);
			
			float3 f2;
			f2.x = f*c22*(t4*dr1.x + t5*dr2.x + t6*dr3.x)/(cD*cB2);
			f2.y = f*c22*(t4*dr1.y + t5*dr2.y + t6*dr3.y)/(cD*cB2);
			f2.z = f*c22*(t4*dr1.z + t5*dr2.z + t6*dr3.z)/(cD*cB2);
			
			//ACHTUNG slow perhaps
			atomicAdd(&(force[a].x), f1.x); 
			atomicAdd(&(force[a].y), f1.y);
			atomicAdd(&(force[a].z), f1.z);
			
			atomicAdd(&(force[b].x), (-(1.0 + cR1)*f1.x + cR2*f2.x)); 
			atomicAdd(&(force[b].y), (-(1.0 + cR1)*f1.y + cR2*f2.y));
			atomicAdd(&(force[b].z), (-(1.0 + cR1)*f1.z + cR2*f2.z));
			
			atomicAdd(&(force[c].x), (cR1*f1.x - (1.0 + cR2)*f2.x));
			atomicAdd(&(force[c].y), (cR1*f1.y - (1.0 + cR2)*f2.y));
			atomicAdd(&(force[c].z), (cR1*f1.z - (1.0 + cR2)*f2.z));
			
			atomicAdd(&(force[d].x), f2.x); 
			atomicAdd(&(force[d].y), f2.y);
			atomicAdd(&(force[d].z), f2.z);
			
      }
	}
		
}

			
			
			

void sep_cuda_force_harmonic(sepcupart *pptr, sepcumol *mptr, int type, float ks, float lbond){
	int nb = mptr->nbondblocks; 
	int nt = pptr->nthreads;
	
	// Notice the change in sequence
	float3 bondinfo = make_float3(ks, lbond, type);
	
	sep_cuda_bond_harmonic<<<nb,nt>>>
		(mptr->dblist, mptr->nbonds, bondinfo, pptr->dx, pptr->df, pptr->lbox);
		
	cudaDeviceSynchronize();
}


void sep_cuda_force_angle(sepcupart *pptr, sepcumol *mptr, int type, float ktheta, float angle0){
	int nb = mptr->nangleblocks; 
	int nt = pptr->nthreads;
	
	// Notice the change in sequence
	float3 angleinfo = make_float3(ktheta, angle0, type);
	
	sep_cuda_angle<<<nb,nt>>>
		(mptr->dalist, mptr->nangles, angleinfo, pptr->dx, pptr->df, pptr->lbox);
		
	cudaDeviceSynchronize();
}

void sep_cuda_force_dihedral(sepcupart *pptr, sepcumol *mptr, int type, float params[6]){
	int nb = mptr->ndihedralblocks; 
	int nt = pptr->nthreads;
	
	
	sep_cuda_ryckertbellemann<<<nb, nt>>>
		(mptr->ddlist, mptr->ndihedrals, type, 
		 params[1], params[2], params[3], params[4], params[5], pptr->dx, pptr->df, pptr->lbox);

	cudaDeviceSynchronize();

}
