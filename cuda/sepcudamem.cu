
#include "sepcudamem.h"


sepcupart* sep_cuda_allocate_memory(unsigned npartPadding){
	sepcupart* ptr;
	
	if ( cudaMallocHost((void **)&ptr, sizeof(sepcupart))== cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	size_t nbytes = npartPadding*sizeof(float4);
	size_t nbytes_excludelist = (1+SEP_MAX_NUMB_EXCLUSION)*npartPadding*sizeof(int);

	// Host
	if ( cudaMallocHost((void **)&(ptr->hx), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hv), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hf), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hx0), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->ht), npartPadding*sizeof(char)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hexclusion), nbytes_excludelist) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hcrossings), npartPadding*sizeof(int3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMallocHost((void **)&(ptr->hmolindex), npartPadding*sizeof(int)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	// Device
	if ( cudaMalloc((void **)&(ptr->dx), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->dv), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->df), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->dx0), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->ddist), npartPadding*sizeof(float)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->epot), npartPadding*sizeof(float)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->press), nbytes) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();

	if ( cudaMalloc((void **)&(ptr->sumpress), sizeof(float4)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->dexclusion), nbytes_excludelist) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();

	ptr->maxneighb = SEP_CUDA_MAXNEIGHBS;
	if ( cudaMalloc(&(ptr->neighblist), sizeof(int)*npartPadding*(ptr->maxneighb)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();

	if ( cudaMalloc((void **)&(ptr->dcrossings), npartPadding*sizeof(int3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(ptr->dmolindex), npartPadding*sizeof(int)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	return ptr;
}

void sep_cuda_free_memory(sepcupart *ptr, sepcusys *sptr){
	
	// Particle structure
	cudaFreeHost(ptr->hx); 	cudaFreeHost(ptr->hv); 
	cudaFreeHost(ptr->hf); 	cudaFreeHost(ptr->hx0);
	cudaFreeHost(ptr->ht);
	
	cudaFreeHost(ptr->hexclusion); cudaFreeHost(ptr->hcrossings); cudaFreeHost(ptr->hmolindex); 
	
	cudaFree(ptr->dx); cudaFree(ptr->dv); cudaFree(ptr->df); cudaFree(ptr->dx0);
	cudaFree(ptr->ddist); cudaFree(ptr->neighblist);
	cudaFree(ptr->epot); cudaFree(ptr->press); cudaFree(ptr->sumpress); 
	
	cudaFree(ptr->dexclusion); cudaFree(ptr->dcrossings); cudaFree(ptr->dmolindex);
	
	
	cudaFreeHost(ptr);
	
	// System structure
	cudaFree(sptr->denergies); cudaFreeHost(sptr->henergies);
	cudaFree(sptr->dalpha); cudaFree(sptr->dupdate); 
	
	cudaFreeHost(sptr);
}


sepcupart* sep_cuda_load_xyz(const char *xyzfile){
	unsigned npart;
	int nthreads = SEP_CUDA_NTHREADS;
	
	FILE *fin = fopen(xyzfile, "r");
	if ( fin == NULL )
		sep_cuda_file_error();
	
	fscanf(fin, "%d\n", &npart);
	
	unsigned nblocks = (npart + nthreads - 1) / nthreads;
	unsigned npartwithPadding = nblocks*nthreads;
	
	sepcupart *ptr = sep_cuda_allocate_memory(npartwithPadding);
		
	ptr->nblocks = nblocks; 
	ptr->nthreads = nthreads;
	ptr->npart = npart; 
	ptr->npart_padding = npartwithPadding;
	ptr->hexclusion_rule = SEP_CUDA_EXCL_NONE;
	
	fscanf(fin, "%f %f %f\n", &(ptr->lbox.x), &(ptr->lbox.y), &(ptr->lbox.z));
	
	for ( unsigned n=0; n<npart; n++ ) {
		fscanf(fin, "%c %f %f %f %f %f %f %f %f\n", 
			   &(ptr->ht[n]), &(ptr->hx[n].x),&(ptr->hx[n].y),&(ptr->hx[n].z), 
			   &(ptr->hv[n].x),&(ptr->hv[n].y),&(ptr->hv[n].z), &(ptr->hx[n].w), &(ptr->hv[n].w));
		ptr->hcrossings[n].x = ptr->hcrossings[n].y = ptr->hcrossings[n].z = 0;
		ptr->hmolindex[n] = -1;
	}
	
	fclose(fin);

	for ( unsigned n=npart; n<npartwithPadding; n++ ){
		ptr->hx[n].x = ptr->hx[n].y = ptr->hx[n].z = 0.0f;
		ptr->hv[n].x = ptr->hv[n].y = ptr->hv[n].z = 0.0f;
		ptr->hv[n].w = 1.0; ptr->ht[n] = 'A';
		ptr->hmolindex[n] = -1;
	}
	
	sep_cuda_copy(ptr, 'x', 'd'); 
	sep_cuda_copy(ptr, 'v', 'd');
	sep_cuda_copy(ptr, 'f', 'd');
	sep_cuda_copy(ptr, 'c', 'd');
	
	cudaMemcpy(ptr->dmolindex, ptr->hmolindex, npartwithPadding*sizeof(int), cudaMemcpyHostToDevice);

	return ptr;
}


sepcusys *sep_cuda_sys_setup(sepcupart *pptr){
	
	sepcusys *sptr;
	if ( cudaMallocHost((void **)&sptr, sizeof(sepcusys)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	sptr->npart = pptr->npart;
	sptr->npart_padding = pptr->npart_padding;
	sptr->nblocks = pptr->nblocks;
	sptr->nthreads = pptr->nthreads;
	sptr->dt = 0.005;
	sptr->skin = 0.3;
	sptr->lbox = pptr->lbox;
	
	if ( cudaMallocHost((void **)&(sptr->henergies), sizeof(float3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(sptr->denergies), sizeof(float3)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	if ( cudaMalloc((void **)&(sptr->dalpha), sizeof(float)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	
	sep_cuda_setvalue<<<1,1>>>(sptr->dalpha, 0.2);
	
	
	if ( cudaMalloc((void **)&(sptr->dupdate), sizeof(int)) == cudaErrorMemoryAllocation )
		sep_cuda_mem_error();
	sep_cuda_setvalue<<<1,1>>>(sptr->dupdate, 1);
	
	return sptr;
}


