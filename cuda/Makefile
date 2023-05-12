


COPTIONS=-Xptxas -O3 -use_fast_math -Xcompiler -O2 --relocatable-device-code=true
#COPTIONS=-Xptxas -O3 -use_fast_math -Xcompiler -O2 -gencode arch=compute_35,code=sm_35 --relocatable-device-code=true

OBJECTFILES = sepcuda.o sepcudamol.o sepcudasampler.o

all: libsepcuda.a tgpu_0 tgpu_1 tgpu_2 tgpu_3 tcpu_0
	
# CPU comparison test 
tcpu_0: tcpu_0.c
	gcc -W -Wextra -fopenmp -Ofast -o tcpu_0 tcpu_0.c -lsep -lm 


# The famous SPC/Fw water model - checks for molecules and charges
tgpu_3: sepcuda.o tgpu_3.cu
	nvcc $(COPTIONS) -o tgpu_3 sepcuda.o sepcudamol.o tgpu_3.cu 

# The famous SPC/Fw water model - checks for molecules and charges
tgpu_2: sepcuda.o sepcudamol.o tgpu_2.cu
	nvcc $(COPTIONS) -o tgpu_2 sepcuda.o sepcudamol.o tgpu_2.cu 
	
# Sanity check - energy, pressure, rdf, sacf
tgpu_1: sepcuda.o sepcudasampler.o tgpu_1.cu 
	nvcc $(COPTIONS) -o tgpu_1 sepcuda.o sepcudasampler.o tgpu_1.cu

# Simplest benchmark test
tgpu_0: sepcuda.o tgpu_0.cu
	nvcc $(COPTIONS) -o tgpu_0 sepcuda.o tgpu_0.cu 

# The library
libsepcuda.a: $(OBJECTFILES)
	ar r libsepcuda.a $(OBJECTFILES)
	ranlib libsepcuda.a

sepcudamol.o: sepcudamol.cu sepcudamol.h
	nvcc -c $(COPTIONS) sepcudamol.cu 

sepcudasampler.o: sepcudasampler.cu sepcudasampler.h
	nvcc -c $(COPTIONS) sepcudasampler.cu 

sepcuda.o: sepcuda.cu sepcuda.h
	nvcc -c $(COPTIONS) sepcuda.cu 
