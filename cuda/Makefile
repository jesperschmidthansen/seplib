

INCLUDE_CUDA = /usr/include/
C_OPTIONS=-Xptxas -O3 -use_fast_math -Xcompiler -Ofast -Xcompiler -Wall   
OCTAVE_OPTIONS=-Xptxas -O3 -use_fast_math -Xcompiler -Wall -Xcompiler -shared -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler  

#COPTIONS=-Xptxas -O3 -use_fast_math -Xcompiler -O2 -gencode arch=compute_35,code=sm_35 -rdc=true


SOURCE_FILES = $(wildcard sep*.cu)
OBJECT_FILES = $(SOURCE_FILES:%.cu=%.o)

all: lib test tcpu_0

# Imagine just how hard I find Makefile syntax... we go manual...
test: libsepcuda.a
	nvcc $(C_OPTIONS)  *.o -o tgpu_0 tgpu_0.cu 
	nvcc $(C_OPTIONS)  *.o -o tgpu_1 tgpu_1.cu 
	nvcc $(C_OPTIONS)  *.o -o tgpu_2 tgpu_2.cu 
	nvcc $(C_OPTIONS)  *.o -o tgpu_3 tgpu_3.cu
	nvcc $(C_OPTIONS)  *.o -o tgpu_4 tgpu_4.cu 


# CPU comparison test 
tcpu_0: tcpu_0.c
	gcc -W -Wextra -fopenmp -Ofast -march=native -o tcpu_0 tcpu_0.c -lsep -lm 

# The library
lib: $(OBJECT_FILES)
	ar -crs libsepcuda.a $(OBJECT_FILES)
	ranlib libsepcuda.a

%.o: %.cu
	nvcc -c $(C_OPTIONS) $<

cmolsim.oct: cmolsim.cpp _cmolsim.cu
	nvcc $(OCTAVE_OPTIONS) sepcudamem.cu -o libsepcudamem.so
	nvcc $(OCTAVE_OPTIONS) sepcudaintgr.cu -o libsepcudaintgr.so
	nvcc $(OCTAVE_OPTIONS) sepcudamisc.cu -o libsepcudamisc.so
	nvcc $(OCTAVE_OPTIONS) sepcudaprfrc.cu -o libsepcudaprfrc.so
	nvcc $(OCTAVE_OPTIONS) sepcudamol.cu -o libsepcudamol.so
	nvcc $(OCTAVE_OPTIONS) _cmolsim.cu -o libcmolsim.so
	mkoctfile --verbose cmolsim.cpp -L. -I. -lcmolsim -lcudart -lsepcudamem -lsepcudaintgr -lsepcudamisc -lsepcudaprfrc -lsepcudamol

