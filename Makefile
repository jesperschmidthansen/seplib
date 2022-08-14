# Makefile.in for sep-library

# Macros

CC          =  gcc
CFLAGS      =  -g -W -Wextra -Wpedantic -Ofast -fgnu89-inline -std=c99 -fPIC #@OMPFLAG@ 
RANLIB      =  ranlib
AR          =  ar 
COMPLEX     =  -DCOMPLEX
OMP         =  -fopenmp
PREFIX      =  /usr/local/
OBJECTS      =  sepmisc.o seputil.o separray.o sepinit.o sepprfrc.o sepintgr.o \
	       sepret.o sepmol.o sepcoulomb.o sepsampler.o sepomp.o	
PRGS        =  test/prg0 test/prg1 test/prg2 test/prg3 test/prg4 test/prg5 \
	       test/prg6 test/prg7 test/prg8 test/prg9
TOOLS       =  tools/sep_sfg tools/sep_lattice 
MARCH       =  

# all
all: libsep.a test tools doc

# Buildning library
libsep.a: $(OBJECTS)
	$(AR) r libsep.a $(OBJECTS) 
	$(RANLIB) libsep.a
	cp libsep.a *.o lib/

%.o:source/%.c 
	$(CC) $(COMPLEX) $(OMP) -c $(CFLAGS) -W -Wall -Iinclude/ $<

# Install (root only)
install: libsep.a
	cp include/*.h $(PREFIX)include/
	cp lib/libsep.a $(PREFIX)lib/

# Cleaning up
clean: 
	rm -f libsep.a *.o
	rm -f lib/*.o lib/*.a
	rm -f $(PRGS)
	rm -f source/*~
	rm -f include/*~
	rm -f mpi/*~
	rm -f doc/*~
	rm -f prgs/*~
	rm -f mydoc/*~
	rm -f *~
	rm -f $(TOOLS)

# Compiling programmes
test: $(PRGS)

test/%:prgs/%.c
	$(CC) $(CFLAGS) $(OMP) -o $@ -Llib/ -Iinclude $^ -lsep -lm

# Compiling tools

# Compiling tools
tools: $(TOOLS)

tools/%:tools/%.c
	$(CC) $(CFLAGS) $(OMP) -Iinclude -c tools/_sep_lattice.c
	$(CC) $(CFLAGS) $(OMP) -Iinclude -c tools/_sep_sfg.c
	$(CC) $(CFLAGS) $(OMP) _sep_lattice.o -o tools/sep_lattice tools/sep_lattice.c -lm
	$(CC) $(CFLAGS) $(OMP) _sep_sfg.o -o tools/sep_sfg tools/sep_sfg.c -lm	
