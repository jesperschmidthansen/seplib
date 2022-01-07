
octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if ( octave )
  mex -Ofast -W -pedantic -c -v -I../include/ task.c
  mex -DOCTAVE -Ofast -W -pedantic -c -v -I../include/ molsim.c 
  mex ../libsep.a task.o molsim.o ../_sep_lattice.o -o molsim

  mkoctfile evcorr.cpp
else
  mex -c -v CFLAGS='$CFLAGS -std=c99 -fPIC -fopenmp -DCOMPLEX -Wall -Ofast -I../include/' task.c
  mex -c -v CFLAGS='$CFLAGS -std=c99 -fPIC -fopenmp -DCOMPLEX -Wall -Ofast -I../include/' molsim.c
  mex -v ../libsep.a task.o molsim.o ../_sep_lattice.o -output molsim -lgomp
end




 
