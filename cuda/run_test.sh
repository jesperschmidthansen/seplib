#!/bin/bash

echo "Testing standard *.cu files"
time ./tgpu_0 start_singleAN1000.xyz
./tgpu_1 0; ./tgpu_1 1
for (( n=2; n<5; n++ ))
do 
	./tgpu_$n
done 

echo "Testing octave wrapper"
export LD_LIBRARY_PATH=.;
for (( n=0; n<3; n++ )) 
do 
	octave -q tgpu_$n.m
done


