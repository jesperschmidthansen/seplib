#! /bin/bash

max_id=8

rm *.dat

for (( n=0; n<=$max_id; n++ ));
do
    echo "Running test $n ... "

    if [ "$n" -eq "4" ]
    then
	time ./prg$n 1
	time ./prg$n 2
	time ./prg$n 4
    else
	./prg$n
    fi
    
    echo "done"
done
