for N in 1000 10000 100000 1000000 10000000
    do
        for M in 128 256 512 1024
            do
                ./test $N $M
            done
    done
