for C in nc MC
    do
        for N in 100 1000 10000
            do
                for T in nt MT
                    do
                        ./test $N gpu $T $C
                    done
            done
    done

# for N in 100 1000 10000
#     do
#         ./test $N cpu nt nc
#     done