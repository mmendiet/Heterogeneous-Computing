# for N in 1000 10000 100000 1000000 10000000
#     do
#         for M in 128 256 512 1024
#             do
#                 #./test $N $M gpu
#                 ./test $N $M cpu
#             done
#     done

# for N in 1000 10000 100000 1000000 10000000
#     do
#         ./test2 $N 128 gpu
#         #./test2 $N 128 cpu
#     done

for M in 128 256 512 1024
    do
        ./test2 1000000 $M gpu
        #./test2 $N 128 cpu
    done

