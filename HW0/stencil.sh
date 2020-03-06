
# for N in 1000 10000 100000 1000000 10000000
#     do
#         ./stencil $N 2 gpu
#         ./stencil $N 2 cpu
#     done


for R in 4 4 4 4
    do
        nvcc 1Dstencil.cu -o stencil
        #./stencil 10000 $R gpu
        ./stencil 10000 $R cpu
    done