# for t in 0 1 2
#     do
#         for s in 0 1
#             do
#                 for k in 3 5 7 9
#                     do
#                         CUDA_VISIBLE_DEVICES=0 ./host.o $k $s $t
#                     done
#             done
#     done

# for t in 1
#     do
#         for s in 0 1
#             do
#                 for k in 3 5 7 9
#                     do
#                         CUDA_VISIBLE_DEVICES=0 ./host.o $k $s $t
#                     done
#             done
#     done

for t in 2
    do
        for s in 0 1
            do
                CUDA_VISIBLE_DEVICES=0 ./host.o 7 $s $t
            done
    done