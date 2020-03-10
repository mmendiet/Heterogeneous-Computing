#compile

g++ -c host.cpp
nvcc -c mycuda.cu
g++ -o test host.o mycuda.o -L/usr/lib/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv`

#g++ host.cpp -o test `pkg-config --cflags --libs opencv`