#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <iostream>
#include <fstream>

//#define N 1000
//#define M 512


__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

void random_ints(int* x, int size)
{
	int i;
	for (i=0;i<size;i++) {
		x[i]=rand()%10;
	}
}

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int *a, *b, *c;
    // host copies of a, b, c
    int *d_a, *d_b, *d_c;
    // device copies of a, b, c
    int size = N * sizeof(int);
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // Setup input values
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);       

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    //printf("a[0]: %i, b[0]: %i, c[0]: %i\nGPU Time: %f\n", a[0], b[0], c[0], time);
    printf("N: %i   M: %i   Time: %f\n", N, M, time);
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}