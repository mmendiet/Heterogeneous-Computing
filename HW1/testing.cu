#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <chrono>

//#define N 1000
//#define M 512
//nvcc testing.cu -o test
//


__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

void cpuAdd(int *a, int *b, int *c, int n) {
    for(int i=0; i<n; ++i) {
        c[0] = a[0] + b[0];
    }
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

    // device copies of a, b, c
    int size = N * sizeof(int);

    // Setup input values
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);  
    if (strcmp(argv[3],"gpu")==0) {
        float total_time = 0;
        for(int i=0; i< 1000; ++i) {
            // host copies of a, b, c
            int *d_a, *d_b, *d_c;
            // Allocate space for device copies of a, b, c
            cudaMalloc((void **)&d_a, size);
            cudaMalloc((void **)&d_b, size);
            cudaMalloc((void **)&d_c, size);

            // Copy inputs to device
            cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
            // Launch add() kernel on GPU

            float time;
            cudaEvent_t start, stop;
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord( start, 0 );
            add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &time, start, stop );
            cudaEventDestroy( start );
            cudaEventDestroy( stop );
            total_time += time;
        
            // Copy result back to host
            cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        }
        float nanosec = (total_time)/1000;
        std::cout << "N: " << N << "   M: " << M << "   GPU time: " << nanosec << "ns" << std::endl;
    }

    else {
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuAdd(a,b,c, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
        std::cout << "N: " << N << "   M: " << M << "   CPU time: " << duration << "ns" << std::endl;
        //time = 100.0;
    }
    //printf("a[0]: %i, b[0]: %i, c[0]: %i\nGPU Time: %f\n", a[0], b[0], c[0], time);
    //printf("N: %i   M: %i   Time: %f\n", N, M, time);
    // Cleanup
    free(a); free(b); free(c);

    return 0;
}
