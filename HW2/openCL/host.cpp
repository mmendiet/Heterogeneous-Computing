#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include "mycuda.h"

#include <stdio.h>                                    
//Add the necessary headers for the file
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
//#include <CL/cl.h>

// g++ -c main.cpp
// nvcc -arch=sm_20 -c test.cu
// g++  -o test main.o test.o -L/usr/local/cuda/lib64 -lcudart

//nvcc test2.cu -o test2 `pkg-config --cflags --libs opencv`



//"__kernel void vecadd(__global int *A, __global int *B, __global int *C, __global unsigned int n) {int idx = get_global_id(0);if(id<n) {C[idx] = A[idx] + B[idx];}}";
int LoadOpenCLKernel(char const* path, char **buf);
void GEMM(int ki, bool shared);
void networkGEMM(bool shared);
void dConv(int ki, bool shared);

void random_ints(int* x, int size)
    {
        //srand(time(0)); 
        int i;
        for (i=0;i<size;i++) {
            x[i]=rand()%10;
            //std::cout << x[i] << " ";
        }
    }

void get_im(int* x, cv::Mat image) {
    cv::Mat bgr[3];
    split(image,bgr);

    bgr[0] = bgr[0].t();
    bgr[1] = bgr[1].t();
    bgr[2] = bgr[2].t();

    // std::cout << bgr[0] << "\n\n\n\n\n\n";
    // std::cout << bgr[1] << "\n\n\n\n\n\n";
    // std::cout << bgr[2] << std::endl;

    int H = image.rows;
    int W = image.cols;
    // int x_size = H*W*3*sizeof(int);
    // int* x = (int*)malloc(x_size);
    for(int c=0; c<3; c++) {
        for (int h=0; h<H; h++){
            for (int w=0; w<W; w++){
                x[(c*H*W)+h*W+w] = (int)bgr[c].data[h*W+w];
                //std::cout << (int)bgr[c].data[h*W+w] << " ";
            }
        }
    }
}

void im2col(int H, int W, int k, int* x, int* im, int stride) {
    int count = 0;
    int limit = (H-(k-1))*(W-(k-1));
    //int l = limit-(k*W-k+1);
    for(int c=0;c<3;c++) {
        int xy = 0;
        for(int y=0; y<limit; y=y+stride) {
            for(int j=0; j<k; j++) {
                for(int i=0; i<k; i++) {
                    x[(c*limit*k*k) + k*k*xy+k*j+i] = im[(c*H*W) + j*W + i + y];
                    count++;
                }
            }
            xy++;
        }
    }
    //std::cout << count << std::endl;    
}

void print(int* x, int size)
    {
        int i;
        for (i=0;i<size;i++) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }



int main(int argc, char* argv[]){
    //int k = 9;
    bool shared=false;
    int k = atoi(argv[1]);
    int s = atoi(argv[2]);
    int t = atoi(argv[3]);
    if(s==1) {
        shared=true;
    }
    if(t==0){
        //GEMM(k, shared);
        std::cout << "Direct:\tk=" <<k <<"\tshared=" << shared << "\t";
        dConv(k,shared);
    }
    else if(t==1) {
        std::cout << "GEMM:\tk=" <<k <<"\tshared=" << shared << "\t";
        GEMM(k, shared);
    }
    else {
        networkGEMM(shared);
    }
    
    return 0;
}

void GEMM(int ki, bool shared) {
    for(int k=ki;k<=ki;k=k+2) {
        //int k = 5;
        float all_micro = 0;
        for(int f=60;f<120;f++) {
            cv::Mat image;
            std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
            image = cv::imread(filename, CV_32SC1);
            // cv::Mat image;
            // std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic0.ppm";
            // image = cv::imread(filename, CV_32SC1);
            //image.convertTo(image,CV_32SC3);
            //my_cuda_func();
            int Hi = image.rows;
            int Wi = image.cols;
            int Ci = 3;
            //int k = 3;
            int K = k*k*Ci;
            int nK = 1;
            int Hc = (Hi-(k-1))*(Wi-(k-1));
            int stride = 1;
            int Ho = (Hi-(k-1));
            int Wo = (Wi-(k-1));

            // Host input vectors
            int *I_cpu, *K_cpu, *O_cpu;

            // Size, in bytes, of each vector
            //size_t bytes = n*sizeof(int);
            size_t I_size =  (K)*(Hc) * sizeof(int);
            size_t K_size =  (K) * (nK) * sizeof(int);
            size_t O_size =  Hc * nK * sizeof(int);

            // Allocate memory for each vector on host
            I_cpu = (int*)malloc(I_size);

            size_t temp_size = Hi*Wi*Ci*sizeof(int);
            int* temp_im = (int*)malloc(temp_size);
            get_im(temp_im, image);
            
            im2col(image.rows, image.cols, k, I_cpu, temp_im, stride);
            free(temp_im);

            K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K*nK) );
            O_cpu = (int*)malloc(O_size);


            size_t globalSize[2], localSize[2];
            cl_int err;

            // Number of work items in each local work group
            // localSize[0] = (size_t)(Hc*sizeof(int));
            // localSize[1] = (size_t)(nK*sizeof(int));
            localSize[0]=16;
            localSize[1]=16;

            // Number of total work items - localSize must be devisor
            globalSize[0] = ceil(Ho/localSize[0])*localSize[0];
            //std::cout << Hc << "  " << globalSize[0] << "   ";
            globalSize[1] = ceil(Wo/localSize[1])*localSize[1];

            // Bind to platform
            cl_uint dev_cnt = 0;
            clGetPlatformIDs(0, 0, &dev_cnt);
            cl_platform_id platform_ids[2];
            clGetPlatformIDs(dev_cnt, platform_ids, NULL);
            int gpu = 1;
            cl_device_id device_id;
            
            err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
            if(err != CL_SUCCESS)
            {
                printf("Error: Failed to create a device group!\n");
                exit(EXIT_FAILURE);
            }

            // Get ID for the device
        
            //err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

            // Create a context
            cl_context context;
            context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            if (!context)
            {
                printf("Error: Failed to create a compute context!\n");
                exit(EXIT_FAILURE);
            }
            // Create a command queue
            cl_command_queue queue;
            queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
            if (!queue)
            {
                printf("Error: Failed to create a command commands!\n");
                exit(EXIT_FAILURE);
            }

            // Create the compute program from the source buffer
            cl_program program;
            char *kernelSource;
            int lFileSize;
            if(shared==true) {
                lFileSize = LoadOpenCLKernel("matrixMulCol_shared.cl", &kernelSource);
            }
            else {
                lFileSize = LoadOpenCLKernel("matrixMulCol.cl", &kernelSource);
            }
            if( lFileSize < 0 ) 
            {
                perror("File read failed");
                exit(EXIT_FAILURE);
            }
            program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
            if (!program)
            {
                printf("Error: Failed to create compute program!\n");
                exit(EXIT_FAILURE);
            }

            // Build the program executable
            err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                size_t len;
                char buffer[2048];
                printf("Error: Failed to build program executable!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n", buffer);
                exit(EXIT_FAILURE);
            }

            // Create the compute kernel in the program we wish to run
            cl_kernel kernel;
            if(shared==true) {
                kernel = clCreateKernel(program, "matrixMulCol_shared", &err);
            }
            else {
                kernel = clCreateKernel(program, "matrixMulCol", &err);
            }
            if (!kernel || err != CL_SUCCESS)
            {
                printf("Error: Failed to create compute kernel: %d\n", err);
                exit(EXIT_FAILURE);
            }


            //device buffers
            cl_mem I_gpu;
            cl_mem K_gpu;
            cl_mem O_gpu;

            // Create the input and output arrays in device memory for our calculation
            I_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, I_size, NULL, NULL);
            K_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, K_size, NULL, NULL);
            O_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, O_size, NULL, NULL);

            // Write our data set into the input array in device memory
            err = clEnqueueWriteBuffer(queue, I_gpu, CL_TRUE, 0, I_size, I_cpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue, K_gpu, CL_TRUE, 0, K_size, K_cpu, 0, NULL, NULL);

            // Set the arguments to our compute kernel
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &I_gpu);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &K_gpu);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &O_gpu);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &Hc);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &nK);
            err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
            if(shared==true) {
                err |= clSetKernelArg(kernel, 6, localSize[0]*K*sizeof(int), NULL);
            }

            // Execute the kernel over the entire range of the data set
            cl_event myevent;
            cl_ulong start;
            cl_ulong end;
            cl_float kernelExecTimeNs = 0;
            err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &myevent);

            // Wait for the command queue to get serviced before reading back results
            clWaitForEvents(1, &myevent);
            clFinish(queue);

            // Read the results from the device
            clEnqueueReadBuffer(queue, O_gpu, CL_TRUE, 0, O_size, O_cpu, 0, NULL, NULL );
            clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
            clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(end), &end, NULL);
            kernelExecTimeNs = (end - start);
            float microseconds = float(kernelExecTimeNs)/1000;
            //std::cout << "f: " << f << "\tk: " << k << "\tGPU time: " << microseconds << "us" << std::endl;
            all_micro += microseconds;


            //printf("\n\nfinal result: %d\n", sum/n);
            //printf("\n\nfinal result CPU: %d\n", sumCPU/n);
            //std::cout << "O: ";
            //print(O_cpu, (Hc*nK));
            // release OpenCL resources
            clReleaseMemObject(I_gpu);
            clReleaseMemObject(K_gpu);
            clReleaseMemObject(O_gpu);
            clReleaseProgram(program);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            //release host memory
            free(I_cpu);
            free(K_cpu);
            free(O_cpu);
        }
        std::cout << "GPU time: " << (all_micro/60.0) << "us" << std::endl;
    }
}

void networkGEMM(bool shared) {
    for(int k=7;k<=7;k=k+2) {
        float all_micro = 0;
        for(int f=60;f<120;f++) {
            cv::Mat image;
            std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
            image = cv::imread(filename, CV_32SC1);
            // cv::Mat image;
            // std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic0.ppm";
            // image = cv::imread(filename, CV_32SC1);
            //image.convertTo(image,CV_32SC3);
            //my_cuda_func();
            int Hi = image.rows;
            int Wi = image.cols*3;
            int Ci = 1;
            //int k = 3;
            int K = k*k*Ci;
            int nK = 96;
            int Hc = (Hi-(k-1))*(Wi-(k-1));
            int stride = 2;

            // Host input vectors
            int *I_cpu, *K_cpu, *O_cpu;

            // Size, in bytes, of each vector
            //size_t bytes = n*sizeof(int);
            size_t I_size =  (K)*(Hc) * sizeof(int);
            size_t K_size =  (K) * (nK) * sizeof(int);
            size_t O_size =  Hc * nK * sizeof(int);

            // Allocate memory for each vector on host
            I_cpu = (int*)malloc(I_size);

            size_t temp_size = Hi*Wi*Ci*sizeof(int);
            int* temp_im = (int*)malloc(temp_size);
            get_im(temp_im, image);
            
            im2col(image.rows, image.cols, k, I_cpu, temp_im, stride);
            free(temp_im);

            K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K*nK) );
            O_cpu = (int*)malloc(O_size);


            size_t globalSize[2], localSize[2];
            cl_int err;


            // Number of work items in each local work group
            // localSize[0] = (size_t)(Hc*sizeof(int));
            // localSize[1] = (size_t)(nK*sizeof(int));
            localSize[0] = 16;
            localSize[1] = 1;

            // Number of total work items - localSize must be devisor
            globalSize[0] = ceil(Hc/localSize[0])*localSize[0];
            //std::cout << Hc << "  " << globalSize[0] << "   ";
            globalSize[1] = ceil(nK/localSize[1])*localSize[1];

            // Bind to platform
            cl_uint dev_cnt = 0;
            clGetPlatformIDs(0, 0, &dev_cnt);
            cl_platform_id platform_ids[2];
            clGetPlatformIDs(dev_cnt, platform_ids, NULL);
            int gpu = 1;
            cl_device_id device_id;
            
            err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
            if(err != CL_SUCCESS)
            {
                printf("Error: Failed to create a device group!\n");
                exit(EXIT_FAILURE);
            }

            // Get ID for the device
        
            //err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

            // Create a context
            cl_context context;
            context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            if (!context)
            {
                printf("Error: Failed to create a compute context!\n");
                exit(EXIT_FAILURE);
            }
            // Create a command queue
            cl_command_queue queue;
            queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
            if (!queue)
            {
                printf("Error: Failed to create a command commands!\n");
                exit(EXIT_FAILURE);
            }

            // Create the compute program from the source buffer
            cl_program program;
            char *kernelSource;
            int lFileSize;
            if(shared==true) {
                lFileSize = LoadOpenCLKernel("matrixMulCol_shared.cl", &kernelSource);
            }
            else {
                lFileSize = LoadOpenCLKernel("matrixMulCol.cl", &kernelSource);
            }
            if( lFileSize < 0 ) 
            {
                perror("File read failed");
                exit(EXIT_FAILURE);
            }
            program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
            if (!program)
            {
                printf("Error: Failed to create compute program!\n");
                exit(EXIT_FAILURE);
            }

            // Build the program executable
            err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                size_t len;
                char buffer[2048];
                printf("Error: Failed to build program executable!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n", buffer);
                exit(EXIT_FAILURE);
            }

            // Create the compute kernel in the program we wish to run
            cl_kernel kernel;
            if(shared==true) {
                kernel = clCreateKernel(program, "matrixMulCol_shared", &err);
            }
            else {
                kernel = clCreateKernel(program, "matrixMulCol", &err);
            }
            if (!kernel || err != CL_SUCCESS)
            {
                printf("Error: Failed to create compute kernel: %d\n", err);
                exit(EXIT_FAILURE);
            }


            //device buffers
            cl_mem I_gpu;
            cl_mem K_gpu;
            cl_mem O_gpu;

            // Create the input and output arrays in device memory for our calculation
            I_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, I_size, NULL, NULL);
            K_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, K_size, NULL, NULL);
            O_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, O_size, NULL, NULL);

            // Write our data set into the input array in device memory
            err = clEnqueueWriteBuffer(queue, I_gpu, CL_TRUE, 0, I_size, I_cpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue, K_gpu, CL_TRUE, 0, K_size, K_cpu, 0, NULL, NULL);

            // Set the arguments to our compute kernel
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &I_gpu);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &K_gpu);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &O_gpu);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &Hc);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &nK);
            err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
            if(shared==true) {
                err |= clSetKernelArg(kernel, 6, localSize[0]*K*sizeof(int), NULL);
            }

            // Execute the kernel over the entire range of the data set
            cl_event myevent;
            cl_ulong start;
            cl_ulong end;
            cl_float kernelExecTimeNs = 0;
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, &myevent);

            // Wait for the command queue to get serviced before reading back results
            clWaitForEvents(1, &myevent);
            clFinish(queue);

            // Read the results from the device
            clEnqueueReadBuffer(queue, O_gpu, CL_TRUE, 0, O_size, O_cpu, 0, NULL, NULL );
            clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
            clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(end), &end, NULL);
            kernelExecTimeNs = (end - start);
            float microseconds = float(kernelExecTimeNs)/1000;
            //std::cout << "f: " << f << "\tk: " << k << "\tGPU time: " << microseconds << "us" << std::endl;
            all_micro += microseconds;


            //printf("\n\nfinal result: %d\n", sum/n);
            //printf("\n\nfinal result CPU: %d\n", sumCPU/n);
            //std::cout << "O: ";
            //print(O_cpu, (Hc*nK));
            // release OpenCL resources
            clReleaseMemObject(I_gpu);
            clReleaseMemObject(K_gpu);
            clReleaseMemObject(O_gpu);
            clReleaseProgram(program);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            //release host memory
            free(I_cpu);
            free(K_cpu);
            free(O_cpu);
        }
        std::cout << "GPU time: " << (all_micro/60.0) << "us" << std::endl;
    }

}

void dConv(int ki, bool shared) {
    for(int k=ki;k<=ki;k=k+2) {
            //int k = 5;
        float all_micro = 0;
        for(int f=60;f<120;f++) {
            cv::Mat image;
            std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
            image = cv::imread(filename, CV_32SC1);
            // cv::Mat image;
            // std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic0.ppm";
            // image = cv::imread(filename, CV_32SC1);
            //image.convertTo(image,CV_32SC3);
            //my_cuda_func();
            int Hi = image.rows;
            int Wi = image.cols;
            int Ci = 3;
            //int k = 3;
            int K = k*k*Ci;
            int nK = 1;
            int Hc = (Hi-(k-1))*(Wi-(k-1));
            int stride = 1;
            int Ho = (Hi-(k-1));
            int Wo = (Wi-(k-1));

            // Host input vectors
            int *I_cpu, *K_cpu, *O_cpu;

            // Size, in bytes, of each vector
            //size_t bytes = n*sizeof(int);
            size_t I_size =  (K)*(Hc) * sizeof(int);
            size_t K_size =  (K) * (nK) * sizeof(int);
            size_t O_size =  Hc * nK * sizeof(int);

            // Allocate memory for each vector on host
            I_cpu = (int*)malloc(I_size);

            size_t temp_size = Hi*Wi*Ci*sizeof(int);
            int* temp_im = (int*)malloc(temp_size);
            get_im(temp_im, image);
            
            im2col(image.rows, image.cols, k, I_cpu, temp_im, stride);
            free(temp_im);

            K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K*nK) );
            O_cpu = (int*)malloc(O_size);


            size_t globalSize[2], localSize[2];
            cl_int err;

            // Number of work items in each local work group
            // localSize[0] = (size_t)(Hc*sizeof(int));
            // localSize[1] = (size_t)(nK*sizeof(int));
            localSize[0]=16;
            localSize[1]=1;

            // Number of total work items - localSize must be devisor
            globalSize[0] = ceil(Hc/localSize[0])*localSize[0];
            //std::cout << Hc << "  " << globalSize[0] << "   ";
            globalSize[1] = ceil(nK/localSize[1])*localSize[1];

            // Bind to platform
            cl_uint dev_cnt = 0;
            clGetPlatformIDs(0, 0, &dev_cnt);
            cl_platform_id platform_ids[2];
            clGetPlatformIDs(dev_cnt, platform_ids, NULL);
            int gpu = 1;
            cl_device_id device_id;
            
            err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
            if(err != CL_SUCCESS)
            {
                printf("Error: Failed to create a device group!\n");
                exit(EXIT_FAILURE);
            }

            // Get ID for the device
        
            //err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

            // Create a context
            cl_context context;
            context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            if (!context)
            {
                printf("Error: Failed to create a compute context!\n");
                exit(EXIT_FAILURE);
            }
            // Create a command queue
            cl_command_queue queue;
            queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
            if (!queue)
            {
                printf("Error: Failed to create a command commands!\n");
                exit(EXIT_FAILURE);
            }

            // Create the compute program from the source buffer
            cl_program program;
            char *kernelSource;
            int lFileSize;
            if(shared==true) {
                lFileSize = LoadOpenCLKernel("directConv_shared.cl", &kernelSource);
            }
            else {
                lFileSize = LoadOpenCLKernel("directConv.cl", &kernelSource);
            }
            if( lFileSize < 0 ) 
            {
                perror("File read failed");
                exit(EXIT_FAILURE);
            }
            program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
            if (!program)
            {
                printf("Error: Failed to create compute program!\n");
                exit(EXIT_FAILURE);
            }

            // Build the program executable
            err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                size_t len;
                char buffer[2048];
                printf("Error: Failed to build program executable!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n", buffer);
                exit(EXIT_FAILURE);
            }

            // Create the compute kernel in the program we wish to run
            cl_kernel kernel;
            if(shared==true) {
                kernel = clCreateKernel(program, "directConv_shared", &err);
            }
            else {
                kernel = clCreateKernel(program, "directConv", &err);
            }
            if (!kernel || err != CL_SUCCESS)
            {
                printf("Error: Failed to create compute kernel: %d\n", err);
                exit(EXIT_FAILURE);
            }


            //device buffers
            cl_mem I_gpu;
            cl_mem K_gpu;
            cl_mem O_gpu;

            // Create the input and output arrays in device memory for our calculation
            I_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, I_size, NULL, NULL);
            K_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, K_size, NULL, NULL);
            O_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, O_size, NULL, NULL);

            // Write our data set into the input array in device memory
            err = clEnqueueWriteBuffer(queue, I_gpu, CL_TRUE, 0, I_size, I_cpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue, K_gpu, CL_TRUE, 0, K_size, K_cpu, 0, NULL, NULL);

            // Set the arguments to our compute kernel
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &I_gpu);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &K_gpu);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &O_gpu);
            //int Ho, int Wo, int k, int nK)
            err |= clSetKernelArg(kernel, 3, sizeof(int), &Ho);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &Wo);
            err |= clSetKernelArg(kernel, 5, sizeof(int), &k);
            err |= clSetKernelArg(kernel, 6, sizeof(int), &nK);
            if(shared==true) {
                err |= clSetKernelArg(kernel, 7, localSize[0]*K*sizeof(int), NULL);
            }

            // Execute the kernel over the entire range of the data set
            cl_event myevent;
            cl_ulong start;
            cl_ulong end;
            cl_float kernelExecTimeNs = 0;
            err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &myevent);

            // Wait for the command queue to get serviced before reading back results
            clWaitForEvents(1, &myevent);
            clFinish(queue);

            // Read the results from the device
            clEnqueueReadBuffer(queue, O_gpu, CL_TRUE, 0, O_size, O_cpu, 0, NULL, NULL );
            clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
            clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(end), &end, NULL);
            kernelExecTimeNs = (end - start);
            float microseconds = float(kernelExecTimeNs)/1000;
            //std::cout << "f: " << f << "\tk: " << k << "\tGPU time: " << microseconds << "us" << std::endl;
            all_micro += microseconds;


            //printf("\n\nfinal result: %d\n", sum/n);
            //printf("\n\nfinal result CPU: %d\n", sumCPU/n);
            //std::cout << "O: ";
            //print(O_cpu, (Hc*nK));
            // release OpenCL resources
            clReleaseMemObject(I_gpu);
            clReleaseMemObject(K_gpu);
            clReleaseMemObject(O_gpu);
            clReleaseProgram(program);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            //release host memory
            free(I_cpu);
            free(K_cpu);
            free(O_cpu);
        }
        std::cout << "GPU time: " << (all_micro/60.0) << "us" << std::endl;
    }
}

int LoadOpenCLKernel(char const* path, char **buf)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (int)fsz;
}



    // unsigned int n = 100000;
    // // Length of vectors

    // // Host input vectors
    // int* h_a;
    // int* h_b;
    // int* h_c;

    // //device buffers
    // cl_mem d_a;
    // cl_mem d_b;
    // cl_mem d_c;

    // // Size, in bytes, of each vector
    // size_t bytes = n*sizeof(int);

    // // Allocate memory for each vector on host
    // h_a = (int*)malloc(bytes);
    // h_b = (int*)malloc(bytes);
    // h_c = (int*)malloc(bytes);

    // // Initialize vectors on host
    // int i;
    // for( i = 0; i < n; i++ ) {
    //     h_a[i] = 1;
    //     h_b[i] = 1;
    // }

    // size_t globalSize, localSize;
    // cl_int err;

    // // Number of work items in each local work group
    // localSize = 64;

    // // Number of total work items - localSize must be devisor
    // globalSize = ceil(n/(float)localSize)*localSize;

    // // Bind to platform
    // cl_uint dev_cnt = 0;
    // clGetPlatformIDs(0, 0, &dev_cnt);
    // cl_platform_id platform_ids[100];
    // clGetPlatformIDs(dev_cnt, platform_ids, NULL);
    // int gpu = 1;
    // cl_device_id device_id;
    
    // err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    // if(err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create a device group!\n");
    //     exit(EXIT_FAILURE);
    // }

    // // Get ID for the device
   
    // //err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // // Create a context
    // cl_context context;
    // context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    // if (!context)
    // {
    //     printf("Error: Failed to create a compute context!\n");
    //     exit(EXIT_FAILURE);
    // }
    // // Create a command queue
    // cl_command_queue queue;
    // queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    // if (!queue)
    // {
    //     printf("Error: Failed to create a command commands!\n");
    //     exit(EXIT_FAILURE);
    // }

    // // Create the compute program from the source buffer
    // cl_program program;
    // char *kernelSource;
    // int lFileSize = LoadOpenCLKernel("vecadd.cl", &kernelSource);
    // if( lFileSize < 0 ) 
    // {
    //     perror("File read failed");
    //     exit(EXIT_FAILURE);
    // }
    // program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
    // if (!program)
    // {
    //     printf("Error: Failed to create compute program!\n");
    //     exit(EXIT_FAILURE);
    // }

    // // Build the program executable
    // err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     size_t len;
    //     char buffer[2048];
    //     printf("Error: Failed to build program executable!\n");
    //     clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    //     printf("%s\n", buffer);
    //     exit(EXIT_FAILURE);
    // }

    // // Create the compute kernel in the program we wish to run
    // cl_kernel kernel;
    // printf("%d\n", err);
    // kernel = clCreateKernel(program, "vecadd", &err);
    // if (!kernel || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel: %d\n", err);
    //     exit(EXIT_FAILURE);
    // }

    // // Create the input and output arrays in device memory for our calculation
    // d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    // d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    // d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // // Write our data set into the input array in device memory
    // err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    // err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);

    // // Set the arguments to our compute kernel
    // err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    // err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    // err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    // err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // // Execute the kernel over the entire range of the data set
    // err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // // Wait for the command queue to get serviced before reading back results
    // clFinish(queue);

    // // Read the results from the device
    // clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL );


    // //Sum up vector c and print result divided by n, this should equal 1 within error
    // int sum = 0;
    // for(i=0; i<n; i++) {
    //     //printf("%f ", h_c[i]);
    //     sum += h_c[i];
    // }

    // int sumCPU = 0.0;
    // for(i=0; i<n; i++) {
    //     //printf("%f ", h_c[i]);
    //     sumCPU += h_a[i] + h_b[i];
    // }
    // printf("\n\nfinal result: %d\n", sum/n);
    // printf("\n\nfinal result CPU: %d\n", sumCPU/n);

    // // release OpenCL resources
    // clReleaseMemObject(d_a);
    // clReleaseMemObject(d_b);
    // clReleaseMemObject(d_c);
    // clReleaseProgram(program);
    // clReleaseKernel(kernel);
    // clReleaseCommandQueue(queue);
    // clReleaseContext(context);
    // //release host memory
    // free(h_a);
    // free(h_b);
    // free(h_c);