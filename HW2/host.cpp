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

#include "mycuda.h"

// g++ -c main.cpp
// nvcc -arch=sm_20 -c test.cu
// g++  -o test main.o test.o -L/usr/local/cuda/lib64 -lcudart

//nvcc test2.cu -o test2 `pkg-config --cflags --libs opencv`

void GEMM(bool shared);
void networkGEMM(bool shared);
void dConv(bool shared);
void GEMM2(bool shared);

void random_ints(int* x, int size)
    {
        //srand(time(0)); 
        int i;
        for (i=0;i<size;i++) {
            x[i]=rand()%10;
            //std::cout << x[i] << " ";
        }
    }

void get_im_old(int* x, cv::Mat image)
    {
        //srand(time(0)); 
        int i;
        for (int i=0; i<(image.rows*image.cols*3); i++) {
            x[i] = (int)image.data[i];
            std::cout << (int)image.data[i] << " ";
        }
        int count = 0;
        int W = image.cols;
        int H = image.rows;
        for (int i = 0; i < W; i++){
            for (int j = 0; j < H; j++){
                x[i+j] = (int)image.data[j*3+3*i+2];
                x[H*W+i+j] = (int)image.data[j*3+3*i+1];
                x[H*W*2+i+j] = (int)image.data[j*3+3*i];
                count = count + 3;
            }
        }
        //std::cout << count << "\n";
        // for (int i = 0; i < H*W; i++){
        //     x[i] = (int)image.data[3*i+2];
        //     x[H*W+i] = (int)image.data[3*i];
        //     x[H*W*2+i] = (int)image.data[3*i+1];
        // }

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
    // bool shared = false;
    // GEMM2(shared);
    // shared = true;
    // GEMM2(shared);

    bool shared = false;
    std::cout << "Direct -- No Shared:\n";
    dConv(shared);
    std::cout << "GEMM -- No Shared:\n";
    GEMM(shared);
    std::cout << "Network -- No Shared:\n";
    networkGEMM(shared);


    shared=true;
    std::cout << "GEMM -- With Shared:\n";
    GEMM(shared);
    std::cout << "Network -- With Shared:\n";
    networkGEMM(shared);
    return 0;
}

void GEMM(bool shared) {
    
    for(int k=3;k<=9;k=k+2) {
        //int k = 5;
        float all_micro = 0.0;
        for(int f=0;f<120;f++) {
            cv::Mat image;
            std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
            image = cv::imread(filename, CV_32SC1);
            //image.convertTo(image,CV_32SC3);
            //my_cuda_func();
            int Hi = image.rows;
            int Wi = image.cols*3;
            int Ci = 1;
            
            int K = k*k*Ci;
            int nK = 1;
            int Hc = (Hi-(k-1))*(Wi-(k-1));
            int stride = 1;
            //define A_cpu, B_cpu, C_cpu in the CPU memory
            int *I_cpu, *K_cpu, *O_cpu;


            int I_size =  (K)*(Hc) * sizeof(int);
            int K_size =  (K) * (nK) * sizeof(int);
            int O_size =  Hc * nK * sizeof(int);

            // Setup input values
            //std::cout << image.rows << " " << image.cols << std::endl;
            I_cpu = (int*)malloc(I_size); //random_ints(I_cpu, (K*Hc) );
            
            int temp_size = Hi*Wi*Ci*sizeof(int);
            int* temp_im = (int*)malloc(temp_size);
            get_im(temp_im, image);
            
            im2col(image.rows, image.cols, k, I_cpu, temp_im, stride);
            free(temp_im);
            
            K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K*nK) );
            O_cpu = (int*)malloc(O_size);
            //std::cout << K << " " << Hc << std::endl;

            //std::cout << "I: ";
            //print(I_cpu, (K*Hc));
            //std::cout << "K: ";
            //print(K_cpu, (K*nK));
            //std::cout << "O: ";
            //print(O_cpu, (Hc*nK));
            float microsec = 0.0;
            if(shared==true) {
                microsec = matMul_shared(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Hc, nK, K);
            }
            else {
                microsec = matMul(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Hc, nK, K);
            }
            all_micro += microsec;
            //std::cout << "\n\nO: ";
            //print(O_cpu, (Hc*nK));
            free(I_cpu); free(K_cpu); free(O_cpu);
        }
        std::cout << "k: " << k << "\tGPU time: " << (all_micro/120) << "us" << std::endl;
    }
}

void GEMM2(bool shared) {
    float all_micro = 0.0;
    for(int k=3;k<=9;k=k+2) {
        //int k = 5;
        for(int f=0;f<120;f++) {
            cv::Mat image;
            std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
            image = cv::imread(filename, CV_32SC1);
            //image.convertTo(image,CV_32SC3);
            //my_cuda_func();
            int Hi = 3;
            int Wi = 3*3;
            int Ci = 1;
            
            int K = k*k*Ci;
            int nK = 1;
            int Hc = (Hi-(k-1))*(Wi-(k-1));
            int stride = 1;
            //define A_cpu, B_cpu, C_cpu in the CPU memory
            int *I_cpu, *K_cpu, *O_cpu;


            int I_size =  (K)*(Hc) * sizeof(int);
            int K_size =  (K) * (nK) * sizeof(int);
            int O_size =  Hc * nK * sizeof(int);

            // Setup input values
            //std::cout << image.rows << " " << image.cols << std::endl;
            I_cpu = (int*)malloc(I_size); random_ints(I_cpu, (K*Hc) );
            
            // int temp_size = Hi*Wi*Ci*sizeof(int);
            // int* temp_im = (int*)malloc(temp_size);
            // get_im(temp_im, image);
            
            // im2col(image.rows, image.cols, k, I_cpu, temp_im, stride);
            // free(temp_im);
            
            K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K*nK) );
            O_cpu = (int*)malloc(O_size);
            //std::cout << K << " " << Hc << std::endl;

            std::cout << "I: ";
            print(I_cpu, (K*Hc));
            std::cout << "K: ";
            print(K_cpu, (K*nK));
            //std::cout << "O: ";
            //print(O_cpu, (Hc*nK));
            float microsec = 0.0;
            if(shared==true) {
                microsec = matMul_shared(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Hc, nK, K);
            }
            else {
                microsec = matMul(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Hc, nK, K);
            }
            all_micro += microsec;
            std::cout << "\n\nO: ";
            print(O_cpu, (Hc*nK));
            free(I_cpu); free(K_cpu); free(O_cpu);
        }
        std::cout << "k: " << k << "\tGPU time: " << (all_micro/120) << "us" << std::endl;
    }
}

void dConv(bool shared) {
    for(int k=3;k<=9;k=k+2) {
        //int k = 5;
        float all_micro = 0.0;
        for(int f=0;f<120;f++) {
            cv::Mat image;
            std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
            image = cv::imread(filename, CV_32SC1);
            //image.convertTo(image,CV_32SC3);
            //my_cuda_func();
            int Hi = image.rows;
            int Wi = image.cols*3;
            int Ci = 1;
            
            int K = k*k;
            int Ho = (Hi-(k-1));
            int Wo = (Wi-(k-1));
            int stride = 1;
            //define A_cpu, B_cpu, C_cpu in the CPU memory
            int *I_cpu, *K_cpu, *O_cpu;


            int I_size =  (Hi) * (Wi) * sizeof(int);
            int K_size =  (K) * sizeof(int);
            int O_size =  Ho*Wo * sizeof(int);

            // Setup input values
            //std::cout << image.rows << " " << image.cols << std::endl;
            I_cpu = (int*)malloc(I_size); //random_ints(I_cpu, (Hi*Wi) );
            int i;
            for (int i=0; i<(image.rows*image.cols*3); i++) {
                I_cpu[i] = (int)image.data[i];
                //std::cout << (int)image.data[i] << " ";
            }

            
            K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K) );
            O_cpu = (int*)malloc(O_size);
            //std::cout << K << " " << Hc << std::endl;

            //std::cout << "I: ";
            //print(I_cpu, (K*Hc));
            //std::cout << "K: ";
            //print(K_cpu, (K*nK));
            //std::cout << "O: ";
            //print(O_cpu, (Ho*Wo));
            float microsec = 0.0;
            if(shared==true) {
                microsec = dirConv(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Ho, Wo, k);
            }
            else {
                microsec = dirConv(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Ho, Wo, k);
            }
            all_micro += microsec;
            //std::cout << "\n\nO: ";
            //print(O_cpu, (Ho*Wo));
            free(I_cpu); free(K_cpu); free(O_cpu);
        }
        std::cout << "k: " << k << "\tGPU time: " << (all_micro/120) << "us" << std::endl;
    }
}



void networkGEMM(bool shared) {
    float all_micro = 0.0;
    int k = 7;
    for(int f=0;f<120;f++) {
        cv::Mat image;
        std::string filename = "/home/matias/Documents/spring2020/Heterogeneous-Computing/HW2/Sample-Video/viptraffic"+std::to_string(f)+".ppm";
        image = cv::imread(filename, CV_32SC1);
        //image.convertTo(image,CV_32SC3);
        //my_cuda_func();
        int Hi = image.rows;
        int Wi = image.cols;
        int Ci = 3;
        
        int K = k*k*Ci;
        int nK = 96;
        int stride = 2;
        int Hc = ((Hi-(k-1))*(Wi-(k-1)) )/stride;
        
        //define A_cpu, B_cpu, C_cpu in the CPU memory
        int *I_cpu, *K_cpu, *O_cpu;


        int I_size =  (K)*(Hc) * sizeof(int);
        int K_size =  (K) * (nK) * sizeof(int);
        int O_size =  Hc * nK * sizeof(int);

        // Setup input values
        //std::cout << image.rows << " " << image.cols << std::endl;
        I_cpu = (int*)malloc(I_size); //random_ints(I_cpu, (K*Hc) );
        
        int temp_size = Hi*Wi*Ci*sizeof(int);
        int* temp_im = (int*)malloc(temp_size);
        get_im(temp_im, image);
        
        im2col(image.rows, image.cols, k, I_cpu, temp_im, stride);
        free(temp_im);
        
        K_cpu = (int*)malloc(K_size); random_ints(K_cpu, (K*nK) );
        O_cpu = (int*)malloc(O_size);
        //std::cout << K << " " << Hc << std::endl;

        //std::cout << "I: ";
        //print(I_cpu, (K*Hc));
        //std::cout << "K: ";
        //print(K_cpu, (K*nK));
        //std::cout << "O: ";
        //print(O_cpu, (Hc*nK));

        float microsec = 0.0;
        if(shared==true) {
            microsec = matMul_shared(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Hc, nK, K);
        }
        else {
            microsec = matMul(I_cpu, K_cpu, O_cpu, I_size, K_size, O_size, Hc, nK, K);
        }
        all_micro += microsec;
        //std::cout << "\n\nO: ";
        //print(O_cpu, (Hc*nK));
        free(I_cpu); free(K_cpu); free(O_cpu);
    }
    std::cout << "k: " << k << "\tGPU time: " << (all_micro/120) << "us" << std::endl;
}