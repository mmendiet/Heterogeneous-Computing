void my_cuda_func();
float matMul(int *I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Hc, int Wc, int K);
float matMul_shared(int* I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Hc, int nK, int K);
float dirConv(int* I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Ho, int Wo, int k, int nK);
float dirConv_shared(int* I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Ho, int Wo, int k, int nK);