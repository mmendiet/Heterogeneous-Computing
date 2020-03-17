__kernel void vecadd(__global int *A, __global int *B, __global int *C, int n) {
    int idx = get_global_id(0);
    if(idx<n) {
        C[idx] = A[idx] + B[idx];
        }
}