#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdio.h>

/*
    To compile, use: nvcc -Xcompiler /wd4819 test1.cu
*/

void vecAddonHost(double *h_A,double *h_B,double *h_C,int n) {
    for (int i=0; i<n; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

// CUDA kernel
// each thread for each element
__global__
void vecAddKernel(double *A, double *B, double *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n) C[i] = A[i] + B[i];
}

void vecAddonDevice(double *h_A,double *h_B,double *h_C,int n) {
    int size = n * sizeof(double);
    double *d_A, *d_B, *d_C;
    cudaError_t error;

    error=cudaMalloc((void **) &d_A, size);
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    } 

    error=cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    } 

    error=cudaMalloc((void **) &d_B, size);
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    } 

    error=cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    } 

    error=cudaMalloc((void **) &d_C, size);
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    } 

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    error=cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    } 

    // free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}    

int main(int argc, char **argv) {
    int N = 1000;

    //double *h_A = new double[N];
    //double *h_B = new double[N];
    //double *h_C = new double[N];
	std::vector<double> h_A(N);
	std::vector<double> h_B(N);
	std::vector<double> h_C(N);
	
    // initialize on host
    for (int i=0; i<N; i++) {
        h_A[i] = i;
        h_B[i] = i*i;
    } 
    for (int i=0; i<N; i++) h_C[i] = 0.0;

    // perform C=A+B on host
    vecAddonHost( &h_A[0], &h_B[0], &h_C[0], N);

    // output
    std::cout << "host answer (first 5 only):" << std::endl;
    for (int i=0; i<5; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // clean up C
    std::cout << "cleaning up" << std::endl;
    for (int i=0; i<N; i++) h_C[i] = 0.0;

    // run cuda add
    std::cout << "run cuda add" << std::endl;
    vecAddonDevice( &h_A[0],  &h_B[0],  &h_C[0], N);

    // output
    std::cout << "device answer (first 5 only):" << std::endl;
    for (int i=0; i<5; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;


//    delete[] h_A;
//    delete[] h_B;
//    delete[] h_C;
    return 0;
}
