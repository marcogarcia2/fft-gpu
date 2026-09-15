#include <iostream>
#include <fstream>

#include "kernels.cuh"
#include "util.cuh"

#define NX      2048 
#define RPT     30

int main() {

    // CUDA Events to measure time
    cudaEvent_t H2D_start, H2D_stop, D2H_start, D2H_stop, kernel_start, kernel_stop;
    float H2D_ms, D2H_ms, kernel_ms;
    cudaEventCreate(&H2D_start);
    cudaEventCreate(&H2D_stop);
    cudaEventCreate(&D2H_start);
    cudaEventCreate(&D2H_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // Results vectors
    float* kernel_times = (float*)malloc(sizeof(float) * RPT);
    float* H2D_times = (float*)malloc(sizeof(float) * RPT);
    float* D2H_times = (float*)malloc(sizeof(float) * RPT);

    // Allocating vectors
    size_t size = NX * sizeof(float2);
    float2* h_X = (float2*)malloc(size);    // host vector
    float2* d_X;                            // device vector
    cudaMalloc(&d_X, size);

    // Threads and Blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (NX + threadsPerBlock - 1) / threadsPerBlock;

    // Start the kernel loop:

    // Check if all kernels are correct with cuFFT before proceeding
    if (!checkKernel(NX)){
        fprintf(stderr, "Error: MyKernel is not correct.\n");
        return 1;
    }
    printf("Kernel is correct!\n");

    // Warmup
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
    FFT_Naive(d_X, NX, blocksPerGrid, threadsPerBlock);
    cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

    for (int rep = 0; rep < RPT; rep++) {

        setVector(h_X, NX);

        // H2D transfer
        cudaEventRecord(H2D_start);
        cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
        cudaEventRecord(H2D_stop);
        cudaEventSynchronize(H2D_stop);
        cudaEventElapsedTime(&H2D_ms, H2D_start, H2D_stop);

        // Kernel
        cudaEventRecord(kernel_start);
        FFT_Naive(d_X, NX, blocksPerGrid, threadsPerBlock);
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

        // D2H transfer
        cudaEventRecord(D2H_start);
        cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(D2H_stop);
        cudaEventSynchronize(D2H_stop);
        cudaEventElapsedTime(&D2H_ms, D2H_start, D2H_stop);

        kernel_times[rep] = kernel_ms;
        H2D_times[rep] = H2D_ms;
        D2H_times[rep] = D2H_ms;
    }

    // Média, Mediana, stddev...
    float mean = 0.0;
    for (int i = 0; i < RPT; i++) {
        mean += kernel_times[i];
    }
    mean /= RPT;
    printf("Tempo médio kernel: %fms\n", mean);
    // write CSV
    // end kernel loop

    // Destroy results vectors
    free(kernel_times);
    free(D2H_times);
    free(H2D_times);

    // Frees memory
    free(h_X);
    cudaFree(d_X);
    
    // Destroy CUDA Events
    cudaEventDestroy(H2D_start);
    cudaEventDestroy(H2D_stop);
    cudaEventDestroy(D2H_start);
    cudaEventDestroy(D2H_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    return 0;
}