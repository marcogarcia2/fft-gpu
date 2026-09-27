#ifndef PROFILER_CUH
#define PROFILER_CUH

#include "util.cuh"
#include "kernels.cuh"

template <typename T, typename T2>
void profiler(kernelFunction<T, T2> kernel) {

    // NX
    int NX_list[] = { 1<<10, 1<<18, 1<<24 };
    int NX_L_SIZE = sizeof(NX_list)/sizeof(NX_list[0]);
    
    for (int nx_idx = 0; nx_idx < NX_L_SIZE; nx_idx++) {
        int NX = NX_list[nx_idx];
        
        // Threads and Blocks
        int threadsPerBlock = 256;
        int blocksPerGrid = (NX + threadsPerBlock - 1) / threadsPerBlock;
        
        // Allocating vectors
        size_t size = NX * sizeof(T2);
        T2* h_X = (T2*)malloc(size);        // host vector
        T2* d_X;                            // device vector
        cudaMalloc(&d_X, size);

        // Fill vector with values
        setVector<T, T2>(h_X, NX);
        
        // Running Kernel
        cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
        kernel(d_X, NX, blocksPerGrid, threadsPerBlock);
        cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

        // Frees memory
        free(h_X);
        cudaFree(d_X);
    }    
    
    return;
}

#endif