#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cufft.h>

#include "util.cuh"

#define PI      3.141592653589793f

// --- KERNELS --- //

// CUFFT KERNEL: BASELINE
__host__ void cuFFT(float2* h_X, int N) {

    cufftHandle plan;
    cufftComplex *d_X;
    cudaMalloc((void**)&d_X, sizeof(cufftComplex)*N);
    cudaMemcpy(d_X, h_X, sizeof(cufftComplex)*N, cudaMemcpyHostToDevice);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }

    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        return;
    }

    if (cufftExecC2C(plan, d_X, d_X, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
        return;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return;
    }

    size_t size = N * sizeof(float2);
    cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cufftDestroy(plan);

    return;
}

__device__ int reverseBits(int i, int log2n) {
    int rev = 0;
    for (int b = 0; b < log2n; b++) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
    }
    return rev;
}

__global__ void reverseBitsLocallyKernel(float2* d_X, int N, int log2n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    int j = reverseBits(i, log2n);
    if (i < j) {
        float2 temp = d_X[i];
        d_X[i] = d_X[j];
        d_X[j] = temp;
    }
}

__global__ void FFT_Naive_butterflyKernel(float2* d_X, int N, int len) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int block = b / (len/2);
    int k = b % (len/2);
    int i = block * len;

    float arg = (float) -2.0f * PI * k / len;
    float2 W = make_float2(cosf(arg), sinf(arg));

    float2 U = d_X[i+k];
    float2 V = cmul(d_X[i+k+(len/2)], W);

    d_X[i+k]         = cadd(U, V);
    d_X[i+k+(len/2)] = csub(U, V);
}

__host__ void FFT_Naive(float2 *d_X, int N, int blocksPerGrid, int threadsPerBlock) {
    int log2n = 0;
    while((1u << log2n) < N) log2n++;

    reverseBitsLocallyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, N, log2n);

    for (int len = 2; len <= N; len <<= 1) {
        FFT_Naive_butterflyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, N, len);
    }
}

#endif