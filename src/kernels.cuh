#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cufft.h>
#include <type_traits>

#include "util.cuh"

#define PI      3.141592653589793

// --- KERNELS --- //

// CUFFT KERNEL: BASELINE
template <typename T, typename T2>
__host__ void cuFFT(T2* d_X, int N, int blocksPerGrid, int threadsPerBlock) {
    // blocks and threads aren't used, they're here just to match the template
    // CUFFT_C2C -> float
    // CUFFT_Z2Z -> double
    
    cufftHandle plan;
    cufftResult planResult, execResult;
    constexpr bool is_float = std::is_same_v<T, float>;

    // Plan creation
    if constexpr (is_float) {
        planResult = cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    }
    else {
        planResult = cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    }

    if (planResult != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        cufftDestroy(plan);
        return;
    }

    // Execution
    if constexpr (is_float) {
        execResult = cufftExecC2C(plan, d_X, d_X, CUFFT_FORWARD);
    }
    else {
        execResult = cufftExecZ2Z(plan, d_X, d_X, CUFFT_FORWARD);
    }

    if (execResult != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
        cufftDestroy(plan);
        return;
    }

    // Sync
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        cufftDestroy(plan);
        return;
    }

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

template <typename T, typename T2>
__global__ void reverseBitsLocallyKernel(T2* d_X, int N, int log2n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    int j = reverseBits(i, log2n);
    if (i < j) {
        T2 temp = d_X[i];
        d_X[i] = d_X[j];
        d_X[j] = temp;
    }
}

template <typename T, typename T2>
__global__ void FFT_Naive_butterflyKernel(T2* d_X, int N, int len) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int block = b / (len/2);
    int k = b % (len/2);
    int i = block * len;

    T arg = (T) -2.0f * PI * k / len;
    T2 W = make_vec2(cos(arg), sin(arg));

    T2 U = d_X[i+k];
    T2 V = cmul(d_X[i+k+(len/2)], W);

    d_X[i+k]         = cadd(U, V);
    d_X[i+k+(len/2)] = csub(U, V);
}

template <typename T, typename T2>
__host__ void FFT_Naive(T2 *d_X, int N, int blocksPerGrid, int threadsPerBlock) {
    int log2n = 0;
    while((1u << log2n) < N) log2n++;

    reverseBitsLocallyKernel<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, log2n);

    for (int len = 2; len <= N; len <<= 1) {
        FFT_Naive_butterflyKernel<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, len);
    }
}

#endif