#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cufft.h>
#include <type_traits>

#include "util.cuh"

#define PI      3.141592653589793

// --- KERNELS --- //

// CUFFT KERNEL: BASELINE
template <typename T, typename T2>
__host__ void cuFFT(T2* d_X, int N, FFTContext& ctx) {
    constexpr bool is_float = std::is_same_v<T, float>;

    // Execution
    cufftResult execResult;
    if constexpr (is_float) // FP32
        execResult = cufftExecC2C(ctx.plan, d_X, d_X, CUFFT_FORWARD);

    else // FP64
        execResult = cufftExecZ2Z(ctx.plan, d_X, d_X, CUFFT_FORWARD);

    if (execResult != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Forward execution failed\n");
        return;
    }

    // Sync
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

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
__global__ void reverseBitsInPlace(T2* d_X, int N, int log2n) {
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
__host__ void FFT_Naive(T2 *d_X, int N, FFTContext& ctx) {
    int log2n = 0;
    while ((1u << log2n) < N) log2n++;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverseBitsInPlace<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, log2n);

    blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    for (int len = 2; len <= N; len <<= 1) {
        FFT_Naive_butterflyKernel<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, len);
    }
}

template <typename T, typename T2>
__global__ void reverseBitsOutOfPlace(T2 *d_X, T2 *d_X_temp, int N, int log2n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    int j = reverseBits(i, log2n);
    d_X_temp[j] = d_X[i];
}

template <typename T, typename T2>
__host__ void FFT_Naive_reverseOutOfPlace(T2 *d_X, int N, FFTContext& ctx) {
    int log2n = 0;
    while ((1u << log2n) < N) log2n++;
    size_t size = N * sizeof(T2);

    T2* d_temp = static_cast<T2*>(ctx.workspace);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    reverseBitsOutOfPlace<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, d_temp, N, log2n);
    
    blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;
    for (int len = 2; len <=N; len <<= 1) {
        FFT_Naive_butterflyKernel<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_temp, N, len);
    }

    cudaMemcpy(d_X, ctx.workspace, size, cudaMemcpyDeviceToDevice);
}

template <typename T, typename T2>
__global__ void FFT_Stockham_stage(T2 *d_in, T2 *d_out, int N, int len){

    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int stride = N / len;
    int s = b / stride;
    int subproblem = b % stride;
    
    T arg = (T) -2.0f * PI * s / len;
    T2 W = make_vec2(cos(arg), sin(arg));

    T2 U = d_in[subproblem + stride*(2*s)];
    T2 V = cmul(d_in[subproblem + stride*(2*s + 1)], W);

    d_out[b]       = cadd(U, V);
    d_out[b + N/2] = csub(U, V);
}

template <typename T, typename T2>
__host__ void FFT_Stockham(T2 *d_A, int N, FFTContext& ctx) {

    size_t size = N * sizeof(T2);
    T2* d_B = static_cast<T2*>(ctx.workspace);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    bool in_A = true;
    for (int len = 2; len <= N; len <<= 1) {
        if (in_A) FFT_Stockham_stage<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, len);
        else      FFT_Stockham_stage<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_B, d_A, N, len);
        in_A = !in_A;
    }

    if (!in_A){
        cudaMemcpy(d_A, d_B, size, cudaMemcpyDeviceToDevice);
    } 
}

#endif