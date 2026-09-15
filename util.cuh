#ifndef UTIL_CUH
#define UTIL_CUH

#include <random>

#include "kernels.cuh"

#define THRESHOLD 1e-4f

void FFT_Naive(float2 *d_X, int N, int blocksPerGrid, int threadsPerBlock);
void cuFFT(float2* h_X, int N);

// --- COMPLEX ARITHMETHIC --- //

// float (device)
__device__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// double (device)
__device__ double2 cmul(double2 a, double2 b) {
    return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ double2 cadd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ double2 csub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

// float (host)
float2 cmul_h(float2 a, float2 b) {
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
float2 cadd_h(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
float2 csub_h(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
float cabs_h(float2 a) {
    return std::sqrt(a.x*a.x + a.y*a.y);
}

// double (host)
double2 cmul_h(double2 a, double2 b) {
    return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
double2 cadd_h(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
double2 csub_h(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
double cabs_h(double2 a) {
    return std::sqrt(a.x*a.x + a.y*a.y);
}


// --- HELPER FUNCTIONS --- //

// Function that resets vector 
void setVector(float2* X, int N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for (int i = 0 ; i < N; i++) {
        X[i].x = dist(rng);
        X[i].y = dist(rng);
    }
}


// Function that checks if a kernel is correct against cuFFT results
bool checkKernel(int N) {

    bool ret = true;
    
    size_t size = N * sizeof(float2);
    float2* cufft_X = (float2*)malloc(size);
    float2* h_myKernel_X = (float2*)malloc(size);
    float2* d_myKernel_X;
    cudaMalloc(&d_myKernel_X, size);
    
    // Fill with values
    setVector(cufft_X, N);
    setVector(h_myKernel_X, N);
    cudaMemcpy(d_myKernel_X, h_myKernel_X, size, cudaMemcpyHostToDevice);
    
    // Invokes kernels
    cuFFT(cufft_X, N);
    FFT_Naive(d_myKernel_X, N, 256, 256);
    cudaMemcpy(h_myKernel_X, d_myKernel_X, size, cudaMemcpyDeviceToHost);
    
    // Assuming cufft_X is the one with correct values
    float max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        max_diff = std::max(max_diff, cabs_h(csub_h(cufft_X[i], h_myKernel_X[i])));
        if (max_diff > THRESHOLD){
            printf("(%f,%f) and (%f,%f) at [%d]", cufft_X[i].x, cufft_X[i].y, h_myKernel_X[i].x, h_myKernel_X[i].y, i);
            ret = false;
            break;
        }
    }

    free(cufft_X);
    free(h_myKernel_X);
    cudaFree(d_myKernel_X);

    return ret;
}


#endif