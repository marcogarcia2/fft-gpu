#ifndef UTIL_CUH
#define UTIL_CUH

#include <random>

#include "kernels.cuh"

#define THRESHOLD 1e-3f

using kernelFunction = void(*)(float2*, int, int, int);

// Struct used to store some constants used in results
struct constants {
    std::string kname;
    std::string precision;
    int nx;
    int rpt;
    int acc;
    float thresh;
    unsigned int temp_begin;
    unsigned int temp_end;
};

// Functions imported from "kernels.cuh"
void FFT_Naive(float2 *d_X, int N, int blocksPerGrid, int threadsPerBlock);
void cuFFT(float2* h_X, int N, int blocksPerGrid, int threadsPerBlock);

// --- COMPLEX ARITHMETHIC --- //

__host__ __device__ inline float2 make_vec2(float x, float y) {
    return make_float2(x, y);
}
__host__ __device__ inline double2 make_vec2(double x, double y) {
    return make_double2(x, y);
}

template <typename T>
__host__ __device__ inline T cmul(T a, T b) {
    return make_vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
template <typename T>
__host__ __device__ inline T cadd(T a, T b) {
    return make_vec2(a.x + b.x, a.y + b.y);
}
template <typename T>
__host__ __device__ inline T csub(T a, T b) {
    return make_vec2(a.x - b.x, a.y - b.y);
}
template <typename T>
__host__ __device__ inline auto cabs(T a) {
    return std::sqrt(a.x*a.x + a.y*a.y);
}


// --- HARNESS HELPER FUNCTIONS --- //

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
bool checkKernel(kernelFunction myKernel, int N) {

    bool ret = true;

    size_t size = N * sizeof(float2);

    // Baseline vectors
    float2* h_cufft_X = (float2*)malloc(size);
    float2* d_cufft_X;
    cudaMalloc(&d_cufft_X, size);
    
    // MyKernel vectors
    float2* h_myKernel_X = (float2*)malloc(size);
    float2* d_myKernel_X;
    cudaMalloc(&d_myKernel_X, size);

    // Calcs threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Fill both data vectors with values
    setVector(h_cufft_X, N);
    setVector(h_myKernel_X, N);
    cudaMemcpy(d_cufft_X, h_cufft_X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_myKernel_X, h_myKernel_X, size, cudaMemcpyHostToDevice);

    // Invokes baseline kernel
    cuFFT(d_cufft_X, N, blocksPerGrid, threadsPerBlock);
    cudaMemcpy(h_cufft_X, d_cufft_X, size, cudaMemcpyDeviceToHost);
    
    // Invokes myKernel
    myKernel(d_myKernel_X, N, blocksPerGrid, threadsPerBlock);
    cudaMemcpy(h_myKernel_X, d_myKernel_X, size, cudaMemcpyDeviceToHost);

    // Assuming h_cufft_X is the one with correct values
    float max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        float diff = cabs(csub(h_cufft_X[i], h_myKernel_X[i]));
        float ref = std::max(cabs(h_cufft_X[i]), 1e-6f);
        float rel = diff / ref;
        max_diff = std::max(max_diff, rel);
        if (max_diff > THRESHOLD){
            printf("(%f,%f) and (%f,%f) at [%d]\n", h_cufft_X[i].x, h_cufft_X[i].y, h_myKernel_X[i].x, h_myKernel_X[i].y, i);
            ret = false;
            break;
        }
    }

    free(h_cufft_X);
    free(h_myKernel_X);
    cudaFree(d_cufft_X);
    cudaFree(d_myKernel_X);

    return ret;
}

// --- STATISTICS FUNCTIONS --- //

// Quicksort
int partition(float* vector, int left, int right) {
    float pivot = vector[right];
    int i = left-1;

    for (int j = left; j < right; j++) {
        if (vector[j] < pivot) {
            i++;
            float temp = vector[i];
            vector[i] = vector[j];
            vector[j] = temp;
        }
    }
    float temp = vector[i+1];
    vector[i+1] = vector[right];
    vector[right] = temp;

    return i+1;
}

void quicksort(float* vector, int left, int right) {
    if (left < right) {
        int pi = partition(vector, left, right);
        quicksort(vector, left, pi-1);
        quicksort(vector, pi+1, right);
    }
}

// Median, mean, stddev and ic95

double median (float* vector, int N) {
    quicksort(vector, 0, N-1);
    if (N & 1) // if odd
        return (double) vector[N/2];

    // if even
    return ((double)vector[(N/2)-1] + (double)vector[N/2]) / 2.0;
}

double mean(float* vector, int N) {
    double sum = vector[0];
    for (int i = 1; i < N; i++)
        sum += vector[i];
    return sum / N;
}

double stddev(float* vector, int N) {
    if (N <= 1) 
        return 0.0;

    double m = mean(vector, N);
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += (vector[i] - m)*(vector[i] - m);
    }
    sum /= (N - 1);
    return std::sqrt(sum);
}

double ic95(float* vector, int N) {
    double t = 1.96;
    double SEM = stddev(vector, N) / std::sqrt((double)N);
    return t * SEM;
}


// --- CSV FILE --- //

// Create a CSV file with the first line with columns names
std::ofstream createCSV() {
    std::ofstream outf { "Results.csv" };
    outf << "kernel_version,N,precision,RPT,ACC,threshold,";              // harness
    outf << "time_mean_ms,time_median_ms,time_std,time_ci95,";            // kernel
    outf << "time_h2d_ms,time_d2h_ms,";                                   // transfers
    outf << "energy_mean_mj,energy_median_mj,energy_std,energy_ci95,";    // energy
    outf << "temp_begin,temp_end,gflops,perf_per_watt,edp" << std::endl;  // other
    return outf;
}

// Function that writes one line of results 
void writeResultsCSV(std::ofstream& outf, float* kernel_times, float* H2D_times, float* D2H_times, float* energies, const constants& c) {
    //* Harness parameters
    outf << c.kname << "," << c.nx << "," << c.precision << "," << c.rpt << "," << c.acc << "," << c.thresh << ",";
    
    //* Kernel statistics (in milliseconds)
    double kernel_mean = mean(kernel_times, c.rpt);  
    double kernel_median = median(kernel_times, c.rpt);
    double kernel_std = stddev(kernel_times, c.rpt);
    double kernel_ic95 = ic95(kernel_times, c.rpt);
    outf << kernel_mean << "," << kernel_median << "," << kernel_std << "," << kernel_ic95 << ",";

    //* Data transfer statistics (in milliseconds)
    double H2D_mean = mean(H2D_times, c.rpt);
    double D2H_mean = mean(D2H_times, c.rpt);
    outf << H2D_mean << "," << D2H_mean << ",";

    //* Energy statistics (in millijoules)
    double energy_mean = mean(energies, c.rpt);
    double energy_median = median(energies, c.rpt);
    double energy_std = stddev(energies, c.rpt);
    double energy_ic95 = ic95(energies, c.rpt);
    outf << energy_mean << "," << energy_median << "," << energy_std << "," << energy_ic95 << ",";

    //* Other data

    // GFLOPS (giga float-operations per second)
    double flops = 5.0 * c.nx * log2(c.nx);
    double kernel_time_in_seconds = kernel_mean / 1000.0;
    double gflops = flops / kernel_time_in_seconds / 1e9; // 1e9 = giga

    // Performance per watt (GFLOPS/W)
    double energy_in_joules = energy_mean / 1000.0;
    double watts = energy_in_joules / kernel_time_in_seconds;
    double perf_per_watt = gflops / watts;
    
    // Energy-Delay Product (J*s)
    double edp = energy_in_joules * kernel_time_in_seconds;

    outf << c.temp_begin << "," << c.temp_end << "," << gflops << "," << perf_per_watt << ","<< edp << std::endl;
}

#endif