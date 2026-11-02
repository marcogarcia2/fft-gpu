#ifndef UTIL_CUH
#define UTIL_CUH

#include <random>
#include <fstream>
#include <limits>

#include "kernels.cuh"

#define THRESHOLD       1e-3f
#define FP32_STRING     "float"
#define FP64_STRING     "double"

// Pi
template <typename T>
constexpr T PI = static_cast<T>(3.141592653589793238462643383279502884L);

// Some FFT Kernels needs additional buffers, cuFFT needs a plan
struct FFTContext {
    void* workspace = nullptr;
    void* twiddles = nullptr;
    cufftHandle plan = 0;
    bool workspaceCreated = false;
    bool tableCreated = false;
    bool planCreated = false;
};

// Kernel Function signature
template <typename T, typename T2>
using kernelFunction = void(*)(T2*, int, FFTContext&);

// Kernel struct
struct kernelEntry {
    const char* name;
    kernelFunction<float, float2> float_fn;
    kernelFunction<double, double2> double_fn;
    bool needsWorkspace;
    bool needsTable;
    bool needsPlan;
};

// Struct used to store some constants used in results
struct constants {
    std::string kname;
    std::string precision;
    int nx;
    int rpt;
    int acc;
    float thresh;
    float time_setup_ms;
    float ctrl_time;
    unsigned int temp_begin;
    unsigned int temp_end;
    unsigned int sm_mhz_begin;
    unsigned int sm_mhz_end;
};

// Functions imported from "kernels.cuh"
template <typename T, typename T2>
void cuFFT(T2* h_X, int N, FFTContext& ctx);


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

template <typename T2>
bool createWorkspace(void** workspace, int N) {
    size_t size = sizeof(T2) * N;
    cudaError_t err = cudaMalloc(workspace, size);
    return err == cudaSuccess;
}
 
template <typename T, typename T2>
bool createTwiddlesTable(void** table, int N) {

    // Calculates table in host
    T2* h_tw = (T2*) malloc(sizeof(T2) * (N/2));
    for (int k = 0; k < (N/2); k++) {
        double arg = -2.0 * PI<T> * k / N;
        double c = cos(arg);
        double s = sin(arg);
        h_tw[k] = make_vec2((T) c, (T) s); // Casts
    }

    // Transfering to GPU
    size_t size = sizeof(T2) * (N/2);
    cudaError_t err = cudaMalloc(table, size);
    if (err == cudaSuccess) {
        cudaMemcpy(*table, h_tw, size, cudaMemcpyHostToDevice);
    }

    free(h_tw);
    return err == cudaSuccess;
}

template <typename T>
bool createCufftPlan(cufftHandle* plan, int N) {
    constexpr bool is_float = std::is_same_v<T, float>;

    // Plan creation
    cufftResult planResult = is_float
        ? cufftPlan1d(plan, N, CUFFT_C2C, 1)   // FP32
        : cufftPlan1d(plan, N, CUFFT_Z2Z, 1);  // FP64

    if (planResult != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        return false;
    }
    return true;
}

template <typename T, typename T2>
FFTContext createFFTContext(int N, bool needsWorkspace, bool needsTable, bool needsPlan) {
    FFTContext ctx;
    if (needsWorkspace) {
        ctx.workspaceCreated = createWorkspace<T2>(&ctx.workspace, N);
    }

    if (needsTable) {
        ctx.tableCreated = createTwiddlesTable<T, T2>(&ctx.twiddles, N);
    }

    if (needsPlan) {
        ctx.planCreated = createCufftPlan<T>(&ctx.plan, N);
    }

    return ctx;
}

void destroyFFTContext(FFTContext& ctx) {
    if (ctx.workspaceCreated) cudaFree(ctx.workspace);
    if (ctx.tableCreated) cudaFree(ctx.twiddles);
    if (ctx.planCreated) cufftDestroy(ctx.plan);
}

bool checkFFTContext(FFTContext& ctx, bool needsWorkspace, bool needsTable, bool needsPlan) {
    bool ret = true;
    if (needsWorkspace && ctx.workspaceCreated == false) {
        ret = false;
        fprintf(stderr, "FFTContext error: failed to create Workspace.\n");
    }
    if (needsTable && ctx.tableCreated == false) {
        ret = false;
        fprintf(stderr, "FFTContext error: failed to create Twiddles Table.\n");
    }
    if (needsPlan && ctx.planCreated == false) {
        ret = false;
        fprintf(stderr, "FFTContext error: failed to create cufftPlan1d.\n");
    }

    return ret;
}

// Function that sets/resets data vector
template <typename T, typename T2>
void setVector(T2* X, int N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (int i = 0 ; i < N; i++) {
        X[i].x = dist(rng);
        X[i].y = dist(rng);
    }
}

// Function that checks if a kernel is correct against cuFFT results
template <typename T, typename T2>
bool checkKernel(kernelFunction<T, T2> myKernel, int N, FFTContext& myKernelCtx) {

    bool ret = true;

    size_t size = N * sizeof(T2);

    // Baseline vectors
    T2* h_cufft_X = (T2*)malloc(size);
    T2* d_cufft_X;
    cudaMalloc(&d_cufft_X, size);
    
    // MyKernel vectors
    T2* h_myKernel_X = (T2*)malloc(size);
    T2* d_myKernel_X;
    cudaMalloc(&d_myKernel_X, size);

    // Context for cuFFT
    FFTContext cufftCtx = createFFTContext<T, T2>(N, false, false, true);

    // Fill both data vectors with values
    setVector<T, T2>(h_cufft_X, N);
    setVector<T, T2>(h_myKernel_X, N);
    cudaMemcpy(d_cufft_X, h_cufft_X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_myKernel_X, h_myKernel_X, size, cudaMemcpyHostToDevice);

    // Invokes cuFFT kernel (baseline)
    cuFFT<T, T2>(d_cufft_X, N, cufftCtx);
    cudaMemcpy(h_cufft_X, d_cufft_X, size, cudaMemcpyDeviceToHost);
    
    // Invokes myKernel
    myKernel(d_myKernel_X, N, myKernelCtx);
    cudaMemcpy(h_myKernel_X, d_myKernel_X, size, cudaMemcpyDeviceToHost);

    // Assuming h_cufft_X is the one with correct values
    T max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        T diff = cabs(csub(h_cufft_X[i], h_myKernel_X[i]));
        T ref = std::max(cabs(h_cufft_X[i]), static_cast<T>(1e-6f));
        T rel = diff / ref;
        max_diff = std::max(max_diff, rel);
        if (max_diff > THRESHOLD){
            printf("(%f,%f) and (%f,%f) at [%d]\n", h_cufft_X[i].x, h_cufft_X[i].y, h_myKernel_X[i].x, h_myKernel_X[i].y, i);
            ret = false;
            break;
        }
    }

    free(h_cufft_X);
    free(h_myKernel_X);
    destroyFFTContext(cufftCtx);
    cudaFree(d_cufft_X);
    cudaFree(d_myKernel_X);

    return ret;
}

// Max consecutive unnormalized FFTs berfore values can overflow T's range
template <typename T>
int safeResetInterval(int N) {
    double maxVal = std::numeric_limits<T>::max();   // 1.8e308 overflowed to inf, making the double SRI 1
    int k = (int)(2.0 * std::log(maxVal / 1e6) / std::log((double)N));
    return k < 1 ? 1 : k;
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
    outf << "kernel_version,N,precision,RPT,ACC,threshold,";                        // harness
    outf << "time_setup_ms,time_mean_ms,time_median_ms,time_std,time_ci95,";        // kernel
    outf << "ctrl_time_ms,time_h2d_ms,time_d2h_ms,";                                             // transfers
    outf << "energy_mean_mj,energy_median_mj,energy_std,energy_ci95,";              // energy
    outf << "temp_begin,temp_end,sm_mhz_begin,sm_mhz_end,gflops,perf_per_watt,edp"; // other
    outf << std::endl;       
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
    outf << c.time_setup_ms << "," << kernel_mean << "," << kernel_median << "," << kernel_std << "," << kernel_ic95 << ",";

    //* Data transfer statistics (in milliseconds)
    double H2D_mean = mean(H2D_times, c.rpt);
    double D2H_mean = mean(D2H_times, c.rpt);
    outf << c.ctrl_time << "," << H2D_mean << "," << D2H_mean << ",";

    //* Energy statistics (in millijoules)
    double energy_mean = mean(energies, c.rpt);
    double energy_median = median(energies, c.rpt);
    double energy_std = stddev(energies, c.rpt);
    double energy_ic95 = ic95(energies, c.rpt);
    outf << energy_mean << "," << energy_median << "," << energy_std << "," << energy_ic95 << ",";

    //* Other data
    
    // Temperature
    outf << c.temp_begin << "," << c.temp_end << ",";

    // Clock
    outf << c.sm_mhz_begin << "," << c.sm_mhz_end << ",";

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

    outf << gflops << "," << perf_per_watt << ","<< edp << std::endl;
}

// 
kernelEntry* selectKernelByName(kernelEntry kernel_list[], int size, std::string input) {
    for (int i = 0; i < size; i++) {
        if (input == kernel_list[i].name)
            return &kernel_list[i];
    }
    return nullptr;
}

#endif