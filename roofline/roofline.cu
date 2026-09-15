#include <cstdio>
#include <fstream>
#include <iostream>

using String = std::string;

// Kernel that stresses memory bandwidth
__global__ void copyKernel(const float* in, float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i];   // 1 leitura + 1 escrita
}

__host__ double bandwidthBenchmark() {

    std::cout << "Running bandwidth benchmark..." << std::endl;

    size_t n = 1 << 26;                  // ~67M floats = 256 MB por buffer
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 1, bytes);

    int threads = 256, blocks = 256;

    // Warmup
    copyKernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int reps = 100;
    cudaEventRecord(start);
    for (int r = 0; r < reps; ++r)
        copyKernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double seconds = (ms / 1000.0);

    double gbps = (double)(2.0 * bytes * reps) / seconds / 1e9;   // 2 times: each element is read AND written

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_in);
    cudaFree(d_out);
    
    return gbps;
}

// Kernel that stresses computation
template <typename T>
__global__ void computeKernel(T* out, int iters) {
    T x1 = threadIdx.x*(T)0.1, x2 = x1+1, x3 = x1+2, x4 = x1+3;
    T b = (T)0.5, c = (T)0.9;
    for (int i = 0; i < iters; ++i) {
        x1 = x1*b + c;   // 1 line = 1 FMA = 2 FLOPs
        x2 = x2*b + c;
        x3 = x3*b + c;
        x4 = x4*b + c;
    } // total: 4 lines of 2 operations
    // Writes once so compiler doesn't cut off this loop
    out[blockIdx.x*blockDim.x + threadIdx.x] = x1+x2+x3+x4;
}

template <typename T>
__host__ double compute_benchmark() {

    std::cout << "Running compute benchmark..." << std::endl;

    int iters = 50000;
    int threads = 256, blocks = 1024;   // test different 'blocks' to saturate GPU
    T* d_out;
    size_t n = (size_t)threads * blocks;
    size_t bytes = n * sizeof(T); 
    cudaMalloc(&d_out, bytes);

    // Warmup
    computeKernel<T><<<blocks, threads>>>(d_out, iters);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int reps = 100;
    cudaEventRecord(start);
    for (int r = 0; r < reps; r++)
        computeKernel<T><<<blocks, threads>>>(d_out, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double seconds = (ms / 1000.0);

    //                                                4 lines * 2 OPs per line
    double total_flops = (double)threads * blocks * reps * iters * 4 * 2;
    double gflops = total_flops / seconds / 1e9;
    
    cudaFree(d_out);

    return gflops;
}

__host__ void write_json(const String gpu_name, const double bw, const double fp32, const double fp64, const String date) {
    
    // Creates JSON file
    std::ofstream outf { "roofline.json" };
    if (!outf) {
        std::cerr << "Couldn't create file 'roofline.json'." << std::endl;
        exit(1);
    }

    outf << "{\n";
    outf << "    \"gpu\": \"" << gpu_name << "\",\n";
    outf << "    \"peak_bandwidth_gbps\": " << bw << ",\n";
    outf << "    \"peak_gflops_fp32\": " << fp32 << ",\n";
    outf << "    \"peak_gflops_fp64\": " << fp64 << ",\n";
    outf << "    \"date\": \"" << date << "\"\n";
    outf << "}\n";

    return;
}

__host__ bool checkGPUAvailability(void) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "CUDA error or no GPU active: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

int main() {

    // Check if GPU is available
    if (!checkGPUAvailability())
        return 1; 

    // Date
    time_t timestamp;
    time(&timestamp);
    String date = ctime(&timestamp);
    date.pop_back(); //removes last character ('\n')
    
    // GPU name
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    String gpu_name = "";
    if (err != cudaSuccess) {
        std::cerr << "Couldn't get GPU name: " << cudaGetErrorString(err) << std::endl;
        gpu_name = "null";
    }
    else gpu_name = prop.name;

    // Benchmarks
    double peak_bandwidth_gbps = bandwidthBenchmark();
    double peak_gflops_fp32 = compute_benchmark<float>();
    double peak_gflops_fp64 = compute_benchmark<double>();

    // Writes down results
    write_json(gpu_name, peak_bandwidth_gbps, peak_gflops_fp32, peak_gflops_fp64, date);
    
    return 0;
}