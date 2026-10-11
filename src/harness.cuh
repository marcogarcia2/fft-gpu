#ifndef HARNESS_CUH
#define HARNESS_CUH

#include <type_traits>
#include <nvml.h>

#include "util.cuh"

#define RPT     20

template <typename T, typename T2>
void runExperiment(std::ofstream& outf, kernelFunction<T, T2> kernel, std::string kname, int NX, int ACC, std::string precision, bool needsWorkspace, bool needsTable, bool needsPlan) {

    // CUDA Events Setup
    cudaEvent_t setup_start, setup_stop, H2D_start, H2D_stop, D2H_start, D2H_stop, kernel_start, kernel_stop;
    cudaEventCreate(&setup_start);
    cudaEventCreate(&setup_stop);
    cudaEventCreate(&H2D_start);
    cudaEventCreate(&H2D_stop);
    cudaEventCreate(&D2H_start);
    cudaEventCreate(&D2H_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    // NVML Setup
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    
    // Time variables
    float time_setup_ms, H2D_ms, D2H_ms, kernel_ms;

    // Energy variables
    unsigned long long energy_start, energy_stop;
    float kernel_energy_mj;
    
    // Temperature and clock variables
    unsigned int temp_begin, temp_end;
    unsigned int sm_mhz_begin, sm_mhz_end;

    // Allocating results vectors
    float* kernel_times = (float*)malloc(sizeof(float) * RPT);
    float* H2D_times    = (float*)malloc(sizeof(float) * RPT);
    float* D2H_times    = (float*)malloc(sizeof(float) * RPT);
    float* energies     = (float*)malloc(sizeof(float) * RPT);

    // Allocating vectors
    size_t size = NX * sizeof(T2);
    T2* h_X = (T2*)malloc(size);        // host vector
    T2* d_X;                            // device vector
    cudaMalloc(&d_X, size);

    // EXPERIMENT STARTED
    std::cout << "Experiment Started: ";
    std::cout << kname << ", ";
    std::cout << precision << ", "; 
    std::cout << "NX=" << NX << ", "; 
    std::cout << "ACC=" << ACC << std::endl; 

    // FFT Context creation
    cudaEventRecord(setup_start);
    FFTContext ctx = createFFTContext<T, T2>(NX, needsWorkspace, needsTable, needsPlan);
    checkFFTContext(ctx, needsWorkspace, needsTable, needsPlan);
    cudaEventRecord(setup_stop);
    cudaEventSynchronize(setup_stop);
    cudaEventElapsedTime(&time_setup_ms, setup_start, setup_stop);

    // Check if all kernels are correct with cuFFT before proceeding
    if (!checkKernel<T, T2>(kernel, NX, ctx)){
        fprintf(stderr, "Error: MyKernel is not correct (NX = %d).\n", NX);
        // return 1;
    }
    else printf("Kernel is correct.\n");

    // Temperature and Clock before experiment
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_begin);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_mhz_begin);

    // BENCHMARKING 
    for (int rep = 0; rep < RPT; rep++) {

        // Resets vector with original values
        setVector<T, T2>(h_X, NX);

        // H2D transfer
        cudaEventRecord(H2D_start);
        cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);             // Mem. Copy
        cudaEventRecord(H2D_stop);
        cudaEventSynchronize(H2D_stop);
        cudaEventElapsedTime(&H2D_ms, H2D_start, H2D_stop);

        // Kernel: Time and Energy
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);
        cudaEventRecord(kernel_start);
        for (int k = 0; k < ACC; k++) {
            kernel(d_X, NX, ctx);            // Kernel execution
        }
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_stop);
        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
        
        // D2H transfer
        cudaEventRecord(D2H_start);
        cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);             // Mem. Copy
        cudaEventRecord(D2H_stop);
        cudaEventSynchronize(D2H_stop);
        cudaEventElapsedTime(&D2H_ms, D2H_start, D2H_stop);

        // Results calculations
        kernel_ms /= (ACC);
        kernel_energy_mj = (float)(energy_stop - energy_start) / (ACC);

        // Saves results
        kernel_times[rep] = kernel_ms;
        H2D_times[rep] = H2D_ms;
        D2H_times[rep] = D2H_ms;
        energies[rep] = kernel_energy_mj;
    }

    // Temperature and Clock after experiment
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_end);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_mhz_end);

    // Constants used in results
    struct constants c {};
    c.kname = kname;
    c.precision = precision;
    c.nx = NX;
    c.rpt = RPT;
    c.acc = ACC;
    c.thresh = THRESHOLD;
    c.time_setup_ms = time_setup_ms;
    c.temp_begin = temp_begin;
    c.temp_end = temp_end;
    c.sm_mhz_begin = sm_mhz_begin;
    c.sm_mhz_end = sm_mhz_end;

    // Writing down results to CSV file
    writeResultsCSV(outf, kernel_times, H2D_times, D2H_times, energies, c);
    printf("Results saved.\n");

    // Destroy FFTContext memory
    destroyFFTContext(ctx);

    // Frees vector memory
    free(h_X);
    cudaFree(d_X);

    // Destroy results vectors
    free(kernel_times);
    free(D2H_times);
    free(H2D_times);
    free(energies);

    // NVML free
    nvmlShutdown();

    // Destroy CUDA Events
    cudaEventDestroy(setup_start);
    cudaEventDestroy(setup_stop);
    cudaEventDestroy(H2D_start);
    cudaEventDestroy(H2D_stop);
    cudaEventDestroy(D2H_start);
    cudaEventDestroy(D2H_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
}

#endif