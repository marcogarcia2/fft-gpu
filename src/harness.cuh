#ifndef HARNESS_CUH
#define HARNESS_CUH

#include <type_traits>
#include <algorithm>
#include <cmath>
#include <nvml.h>

#include "util.cuh"

#define RPT     30
#define CTRL_WINDOW_MS 1000.0f   // length of the control energy window (ms)

// Function that performs kernel measurements
template <typename T, typename T2>
void runExperiment(std::ofstream& outf, kernelFunction<T, T2> kernel, std::string kname, int NX, int ACC, controlResult ctrl, std::string precision, bool needsWorkspace, bool needsTable, bool needsPlan) {

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
    float* window_times = (float*)malloc(sizeof(float) * RPT);   // raw energy window (ms), resets included

    // Allocating vectors
    size_t size = NX * sizeof(T2);
    T2* h_X_in  = (T2*)malloc(size);        // host vector (input)
    T2* h_X_out = (T2*)malloc(size);        // host vector (output)
    T2* d_X;                                // device vector
    T2* d_Pristine;                         // Pristine copy
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Pristine, size);

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
    // else printf("Kernel is correct.\n");

    // Temperature and Clock before experiment
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_begin);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_mhz_begin);
    
    // Sets up vectors, pristine copy and safe reset interval
    setVector<T, T2>(h_X_in, NX);   // sets up h_X_in values
    memset(h_X_out, 0, size);       // touches h_X_out once
    cudaMemcpy(d_Pristine, h_X_in, size, cudaMemcpyHostToDevice);   // sets up pristine copy
    int SRI = safeResetInterval<T>(NX);   // Calculates reset range so it doesn't overflows

    // BENCHMARKING 
    for (int rep = 0; rep < RPT; rep++) {

        // H2D transfer
        cudaEventRecord(H2D_start);
        cudaMemcpy(d_X, h_X_in, size, cudaMemcpyHostToDevice);             // Mem. Copy
        cudaEventRecord(H2D_stop);
        cudaEventSynchronize(H2D_stop);
        cudaEventElapsedTime(&H2D_ms, H2D_start, H2D_stop);

        // Kernel: Time and Energy
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);
        cudaEventRecord(kernel_start);
        for (int k = 0; k < ACC; k++) {
            kernel(d_X, NX, ctx); // Kernel execution
            if ((k + 1) % SRI == 0) // resets after SRI reps so it doesn't overflows
                cudaMemcpy(d_X, d_Pristine, size, cudaMemcpyDeviceToDevice); // D2D Copy
        }
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_stop);
        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
        window_times[rep] = kernel_ms;   // same interval as the NVML energy reading
        
        // D2H transfer
        cudaEventRecord(D2H_start);
        cudaMemcpy(h_X_out, d_X, size, cudaMemcpyDeviceToHost);             // Mem. Copy
        cudaEventRecord(D2H_stop);
        cudaEventSynchronize(D2H_stop);
        cudaEventElapsedTime(&D2H_ms, D2H_start, D2H_stop);

        // Results calculations
        kernel_ms = (float)(kernel_ms / ACC) - ctrl.time_ms * (ACC / SRI) / ACC;
        kernel_energy_mj = (float)((energy_stop - energy_start) - ctrl.energy_mj * (ACC / SRI)) / ACC;

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
    c.ctrl_time = ctrl.time_ms;
    c.ctrl_energy = ctrl.energy_mj;
    c.window_ms = mean(window_times, RPT);

    // Statistics of the RPT repetitions
    c.kernel_mean   = mean(kernel_times, RPT);
    c.kernel_median = median(kernel_times, RPT);
    c.kernel_std    = stddev(kernel_times, RPT);
    c.kernel_ic95   = ic95(kernel_times, RPT);
    c.H2D_mean      = mean(H2D_times, RPT);
    c.D2H_mean      = mean(D2H_times, RPT);
    c.energy_mean   = mean(energies, RPT);
    c.energy_median = median(energies, RPT);
    c.energy_std    = stddev(energies, RPT);
    c.energy_ic95   = ic95(energies, RPT);

    // Writing down results to CSV file
    writeResultsCSV(outf, c);
    printf("Results saved.\n");

    // Destroy FFTContext memory
    destroyFFTContext(ctx);

    // Frees vector memory
    free(h_X_in);
    free(h_X_out);
    cudaFree(d_X);
    cudaFree(d_Pristine);

    // Destroy results vectors
    free(kernel_times);
    free(D2H_times);
    free(H2D_times);
    free(energies);
    free(window_times);

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

// Function that calculates the D2D reset copy cost (time and energy) to subtract later.
// First ACC_ctrl copies give the time of one copy; then enough copies to fill CTRL_WINDOW_MS
// are timed and metered, so the NVML energy counter (~100 ms updates) gets a long window.
template <typename T, typename T2>
controlResult runControl(int NX, int ACC_ctrl, std::string precision) {
    // CUDA Events Setup
    cudaEvent_t ctrl_event_start, ctrl_event_stop;    
    cudaEventCreate(&ctrl_event_stop);
    cudaEventCreate(&ctrl_event_start);

    // NVML Setup
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    
    // Time and Energy Variables
    float total_time;
    unsigned long long energy_start, energy_stop;
    controlResult ctrl {};

    // Allocating vectors
    size_t size = NX * sizeof(T2);
    T2* h_X = (T2*)malloc(size);        // host vector
    T2* d_X;                            // device vector
    T2* d_Pristine;                     // pristine copy
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Pristine, size);

    setVector<T, T2>(h_X, NX);
    cudaMemcpy(d_Pristine, h_X, size, cudaMemcpyHostToDevice);

    // Calibration: time of one copy, to size the energy window
    cudaEventRecord(ctrl_event_start);
    for (int k = 0; k < ACC_ctrl; k++) {
        cudaMemcpy(d_X, d_Pristine, size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(ctrl_event_stop);
    cudaEventSynchronize(ctrl_event_stop);
    cudaEventElapsedTime(&total_time, ctrl_event_start, ctrl_event_stop);
    int ACC_energy = std::max(ACC_ctrl, (int)std::ceil(CTRL_WINDOW_MS / (total_time / ACC_ctrl)));

    std::cout << "Starting Control Experiment: ";
    std::cout << "("<< "NX=" << NX << ", ";
    std::cout << precision << ", ";
    std::cout << "ACC_ctrl=" << ACC_energy << ")" << std::endl; 

    // Control Data Flow: Time and Energy
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);
    cudaEventRecord(ctrl_event_start);
    for (int k = 0; k < ACC_energy; k++) {
        cudaMemcpy(d_X, d_Pristine, size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(ctrl_event_stop);
    cudaEventSynchronize(ctrl_event_stop);
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_stop);
    cudaEventElapsedTime(&total_time, ctrl_event_start, ctrl_event_stop);

    // Calculating results per copy
    ctrl.time_ms = (float)total_time / ACC_energy;
    ctrl.energy_mj = (float)(energy_stop - energy_start) / ACC_energy;
    
    std::cout << "Control Time Elapsed = " << total_time << " ms" << std::endl;
    std::cout << "D2D Time = " << ctrl.time_ms << " ms, D2D Energy = " << ctrl.energy_mj << " mJ" << std::endl;

    // Freeing memory
    cudaFree(d_X);
    cudaFree(d_Pristine);
    free(h_X);

    nvmlShutdown();

    cudaEventDestroy(ctrl_event_start);
    cudaEventDestroy(ctrl_event_stop);

    return ctrl;
}

#endif