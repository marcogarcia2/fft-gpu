#include <iostream>
#include <fstream>

#include <nvml.h>

#include "kernels.cuh"
#include "util.cuh"

#define RPT     100

int main() {

    setvbuf(stdout, NULL, _IOLBF, 0);
    std::ofstream outf { createCSV() };

    // CUDA Events Setup
    cudaEvent_t H2D_start, H2D_stop, D2H_start, D2H_stop, kernel_start, kernel_stop;
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
    float H2D_ms, D2H_ms, kernel_ms;

    // Energy variables
    unsigned long long energy_start, energy_stop;
    float kernel_energy_mj;

    // Temperature variables
    unsigned int temp_begin, temp_end;

    // Results vectors
    float* kernel_times = (float*)malloc(sizeof(float) * RPT);
    float* H2D_times    = (float*)malloc(sizeof(float) * RPT);
    float* D2H_times    = (float*)malloc(sizeof(float) * RPT);
    float* energies     = (float*)malloc(sizeof(float) * RPT);

    // Start float-double loop

    // NX and ACC
    int ACC_list[] = { 20000, 10000, 2500, 40, 10 };
    int ACC = ACC_list[0];
    int NX_list[] = { 1<<10, 1<<14, 1<<18, 1<<22, 1<<24 };
    int NX = NX_list[0];

    // Start NX Loop
    for (int nx_idx = 0; nx_idx < 5; nx_idx++) {
        NX = NX_list[nx_idx];
        ACC = ACC_list[nx_idx];
        
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_begin);
        
        printf("Experiment Started: FFT_Naive, float, NX = %d, ACC = %d\n", NX, ACC);
        // Start kernel loop

        // Allocating vectors
        size_t size = NX * sizeof(float2);
        float2* h_X = (float2*)malloc(size);    // host vector
        float2* d_X;                            // device vector
        cudaMalloc(&d_X, size);

        // Threads and Blocks
        int threadsPerBlock = 256;
        int blocksPerGrid = (NX + threadsPerBlock - 1) / threadsPerBlock;

        // Check if all kernels are correct with cuFFT before proceeding
        if (!checkKernel(NX)){
            fprintf(stderr, "Error: MyKernel is not correct (NX = %d).\n", NX);
            // return 1;
        }

        // Warmup
        cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
        FFT_Naive(d_X, NX, blocksPerGrid, threadsPerBlock);
        cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

        for (int rep = 0; rep < RPT; rep++) {

            setVector(h_X, NX);

            // H2D transfer
            cudaEventRecord(H2D_start);
            cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
            cudaEventRecord(H2D_stop);
            cudaEventSynchronize(H2D_stop);
            cudaEventElapsedTime(&H2D_ms, H2D_start, H2D_stop);

            // Kernel: Time and Energy
            nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);
            cudaEventRecord(kernel_start);
            for (int k = 0; k < ACC; k++) {
                FFT_Naive(d_X, NX, blocksPerGrid, threadsPerBlock);
            }
            cudaEventRecord(kernel_stop);
            cudaEventSynchronize(kernel_stop);
            nvmlDeviceGetTotalEnergyConsumption(device, &energy_stop);
            cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
            
            // D2H transfer
            cudaEventRecord(D2H_start);
            cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);
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

        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_end);

        // Constants used in results
        struct constants c {};
        c.kname = "FFT_Naive";
        c.precision = "float";
        c.nx = NX;
        c.rpt = RPT;
        c.acc = ACC;
        c.thresh = THRESHOLD;
        c.temp_begin = temp_begin;
        c.temp_end = temp_end;

        // Saving results
        writeResultsCSV(outf, kernel_times, H2D_times, D2H_times, energies, c);

        // Frees memory
        free(h_X);
        cudaFree(d_X);
        
        // end kernel loop
    }
    // end float-double loop

    // Destroy results vectors
    free(kernel_times);
    free(D2H_times);
    free(H2D_times);
    free(energies);

    // NVML free
    nvmlShutdown();
    
    // Destroy CUDA Events
    cudaEventDestroy(H2D_start);
    cudaEventDestroy(H2D_stop);
    cudaEventDestroy(D2H_start);
    cudaEventDestroy(D2H_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    return 0;
}