#include <iostream>
#include <fstream>

#include <nvml.h>

#include "kernels.cuh"
#include "util.cuh"

#define RPT     100

int main() {

    setvbuf(stdout, NULL, _IOLBF, 0); // synchronous prints
    
    std::ofstream outf { createCSV() }; // CSV file that stores results

    // NX
    int NX_list[] = { 1<<10, 1<<14, 1<<18, 1<<22, 1<<24 };
    int NX = NX_list[0];
    const int NX_L_SIZE = sizeof(NX_list)/sizeof(NX_list[0]);
    
    // ACC
    int ACC_list[] = { 10000, 7500, 2500, 40, 10 };
    int ACC = ACC_list[0];

    // Kernels
    const char* kernel_names[] = { "cuFFT", "FFT_Naive" };
    kernelFunction kernel_list[] = { cuFFT, FFT_Naive };
    kernelFunction kernel = kernel_list[0];
    const int KN_L_SIZE = sizeof(kernel_list)/sizeof(kernel_list[0]);

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

    // Allocating results vectors
    float* kernel_times = (float*)malloc(sizeof(float) * RPT);
    float* H2D_times    = (float*)malloc(sizeof(float) * RPT);
    float* D2H_times    = (float*)malloc(sizeof(float) * RPT);
    float* energies     = (float*)malloc(sizeof(float) * RPT);

    // NX Loop
    for (int nx_idx = 0; nx_idx < NX_L_SIZE; nx_idx++) {
        NX = NX_list[nx_idx];
        ACC = ACC_list[nx_idx];

        // Allocating vectors
        size_t size = NX * sizeof(float2);
        float2* h_X = (float2*)malloc(size);    // host vector
        float2* d_X;                            // device vector
        cudaMalloc(&d_X, size);

        // Threads and Blocks
        int threadsPerBlock = 256;
        int blocksPerGrid = (NX + threadsPerBlock - 1) / threadsPerBlock;

        // Kernels Loop
        for (int kernel_idx = 0; kernel_idx < KN_L_SIZE; kernel_idx ++) {
            kernel = kernel_list[kernel_idx];

            // EXPERIMENT STARTED
            printf("Experiment Started: %s, float, NX = %d, ACC = %d\n", kernel_names[kernel_idx], NX, ACC);
            
            // Temperature before experiment
            nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_begin);

            // Check if all kernels are correct with cuFFT before proceeding
            if (!checkKernel(kernel, NX)){
                fprintf(stderr, "Error: MyKernel is not correct (NX = %d).\n", NX);
                // return 1;
            }
            else printf("Kernel is correct.\n");

            // Warmup (not necessary because checkKernel() is already being called)
            // cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
            // kernel(d_X, NX, blocksPerGrid, threadsPerBlock);
            // cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

            for (int rep = 0; rep < RPT; rep++) {

                // Resets vector with original values
                setVector(h_X, NX);

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
                    kernel(d_X, NX, blocksPerGrid, threadsPerBlock);            // Kernel execution
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

            // Temperature after experiment
            nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_end);

            // Constants used in results
            struct constants c {};
            c.kname = kernel_names[kernel_idx];
            c.precision = "float";
            c.nx = NX;
            c.rpt = RPT;
            c.acc = ACC;
            c.thresh = THRESHOLD;
            c.temp_begin = temp_begin;
            c.temp_end = temp_end;

            // Writing down results to CSV file
            writeResultsCSV(outf, kernel_times, H2D_times, D2H_times, energies, c);
            
        }   // end kernel loop
        
        // Frees vector memory
        free(h_X);
        cudaFree(d_X);
        
    }   // end NX loop

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