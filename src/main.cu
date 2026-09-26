#include <iostream>

#include <nvml.h>

#include "kernels.cuh"
#include "harness.cuh"

int main() {

    setvbuf(stdout, NULL, _IOLBF, 0); // synchronous prints

    std::ofstream outf { createCSV() }; // CSV file that stores results
    
    // NX
    int NX = 0;
    int NX_list[] = { 1<<10, 1<<14, 1<<18, 1<<22, 1<<24 };
    const int NX_L_SIZE = sizeof(NX_list)/sizeof(NX_list[0]);
    
    // ACC (different values for float and double)
    int ACC_f = 0, ACC_d = 0;
    int ACC_float_list[] = { 10000, 7000, 1000, 40, 10 };
    int ACC_double_list[] = { 5000, 2000, 300, 15, 5 };
    
    // Kernels
    kernelEntry kernel_list [] {
        { "FFT_Naive", FFT_Naive<float, float2>, FFT_Naive<double, double2> },
    };
    const int KN_L_SIZE = sizeof(kernel_list)/sizeof(kernel_list[0]);

    // Precisions
    std::string fff = "float";
    std::string ddd = "double";

    // NX Loop
    for (int n_idx = 0; n_idx < NX_L_SIZE; n_idx++) {
        NX = NX_list[n_idx];
        ACC_f = ACC_float_list[n_idx];
        ACC_d = ACC_double_list[n_idx];

        // KERNEL Loop 
        for (int k_idx = 0; k_idx < KN_L_SIZE; k_idx++) {
            kernelEntry kernel = kernel_list[k_idx];
            
            // Run experiments
            runExperiment<float, float2>(outf, kernel.float_fn, kernel.name, NX, ACC_f, fff);
            runExperiment<double, double2>(outf, kernel.double_fn, kernel.name, NX, ACC_d, ddd);
        }
    }

    return 0;
}
