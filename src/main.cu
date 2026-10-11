#include <iostream>

#include <nvml.h>

#include "kernels.cuh"
#include "harness.cuh"
#include "profiler.cuh"

int main(int argc, char *argv[]) {
    
    setvbuf(stdout, NULL, _IOLBF, 0); // synchronous prints
    
    // Kernels:                                                                 BOOLEAN VALUES LIST: (needsWorkspace, needsTable, needsPlan)
    static kernelEntry kernel_list [] {
        { "FFT_Naive",           FFT_Naive<float, float2>,           FFT_Naive<double, double2>,           false, false, false },
        { "FFT_Naive_Tabled",    FFT_Naive_Tabled<float, float2>,    FFT_Naive_Tabled<double, double2>,    false, true,  false },
        { "FFT_Stockham_Tabled", FFT_Stockham_Tabled<float, float2>, FFT_Stockham_Tabled<double, double2>, true,  true,  false },
        { "cuFFT",               cuFFT<float, float2>,               cuFFT<double, double2>,               false, false, true  },
    };
    const int KN_L_SIZE = sizeof(kernel_list)/sizeof(kernel_list[0]);

    // --- PROFILE MODE (NCU) --- //
    if (argc == 4) {
        return profiler(argc, argv, kernel_list, KN_L_SIZE);
    } 
    else if (argc == 2 || argc == 3 || argc >= 5) {
        fprintf(stderr, "Usage: %s <kernel_name> <N> <precision>\n", argv[0]);
        return 1;
    }



    // --- REGULAR MODE (EXPERIMENTS) --- //
    std::ofstream outf { createCSV() }; // CSV file that stores results
    
    // NX
    int NX_list[] = { 1<<10, 1<<14, 1<<18, 1<<22, 1<<24 };
    const int NX_L_SIZE = sizeof(NX_list)/sizeof(NX_list[0]);
    
    // ACC (different values for float and double)
    int ACC_float_list[] = { 20000, 10000, 2000, 100, 30 };
    int ACC_double_list[] = { 10000, 5000, 600, 30, 10 };
    // int ACC_float_list[] = { 500, 300, 100, 10, 3 }; // testing purposes
    // int ACC_double_list[] = { 100, 75, 50, 2, 1 };
    
    // NX Loop
    for (int n_idx = 0; n_idx < NX_L_SIZE; n_idx++) {
        int NX = NX_list[n_idx];
        int ACC_f = ACC_float_list[n_idx];
        int ACC_d = ACC_double_list[n_idx];

        // KERNEL Loop 
        for (int k_idx = 0; k_idx < KN_L_SIZE; k_idx++) {
            kernelEntry kernel = kernel_list[k_idx];

            // Run experiments
            runExperiment<float, float2>(outf, kernel.float_fn, kernel.name, NX, ACC_f, FP32_STRING,
                                         kernel.needsWorkspace, kernel.needsTable, kernel.needsPlan);
            runExperiment<double, double2>(outf, kernel.double_fn, kernel.name, NX, ACC_d, FP64_STRING,
                                           kernel.needsWorkspace, kernel.needsTable, kernel.needsPlan);
        }
    }

    return 0;
}
