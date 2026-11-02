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
        { "FFT_Stockham",        FFT_Stockham<float, float2>,        FFT_Stockham<double, double2>,        true,  false, false },
        { "FFT_Stockham_Tabled", FFT_Stockham_Tabled<float, float2>, FFT_Stockham_Tabled<double, double2>, true,  true,  false },
        { "FFT_Shared",          FFT_Shared<float, float2>,          FFT_Shared<double, double2>,          true,  false, false },
        { "FFT_Shared_Tabled",   FFT_Shared_Tabled<float, float2>,   FFT_Shared_Tabled<double, double2>,   true,  true,  false },
        { "FFT_BCF",             FFT_BCF<float, float2>,             FFT_BCF<double, double2>,             true,  false, false },
        { "FFT_BCF_Tabled",      FFT_BCF_Tabled<float, float2>,      FFT_BCF_Tabled<double, double2>,      true,  true,  false },
        { "FFT_Radix4",          FFT_Radix4<float, float2>,          FFT_Radix4<double, double2>,          true,  false, false },
        { "FFT_Radix4_Tabled",   FFT_Radix4_Tabled<float, float2>,   FFT_Radix4_Tabled<double, double2>,   true,  true,  false },
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
    int NX_list[] = { 1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24 };
    const int NX_L_SIZE = sizeof(NX_list)/sizeof(NX_list[0]);
    
    // ACC (handpicked per kernel, precision and N for a ~1 s energy window)
    //                       FP32 { N = 2^10 ... 2^24 }
    //                       FP64 { N = 2^10 ... 2^24 }
    static accEntry ACC_list [] {
        { "FFT_Naive",           {  44000,  37000,  31000, 20000,  6800,  740,  170,  39 },
                                 {  16000,  13000,   6800,  2600,   610,  140,   33,   8 } },
        { "FFT_Naive_Tabled",    {  45000,  37000,  31000, 20000,  4600,  640,  130,  27 },
                                 {  37000,  31000,  20000, 11000,  1600,  360,   77,  17 } },
        { "FFT_Stockham",        {  48000,  40000,  33000, 23000,  4200,  940,  220,  52 },
                                 {  17000,  14000,   7100,  2600,   670,  160,   37,   8 } },
        { "FFT_Stockham_Tabled", {  49000,  41000,  34000, 25000,  4100,  910,  210,  50 },
                                 {  41000,  34000,  23000, 11000,  1900,  450,  100,  24 } },
        { "FFT_Shared",          { 150000, 140000, 110000, 54000, 15000, 3400,  830, 200 },
                                 {  46000,  30000,  11000,  2800,   660,  160,   36,   8 } },
        { "FFT_Shared_Tabled",   { 190000, 170000, 140000, 77000, 14000, 3500,  810, 170 },
                                 { 130000,  93000,  50000, 15000,  3700,  870,  210,  48 } },
        { "FFT_BCF",             { 150000, 140000, 100000, 49000, 14000, 3100,  760, 190 },
                                 {  46000,  30000,  11000,  2800,   660,  160,   36,   8 } },
        { "FFT_BCF_Tabled",      { 190000, 170000, 140000, 72000, 14000, 3400,  800, 170 },
                                 { 130000,  93000,  51000, 15000,  3700,  870,  210,  48 } },
        { "FFT_Radix4",          { 160000, 150000, 120000, 72000, 18000, 4000,  970, 230 },
                                 {  44000,  35000,  14000,  4200,   910,  220,   51,  12 } },
        { "FFT_Radix4_Tabled",   { 180000, 180000, 150000, 87000, 19000, 3500,  820, 170 },
                                 { 120000,  98000,  51000, 16000,  3900,  920,  220,  51 } },
        { "cuFFT",               { 130000, 100000,  85000, 73000, 30000, 6800, 1800, 370 },
                                 {  38000,  14000,  17000,  9100,  2900,  700,  190,  49 } },
    };
    const int ACC_L_SIZE = sizeof(ACC_list)/sizeof(ACC_list[0]);
    static_assert(NX_L_SIZE == NX_COUNT, "ACC_list rows need one value per N in NX_list");

    // Every kernel must have an ACC row: fail now, not in the middle of the run
    for (int k_idx = 0; k_idx < KN_L_SIZE; k_idx++) {
        if (!selectAccByName(ACC_list, ACC_L_SIZE, kernel_list[k_idx].name)) {
            fprintf(stderr, "No ACC entry for kernel %s\n", kernel_list[k_idx].name);
            return 1;
        }
    }

    // NX Loop
    for (int n_idx = 0; n_idx < NX_L_SIZE; n_idx++) {
        int NX = NX_list[n_idx];
        
        const int ACC_CTRL = 1000;
        controlResult ctrl_f = runControl<float, float2>(NX, ACC_CTRL, FP32_STRING);
        controlResult ctrl_d = runControl<double, double2>(NX, ACC_CTRL, FP64_STRING);
        
        // KERNEL Loop 
        for (int k_idx = 0; k_idx < KN_L_SIZE; k_idx++) {
            kernelEntry kernel = kernel_list[k_idx];
            
            accEntry* acc = selectAccByName(ACC_list, ACC_L_SIZE, kernel.name);
            int ACC_f = acc->float_acc[n_idx];
            int ACC_d = acc->double_acc[n_idx];

            // Run experiments
            runExperiment<float, float2>(outf, kernel.float_fn, kernel.name, NX, ACC_f, ctrl_f, FP32_STRING,
                                         kernel.needsWorkspace, kernel.needsTable, kernel.needsPlan);
            runExperiment<double, double2>(outf, kernel.double_fn, kernel.name, NX, ACC_d, ctrl_d, FP64_STRING,
                                           kernel.needsWorkspace, kernel.needsTable, kernel.needsPlan);
        }
    }

    return 0;
}
