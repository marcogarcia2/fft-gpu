#ifndef PROFILER_CUH
#define PROFILER_CUH

#include "util.cuh"
#include "kernels.cuh"

// Runs a profile set
template <typename T, typename T2>
void runProfiler(kernelFunction<T, T2> kernel, int NX, bool needsWorkspace, bool needsTable, bool needsPlan) {
        
    // Allocating vectors
    size_t size = NX * sizeof(T2);
    T2* h_X = (T2*)malloc(size);        // host vector
    T2* d_X;                            // device vector
    cudaMalloc(&d_X, size);

    // FFT Context Creation
    FFTContext ctx = createFFTContext<T, T2>(NX, needsWorkspace, needsTable, needsPlan);
    checkFFTContext(ctx, needsWorkspace, needsTable, needsPlan);

    // Fill vector with values
    setVector<T, T2>(h_X, NX);

    // Running Kernel
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
    kernel(d_X, NX, ctx);
    cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

    // Frees memory
    destroyFFTContext(ctx);
    free(h_X);
    cudaFree(d_X);   
    
    return;
}

// Function that verifies argc and argv, moved from main.cu
int profiler(int argc, char *argv[], kernelEntry kernel_list[], int ksize) {
        kernelEntry* kernel = selectKernelByName(kernel_list, ksize, argv[1]);
        
        // Check argv[2]: NX must be positive integer
        int NX = atoi(argv[2]);
        if (NX <= 0) {
            fprintf(stderr, "N value must be positive.\n");
            return 1;
        }

        // Check argv[3]: "float" or "double"
        bool is_float;
        if      (!strcmp(FP32_STRING, argv[3])) is_float = true;
        else if (!strcmp(FP64_STRING, argv[3])) is_float = false;
        else {
            fprintf(stderr, "Error: Precision must be \"float\" or \"double\".\n");
            return 1;
        }
        
        if (kernel) {
            if (is_float) runProfiler<float, float2>(kernel->float_fn, NX, kernel->needsWorkspace, kernel->needsTable, kernel->needsPlan);
            else          runProfiler<double, double2>(kernel->double_fn, NX, kernel->needsWorkspace, kernel->needsTable, kernel->needsPlan);
        }
        
        else {
            printf("Errror: Kernel '%s' not found.", argv[1]);
            return 1;
        }
        
        return 0;
}

#endif