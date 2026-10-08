#include "cpu_kernels.h"
#include "util.h"

#define ARRAY_SZ 1 << 20
#define RPT 10

int main(int argc, char *argv[]) {

    //Kernels list
    std::vector<KernelFn> kernels = { parallel_FFT, collapsed_parallel_FFT, stockham_FFT };
    std::vector<std::string> kernel_names = { "Parallel FFT", "Collapsed Parallel FFT", "Stockham FFT" };
    KernelFn kernel;

    // Signals and N
    const std::size_t N = ARRAY_SZ;
    std::vector<double> x(N);
    std::vector<Complex> X(N);
    set_vector(x);

    // Check if kernels are returning the same answer
    std::string ret = check_correctude(kernels, kernel_names, x);    
    if (ret.length() > 0) {
        std::cout << "Incorrect kernels: " << ret << std::endl;
        std::cout << "Aborting" << std::endl;
        return 1;
    }

    // debug("All kernels are correct.");

    // For each kernel
    for (std::size_t k = 0; k < kernels.size(); k++) {
        
        // Switching kernels
        kernel = kernels[k];

        // Zeroing times vector
        std::vector<double> times(RPT);

        // Running benchmark
        for (int r = 0; r < RPT; r++) {
            clean_cache();

            tp begin = std::chrono::steady_clock::now();
            kernel(x);
            tp end = std::chrono::steady_clock::now();
            
            times[r] = std::chrono::duration<double>(end - begin).count();
        }        
        
        double avgtime = 0.0;
        for (int r = 1; r < RPT; r++) {
            avgtime += times[r];
        }
        avgtime = avgtime / (RPT-1);
        std::cout << "Kernel:\t\t" << kernel_names[k] << std::endl;
        std::cout << "Avg Time:\t" << avgtime << std::endl;
        std::cout << std::endl;
    }

    return 0;
}