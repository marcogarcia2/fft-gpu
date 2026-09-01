#ifndef UTIL_H
#define UTIL_H

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>


#define ARRAY_SZ 2048
#define RPT 150

#define KB (1024)
#define MB (1024 * KB)
#define GB (1024 * MB)
#define LARGEST_CACHE_SZ (16 * MB)

// type abbreviations
using tp = std::chrono::steady_clock::time_point;
using duration = std::chrono::duration<double>;
using Complex = std::complex<double>;
using KernelFn = std::vector<Complex>(*)(const std::vector<double>&, std::size_t);

// cleaning cache fn
static unsigned char dummy_buffer[LARGEST_CACHE_SZ];
unsigned long long clean_cache() {
    unsigned long long acc { 0 };
    unsigned long long i;
    for (i = 0; i < LARGEST_CACHE_SZ; i++) {
        dummy_buffer[i] += 1;
        acc += dummy_buffer[i];
    }
    return acc;
}

void debug(std::string message) {
    std::cout << message << std::endl;
}

void set_vector(std::vector<double>& f, const std::size_t Np) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    // for (auto& i : f) i = dist(rng); 
    for (std::size_t i { 0 }; i < Np; i++) {
        f[i] = dist(rng);
    }
}

bool compare_vectors(const std::vector<Complex>& X, const std::vector<Complex>& Y, const std::size_t Np) {
    double threshold = 1e-9;
    double max_diff = 0.0;
    for (std::size_t i { 0 }; i < Np; i++) {
        max_diff = std::max(max_diff, std::abs(X[i] - Y[i]));
        if (max_diff > threshold) {
            std::cout << X[i] << " and " << Y[i] << " at element " << i << std::endl;
            return false;
        }
    }
    return true;
}

std::string check_correctude(const std::vector<KernelFn>& kernels, const std::vector<std::string>& kernel_names, const std::vector<double>& f, const std::size_t Np) {
    // Return variable
    std::string ret = "";

    // Edge case: only one kernel
    if (kernels.size() <= 1)
        return ret;

    // Assumes first kernel is correct
    KernelFn kernel = kernels[0];
    std::vector<Complex> F = kernel(f, Np);

    for (std::size_t k = 1; k < kernels.size(); k++) {
        
        kernel = kernels[k];
        std::vector<Complex> F_aux = kernel(f, Np);

        if (!compare_vectors(F, F_aux, Np)) {
            // At least one kernel is incorrect
            if (ret.length() == 0)
                ret = kernel_names[k];
            else
                ret += ", " + kernel_names[k];
            
            continue;
        }
    }
    return ret;
}

#endif