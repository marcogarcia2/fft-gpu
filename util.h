#ifndef UTIL_H
#define UTIL_H

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cassert>

#define KB (1024)
#define MB (1024 * KB)
#define GB (1024 * MB)
#define LARGEST_CACHE_SZ (16 * MB)

// type abbreviations
using tp = std::chrono::steady_clock::time_point;
using duration = std::chrono::duration<double>;
using Complex = std::complex<double>;
using KernelFn = std::vector<Complex>(*)(const std::vector<double>&);

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

template <typename T>
void print_vector(const std::vector<T>& f) {
    for (std::size_t i { 0 }; i < f.size(); i++) {
        std::cout << f[i];
        if (i < (f.size() - 1)) std::cout << ", ";
    }
    std::cout << std::endl;
}

void set_vector(std::vector<double>& f) {
    std::size_t N = f.size();
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    // for (auto& i : f) i = dist(rng); 
    for (std::size_t i { 0 }; i < N; i++) {
        f[i] = dist(rng);
    }
}

bool compare_vectors(const std::vector<Complex>& X, const std::vector<Complex>& Y) {
    assert(X.size() == Y.size());
    std::size_t N = X.size();
    double threshold = 1e-9;
    double max_diff = 0.0;
    for (std::size_t i { 0 }; i < N; i++) {
        max_diff = std::max(max_diff, std::abs(X[i] - Y[i]));
        if (max_diff > threshold) {
            std::cout << X[i] << " and " << Y[i] << " at element " << i << std::endl;
            return false;
        }
    }
    return true;
}

std::string check_correctude(const std::vector<KernelFn>& kernels, const std::vector<std::string>& kernel_names, const std::vector<double>& f) {
    // Return variable
    std::string ret = "";

    // Edge case: only one kernel
    if (kernels.size() <= 1)
        return ret;

    // Assumes first kernel is correct
    KernelFn kernel = kernels[0];
    std::vector<Complex> F = kernel(f);

    for (std::size_t k = 1; k < kernels.size(); k++) {
        
        kernel = kernels[k];
        std::vector<Complex> F_aux = kernel(f);

        if (!compare_vectors(F, F_aux)) {
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