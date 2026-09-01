#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include "util.h"

std::vector<Complex> DFT(const std::vector<double>& f, const std::size_t Np) {

    std::vector<Complex> F(Np);

    for (std::size_t k { 0 }; k < Np; k++) {
        Complex acc { 0.0 };
        double common = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(Np);
        for (std::size_t n { 0 }; n < Np; n++) {
            double arg = common * static_cast<double>(n);
            acc += f[n] * Complex(std::cos(arg), std::sin(arg));
        }
        F[k] = acc;
    }
    return F;
}

std::vector<Complex> FFT_recursion(const std::vector<Complex>& f, const std::size_t N) {

    // Recursion base case
    if (N == 1) {
        return f;
    }

    // Even indexes of `f`
    std::vector<Complex> f_even(N/2);
    for (std::size_t i { 0 }; i < N/2; i++) {
        f_even[i] = f[2*i];
    }

    // Odd indexes of `f`
    std::vector<Complex> f_odd(N/2);
    for (std::size_t i { 0 }; i < N/2; i++) {
        f_odd[i] = f[(2*i)+1];
    }

    // Recursive call
    std::vector<Complex> E = FFT_recursion(f_even, N/2);
    std::vector<Complex> O = FFT_recursion(f_odd, N/2);

    // Final Vector
    std::vector<Complex> F(N);

    // Computing answer
    for (std::size_t k { 0 }; k < (N/2); k++){
        double arg = -2.0 * M_PI * k / N;
        Complex Wkn = Complex(std::cos(arg), std::sin(arg));
        F[k]        = E[k] + Wkn * O[k];
        F[k+(N/2)]  = E[k] - Wkn * O[k];
    }
    
    return F;
}

std::vector<Complex> FFT(const std::vector<double>& f, const std::size_t N) {
    std::vector<Complex> f_complex(N);
    // Converts f to complex
    for (std::size_t i { 0 }; i < N; i++) {
        f_complex[i] = Complex(f[i], 0.0);
    }
    return FFT_recursion(f_complex, N);
}

#endif