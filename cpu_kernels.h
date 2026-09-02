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

std::vector<Complex> rec_FFT(const std::vector<double>& f) {
    // Converts to complex first
    std::vector<Complex> f_complex(f.begin(), f.end());
    return FFT_recursion(f_complex, f.size());
}

std::size_t reverse_bits(std::size_t i, int log2n) {
    // i is the number that is going to be reversed
    // log2n is the number of bits considered in the reversion
    std::size_t rev = 0;
    for (std::size_t b { 0 }; b < log2n; b++) {
        rev = (rev << 1) | (i & 1);
        i = i >> 1;
    }
    return rev;
}

std::vector<Complex> iter_FFT(const std::vector<double>& f_real) {

	std::size_t N = f_real.size();
	std::vector<Complex> f(f_real.begin(), f_real.end());
	int log2n = 0;
	while ((1u << log2n) < N) log2n++;

	// Reordena vetor com bit-reversal
	for (std::size_t i { 0 }; i < N; i++) {
		std::size_t j = reverse_bits(i, log2n);
		if (i < j) std::swap(f[i], f[j]);
	}	
	
	// Loop externo: representa cada nível de recursão
	for (std::size_t len { 2 }; len <= N; len <<= 1) {
		double arg = -2.0 * M_PI / len;
		Complex W_base(std::cos(arg), std::sin(arg));
		
		// Laço intermediário: delimita os blocos
		for (std::size_t i { 0 }; i < N; i += len) {
			Complex W (1.0, 0.0);	

			// Laço interno: percorre cada elemento do bloco
			for (std::size_t k { 0 }; k < len/2; k++) {
				
				// Borboletas
				Complex u = f[i+k];
				Complex v = f[i+k+(len/2)] * W;
				
				// Recombinando
				f[i+k]         = u + v;
				f[i+k+(len/2)] = u - v;
				
				W *= W_base;
			}
		}
	}

	return f;
}

#endif