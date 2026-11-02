#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cufft.h>
#include <type_traits>

#include "util.cuh"

// --- KERNELS --- //
//
// Main line (each one adds a single improvement to the previous one):
//   FFT_Naive -> FFT_Stockham -> FFT_Shared -> FFT_BCF -> FFT_Radix4
// Every main-line kernel has a twin FFT_*_Tabled that reads the twiddles from a table
// instead of computing them with sin/cos.

// CUFFT KERNEL: BASELINE
template <typename T, typename T2>
__host__ void cuFFT(T2* d_X, int N, FFTContext& ctx) {
    constexpr bool is_float = std::is_same_v<T, float>;

    // Execution
    cufftResult execResult;
    if constexpr (is_float) // FP32
        execResult = cufftExecC2C(ctx.plan, d_X, d_X, CUFFT_FORWARD);

    else // FP64
        execResult = cufftExecZ2Z(ctx.plan, d_X, d_X, CUFFT_FORWARD);

    if (execResult != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Forward execution failed\n");
        return;
    }

    // Sync
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    return;
}

__device__ int reverseBits(int i, int log2n) {
    int rev = 0;
    for (int b = 0; b < log2n; b++) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
    }
    return rev;
}

template <typename T, typename T2>
__global__ void bitReversal(T2* d_X, int N, int log2n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    int j = reverseBits(i, log2n);
    if (i < j) {
        T2 temp = d_X[i];
        d_X[i] = d_X[j];
        d_X[j] = temp;
    }
}


// ============================================================================
// FFT_Naive: in place, bit reversal + one global kernel per stage
// ============================================================================
template <typename T, typename T2>
__global__ void FFT_Naive_butterfly(T2* d_X, int N, int len) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int block = b / (len/2);
    int k = b % (len/2);
    int i = block * len;

    T arg = (T) -2.0f * PI<T> * k / len;
    T2 W = make_vec2(cos(arg), sin(arg));

    T2 U = d_X[i+k];
    T2 V = cmul(d_X[i+k+(len/2)], W);

    d_X[i+k]         = cadd(U, V);
    d_X[i+k+(len/2)] = csub(U, V);
}

template <typename T, typename T2>
__host__ void FFT_Naive(T2 *d_X, int N, FFTContext& ctx) {
    int log2n = 0;
    while ((1u << log2n) < N) log2n++;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    bitReversal<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, log2n);

    blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    for (int len = 2; len <= N; len <<= 1) {
        FFT_Naive_butterfly<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, len);
    }
}

template <typename T, typename T2>
__global__ void FFT_Naive_Tabled_butterfly(T2* d_X, int N, int len, const T2* __restrict__ table) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int block = b / (len/2);
    int k = b % (len/2);
    int i = block * len;

    T2 W = table[k * (N / len)];

    T2 U = d_X[i+k];
    T2 V = cmul(d_X[i+k+(len/2)], W);

    d_X[i+k]         = cadd(U, V);
    d_X[i+k+(len/2)] = csub(U, V);
}

template <typename T, typename T2>
__host__ void FFT_Naive_Tabled(T2 *d_X, int N, FFTContext& ctx) {
    const T2* table = static_cast<const T2*>(ctx.twiddles);

    int log2n = 0;
    while ((1u << log2n) < N) log2n++;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    bitReversal<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, log2n);

    blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    for (int len = 2; len <= N; len <<= 1) {
        FFT_Naive_Tabled_butterfly<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_X, N, len, table);
    }
}


// ============================================================================
// FFT_Stockham: out of place, ping-pong between two buffers, no bit reversal
// ============================================================================
template <typename T, typename T2>
__global__ void FFT_Stockham_stage(T2 *d_in, T2 *d_out, int N, int len){

    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int stride = N / len;
    int s = b / stride;
    int subproblem = b % stride;

    T arg = (T) -2.0f * PI<T> * s / len;
    T2 W = make_vec2(cos(arg), sin(arg));

    T2 U = d_in[subproblem + stride*(2*s)];
    T2 V = cmul(d_in[subproblem + stride*(2*s + 1)], W);

    d_out[b]       = cadd(U, V);
    d_out[b + N/2] = csub(U, V);
}

template <typename T, typename T2>
__host__ void FFT_Stockham(T2 *d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);
    size_t size = N * sizeof(T2);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    bool in_A = true;
    for (int len = 2; len <= N; len <<= 1) {
        if (in_A) FFT_Stockham_stage<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, len);
        else      FFT_Stockham_stage<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_B, d_A, N, len);
        in_A = !in_A;
    }

    if (!in_A){
        cudaMemcpy(d_A, d_B, size, cudaMemcpyDeviceToDevice);
    }
}

template <typename T, typename T2>
__global__ void FFT_Stockham_Tabled_stage(T2 *d_in, T2 *d_out, int N, int len, const T2* __restrict__ table){

    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= N/2) return;
    int stride = N / len;
    int s = b / stride;
    int subproblem = b % stride;

    T2 W = table[s * (N / len)];

    T2 U = d_in[subproblem + stride*(2*s)];
    T2 V = cmul(d_in[subproblem + stride*(2*s + 1)], W);

    d_out[b]       = cadd(U, V);
    d_out[b + N/2] = csub(U, V);
}

template <typename T, typename T2>
__host__ void FFT_Stockham_Tabled(T2 *d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);
    const T2* table = static_cast<const T2*>(ctx.twiddles);
    size_t size = N * sizeof(T2);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    bool in_A = true;
    for (int len = 2; len <= N; len <<= 1) {
        if (in_A) FFT_Stockham_Tabled_stage<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, len, table);
        else      FFT_Stockham_Tabled_stage<T, T2><<<blocksPerGrid, threadsPerBlock>>>(d_B, d_A, N, len, table);
        in_A = !in_A;
    }

    if (!in_A){
        cudaMemcpy(d_A, d_B, size, cudaMemcpyDeviceToDevice);
    }
}


// ============================================================================
// FFT_Shared: Stockham + shared memory, all stages grouped into passes
//
// After the Stockham stage of length L, position k*(N/L) + sub holds output k of the
// L-point DFT of the subsequence { x[sub + (N/L)*m] }. A pass takes the data after
// stage L and runs the next log2(R) stages at once. Its butterflies split into N/R
// independent groups g = kl*S + sub (kl in [0, L), sub in [0, S), S = N/(L*R)):
//   reads  the R elements  in[kl*(N/L) + sub + j*S],  j in [0, R)
//   writes the R elements out[g + t*(N/R)],            t in [0, R)
// Inside a group the log2(R) stages are a Stockham FFT of size R whose twiddles carry
// the group offset kl. Each block takes W adjacent groups (W*sizeof(T2) = one 32 B
// sector, so the strided accesses still use whole sectors), runs the stages
// ping-ponging between two shared-memory buffers and writes the result back.
// log2(N) stages are split evenly into ceil(log2(N) / log2(CHUNK)) passes, so the data
// crosses global memory once per pass instead of once per stage.
//
// FFT_Shared_Tabled is the same with the twiddles read from the table:
// W_len^s = W_N^(s*N/len) = table[s*(N/len)].
// ============================================================================
constexpr int STOCKHAM_SHARED_CHUNK = 512;     // max group size R

// Block layout: threadIdx.x = w (group inside the block), threadIdx.y walks the group.
template <typename T, typename T2>
__global__ void FFT_Shared_pass(T2* d_in, T2* d_out, int logN, int logL, int logR) {
    extern __shared__ __align__(16) unsigned char sharedMem[];
    int W = blockDim.x;
    int w = threadIdx.x;
    int R = 1 << logR;
    T2* in  = reinterpret_cast<T2*>(sharedMem);  // layout: buf[q * W + w]
    T2* out = in + R * W;

    int logS = logN - logL - logR;
    int g    = blockIdx.x * W + w;              // this thread's group
    int kl   = g >> logS;
    int sub  = g & ((1 << logS) - 1);

    // Load: group element j comes from in[kl*(N/L) + sub + j*S]
    const T2* src = d_in + (kl << (logN - logL)) + sub;
    for (int j = threadIdx.y; j < R; j += blockDim.y)
        in[j * W + w] = src[j << logS];
    __syncthreads();

    // Local Stockham stages; the global stage length is L * 2^logLen
    for (int logLen = 1; logLen <= logR; logLen++) {
        int logStride = logR - logLen;
        for (int b = threadIdx.y; b < R / 2; b += blockDim.y) {
            int s  = b >> logStride;
            int jj = b & ((1 << logStride) - 1);

            int sGlobal = kl + (s << logL);
            T arg = (T) -2.0f * PI<T> * sGlobal / (1 << (logL + logLen));
            T2 tw = make_vec2(cos(arg), sin(arg));

            T2 U = in[(jj + ((2 * s) << logStride)) * W + w];
            T2 V = cmul(in[(jj + ((2 * s + 1) << logStride)) * W + w], tw);

            out[b * W + w]           = cadd(U, V);
            out[(b + R / 2) * W + w] = csub(U, V);
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
    }

    // Store: local position t goes to out[g + t*(N/R)]
    T2* dst = d_out + g;
    for (int t = threadIdx.y; t < R; t += blockDim.y)
        dst[t << (logN - logR)] = in[t * W + w];
}

// Assumes N is a power of two and N >= 2.
template <typename T, typename T2>
__host__ void FFT_Shared(T2* d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);

    int logN = 0, logC = 0;
    while ((1 << logN) < N) logN++;
    while ((1 << logC) < STOCKHAM_SHARED_CHUNK) logC++;
    int passes  = (logN + logC - 1) / logC;
    int threads = 256;

    T2* src = d_A;
    T2* dst = d_B;
    int logL = 0;
    for (int p = 0; p < passes; p++) {
        int logR = (logN - logL) / (passes - p);   // spread the remaining stages evenly
        int R = 1 << logR;
        int W = 32 / (int)sizeof(T2);              // groups per block: one 32 B sector
        if (W > N / R) W = N / R;
        int rows = (threads / W < R / 2) ? threads / W : R / 2;
        size_t sharedBytes = 2 * (size_t)R * W * sizeof(T2);

        FFT_Shared_pass<T, T2><<<(N / R) / W, dim3(W, rows), sharedBytes>>>(src, dst, logN, logL, logR);

        T2* tmp = src; src = dst; dst = tmp;
        logL += logR;
    }

    if (src != d_A) {
        cudaMemcpy(d_A, src, (size_t)N * sizeof(T2), cudaMemcpyDeviceToDevice);
    }
}

// Same as FFT_Shared_pass, but the twiddles are read from the table (W_N^m = table[m]).
// Block layout: threadIdx.x = w (group inside the block), threadIdx.y walks the group.
template <typename T, typename T2>
__global__ void FFT_Shared_Tabled_pass(T2* d_in, T2* d_out, int logN, int logL, int logR, const T2* __restrict__ table) {
    extern __shared__ __align__(16) unsigned char sharedMem[];
    int W = blockDim.x;
    int w = threadIdx.x;
    int R = 1 << logR;
    T2* in  = reinterpret_cast<T2*>(sharedMem);  // layout: buf[q * W + w]
    T2* out = in + R * W;

    int logS = logN - logL - logR;
    int g    = blockIdx.x * W + w;              // this thread's group
    int kl   = g >> logS;
    int sub  = g & ((1 << logS) - 1);

    // Load: group element j comes from in[kl*(N/L) + sub + j*S]
    const T2* src = d_in + (kl << (logN - logL)) + sub;
    for (int j = threadIdx.y; j < R; j += blockDim.y)
        in[j * W + w] = src[j << logS];
    __syncthreads();

    // Local Stockham stages; the global stage length is L * 2^logLen
    for (int logLen = 1; logLen <= logR; logLen++) {
        int logStride = logR - logLen;
        for (int b = threadIdx.y; b < R / 2; b += blockDim.y) {
            int s  = b >> logStride;
            int jj = b & ((1 << logStride) - 1);

            int sGlobal = kl + (s << logL);
            T2 tw = table[sGlobal << (logN - logL - logLen)];    // W_len^s = W_N^(s*N/len)

            T2 U = in[(jj + ((2 * s) << logStride)) * W + w];
            T2 V = cmul(in[(jj + ((2 * s + 1) << logStride)) * W + w], tw);

            out[b * W + w]           = cadd(U, V);
            out[(b + R / 2) * W + w] = csub(U, V);
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
    }

    // Store: local position t goes to out[g + t*(N/R)]
    T2* dst = d_out + g;
    for (int t = threadIdx.y; t < R; t += blockDim.y)
        dst[t << (logN - logR)] = in[t * W + w];
}

// Assumes N is a power of two and N >= 2.
template <typename T, typename T2>
__host__ void FFT_Shared_Tabled(T2* d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);
    const T2* table = static_cast<const T2*>(ctx.twiddles);

    int logN = 0, logC = 0;
    while ((1 << logN) < N) logN++;
    while ((1 << logC) < STOCKHAM_SHARED_CHUNK) logC++;
    int passes  = (logN + logC - 1) / logC;
    int threads = 256;

    T2* src = d_A;
    T2* dst = d_B;
    int logL = 0;
    for (int p = 0; p < passes; p++) {
        int logR = (logN - logL) / (passes - p);   // spread the remaining stages evenly
        int R = 1 << logR;
        int W = 32 / (int)sizeof(T2);              // groups per block: one 32 B sector
        if (W > N / R) W = N / R;
        int rows = (threads / W < R / 2) ? threads / W : R / 2;
        size_t sharedBytes = 2 * (size_t)R * W * sizeof(T2);

        FFT_Shared_Tabled_pass<T, T2><<<(N / R) / W, dim3(W, rows), sharedBytes>>>(src, dst, logN, logL, logR, table);

        T2* tmp = src; src = dst; dst = tmp;
        logL += logR;
    }

    if (src != d_A) {
        cudaMemcpy(d_A, src, (size_t)N * sizeof(T2), cudaMemcpyDeviceToDevice);
    }
}


// ============================================================================
// RADIX-4 building blocks (shared by FFT_Radix4 and FFT_Radix4_Tabled)
// ============================================================================
// 4-point combine of b0 = x0, b1, b2, b3 (b1..b3 already twiddled)
template <typename T2>
__device__ __forceinline__ void radix4Combine(T2& x0, T2& x1, T2& x2, T2& x3, T2 b1, T2 b2, T2 b3) {
    T2 b0  = x0;
    T2 s01 = cadd(b0, b1), d01 = csub(b0, b1);
    T2 s23 = cadd(b2, b3), d23 = csub(b2, b3);
    x0 = cadd(s01, s23);
    x2 = csub(s01, s23);
    x1 = make_vec2(d01.x + d23.y, d01.y - d23.x);    // d01 - i*d23
    x3 = make_vec2(d01.x - d23.y, d01.y + d23.x);    // d01 + i*d23
}

template <typename T2>
__device__ __forceinline__ void radix4(T2& x0, T2& x1, T2& x2, T2& x3, T2 Wk, T2 W2k, T2 W3k) {
    radix4Combine(x0, x1, x2, x3, cmul(x1, W2k), cmul(x2, Wk), cmul(x3, W3k));
}

// Twiddles of one radix-4 pass of size S, k in [0, S/4): W_S^k, W_S^2k, W_S^3k, computed with sin/cos
template <typename T, typename T2>
__device__ __forceinline__ void r4TwiddlesTrig(int S, int k, T2& Wk, T2& W2k, T2& W3k) {
    T arg = (T) -2.0f * PI<T> * k / S;
    Wk  = make_vec2(cos(arg),           sin(arg));
    W2k = make_vec2(cos((T) 2 * arg),   sin((T) 2 * arg));
    W3k = make_vec2(cos((T) 3 * arg),   sin((T) 3 * arg));
}

// Same twiddles read from the table. The table holds W_N^m only for m < N/2;
// W_N^(m) = -W_N^(m - N/2) covers 3k*(N/S) >= N/2.
template <typename T, typename T2>
__device__ __forceinline__ void r4TwiddlesTable(const T2* __restrict__ table, int N, int S, int k,
                                                T2& Wk, T2& W2k, T2& W3k) {
    const int m     = k * (N / S);
    const int halfN = N >> 1;
    Wk  = table[m];
    W2k = table[2 * m];
    const int m3 = 3 * m;
    if (m3 < halfN) {
        W3k = table[m3];
    } else {
        T2 t = table[m3 - halfN];
        W3k = make_vec2(-t.x, -t.y);
    }
}


// ============================================================================
// FFT_BCF: FFT_Shared + bank-conflict-free (BCF) shared-memory layout
// FFT_Radix4: FFT_BCF + radix-4
//
//  - BCF: FFT_BCF_pass is FFT_Shared_pass with element q of a group stored at slot
//    bcfSwz(q) = q ^ ((q >> 2) & 3); nothing else changes. A slot is W groups wide
//    (32 B = 8 banks), so a quarter/half warp touching 4 consecutive butterflies must hit
//    4 different slots mod 4. The plain layout breaks this in the last stages (small
//    stride). It is a permutation of [0, R): no extra memory.
//  - RADIX4: FFT_Radix4_pass is FFT_BCF_pass with the local stages fused two at a time
//    (len, 2*len). With A_r[k] = in[sub + S2*(4k + r)], S2 = R/(2*len), k in [0, len/2)
//    and w = W_{2 len}^k (global length and index, as in the radix-2 stages):
//      Z[k]          = (A0 + w^2 A2) + (w A1 + w^3 A3)
//      Z[k + len]    = (A0 + w^2 A2) - (w A1 + w^3 A3)
//      Z[k + len/2]  = (A0 - w^2 A2) - i (w A1 - w^3 A3)
//      Z[k + 3len/2] = (A0 - w^2 A2) + i (w A1 - w^3 A3)
//    and Z[k + j*len/2] is written at out[b + j*R/4], b = sub + S2*k.
//    That is radix4(A0, A2, A1, A3, w, w^2, w^3). If log2(R) is odd, one radix-2 stage
//    is done first.
//
// The _Tabled twins only replace sin/cos by table reads (r4TwiddlesTable for radix-4).
// ============================================================================
__device__ __forceinline__ int bcfSwz(int q) { return q ^ ((q >> 2) & 3); }

// Same as FFT_Shared_pass; the only change is slot bcfSwz(q) for every shared index q.
template <typename T, typename T2>
__global__ void FFT_BCF_pass(T2* d_in, T2* d_out, int logN, int logL, int logR) {
    extern __shared__ __align__(16) unsigned char sharedMem[];
    int W = blockDim.x;
    int w = threadIdx.x;
    int R = 1 << logR;
    T2* in  = reinterpret_cast<T2*>(sharedMem);  // layout: buf[bcfSwz(q) * W + w]
    T2* out = in + R * W;

    int logS = logN - logL - logR;
    int g    = blockIdx.x * W + w;              // this thread's group
    int kl   = g >> logS;
    int sub  = g & ((1 << logS) - 1);

    // Load: group element j comes from in[kl*(N/L) + sub + j*S]
    const T2* src = d_in + (kl << (logN - logL)) + sub;
    for (int j = threadIdx.y; j < R; j += blockDim.y)
        in[bcfSwz(j) * W + w] = src[j << logS];
    __syncthreads();

    // Local Stockham stages; the global stage length is L * 2^logLen
    for (int logLen = 1; logLen <= logR; logLen++) {
        int logStride = logR - logLen;
        for (int b = threadIdx.y; b < R / 2; b += blockDim.y) {
            int s  = b >> logStride;
            int jj = b & ((1 << logStride) - 1);

            int sGlobal = kl + (s << logL);
            T arg = (T) -2.0f * PI<T> * sGlobal / (1 << (logL + logLen));
            T2 tw = make_vec2(cos(arg), sin(arg));

            T2 U = in[bcfSwz(jj + ((2 * s) << logStride)) * W + w];
            T2 V = cmul(in[bcfSwz(jj + ((2 * s + 1) << logStride)) * W + w], tw);

            out[bcfSwz(b) * W + w]         = cadd(U, V);
            out[bcfSwz(b + R / 2) * W + w] = csub(U, V);
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
    }

    // Store: local position t goes to out[g + t*(N/R)]
    T2* dst = d_out + g;
    for (int t = threadIdx.y; t < R; t += blockDim.y)
        dst[t << (logN - logR)] = in[bcfSwz(t) * W + w];
}

// Same pass schedule as FFT_Shared, launching FFT_BCF_pass.
// Assumes N is a power of two and N >= 2.
template <typename T, typename T2>
__host__ void FFT_BCF(T2* d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);

    int logN = 0, logC = 0;
    while ((1 << logN) < N) logN++;
    while ((1 << logC) < STOCKHAM_SHARED_CHUNK) logC++;
    int passes  = (logN + logC - 1) / logC;
    int threads = 256;

    T2* src = d_A;
    T2* dst = d_B;
    int logL = 0;
    for (int p = 0; p < passes; p++) {
        int logR = (logN - logL) / (passes - p);   // spread the remaining stages evenly
        int R = 1 << logR;
        int W = 32 / (int)sizeof(T2);              // groups per block: one 32 B sector
        if (W > N / R) W = N / R;
        int rows = (threads / W < R / 2) ? threads / W : R / 2;
        size_t sharedBytes = 2 * (size_t)R * W * sizeof(T2);

        FFT_BCF_pass<T, T2><<<(N / R) / W, dim3(W, rows), sharedBytes>>>(src, dst, logN, logL, logR);

        T2* tmp = src; src = dst; dst = tmp;
        logL += logR;
    }

    if (src != d_A) {
        cudaMemcpy(d_A, src, (size_t)N * sizeof(T2), cudaMemcpyDeviceToDevice);
    }
}
// Same as FFT_BCF_pass, but the twiddles are read from the table (W_N^m = table[m]).
template <typename T, typename T2>
__global__ void FFT_BCF_Tabled_pass(T2* d_in, T2* d_out, int logN, int logL, int logR, const T2* __restrict__ table) {
    extern __shared__ __align__(16) unsigned char sharedMem[];
    int W = blockDim.x;
    int w = threadIdx.x;
    int R = 1 << logR;
    T2* in  = reinterpret_cast<T2*>(sharedMem);  // layout: buf[bcfSwz(q) * W + w]
    T2* out = in + R * W;

    int logS = logN - logL - logR;
    int g    = blockIdx.x * W + w;              // this thread's group
    int kl   = g >> logS;
    int sub  = g & ((1 << logS) - 1);

    // Load: group element j comes from in[kl*(N/L) + sub + j*S]
    const T2* src = d_in + (kl << (logN - logL)) + sub;
    for (int j = threadIdx.y; j < R; j += blockDim.y)
        in[bcfSwz(j) * W + w] = src[j << logS];
    __syncthreads();

    // Local Stockham stages; the global stage length is L * 2^logLen
    for (int logLen = 1; logLen <= logR; logLen++) {
        int logStride = logR - logLen;
        for (int b = threadIdx.y; b < R / 2; b += blockDim.y) {
            int s  = b >> logStride;
            int jj = b & ((1 << logStride) - 1);

            int sGlobal = kl + (s << logL);
            T2 tw = table[sGlobal << (logN - logL - logLen)];    // W_len^s = W_N^(s*N/len)

            T2 U = in[bcfSwz(jj + ((2 * s) << logStride)) * W + w];
            T2 V = cmul(in[bcfSwz(jj + ((2 * s + 1) << logStride)) * W + w], tw);

            out[bcfSwz(b) * W + w]         = cadd(U, V);
            out[bcfSwz(b + R / 2) * W + w] = csub(U, V);
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
    }

    // Store: local position t goes to out[g + t*(N/R)]
    T2* dst = d_out + g;
    for (int t = threadIdx.y; t < R; t += blockDim.y)
        dst[t << (logN - logR)] = in[bcfSwz(t) * W + w];
}

// Same pass schedule as FFT_Shared, launching FFT_BCF_Tabled_pass.
// Assumes N is a power of two and N >= 2.
template <typename T, typename T2>
__host__ void FFT_BCF_Tabled(T2* d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);
    const T2* table = static_cast<const T2*>(ctx.twiddles);

    int logN = 0, logC = 0;
    while ((1 << logN) < N) logN++;
    while ((1 << logC) < STOCKHAM_SHARED_CHUNK) logC++;
    int passes  = (logN + logC - 1) / logC;
    int threads = 256;

    T2* src = d_A;
    T2* dst = d_B;
    int logL = 0;
    for (int p = 0; p < passes; p++) {
        int logR = (logN - logL) / (passes - p);   // spread the remaining stages evenly
        int R = 1 << logR;
        int W = 32 / (int)sizeof(T2);              // groups per block: one 32 B sector
        if (W > N / R) W = N / R;
        int rows = (threads / W < R / 2) ? threads / W : R / 2;
        size_t sharedBytes = 2 * (size_t)R * W * sizeof(T2);

        FFT_BCF_Tabled_pass<T, T2><<<(N / R) / W, dim3(W, rows), sharedBytes>>>(src, dst, logN, logL, logR, table);

        T2* tmp = src; src = dst; dst = tmp;
        logL += logR;
    }

    if (src != d_A) {
        cudaMemcpy(d_A, src, (size_t)N * sizeof(T2), cudaMemcpyDeviceToDevice);
    }
}

// Same as FFT_BCF_pass, but the local stages run two at a time as radix-4 butterflies
// (one radix-2 stage first when log2(R) is odd).
template <typename T, typename T2>
__global__ void FFT_Radix4_pass(T2* d_in, T2* d_out, int logN, int logL, int logR) {
    extern __shared__ __align__(16) unsigned char sharedMem[];
    int W = blockDim.x;
    int w = threadIdx.x;
    int R = 1 << logR;
    T2* in  = reinterpret_cast<T2*>(sharedMem);  // layout: buf[bcfSwz(q) * W + w]
    T2* out = in + R * W;

    int logS = logN - logL - logR;
    int g    = blockIdx.x * W + w;              // this thread's group
    int kl   = g >> logS;
    int sub  = g & ((1 << logS) - 1);

    // Load: group element j comes from in[kl*(N/L) + sub + j*S]
    const T2* src = d_in + (kl << (logN - logL)) + sub;
    for (int j = threadIdx.y; j < R; j += blockDim.y)
        in[bcfSwz(j) * W + w] = src[j << logS];
    __syncthreads();

    // Odd stage count: one radix-2 stage (local length 2) first, as in FFT_BCF_pass
    int logLen = 1;
    if (logR & 1) {
        int logStride = logR - 1;
        for (int b = threadIdx.y; b < R / 2; b += blockDim.y) {
            int s  = b >> logStride;
            int jj = b & ((1 << logStride) - 1);

            int sGlobal = kl + (s << logL);
            T arg = (T) -2.0f * PI<T> * sGlobal / (1 << (logL + 1));
            T2 tw = make_vec2(cos(arg), sin(arg));

            T2 U = in[bcfSwz(jj + ((2 * s) << logStride)) * W + w];
            T2 V = cmul(in[bcfSwz(jj + ((2 * s + 1) << logStride)) * W + w], tw);

            out[bcfSwz(b) * W + w]         = cadd(U, V);
            out[bcfSwz(b + R / 2) * W + w] = csub(U, V);
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
        logLen = 2;
    }

    // Radix-4: fuses the local stages of length 2^logLen and 2^(logLen+1)
    for (; logLen < logR; logLen += 2) {
        int logS2 = logR - logLen - 1;          // local S2 = R / (2 * len)
        for (int b = threadIdx.y; b < R / 4; b += blockDim.y) {
            int s  = b >> logS2;
            int jj = b & ((1 << logS2) - 1);

            T2 Wk, W2k, W3k;
            r4TwiddlesTrig<T, T2>(1 << (logL + logLen + 1), kl + (s << logL), Wk, W2k, W3k);

            T2 x0 = in[bcfSwz(jj + ((4 * s)     << logS2)) * W + w];
            T2 x1 = in[bcfSwz(jj + ((4 * s + 2) << logS2)) * W + w];
            T2 x2 = in[bcfSwz(jj + ((4 * s + 1) << logS2)) * W + w];
            T2 x3 = in[bcfSwz(jj + ((4 * s + 3) << logS2)) * W + w];
            radix4(x0, x1, x2, x3, Wk, W2k, W3k);

            out[bcfSwz(b)               * W + w] = x0;
            out[bcfSwz(b + R / 4)       * W + w] = x1;
            out[bcfSwz(b + 2 * (R / 4)) * W + w] = x2;
            out[bcfSwz(b + 3 * (R / 4)) * W + w] = x3;
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
    }

    // Store: local position t goes to out[g + t*(N/R)]
    T2* dst = d_out + g;
    for (int t = threadIdx.y; t < R; t += blockDim.y)
        dst[t << (logN - logR)] = in[bcfSwz(t) * W + w];
}

// Same pass schedule as FFT_Shared and FFT_BCF, launching FFT_Radix4_pass.
// Assumes N is a power of two and N >= 2.
template <typename T, typename T2>
__host__ void FFT_Radix4(T2* d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);

    int logN = 0, logC = 0;
    while ((1 << logN) < N) logN++;
    while ((1 << logC) < STOCKHAM_SHARED_CHUNK) logC++;
    int passes  = (logN + logC - 1) / logC;
    int threads = 256;

    T2* src = d_A;
    T2* dst = d_B;
    int logL = 0;
    for (int p = 0; p < passes; p++) {
        int logR = (logN - logL) / (passes - p);   // spread the remaining stages evenly
        int R = 1 << logR;
        int W = 32 / (int)sizeof(T2);              // groups per block: one 32 B sector
        if (W > N / R) W = N / R;
        int rows = (threads / W < R / 2) ? threads / W : R / 2;
        size_t sharedBytes = 2 * (size_t)R * W * sizeof(T2);

        FFT_Radix4_pass<T, T2><<<(N / R) / W, dim3(W, rows), sharedBytes>>>(src, dst, logN, logL, logR);

        T2* tmp = src; src = dst; dst = tmp;
        logL += logR;
    }

    if (src != d_A) {
        cudaMemcpy(d_A, src, (size_t)N * sizeof(T2), cudaMemcpyDeviceToDevice);
    }
}
// Same as FFT_Radix4_pass, but the twiddles are read from the table (W_N^m = table[m]).
template <typename T, typename T2>
__global__ void FFT_Radix4_Tabled_pass(T2* d_in, T2* d_out, int logN, int logL, int logR, const T2* __restrict__ table) {
    extern __shared__ __align__(16) unsigned char sharedMem[];
    int W = blockDim.x;
    int w = threadIdx.x;
    int R = 1 << logR;
    T2* in  = reinterpret_cast<T2*>(sharedMem);  // layout: buf[bcfSwz(q) * W + w]
    T2* out = in + R * W;

    int logS = logN - logL - logR;
    int g    = blockIdx.x * W + w;              // this thread's group
    int kl   = g >> logS;
    int sub  = g & ((1 << logS) - 1);

    // Load: group element j comes from in[kl*(N/L) + sub + j*S]
    const T2* src = d_in + (kl << (logN - logL)) + sub;
    for (int j = threadIdx.y; j < R; j += blockDim.y)
        in[bcfSwz(j) * W + w] = src[j << logS];
    __syncthreads();

    // Odd stage count: one radix-2 stage (local length 2) first, as in FFT_BCF_pass
    int logLen = 1;
    if (logR & 1) {
        int logStride = logR - 1;
        for (int b = threadIdx.y; b < R / 2; b += blockDim.y) {
            int s  = b >> logStride;
            int jj = b & ((1 << logStride) - 1);

            int sGlobal = kl + (s << logL);
            T2 tw = table[sGlobal << (logN - logL - 1)];         // W_2L^s = W_N^(s*N/(2L))

            T2 U = in[bcfSwz(jj + ((2 * s) << logStride)) * W + w];
            T2 V = cmul(in[bcfSwz(jj + ((2 * s + 1) << logStride)) * W + w], tw);

            out[bcfSwz(b) * W + w]         = cadd(U, V);
            out[bcfSwz(b + R / 2) * W + w] = csub(U, V);
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
        logLen = 2;
    }

    // Radix-4: fuses the local stages of length 2^logLen and 2^(logLen+1)
    for (; logLen < logR; logLen += 2) {
        int logS2 = logR - logLen - 1;          // local S2 = R / (2 * len)
        for (int b = threadIdx.y; b < R / 4; b += blockDim.y) {
            int s  = b >> logS2;
            int jj = b & ((1 << logS2) - 1);

            T2 Wk, W2k, W3k;
            r4TwiddlesTable<T, T2>(table, 1 << logN, 1 << (logL + logLen + 1), kl + (s << logL), Wk, W2k, W3k);

            T2 x0 = in[bcfSwz(jj + ((4 * s)     << logS2)) * W + w];
            T2 x1 = in[bcfSwz(jj + ((4 * s + 2) << logS2)) * W + w];
            T2 x2 = in[bcfSwz(jj + ((4 * s + 1) << logS2)) * W + w];
            T2 x3 = in[bcfSwz(jj + ((4 * s + 3) << logS2)) * W + w];
            radix4(x0, x1, x2, x3, Wk, W2k, W3k);

            out[bcfSwz(b)               * W + w] = x0;
            out[bcfSwz(b + R / 4)       * W + w] = x1;
            out[bcfSwz(b + 2 * (R / 4)) * W + w] = x2;
            out[bcfSwz(b + 3 * (R / 4)) * W + w] = x3;
        }
        __syncthreads();
        T2* tmp = in; in = out; out = tmp;
    }

    // Store: local position t goes to out[g + t*(N/R)]
    T2* dst = d_out + g;
    for (int t = threadIdx.y; t < R; t += blockDim.y)
        dst[t << (logN - logR)] = in[bcfSwz(t) * W + w];
}

// Same pass schedule as FFT_Shared and FFT_BCF, launching FFT_Radix4_Tabled_pass.
// Assumes N is a power of two and N >= 2.
template <typename T, typename T2>
__host__ void FFT_Radix4_Tabled(T2* d_A, int N, FFTContext& ctx) {
    T2* d_B = static_cast<T2*>(ctx.workspace);
    const T2* table = static_cast<const T2*>(ctx.twiddles);

    int logN = 0, logC = 0;
    while ((1 << logN) < N) logN++;
    while ((1 << logC) < STOCKHAM_SHARED_CHUNK) logC++;
    int passes  = (logN + logC - 1) / logC;
    int threads = 256;

    T2* src = d_A;
    T2* dst = d_B;
    int logL = 0;
    for (int p = 0; p < passes; p++) {
        int logR = (logN - logL) / (passes - p);   // spread the remaining stages evenly
        int R = 1 << logR;
        int W = 32 / (int)sizeof(T2);              // groups per block: one 32 B sector
        if (W > N / R) W = N / R;
        int rows = (threads / W < R / 2) ? threads / W : R / 2;
        size_t sharedBytes = 2 * (size_t)R * W * sizeof(T2);

        FFT_Radix4_Tabled_pass<T, T2><<<(N / R) / W, dim3(W, rows), sharedBytes>>>(src, dst, logN, logL, logR, table);

        T2* tmp = src; src = dst; dst = tmp;
        logL += logR;
    }

    if (src != d_A) {
        cudaMemcpy(d_A, src, (size_t)N * sizeof(T2), cudaMemcpyDeviceToDevice);
    }
}


#endif
