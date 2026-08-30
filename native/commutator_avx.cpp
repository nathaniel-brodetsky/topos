#include <immintrin.h>
#include <cstring>
#include <chrono>
#include <random>
#include <cstdio>
#include <vector>
#include <algorithm>


constexpr int N = 20;
constexpr int NP = 24; // padded to multiple of 8 for AVX2 alignment

alignas(64) float A[NP * NP];
alignas(64) float A_prev[NP * NP];
alignas(64) float dA[NP * NP];
alignas(64) float F[NP * NP];
alignas(64) float tmp1[NP * NP]; // A @ dA
alignas(64) float tmp2[NP * NP]; // dA @ A

inline void compute_delta(const float* a, const float* a_prev, float* out) {
    for (int i = 0; i < NP * NP; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vap = _mm256_load_ps(a_prev + i);
        __m256 vd = _mm256_sub_ps(va, vap);
        _mm256_store_ps(out + i, vd);
    }
}

inline void matmul_avx(const float* lhs, const float* rhs, float* out) {
    for (int i = 0; i < NP * NP; i += 8) {
        _mm256_store_ps(out + i, _mm256_setzero_ps());
    }
    for (int i = 0; i < NP; ++i) {
        for (int k = 0; k < NP; ++k) {
            __m256 lhs_ik = _mm256_set1_ps(lhs[i * NP + k]);
            for (int j = 0; j < NP; j += 8) {
                __m256 rhs_kj = _mm256_load_ps(rhs + k * NP + j);
                __m256 out_ij = _mm256_load_ps(out + i * NP + j);
                out_ij = _mm256_fmadd_ps(lhs_ik, rhs_kj, out_ij);
                _mm256_store_ps(out + i * NP + j, out_ij);
            }
        }
    }
}

inline void compute_commutator_avx() {
    compute_delta(A, A_prev, dA);
    matmul_avx(A, dA, tmp1);
    matmul_avx(dA, A, tmp2);
    for (int i = 0; i < NP * NP; i += 8) {
        __m256 t1 = _mm256_load_ps(tmp1 + i);
        __m256 t2 = _mm256_load_ps(tmp2 + i);
        __m256 f = _mm256_sub_ps(t1, t2);
        _mm256_store_ps(F + i, f);
    }
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::memset(A, 0, sizeof(A));
    std::memset(A_prev, 0, sizeof(A_prev));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * NP + j] = dist(rng);
            A_prev[i * NP + j] = dist(rng);
        }
    }

    const int n_warmup = 1000;
    const int n_trials = 100000;

    for (int i = 0; i < n_warmup; ++i) {
        compute_commutator_avx();
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        compute_commutator_avx();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(us);
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double mean = 0.0;
    for (double v : latencies_us) mean += v;
    mean /= latencies_us.size();

    double p50 = latencies_us[latencies_us.size() * 50 / 100];
    double p99 = latencies_us[latencies_us.size() * 99 / 100];

    printf("AVX2 commutator: mean=%.4f us  p50=%.4f us  p99=%.4f us\n", mean, p50, p99);
    printf("F[0][0]=%.6f (sanity check, non-zero expected)\n", F[0]);

    return 0;
}