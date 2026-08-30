#include <immintrin.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

constexpr int K = 100;
constexpr int KP = 104; // padded to multiple of 8
constexpr int DIM = 5;

alignas(64) float centroids[KP][DIM];
alignas(64) float epsilon[KP];
alignas(32) float query[DIM];

struct LookupResult {
    int best_idx;
    float best_dist;
    bool anomaly;
};

inline LookupResult centroid_lookup_avx(const float* q) {
    alignas(32) float distances[KP];

    for (int k = 0; k < KP; k += 8) {
        __m256 sum_sq = _mm256_setzero_ps();
        for (int d = 0; d < DIM; ++d) {
            __m256 c = _mm256_set_ps(
                centroids[k + 7][d], centroids[k + 6][d], centroids[k + 5][d], centroids[k + 4][d],
                centroids[k + 3][d], centroids[k + 2][d], centroids[k + 1][d], centroids[k + 0][d]
            );
            __m256 qv = _mm256_set1_ps(q[d]);
            __m256 diff = _mm256_sub_ps(c, qv);
            sum_sq = _mm256_fmadd_ps(diff, diff, sum_sq);
        }
        _mm256_store_ps(distances + k, sum_sq);
    }

    int best_idx = 0;
    float best_dist_sq = distances[0];
    for (int k = 1; k < K; ++k) {
        if (distances[k] < best_dist_sq) {
            best_dist_sq = distances[k];
            best_idx = k;
        }
    }

    float best_dist = std::sqrt(best_dist_sq);
    bool anomaly = best_dist > epsilon[best_idx];

    return {best_idx, best_dist, anomaly};
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> eps_dist(0.1f, 2.0f);

    std::memset(centroids, 0, sizeof(centroids));
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < DIM; ++d) {
            centroids[k][d] = dist(rng);
        }
        epsilon[k] = eps_dist(rng);
    }
    for (int k = K; k < KP; ++k) {
        for (int d = 0; d < DIM; ++d) {
            centroids[k][d] = 1e6f; // push padding centroids far away, never selected as argmin
        }
        epsilon[k] = 0.0f;
    }

    for (int d = 0; d < DIM; ++d) {
        query[d] = dist(rng);
    }

    const int n_warmup = 1000;
    const int n_trials = 1000000;

    for (int i = 0; i < n_warmup; ++i) {
        centroid_lookup_avx(query);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        LookupResult r = centroid_lookup_avx(query);
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

    LookupResult final_result = centroid_lookup_avx(query);

    printf("Centroid lookup (K=%d): mean=%.4f us  p50=%.4f us  p99=%.4f us\n", K, mean, p50, p99);
    printf("best_idx=%d  best_dist=%.4f  anomaly=%d (sanity check)\n",
           final_result.best_idx, final_result.best_dist, final_result.anomaly);

    return 0;
}