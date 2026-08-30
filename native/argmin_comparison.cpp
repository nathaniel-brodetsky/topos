#include <immintrin.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

constexpr int K = 100;
constexpr int DIM = 5;

alignas(64) float centroids[K][DIM];
alignas(64) float distances[K];

struct ArgminResult {
    int idx;
    float val;
};

inline ArgminResult argmin_scalar(const float* dist, int n) {
    int best_idx = 0;
    float best_val = dist[0];
    for (int k = 1; k < n; ++k) {
        if (dist[k] < best_val) {
            best_val = dist[k];
            best_idx = k;
        }
    }
    return {best_idx, best_val};
}

inline ArgminResult argmin_avx2(const float* dist, int n) {
    __m256 min_vals = _mm256_loadu_ps(dist);
    __m256i min_idxs = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i idx_increment = _mm256_set1_epi32(8);
    __m256i current_idxs = min_idxs;

    int k = 8;
    for (; k + 8 <= n; k += 8) {
        current_idxs = _mm256_add_epi32(current_idxs, idx_increment);
        __m256 vals = _mm256_loadu_ps(dist + k);
        __m256 mask = _mm256_cmp_ps(vals, min_vals, _CMP_LT_OQ);
        min_vals = _mm256_blendv_ps(min_vals, vals, mask);
        min_idxs = _mm256_blendv_epi8(min_idxs, current_idxs, _mm256_castps_si256(mask));
    }

    alignas(32) float vals_arr[8];
    alignas(32) int idxs_arr[8];
    _mm256_store_ps(vals_arr, min_vals);
    _mm256_store_si256((__m256i*)idxs_arr, min_idxs);

    int best_idx = idxs_arr[0];
    float best_val = vals_arr[0];
    for (int i = 1; i < 8; ++i) {
        if (vals_arr[i] < best_val) {
            best_val = vals_arr[i];
            best_idx = idxs_arr[i];
        }
    }

    for (; k < n; ++k) {
        if (dist[k] < best_val) {
            best_val = dist[k];
            best_idx = k;
        }
    }

    return {best_idx, best_val};
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int k = 0; k < K; ++k) distances[k] = dist(rng);

    const int n_warmup = 1000;
    const int n_trials = 1000000;

    for (int i = 0; i < n_warmup; ++i) { argmin_scalar(distances, K); argmin_avx2(distances, K); }

    std::vector<double> scalar_times, avx2_times;
    scalar_times.reserve(n_trials);
    avx2_times.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        ArgminResult r = argmin_scalar(distances, K);
        auto end = std::chrono::high_resolution_clock::now();
        scalar_times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        asm volatile("" : : "r"(r.idx) : "memory");
    }

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        ArgminResult r = argmin_avx2(distances, K);
        auto end = std::chrono::high_resolution_clock::now();
        avx2_times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        asm volatile("" : : "r"(r.idx) : "memory");
    }

    auto stats = [](std::vector<double>& v, const char* name) {
        std::sort(v.begin(), v.end());
        double mean = 0.0;
        for (double x : v) mean += x;
        mean /= v.size();
        printf("%-12s mean=%.5f us  p50=%.5f us  p99=%.5f us\n", name, mean, v[v.size()*50/100], v[v.size()*99/100]);
    };

    stats(scalar_times, "scalar");
    stats(avx2_times, "avx2");

    ArgminResult r1 = argmin_scalar(distances, K);
    ArgminResult r2 = argmin_avx2(distances, K);
    printf("correctness check: scalar_idx=%d avx2_idx=%d (should match): %s\n",
           r1.idx, r2.idx, (r1.idx == r2.idx) ? "MATCH" : "MISMATCH - BUG");

    return 0;
}
