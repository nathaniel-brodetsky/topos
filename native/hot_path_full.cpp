#include <immintrin.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

constexpr int N = 20;
constexpr int NP = 24;
constexpr int K = 100;
constexpr int KP = 104;
constexpr int DIM = 5;
constexpr float EPS_STABILIZER = 0.01f;

enum class Decision { CANCEL_ALL, MARKET_TAKER, LIQUIDITY_PROVISION, HOLD };

alignas(64) float dV[NP];
alignas(64) float P[NP];
alignas(64) float A[NP * NP];
alignas(64) float A_prev[NP * NP];
alignas(64) float dA[NP * NP];
alignas(64) float F[NP * NP];
alignas(64) float tmp1[NP * NP];
alignas(64) float tmp2[NP * NP];

alignas(64) float centroids[KP][DIM];
alignas(64) float epsilon[KP];
alignas(64) int centroid_regime_id[KP];

inline void build_adjacency_matrix_avx() {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff_v = dV[i] - dV[j];
            float diff_p = std::fabs(P[i] - P[j]) + EPS_STABILIZER;
            A[i * NP + j] = diff_v / diff_p;
        }
        for (int j = N; j < NP; ++j) A[i * NP + j] = 0.0f;
    }
    for (int i = N; i < NP; ++i)
        for (int j = 0; j < NP; ++j) A[i * NP + j] = 0.0f;
}

inline void compute_delta(const float* a, const float* a_prev, float* out) {
    for (int i = 0; i < NP * NP; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vap = _mm256_load_ps(a_prev + i);
        _mm256_store_ps(out + i, _mm256_sub_ps(va, vap));
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

inline float compute_commutator_norm() {
    compute_delta(A, A_prev, dA);
    matmul_avx(A, dA, tmp1);
    matmul_avx(dA, A, tmp2);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < NP * NP; i += 8) {
        __m256 t1 = _mm256_load_ps(tmp1 + i);
        __m256 t2 = _mm256_load_ps(tmp2 + i);
        __m256 f = _mm256_sub_ps(t1, t2);
        _mm256_store_ps(F + i, f);
        acc = _mm256_fmadd_ps(f, f, acc);
    }

    alignas(32) float partial[8];
    _mm256_store_ps(partial, acc);
    float sum_sq = 0.0f;
    for (int i = 0; i < 8; ++i) sum_sq += partial[i];
    return std::sqrt(sum_sq);
}

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

inline Decision decide(int regime_id, bool anomaly_triggered) {
    if (anomaly_triggered) return Decision::CANCEL_ALL;
    if (regime_id == 1) return Decision::MARKET_TAKER;
    if (regime_id == 0) return Decision::LIQUIDITY_PROVISION;
    return Decision::HOLD;
}

struct FullTickResult {
    float commutator_norm;
    LookupResult regime_lookup;
    Decision decision;
};

inline FullTickResult run_full_tick(const float* query_5d) {
    build_adjacency_matrix_avx();
    float f_norm = compute_commutator_norm();
    LookupResult lookup = centroid_lookup_avx(query_5d);
    Decision decision = decide(centroid_regime_id[lookup.best_idx], lookup.anomaly);
    return {f_norm, lookup, decision};
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> price_dist(99.0f, 101.0f);
    std::uniform_real_distribution<float> eps_dist(0.1f, 2.0f);
    std::uniform_int_distribution<int> regime_dist(0, 2);

    for (int i = 0; i < N; ++i) {
        dV[i] = dist(rng);
        P[i] = price_dist(rng);
    }
    for (int i = N; i < NP; ++i) { dV[i] = 0.0f; P[i] = 0.0f; }

    std::memset(A_prev, 0, sizeof(A_prev));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_prev[i * NP + j] = dist(rng);

    std::memset(centroids, 0, sizeof(centroids));
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < DIM; ++d) centroids[k][d] = dist(rng) * 10.0f;
        epsilon[k] = eps_dist(rng);
        centroid_regime_id[k] = regime_dist(rng);
    }
    for (int k = K; k < KP; ++k) {
        for (int d = 0; d < DIM; ++d) centroids[k][d] = 1e6f;
        epsilon[k] = 0.0f;
        centroid_regime_id[k] = -1;
    }

    alignas(32) float query_5d[DIM];
    for (int d = 0; d < DIM; ++d) query_5d[d] = dist(rng) * 10.0f;

    const int n_warmup = 1000;
    const int n_trials = 1000000;

    for (int i = 0; i < n_warmup; ++i) {
        run_full_tick(query_5d);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        FullTickResult r = run_full_tick(query_5d);
        auto end = std::chrono::high_resolution_clock::now();
        latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double mean = 0.0;
    for (double v : latencies_us) mean += v;
    mean /= latencies_us.size();

    double p50 = latencies_us[latencies_us.size() * 50 / 100];
    double p99 = latencies_us[latencies_us.size() * 99 / 100];
    double p999 = latencies_us[(size_t)(latencies_us.size() * 0.999)];

    FullTickResult final_r = run_full_tick(query_5d);

    const char* decision_names[] = {"CANCEL_ALL", "MARKET_TAKER", "LIQUIDITY_PROVISION", "HOLD"};

    printf("FULL tick-to-trade (Layer0+1+2+4, single measured call):\n");
    printf("  mean=%.4f us  p50=%.4f us  p99=%.4f us  p999=%.4f us\n", mean, p50, p99, p999);
    printf("  commutator_norm=%.4f  best_idx=%d  anomaly=%d  decision=%s (sanity check)\n",
           final_r.commutator_norm, final_r.regime_lookup.best_idx,
           final_r.regime_lookup.anomaly, decision_names[(int)final_r.decision]);

    return 0;
}
