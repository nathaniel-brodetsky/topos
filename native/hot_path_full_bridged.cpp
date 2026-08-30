#include <immintrin.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <sched.h>
#include <pthread.h>
#include "attractor_bridge.hpp"

constexpr int N = 20;
constexpr int NP = 24;
constexpr int K = 100;
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

inline LookupResult centroid_lookup_from_bridge(const AttractorSnapshot* snap, const float* q) {
    int n = snap->n_centroids;
    int best_idx = 0;
    float best_dist_sq = 1e18f;

    for (int k = 0; k < n; ++k) {
        float sum_sq = 0.0f;
        for (int d = 0; d < CENTROID_DIM; ++d) {
            float diff = snap->centroids[k][d] - q[d];
            sum_sq += diff * diff;
        }
        if (sum_sq < best_dist_sq) {
            best_dist_sq = sum_sq;
            best_idx = k;
        }
    }

    float best_dist = std::sqrt(best_dist_sq);
    bool anomaly = best_dist > snap->epsilon[best_idx];
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
    uint64_t snapshot_version_used;
};

inline FullTickResult run_full_tick(AttractorBridge& bridge, const float* query_5d) {
    build_adjacency_matrix_avx();
    float f_norm = compute_commutator_norm();

    const AttractorSnapshot* snap = bridge.current();
    LookupResult lookup = centroid_lookup_from_bridge(snap, query_5d);
    Decision decision = decide(snap->regime_id[lookup.best_idx], lookup.anomaly);

    return {f_norm, lookup, decision, snap->version};
}

int main() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

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

    AttractorBridge bridge;
    AttractorSnapshot* initial = bridge.staging();
    initial->n_centroids = K;
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < CENTROID_DIM; ++d) initial->centroids[k][d] = dist(rng) * 10.0f;
        initial->epsilon[k] = eps_dist(rng);
        initial->regime_id[k] = regime_dist(rng);
    }
    bridge.publish(1);

    alignas(32) float query_5d[CENTROID_DIM];
    for (int d = 0; d < CENTROID_DIM; ++d) query_5d[d] = dist(rng) * 10.0f;

    const int n_warmup = 1000;
    const int n_trials = 1000000;

    for (int i = 0; i < n_warmup; ++i) {
        run_full_tick(bridge, query_5d);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        FullTickResult r = run_full_tick(bridge, query_5d);
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

    FullTickResult final_r = run_full_tick(bridge, query_5d);
    const char* decision_names[] = {"CANCEL_ALL", "MARKET_TAKER", "LIQUIDITY_PROVISION", "HOLD"};

    printf("FULL tick-to-trade WITH AttractorBridge (Layer0+1+2+4, reading from lock-free snapshot):\n");
    printf("  mean=%.4f us  p50=%.4f us  p99=%.4f us  p999=%.4f us\n", mean, p50, p99, p999);
    printf("  commutator_norm=%.4f  best_idx=%d  anomaly=%d  decision=%s  snapshot_version=%lu (sanity check)\n",
           final_r.commutator_norm, final_r.regime_lookup.best_idx,
           final_r.regime_lookup.anomaly, decision_names[(int)final_r.decision],
           final_r.snapshot_version_used);

    return 0;
}
