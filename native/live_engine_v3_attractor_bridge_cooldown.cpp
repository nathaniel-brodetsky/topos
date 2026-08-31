#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <nlohmann/json.hpp>
#include <immintrin.h>
#include <sched.h>
#include <pthread.h>
#include <dlfcn.h>
#include <thread>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <array>
#include <chrono>
#include "orderbook_state.hpp"
#include "kmeans_calibration.hpp"
#include "regime_labeling.hpp"
#include "query_ring_buffer.hpp"
#include "live_attractor_bridge.hpp"

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

struct BookUpdate {
    float dV[20];
    float P[20];
};
static_assert(std::is_trivially_copyable<BookUpdate>::value, "BookUpdate must be trivially copyable");

boost::lockfree::spsc_queue<BookUpdate, boost::lockfree::capacity<4096>> ring_buffer;
std::atomic<bool> shutdown_flag{false};
std::atomic<uint64_t> messages_processed{0};

constexpr int N = 20;
constexpr int NP = 24;
constexpr int K = 20;
constexpr float EPS_STABILIZER = 0.01f;
constexpr int INITIAL_CALIBRATION_SAMPLES = 100;
constexpr uint64_t DRIFT_REFIT_TICK_INTERVAL = 10000;
constexpr uint64_t COOLDOWN_TICKS = 200;

enum class Decision { CANCEL_ALL, MARKET_TAKER, LIQUIDITY_PROVISION, HOLD };
const char* decision_names[] = {"CANCEL_ALL", "MARKET_TAKER", "LIQUIDITY_PROVISION", "HOLD"};

alignas(64) float A[NP * NP];
alignas(64) float A_prev[NP * NP];
alignas(64) float dA[NP * NP];
alignas(64) float F[NP * NP];
alignas(64) float tmp1[NP * NP];
alignas(64) float tmp2[NP * NP];

QueryRingBuffer query_ring;
LiveAttractorBridge attractor_bridge;
std::atomic<bool> refit_requested{false};
std::atomic<bool> refit_in_progress{false};
std::atomic<uint64_t> refit_trigger_count{0};
std::atomic<uint64_t> anomaly_ignored_cooldown_count{0};

union Entry { int missing; float fvalue; int qvalue; };
typedef void (*predict_fn_t)(Entry*, int, float*);
predict_fn_t g_value_predict_fn = nullptr;
void* g_value_model_handle = nullptr;

bool load_value_model() {
    g_value_model_handle = dlopen("/tmp/live_value_model.so", RTLD_NOW);
    if (!g_value_model_handle) {
        fprintf(stderr, "[hot-path] WARNING: failed to load value model: %s\n", dlerror());
        return false;
    }
    g_value_predict_fn = (predict_fn_t)dlsym(g_value_model_handle, "predict");
    if (!g_value_predict_fn) {
        fprintf(stderr, "[hot-path] WARNING: failed to find predict symbol: %s\n", dlerror());
        return false;
    }
    printf("[hot-path] value model loaded via tl2cgen dlopen/dlsym\n");
    return true;
}

inline void build_adjacency_matrix_avx(const float* dV, const float* P) {
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
    for (int i = 0; i < NP * NP; i += 8) _mm256_store_ps(out + i, _mm256_setzero_ps());
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

inline std::array<float, CALIB_DIM> compute_query_vector(const BookUpdate& update, float commutator_norm) {
    return {
        commutator_norm,
        update.dV[0] - update.dV[10],
        update.P[10] - update.P[0],
        update.dV[0],
        update.dV[10]
    };
}

inline int centroid_lookup(const LiveAttractorSnapshot* snap, const std::array<float, CALIB_DIM>& query, float* out_dist) {
    int best_idx = 0;
    float best_dist_sq = 1e18f;
    for (int k = 0; k < snap->n_centroids; ++k) {
        float sum_sq = 0.0f;
        for (int d = 0; d < CALIB_DIM; ++d) {
            float diff = snap->centroid(k, d) - query[d];
            sum_sq += diff * diff;
        }
        if (sum_sq < best_dist_sq) { best_dist_sq = sum_sq; best_idx = k; }
    }
    *out_dist = std::sqrt(best_dist_sq);
    return best_idx;
}

inline Decision decide(SemanticRegime regime, bool anomaly) {
    if (anomaly) return Decision::CANCEL_ALL;
    if (regime == SemanticRegime::IMPULSE) return Decision::MARKET_TAKER;
    if (regime == SemanticRegime::EQUILIBRIUM) return Decision::LIQUIDITY_PROVISION;
    return Decision::HOLD;
}

void apply_depth_update(OrderBookState& book, const json& j) {
    for (auto& level : j["b"]) {
        double price = std::stod(level[0].get<std::string>());
        double quantity = std::stod(level[1].get<std::string>());
        book.apply_bid_update(price, quantity);
    }
    for (auto& level : j["a"]) {
        double price = std::stod(level[0].get<std::string>());
        double quantity = std::stod(level[1].get<std::string>());
        book.apply_ask_update(price, quantity);
    }
}

void network_thread_fn() {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";
    const std::string target = "/ws/btcusdt@depth@100ms";

    try {
        net::io_context ioc;
        ssl::context ctx{ssl::context::tlsv12_client};
        ctx.set_default_verify_paths();
        tcp::resolver resolver{ioc};
        websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws{ioc, ctx};
        auto const results = resolver.resolve(host, port);
        beast::get_lowest_layer(ws).connect(results);

        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host.c_str())) {
            throw beast::system_error(
                beast::error_code(static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()),
                "Failed to set SNI hostname");
        }
        ws.next_layer().handshake(ssl::stream_base::client);
        ws.handshake(host, target);
        printf("[network] connected to %s%s\n", host.c_str(), target.c_str());

        OrderBookState book;
        while (!shutdown_flag.load(std::memory_order_relaxed)) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            std::string raw = beast::buffers_to_string(buffer.data());
            json j = json::parse(raw);
            apply_depth_update(book, j);

            auto top_bids = book.top_bids(10);
            auto top_asks = book.top_asks(10);
            if (top_bids.size() < 10 || top_asks.size() < 10) continue;

            BookUpdate update;
            for (size_t k = 0; k < 10; ++k) {
                update.P[k] = static_cast<float>(top_bids[k].price);
                update.dV[k] = static_cast<float>(top_bids[k].quantity);
            }
            for (size_t k = 0; k < 10; ++k) {
                update.P[10 + k] = static_cast<float>(top_asks[k].price);
                update.dV[10 + k] = static_cast<float>(top_asks[k].quantity);
            }
            while (!ring_buffer.push(update) && !shutdown_flag.load(std::memory_order_relaxed)) {}
        }
        ws.close(websocket::close_code::normal);
    } catch (std::exception const& e) {
        fprintf(stderr, "[network] ERROR: %s\n", e.what());
    }
}

void refit_thread_fn() {
    uint64_t version_counter = 1;
    std::mt19937 unused_rng(1);

    while (!shutdown_flag.load(std::memory_order_relaxed)) {
        if (refit_requested.load(std::memory_order_acquire)) {
            refit_in_progress.store(true, std::memory_order_release);

            auto data = query_ring.snapshot();
            if (data.size() >= 50) {
                int k = std::min(K, static_cast<int>(data.size() / 5));
                CalibrationResult result = run_kmeans_calibration(data, k, 15, 0.90f);
                auto regime = label_centroids_by_pressure(data, result.final_assignment, k);

                attractor_bridge.publish(result.centroids, result.epsilon, regime, version_counter);
                printf("[refit] published snapshot version=%lu, k=%d, samples_used=%zu\n",
                       version_counter, k, data.size());
                version_counter++;
            } else {
                printf("[refit] skipped: insufficient ring buffer samples (%zu < 50)\n", data.size());
            }

            refit_requested.store(false, std::memory_order_release);
            refit_in_progress.store(false, std::memory_order_release);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void hot_path_thread_fn() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    bool value_model_available = load_value_model();
    bool first_tick = true;
    BookUpdate update;
    uint64_t tick_count_since_last_refit = 0;
    uint64_t ticks_since_any_trigger = COOLDOWN_TICKS;

    printf("[hot-path] entering INITIAL CALIBRATION phase, collecting %d samples...\n", INITIAL_CALIBRATION_SAMPLES);

    std::vector<std::array<float, CALIB_DIM>> initial_data;
    while (!shutdown_flag.load(std::memory_order_relaxed) && (int)initial_data.size() < INITIAL_CALIBRATION_SAMPLES) {
        if (ring_buffer.pop(update)) {
            if (first_tick) {
                build_adjacency_matrix_avx(update.dV, update.P);
                std::memcpy(A_prev, A, sizeof(A));
                first_tick = false;
                continue;
            }
            std::memcpy(A_prev, A, sizeof(A));
            build_adjacency_matrix_avx(update.dV, update.P);
            float cn = compute_commutator_norm();
            auto q = compute_query_vector(update, cn);
            initial_data.push_back(q);
            query_ring.push(q);
        }
    }
    if (shutdown_flag.load(std::memory_order_relaxed)) return;

    printf("[hot-path] running initial k-means on %zu samples...\n", initial_data.size());
    CalibrationResult initial_result = run_kmeans_calibration(initial_data, K, 20, 0.90f);
    auto initial_regime = label_centroids_by_pressure(initial_data, initial_result.final_assignment, K);
    attractor_bridge.publish(initial_result.centroids, initial_result.epsilon, initial_regime, 0);
    printf("[hot-path] initial calibration published as version 0. entering LIVE decision phase\n\n");

    while (!shutdown_flag.load(std::memory_order_relaxed)) {
        if (ring_buffer.pop(update)) {
            std::memcpy(A_prev, A, sizeof(A));
            build_adjacency_matrix_avx(update.dV, update.P);
            float commutator_norm = compute_commutator_norm();
            auto query = compute_query_vector(update, commutator_norm);
            query_ring.push(query);

            const LiveAttractorSnapshot* snap = attractor_bridge.current();
            float best_dist;
            int best_idx = centroid_lookup(snap, query, &best_dist);
            bool anomaly = best_dist > snap->epsilon[best_idx];
            Decision decision = decide(snap->regime[best_idx], anomaly);

            tick_count_since_last_refit++;
            ticks_since_any_trigger++;

            bool in_cooldown = ticks_since_any_trigger < COOLDOWN_TICKS;
            bool refit_busy = refit_in_progress.load(std::memory_order_acquire) || refit_requested.load(std::memory_order_acquire);

            bool should_trigger_refit = false;
            const char* trigger_reason = "";

            if (anomaly && !in_cooldown && !refit_busy) {
                should_trigger_refit = true;
                trigger_reason = "ANOMALY";
            } else if (anomaly && in_cooldown) {
                anomaly_ignored_cooldown_count.fetch_add(1, std::memory_order_relaxed);
            } else if (tick_count_since_last_refit >= DRIFT_REFIT_TICK_INTERVAL && !refit_busy) {
                should_trigger_refit = true;
                trigger_reason = "DRIFT_TTL";
            }

            if (should_trigger_refit) {
                refit_requested.store(true, std::memory_order_release);
                refit_trigger_count.fetch_add(1, std::memory_order_relaxed);
                tick_count_since_last_refit = 0;
                ticks_since_any_trigger = 0;
                printf("[hot-path] >>> REFIT TRIGGERED (%s) at tick %lu, snapshot_version_before=%lu\n",
                       trigger_reason, messages_processed.load(), snap->version);
            }

            float predicted_value = 0.0f;
            if (value_model_available) {
                Entry entry_data[5];
                entry_data[0].fvalue = commutator_norm;
                entry_data[1].fvalue = query[1];
                entry_data[2].fvalue = query[2];
                entry_data[3].fvalue = query[3];
                entry_data[4].fvalue = query[4];
                float result[1];
                g_value_predict_fn(entry_data, 0, result);
                predicted_value = result[0];
            }

            uint64_t count = messages_processed.fetch_add(1, std::memory_order_relaxed) + 1;

            printf("[tick %lu] mid=%.2f snap_v=%lu commutator=%.2f dist=%.2f eps=%.2f decision=%s value_pred=%.2f\n",
                   count, (update.P[0] + update.P[10]) / 2.0f, snap->version,
                   commutator_norm, best_dist, snap->epsilon[best_idx],
                   decision_names[(int)decision], predicted_value);
        }
    }
}

int main() {
    printf("=== TOPOS Live Engine v3: AttractorBridge with hybrid (anomaly+drift) refit trigger ===\n");
    printf("WARNING: order book has no REST snapshot bootstrap (geo-blocked on this infra, logic verified via synthetic test)\n\n");

    std::thread network_thread(network_thread_fn);
    std::thread refit_thread(refit_thread_fn);
    std::thread hot_path_thread(hot_path_thread_fn);

    std::this_thread::sleep_for(std::chrono::seconds(90));
    shutdown_flag.store(true, std::memory_order_relaxed);

    network_thread.join();
    refit_thread.join();
    hot_path_thread.join();

    printf("\n=== shutdown ===\n");
    printf("total live-phase ticks processed: %lu\n", messages_processed.load());
    printf("total refit triggers fired: %lu\n", refit_trigger_count.load());
    printf("total anomalies ignored due to cooldown: %lu\n", anomaly_ignored_cooldown_count.load());
    printf("effective trigger rate: %.2f%%\n", 100.0 * refit_trigger_count.load() / (double)messages_processed.load());
    printf("final attractor snapshot version: %lu\n", attractor_bridge.current()->version);

    return 0;
}
