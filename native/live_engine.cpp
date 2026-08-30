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
#include <thread>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "orderbook_state.hpp"

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

static_assert(std::is_trivially_copyable<BookUpdate>::value, "BookUpdate must be trivially copyable for spsc_queue");

boost::lockfree::spsc_queue<BookUpdate, boost::lockfree::capacity<1024>> ring_buffer;
std::atomic<bool> shutdown_flag{false};
std::atomic<uint64_t> messages_processed{0};

constexpr int N = 20;
constexpr int NP = 24;
constexpr int K = 100;
constexpr float EPS_STABILIZER = 0.01f;

enum class Decision { CANCEL_ALL, MARKET_TAKER, LIQUIDITY_PROVISION, HOLD };
const char* decision_names[] = {"CANCEL_ALL", "MARKET_TAKER", "LIQUIDITY_PROVISION", "HOLD"};

alignas(64) float A[NP * NP];
alignas(64) float A_prev[NP * NP];
alignas(64) float dA[NP * NP];
alignas(64) float F[NP * NP];
alignas(64) float tmp1[NP * NP];
alignas(64) float tmp2[NP * NP];

alignas(64) float centroids[K][5];
alignas(64) float epsilon_arr[K];
alignas(64) int centroid_regime_id[K];

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

inline int centroid_lookup(const float* query_5d, float* out_dist) {
    int best_idx = 0;
    float best_dist_sq = 1e18f;
    for (int k = 0; k < K; ++k) {
        float sum_sq = 0.0f;
        for (int d = 0; d < 5; ++d) {
            float diff = centroids[k][d] - query_5d[d];
            sum_sq += diff * diff;
        }
        if (sum_sq < best_dist_sq) {
            best_dist_sq = sum_sq;
            best_idx = k;
        }
    }
    *out_dist = std::sqrt(best_dist_sq);
    return best_idx;
}

inline Decision decide(int regime_id, bool anomaly) {
    if (anomaly) return Decision::CANCEL_ALL;
    if (regime_id == 1) return Decision::MARKET_TAKER;
    if (regime_id == 0) return Decision::LIQUIDITY_PROVISION;
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

            if (top_bids.size() < 10 || top_asks.size() < 10) {
                continue;
            }

            BookUpdate update;
            for (size_t k = 0; k < 10; ++k) {
                update.P[k] = static_cast<float>(top_bids[k].price);
                update.dV[k] = static_cast<float>(top_bids[k].quantity);
            }
            for (size_t k = 0; k < 10; ++k) {
                update.P[10 + k] = static_cast<float>(top_asks[k].price);
                update.dV[10 + k] = static_cast<float>(top_asks[k].quantity);
            }

            while (!ring_buffer.push(update) && !shutdown_flag.load(std::memory_order_relaxed)) {
            }
        }

        ws.close(websocket::close_code::normal);

    } catch (std::exception const& e) {
        fprintf(stderr, "[network] ERROR: %s\n", e.what());
    }
}

void hot_path_thread_fn() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> eps_dist(0.5f, 5.0f);
    std::uniform_int_distribution<int> regime_dist(0, 2);

    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < 5; ++d) centroids[k][d] = dist(rng) * 50.0f;
        epsilon_arr[k] = eps_dist(rng);
        centroid_regime_id[k] = regime_dist(rng);
    }

    bool first_tick = true;
    BookUpdate update;

    while (!shutdown_flag.load(std::memory_order_relaxed)) {
        if (ring_buffer.pop(update)) {
            if (first_tick) {
                std::memcpy(A_prev, A, sizeof(A));
                build_adjacency_matrix_avx(update.dV, update.P);
                std::memcpy(A_prev, A, sizeof(A));
                first_tick = false;
                continue;
            }

            std::memcpy(A_prev, A, sizeof(A));
            build_adjacency_matrix_avx(update.dV, update.P);

            float commutator_norm = compute_commutator_norm();

            float query_5d[5] = {
                commutator_norm,
                update.dV[0] - update.dV[10],
                update.P[10] - update.P[0],
                update.dV[0],
                update.dV[10]
            };

            float best_dist;
            int best_idx = centroid_lookup(query_5d, &best_dist);
            bool anomaly = best_dist > epsilon_arr[best_idx];
            Decision decision = decide(centroid_regime_id[best_idx], anomaly);

            uint64_t count = messages_processed.fetch_add(1, std::memory_order_relaxed) + 1;

            printf("[tick %lu] mid=%.2f spread=%.2f commutator=%.4f decision=%s\n",
                   count,
                   (update.P[0] + update.P[10]) / 2.0f,
                   update.P[10] - update.P[0],
                   commutator_norm,
                   decision_names[(int)decision]);
        }
    }
}

int main() {
    printf("=== TOPOS Live Engine: Network thread + Hot-Path thread, SPSC bridge ===\n");
    printf("WARNING: order book has no REST snapshot bootstrap, first ~100-200 messages may be incomplete\n\n");

    std::thread network_thread(network_thread_fn);
    std::thread hot_path_thread(hot_path_thread_fn);

    std::this_thread::sleep_for(std::chrono::seconds(15));
    shutdown_flag.store(true, std::memory_order_relaxed);

    network_thread.join();
    hot_path_thread.join();

    printf("\n=== shutdown, total ticks processed: %lu ===\n", messages_processed.load());

    return 0;
}
