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
#include <fstream>
#include "orderbook_state.hpp"
#include "kmeans_calibration.hpp"
#include "regime_labeling.hpp"
#include "query_ring_buffer.hpp"
#include "live_attractor_bridge.hpp"
#include "layer5_execution/order.hpp"
#include "layer5_execution/shadow_matcher.hpp"

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
constexpr int TRADE_BATCH_INTERVAL_MS = 3;
constexpr double TICK_SIZE = 0.1;
constexpr double ORDER_SIZE = 0.001;

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
std::atomic<uint64_t> vortex_cancel_calls{0};
std::atomic<uint64_t> impulse_replace_calls{0};

ShadowMatcher matcher;
std::atomic<uint64_t> current_tick_global{0};

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

// STRICT MAKER EXECUTION MODEL (Sniper Maker paradigm)
// place_market_order is NEVER called in this file. Zero Taker Fee Rule.
void execute_decision(Decision decision, float directed_pressure_proxy, const BookUpdate& update) {
    uint64_t tick = current_tick_global.load(std::memory_order_relaxed);
    double best_bid = update.P[0];
    double best_ask = update.P[10];
    double bid_vol = update.dV[0];
    double ask_vol = update.dV[10];

    switch (decision) {
        case Decision::CANCEL_ALL: {
            // VORTEX shield: immediate cancel, no debounce. This is our defense
            // against toxic flow, so it must fire without delay.
            vortex_cancel_calls.fetch_add(1, std::memory_order_relaxed);
            matcher.cancel_all(tick);
            break;
        }

        case Decision::MARKET_TAKER: {
            // IMPULSE regime, reinterpreted as "Sniper Maker": we never cross
            // the spread. We post a price-improved limit order (one tick
            // better than current best) on the side we expect to be filled
            // from, which makes us the new best price -> queue_position = 0.
            //
            // Debounce: if we already have an open order on the CORRECT side,
            // do nothing -- leave it resting. Only cancel and re-place if the
            // desired side has flipped (signal reversed) or we have no open
            // order at all. Without this, IMPULSE persisting across multiple
            // ticks (the normal case) causes cancel-and-replace churn every
            // tick, giving the order zero time to ever be reached by trade
            // flow (this was tested and confirmed: 513 cancels vs 1 fill).
            bool expect_up = directed_pressure_proxy > 0;
            Side desired_side = expect_up ? Side::BUY : Side::SELL;

            Side current_open_side;
            bool has_open_order = matcher.get_single_open_order_side(current_open_side);

            bool need_replace = !has_open_order || (current_open_side != desired_side);

            if (need_replace) {
                impulse_replace_calls.fetch_add(1, std::memory_order_relaxed);
                matcher.cancel_all(tick);

                double improved_price;
                if (desired_side == Side::BUY) {
                    improved_price = best_bid + TICK_SIZE;
                    if (improved_price >= best_ask) improved_price = best_ask - TICK_SIZE;
                } else {
                    improved_price = best_ask - TICK_SIZE;
                    if (improved_price <= best_bid) improved_price = best_bid + TICK_SIZE;
                }

                matcher.place_limit_order(desired_side, improved_price, ORDER_SIZE, 0.0, tick);
            }
            break;
        }

        case Decision::LIQUIDITY_PROVISION: {
            // EQUILIBRIUM regime: post both sides. Thin-level quoting (scan
            // top-3 levels, join the level with least resting volume) to
            // reduce, though not eliminate, queue depth ahead of us.
            if (matcher.n_open_orders() == 0) {
                constexpr int SCAN_LEVELS = 3;

                int best_bid_level = 0;
                double min_bid_vol = update.dV[0];
                for (int lvl = 1; lvl < SCAN_LEVELS; ++lvl) {
                    if (update.dV[lvl] < min_bid_vol) {
                        min_bid_vol = update.dV[lvl];
                        best_bid_level = lvl;
                    }
                }

                int best_ask_level = 0;
                double min_ask_vol = update.dV[10];
                for (int lvl = 1; lvl < SCAN_LEVELS; ++lvl) {
                    if (update.dV[10 + lvl] < min_ask_vol) {
                        min_ask_vol = update.dV[10 + lvl];
                        best_ask_level = lvl;
                    }
                }

                double bid_price = update.P[best_bid_level];
                double ask_price = update.P[10 + best_ask_level];

                matcher.place_limit_order(Side::BUY, bid_price, ORDER_SIZE, std::max((double)min_bid_vol, 0.001), tick);
                matcher.place_limit_order(Side::SELL, ask_price, ORDER_SIZE, std::max((double)min_ask_vol, 0.001), tick);
            }
            break;
        }

        case Decision::HOLD: {
            // No new action; leave any resting orders as-is.
            break;
        }
    }
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

void depth_thread_fn() {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";
    const std::string target = "/public/ws/btcusdt@depth@100ms";

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
        printf("[depth] connected to %s%s\n", host.c_str(), target.c_str());

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
        fprintf(stderr, "[depth] ERROR: %s\n", e.what());
    }
}

void trade_thread_fn() {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";
    const std::string target = "/market/ws/btcusdt@aggTrade";

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
        printf("[trade] connected to %s%s\n", host.c_str(), target.c_str());

        std::vector<TradeEvent> batch;
        auto last_flush = std::chrono::steady_clock::now();

        while (!shutdown_flag.load(std::memory_order_relaxed)) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            std::string raw = beast::buffers_to_string(buffer.data());
            json j = json::parse(raw);

            double price = std::stod(j["p"].get<std::string>());
            double qty = std::stod(j["q"].get<std::string>());
            bool buyer_is_maker = j["m"].get<bool>();

            TradeEvent ev;
            ev.price = price;
            ev.quantity = qty;
            ev.sell_side_aggressor = buyer_is_maker;
            ev.tick = current_tick_global.load(std::memory_order_relaxed);
            batch.push_back(ev);

            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush).count();

            if (elapsed_ms >= TRADE_BATCH_INTERVAL_MS && !batch.empty()) {
                matcher.on_trade_batch(batch);
                batch.clear();
                last_flush = now;
            }
        }

        if (!batch.empty()) {
            matcher.on_trade_batch(batch);
        }

        ws.close(websocket::close_code::normal);
    } catch (std::exception const& e) {
        fprintf(stderr, "[trade] ERROR: %s\n", e.what());
    }
}

void refit_thread_fn() {
    uint64_t version_counter = 1;
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
    printf("[hot-path] initial calibration published as version 0. entering LIVE decision phase (STRICT MAKER MODE)\n\n");

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
            float severity_ratio = (snap->epsilon[best_idx] > 0.0f) ? (best_dist / snap->epsilon[best_idx]) : 0.0f;
            if (anomaly) {
                printf("[SEVERITY_DEBUG] tick=%lu best_dist=%.4f epsilon=%.4f ratio=%.4f\n",
                       messages_processed.load() + 1, best_dist, snap->epsilon[best_idx], severity_ratio);
            }
            Decision decision = decide(snap->regime[best_idx], anomaly);

            uint64_t count = messages_processed.fetch_add(1, std::memory_order_relaxed) + 1;
            current_tick_global.store(count, std::memory_order_relaxed);

            tick_count_since_last_refit++;
            ticks_since_any_trigger++;
            bool in_cooldown = ticks_since_any_trigger < COOLDOWN_TICKS;
            bool refit_busy = refit_in_progress.load(std::memory_order_acquire) || refit_requested.load(std::memory_order_acquire);

            if (anomaly && !in_cooldown && !refit_busy) {
                refit_requested.store(true, std::memory_order_release);
                refit_trigger_count.fetch_add(1, std::memory_order_relaxed);
                tick_count_since_last_refit = 0;
                ticks_since_any_trigger = 0;
            } else if (anomaly && in_cooldown) {
                anomaly_ignored_cooldown_count.fetch_add(1, std::memory_order_relaxed);
            } else if (tick_count_since_last_refit >= DRIFT_REFIT_TICK_INTERVAL && !refit_busy) {
                refit_requested.store(true, std::memory_order_release);
                refit_trigger_count.fetch_add(1, std::memory_order_relaxed);
                tick_count_since_last_refit = 0;
                ticks_since_any_trigger = 0;
            }

            execute_decision(decision, query[1], update);

            if (count % 200 == 0) {
                Position pos = matcher.position();
                printf("[tick %lu] decision=%s open_orders=%zu filled=%zu net_pnl=%.6f fees=%.6f\n",
                       count, decision_names[(int)decision], matcher.n_open_orders(),
                       matcher.n_filled_orders(), pos.net_pnl(), pos.fees_paid_total);
            }
        }
    }
}

int main() {
    printf("=== TOPOS Sniper Maker: Zero Taker Fee Rule enforced, strict maker-only execution ===\n");
    printf("WARNING: order book has no REST snapshot bootstrap; TICK_SIZE=0.1 is an assumed constant, not queried from exchange\n\n");

    std::thread depth_thread(depth_thread_fn);
    std::thread trade_thread(trade_thread_fn);
    std::thread refit_thread(refit_thread_fn);
    std::thread hot_path_thread(hot_path_thread_fn);

    std::this_thread::sleep_for(std::chrono::seconds(60));
    shutdown_flag.store(true, std::memory_order_relaxed);

    depth_thread.join();
    trade_thread.join();
    refit_thread.join();
    hot_path_thread.join();

    matcher.cancel_all(current_tick_global.load());

    Position final_pos = matcher.position();

    printf("\n=== SHADOW SESSION FINAL REPORT (STRICT MAKER) ===\n");
    printf("total ticks processed: %lu\n", messages_processed.load());
    printf("refit triggers fired: %lu\n", refit_trigger_count.load());
    printf("anomalies ignored (cooldown): %lu\n", anomaly_ignored_cooldown_count.load());
    printf("DEBUG: vortex cancel_all calls: %lu\n", vortex_cancel_calls.load());
    printf("DEBUG: impulse replace calls: %lu\n", impulse_replace_calls.load());
    printf("\n--- OMS / PnL ---\n");
    printf("filled orders: %zu\n", matcher.n_filled_orders());
    printf("canceled orders: %zu\n", matcher.n_canceled_orders());
    printf("open orders at shutdown: %zu\n", matcher.n_open_orders());
    printf("net position quantity: %.6f\n", final_pos.net_quantity);
    printf("avg entry price: %.2f\n", final_pos.avg_entry_price);
    printf("realized PnL (gross): %.6f\n", final_pos.realized_pnl);
    printf("total fees paid: %.6f\n", final_pos.fees_paid_total);
    printf("NET PnL: %.6f\n", final_pos.net_pnl());

    int maker_fills = 0, taker_fills = 0;
    for (auto& o : matcher.filled_orders()) {
        if (o.is_maker()) maker_fills++; else taker_fills++;
    }
    int total_fills = maker_fills + taker_fills;
    printf("maker fills: %d  taker fills: %d  maker_ratio=%.1f%%\n",
           maker_fills, taker_fills, total_fills > 0 ? 100.0 * maker_fills / total_fills : 0.0);

    return 0;
}
