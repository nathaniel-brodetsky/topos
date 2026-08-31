#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include "orderbook_state.hpp"
#include "rest_snapshot.hpp"
#include <iostream>
#include <vector>
#include <string>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

struct BufferedEvent {
    uint64_t U, u, pu;
    json raw;
};

void apply_full_snapshot(OrderBookState& book, const DepthSnapshot& snap) {
    book.clear();
    for (auto& level : snap.raw_bids) {
        double price = std::stod(level[0].get<std::string>());
        double qty = std::stod(level[1].get<std::string>());
        if (qty > 0.0) book.apply_bid_update(price, qty);
    }
    for (auto& level : snap.raw_asks) {
        double price = std::stod(level[0].get<std::string>());
        double qty = std::stod(level[1].get<std::string>());
        if (qty > 0.0) book.apply_ask_update(price, qty);
    }
}

void apply_diff_event(OrderBookState& book, const json& j) {
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

int main() {
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
        std::cout << "[step 1] WS connected, buffering events before applying snapshot\n";

        std::vector<BufferedEvent> event_buffer;
        const int buffer_count = 30;

        for (int i = 0; i < buffer_count; ++i) {
            beast::flat_buffer buf;
            ws.read(buf);
            std::string raw = beast::buffers_to_string(buf.data());
            json j = json::parse(raw);

            BufferedEvent ev;
            ev.U = j["U"].get<uint64_t>();
            ev.u = j["u"].get<uint64_t>();
            ev.pu = j["pu"].get<uint64_t>();
            ev.raw = j;
            event_buffer.push_back(ev);
        }

        std::cout << "[step 2] buffered " << event_buffer.size() << " events (U range: "
                  << event_buffer.front().U << " - " << event_buffer.back().u << ")\n";

        std::cout << "[step 3] fetching REST snapshot...\n";
        DepthSnapshot snapshot = fetch_depth_snapshot("BTCUSDT");
        std::cout << "[step 3] snapshot lastUpdateId=" << snapshot.last_update_id << "\n";

        OrderBookState book;
        apply_full_snapshot(book, snapshot);
        std::cout << "[step 4] applied snapshot: bid_levels=" << book.n_bid_levels()
                  << " ask_levels=" << book.n_ask_levels() << "\n";

        int discarded = 0;
        int applied = 0;
        bool first_applied = false;
        uint64_t last_u = 0;
        bool gap_detected = false;

        for (auto& ev : event_buffer) {
            if (ev.u <= snapshot.last_update_id) {
                discarded++;
                continue;
            }

            if (!first_applied) {
                if (!(ev.U <= snapshot.last_update_id + 1 && snapshot.last_update_id + 1 <= ev.u)) {
                    std::cerr << "[WARNING] first applicable event does not bracket lastUpdateId+1 as expected: "
                              << "U=" << ev.U << " u=" << ev.u << " lastUpdateId+1=" << (snapshot.last_update_id + 1) << "\n";
                }
                apply_diff_event(book, ev.raw);
                first_applied = true;
                last_u = ev.u;
                applied++;
                continue;
            }

            if (ev.pu != last_u) {
                std::cerr << "[GAP DETECTED] event pu=" << ev.pu << " does not match last applied u=" << last_u << "\n";
                gap_detected = true;
            }

            apply_diff_event(book, ev.raw);
            last_u = ev.u;
            applied++;
        }

        std::cout << "[step 5] reconciliation complete: discarded=" << discarded
                  << " applied=" << applied << " gap_detected=" << (gap_detected ? "YES" : "no") << "\n";

        auto top_bids = book.top_bids(5);
        auto top_asks = book.top_asks(5);

        std::cout << "\nfinal top-5 bids:\n";
        for (auto& lvl : top_bids) std::cout << "  " << lvl.price << " @ " << lvl.quantity << "\n";
        std::cout << "final top-5 asks:\n";
        for (auto& lvl : top_asks) std::cout << "  " << lvl.price << " @ " << lvl.quantity << "\n";

        std::cout << "\ntotal bid_levels=" << book.n_bid_levels() << " total ask_levels=" << book.n_ask_levels() << "\n";
        std::cout << (top_bids[0].price < top_asks[0].price ? "SANITY OK: best bid < best ask\n" : "SANITY FAILED: crossed book!\n");

        ws.close(websocket::close_code::normal);

    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
