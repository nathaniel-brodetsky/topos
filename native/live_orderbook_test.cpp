#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include "orderbook_state.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

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

int main() {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";
    const std::string target = "/ws/btcusdt@depth@100ms";

    std::cout << "NOTE: this test skips the required REST snapshot bootstrap step;\n";
    std::cout << "the resulting order book is INCOMPLETE and not production-valid,\n";
    std::cout << "this only verifies that JSON parsing and top-N extraction work.\n\n";

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
        std::cout << "connected, subscribed to " << target << "\n\n";

        OrderBookState book;

        const int n_messages = 20;
        for (int i = 0; i < n_messages; ++i) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            std::string raw = beast::buffers_to_string(buffer.data());

            json j = json::parse(raw);
            apply_depth_update(book, j);

            if (i == n_messages - 1) {
                std::cout << "=== FINAL STATE after " << n_messages << " diff messages ===\n";
                std::cout << "total bid_levels=" << book.n_bid_levels()
                          << " total ask_levels=" << book.n_ask_levels() << "\n\n";

                float dV[20];
                float P[20];

                auto top_bids = book.top_bids(10);
                auto top_asks = book.top_asks(10);

                std::cout << "top " << top_bids.size() << " bids:\n";
                for (size_t k = 0; k < top_bids.size(); ++k) {
                    P[k] = static_cast<float>(top_bids[k].price);
                    dV[k] = static_cast<float>(top_bids[k].quantity);
                    std::cout << "  [" << k << "] price=" << P[k] << " qty=" << dV[k] << "\n";
                }

                std::cout << "top " << top_asks.size() << " asks:\n";
                for (size_t k = 0; k < top_asks.size(); ++k) {
                    size_t idx = 10 + k;
                    P[idx] = static_cast<float>(top_asks[k].price);
                    dV[idx] = static_cast<float>(top_asks[k].quantity);
                    std::cout << "  [" << idx << "] price=" << P[idx] << " qty=" << dV[idx] << "\n";
                }

                if (top_bids.size() < 10 || top_asks.size() < 10) {
                    std::cout << "\nWARNING: fewer than 10 levels on one side; this is expected\n";
                    std::cout << "given we skipped the REST snapshot bootstrap.\n";
                }
            }
        }

        ws.close(websocket::close_code::normal);

    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
