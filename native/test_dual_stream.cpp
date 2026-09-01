#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <string>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

std::atomic<uint64_t> depth_count{0};
std::atomic<uint64_t> trade_count{0};
std::atomic<bool> shutdown_flag{false};

void stream_thread(const std::string& target, std::atomic<uint64_t>& counter, const char* label) {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";

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
        printf("[%s] connected\n", label);

        while (!shutdown_flag.load(std::memory_order_relaxed)) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            counter.fetch_add(1, std::memory_order_relaxed);
        }
        ws.close(websocket::close_code::normal);
    } catch (std::exception const& e) {
        fprintf(stderr, "[%s] ERROR: %s\n", label, e.what());
    }
}

int main() {
    std::thread depth_thread(stream_thread, "/public/ws/btcusdt@depth@100ms", std::ref(depth_count), "depth");
    std::thread trade_thread(stream_thread, "/market/ws/btcusdt@aggTrade", std::ref(trade_count), "trade");

    std::this_thread::sleep_for(std::chrono::seconds(20));
    shutdown_flag.store(true, std::memory_order_relaxed);

    depth_thread.join();
    trade_thread.join();

    printf("\nafter 20 seconds concurrently: depth_messages=%lu trade_messages=%lu\n",
           depth_count.load(), trade_count.load());

    return 0;
}
