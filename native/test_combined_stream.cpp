#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

int main() {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";
    const std::string target = "/stream?streams=btcusdt@depth@100ms/btcusdt@aggTrade";

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
        std::cout << "connected to combined stream: " << target << "\n\n";

        int depth_count = 0;
        int trade_count = 0;

        for (int i = 0; i < 100; ++i) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            std::string raw = beast::buffers_to_string(buffer.data());
            json j = json::parse(raw);

            std::string stream_name = j["stream"].get<std::string>();
            json data = j["data"];

            if (stream_name.find("depth") != std::string::npos) {
                depth_count++;
            } else if (stream_name.find("aggTrade") != std::string::npos) {
                trade_count++;
                if (trade_count <= 3) {
                    double price = std::stod(data["p"].get<std::string>());
                    double qty = std::stod(data["q"].get<std::string>());
                    bool is_buyer_maker = data["m"].get<bool>();
                    std::cout << "TRADE sample: price=" << price << " qty=" << qty
                              << " buyer_is_maker=" << is_buyer_maker << "\n";
                }
            }
        }

        std::cout << "\nafter 100 combined messages: depth_updates=" << depth_count
                  << " trades=" << trade_count << "\n";

        ws.close(websocket::close_code::normal);

    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
